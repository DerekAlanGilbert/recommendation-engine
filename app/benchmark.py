"""Focused Precision@5 benchmark for the one methodology this repo serves.

Runs Targeted Learning — Current Relationships (the exact live stack: the
pretrained Current tower, the targeted `select_probe`, and the baseline
`rank_recommendations`) against five broad preference proxies defined over
frozen EPA attributes, and measures how many swipes (thumbs up or down, each
counted once) the top-5 ranking needs to reach and hold Precision@5 >= 0.80
(at least 4 of 5 matches).

The proxies use honest, data-supported names. EPA data has no exterior
color, price, horsepower, or body-style marketing labels, so:

- "passenger_car" is an EPA car-class proxy (compact/midsize/large and
  similar; not provably sedans),
- "premium_brand" is a brand-set proxy, not a trim-level luxury signal,
- "performance_suv" is a cylinders/trim-token proxy, not measured 0-60.

Every case starts from a cold uniform posterior with empty feedback; probes
are thumbed deterministically by the case predicate; the posterior is
recomputed exactly from the complete history after each swipe. Artifacts are
content-addressed by the configuration hash and land under
``artifacts/benchmarks/targeted-precision-<hash>/`` (generated output, not
source): ``events.csv``, ``summary.json``, and ``precision.png``.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")  # headless: charts render identically with no display

import matplotlib.pyplot as plt

from app.methodology import METHODOLOGY_ID, METHODOLOGY_NAME
from app.model import load_catalog_features, pretrain
from app.preference import (
    COSINE_WEIGHT,
    SHARPNESS,
    THETA_GRID,
    TOKEN_WEIGHT,
    build_engine,
    posterior_from_feedback,
    rank_recommendations,
    select_probe,
)

MAX_SWIPES = 20
TOP_K = 5
# Precision@5 >= 0.80 means at least 4 of 5 matches; comparing integer match
# counts avoids any float-threshold ambiguity.
TARGET_MATCHES = 4
PRECISION_TARGET = TARGET_MATCHES / TOP_K
STABLE_RUN = 3  # consecutive evaluated swipes (swipe 0 may start the run)

MODEL_LABEL = "app.model.pretrain(features, seed=0)"

PREMIUM_MAKES = frozenset({"BMW", "Lexus", "Mercedes-Benz", "Cadillac", "Audi", "Acura"})

# Frozen case-insensitive EPA model-string performance markers for the
# performance-SUV proxy. EPA carries no horsepower or acceleration data, so
# these trim tokens plus a six-cylinder floor are the honest handle.
PERFORMANCE_MARKERS = (
    "AMG", " M", "M40", "M50", "M60", "RS ", "SQ", "SRT", "Trackhawk",
    "Type S", "N Line", " N", "Turbo", "SVR", "Quadrifoglio",
)

CAR_CLASS_CAVEAT = ("EPA vehicle classes group cars by size, not body style: "
                    "'passenger car' can include non-sedans, and EPA has no "
                    "exterior color attribute at all.")


def is_passenger_car(feature):
    vehicle_class = feature["vehicle_class"].lower()
    return ("cars" in vehicle_class
            and "station wagon" not in vehicle_class
            and "two seater" not in vehicle_class)


def is_pickup_truck(feature):
    return "pickup truck" in feature["vehicle_class"].lower()


def is_premium_brand(feature):
    return feature["make"] in PREMIUM_MAKES


def is_electric_passenger_car(feature):
    return (is_passenger_car(feature)
            and feature["electric_range"] is not None
            and feature["electric_range"] > 0
            and feature["cylinders"] is None)


def is_performance_suv(feature):
    if "sport utility" not in feature["vehicle_class"].lower():
        return False
    cylinders = feature["cylinders"]
    if cylinders is not None and cylinders >= 6:
        return True
    model = feature["model"].upper()
    return any(marker.upper() in model for marker in PERFORMANCE_MARKERS)


class CaseSpec(NamedTuple):
    case_id: str
    label: str
    proxy: str
    caveat: str
    predicate: object


CASES = (
    CaseSpec(
        "passenger_car", "Passenger-car shopper",
        "vehicle_class contains 'cars' (case-insensitive), excluding "
        "'station wagon' and 'two seater'",
        CAR_CLASS_CAVEAT, is_passenger_car),
    CaseSpec(
        "pickup_truck", "Pickup-truck shopper",
        "vehicle_class contains 'pickup truck' (case-insensitive)",
        "EPA pickup classes carry no bed/cab or color detail.", is_pickup_truck),
    CaseSpec(
        "premium_brand", "Premium-brand shopper",
        "make in {BMW, Lexus, Mercedes-Benz, Cadillac, Audi, Acura}; all "
        "body types from those makes",
        "Price/luxury tier is unavailable in EPA data; this is a "
        "premium-brand proxy, not a trim-level luxury signal.",
        is_premium_brand),
    CaseSpec(
        "electric_passenger_car", "Electric passenger-car shopper",
        "passenger-car proxy AND electric_range > 0 AND cylinders is None "
        "(battery-electric proxy; excludes combustion and PHEV)",
        CAR_CLASS_CAVEAT + " Battery-electric is proxied by missing "
        "cylinders with positive electric range.", is_electric_passenger_car),
    CaseSpec(
        "performance_suv", "Performance-SUV shopper",
        "vehicle_class contains 'sport utility' AND (cylinders >= 6 OR the "
        "model string contains a frozen case-insensitive performance "
        f"marker: {', '.join(PERFORMANCE_MARKERS)})",
        "Performance-oriented SUV proxy: EPA has no horsepower or 0-60 "
        "data, so sportiness is inferred from cylinders and trim tokens.",
        is_performance_suv),
)


def case_match_counts(features):
    return {
        case.case_id: sum(1 for f in features if case.predicate(f))
        for case in CASES
    }


def first_reach(match_counts, target_matches=TARGET_MATCHES):
    """First swipe index (0 included) with match count >= target, else None."""
    return next((n for n, count in enumerate(match_counts)
                 if count >= target_matches), None)


def stable_reach(match_counts, target_matches=TARGET_MATCHES,
                 stable_run=STABLE_RUN):
    """Earliest start of `stable_run` consecutive qualifying swipes, else None.

    Swipe 0 has no feedback but is an evaluated checkpoint, so a stable run
    may begin at 0 only when the first `stable_run` evaluations all qualify.
    A qualifying tail shorter than `stable_run` cannot confirm stability.
    """
    for start in range(len(match_counts) - stable_run + 1):
        if all(count >= target_matches
               for count in match_counts[start:start + stable_run]):
            return start
    return None


def run_case(engine, features_by_id, predicate, max_swipes=MAX_SWIPES,
             top_k=TOP_K):
    """Deterministic swipe trajectory for one case against the live stack.

    Swipe 0 is the cold start (uniform posterior, no probe); each later swipe
    probes with `select_probe`, thumbs by the case predicate, and recomputes
    the exact posterior from the complete history.
    """
    feedback = []
    rows = []

    def record(swipe, probe_id, liked):
        rated = frozenset(variant_id for variant_id, _ in feedback)
        top = rank_recommendations(engine, posterior, exclude=rated, limit=top_k)
        top_ids = [variant_id for variant_id, _ in top]
        matches = [bool(predicate(features_by_id[variant_id]))
                   for variant_id in top_ids]
        rows.append({
            "swipe": swipe,
            "probe_id": probe_id,
            "liked": liked,
            "ups": sum(1 for _, up in feedback if up),
            "downs": sum(1 for _, up in feedback if not up),
            "top5": top_ids,
            "matches": matches,
            "match_count": sum(matches),
            "precision": sum(matches) / top_k,
        })

    posterior = posterior_from_feedback(engine, feedback)
    record(0, None, None)
    for swipe in range(1, max_swipes + 1):
        probe = select_probe(engine, posterior, feedback)
        if probe is None:  # every variant rated; impossible at 20 of 6,606
            break
        probe_id = probe["variant_id"]
        liked = bool(predicate(features_by_id[probe_id]))
        feedback.append((probe_id, liked))
        posterior = posterior_from_feedback(engine, feedback)
        record(swipe, probe_id, liked)
    return rows


def summarize_case(case, rows, catalog_matches):
    match_counts = [row["match_count"] for row in rows]
    precisions = [row["precision"] for row in rows]
    last = rows[-1]
    return {
        "label": case.label,
        "proxy": case.proxy,
        "caveat": case.caveat,
        "catalog_matches": catalog_matches,
        "first_80_swipe": first_reach(match_counts),
        "stable_80_swipe": stable_reach(match_counts),
        "final_precision": precisions[-1],
        "max_precision": max(precisions),
        "ups": last["ups"],
        "downs": last["downs"],
    }


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


def summarize_overall(case_summaries):
    firsts = [block["first_80_swipe"] for block in case_summaries.values()]
    stables = [block["stable_80_swipe"] for block in case_summaries.values()]
    achieved_first = [n for n in firsts if n is not None]
    achieved_stable = [n for n in stables if n is not None]
    return {
        "cases": len(case_summaries),
        "median_first_80_swipe": _median(achieved_first),
        "worst_first_80_swipe": max(achieved_first, default=None),
        "cases_missing_first_80": firsts.count(None),
        "median_stable_80_swipe": _median(achieved_stable),
        "worst_stable_80_swipe": max(achieved_stable, default=None),
        "cases_missing_stable_80": stables.count(None),
        "note": "medians and worst cases cover only cases that reached the "
                "target within max_swipes; the *_missing_* counts state how "
                "many did not. No weighted overall score is computed.",
    }


# Validated categorical palette (dataviz six-checks, light surface), one
# fixed color per case; distinct markers are the secondary encoding.
CASE_COLORS = {
    "passenger_car": "#2a78d6",
    "pickup_truck": "#eb6834",
    "premium_brand": "#4a3aa7",
    "electric_passenger_car": "#1baf7a",
    "performance_suv": "#e34948",
}

CASE_MARKERS = {
    "passenger_car": "o",
    "pickup_truck": "s",
    "premium_brand": "^",
    "electric_passenger_car": "D",
    "performance_suv": "v",
}


def _write_atomic_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_atomic_bytes(path, payload):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _chart_precision(path, benchmark_id, rows_by_case, max_swipes):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for case in CASES:
        rows = rows_by_case[case.case_id]
        ax.plot([row["swipe"] for row in rows],
                [row["precision"] for row in rows],
                label=case.label, color=CASE_COLORS[case.case_id],
                marker=CASE_MARKERS[case.case_id], markersize=4.5,
                linewidth=2)
    ax.axhline(PRECISION_TARGET, color="#666666", linewidth=1.2,
               linestyle="--", label="80% target")
    ax.set_xlim(-0.4, max_swipes + 0.4)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks(range(0, max_swipes + 1, 5 if max_swipes >= 5 else 1))
    ax.set_xlabel("total swipes (thumbs up or down)")
    ax.set_ylabel("precision@5 of the live top-5")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9)
    fig.suptitle("Targeted Learning — broad-preference Precision@5 vs swipes\n"
                 "proxies over frozen EPA attributes (no color/price/HP data) "
                 f"— run {benchmark_id}", fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.90))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    _write_atomic_bytes(path, buffer.getvalue())


EVENT_FIELDS = (
    "benchmark_id", "case", "swipe", "probe_id", "thumb", "ups", "downs",
    "top5_json", "matches_json", "match_count", "precision_at_5",
)


def _events_csv(benchmark_id, rows_by_case):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EVENT_FIELDS, lineterminator="\n")
    writer.writeheader()
    for case in CASES:
        for row in rows_by_case[case.case_id]:
            writer.writerow({
                "benchmark_id": benchmark_id,
                "case": case.case_id,
                "swipe": row["swipe"],
                "probe_id": row["probe_id"] or "",
                "thumb": ("" if row["liked"] is None
                          else "up" if row["liked"] else "down"),
                "ups": row["ups"],
                "downs": row["downs"],
                "top5_json": json.dumps(row["top5"], separators=(",", ":")),
                "matches_json": json.dumps(row["matches"],
                                           separators=(",", ":")),
                "match_count": row["match_count"],
                "precision_at_5": f"{row['precision']:.2f}",
            })
    return buffer.getvalue()


def benchmark_config(model_label, max_swipes, top_k):
    """Pure configuration dictionary; its content hash is the benchmark id."""
    return {
        "schema_version": 1,
        "name": "targeted-precision",
        "max_swipes": max_swipes,
        "top_k": top_k,
        "target_matches": TARGET_MATCHES,
        "stable_run": STABLE_RUN,
        "system": {
            "methodology_id": METHODOLOGY_ID,
            "methodology_name": METHODOLOGY_NAME,
            "content_tower": model_label,
            "probe_policy": "app.preference.select_probe (targeted "
                            "information gain, threshold as nuisance)",
            "ranking_policy": "app.preference.rank_recommendations "
                              "(marginal posterior, family-aware)",
        },
        "cases": [
            {"case_id": case.case_id, "label": case.label,
             "proxy": case.proxy, "caveat": case.caveat}
            for case in CASES
        ],
        "likelihood": {
            "sharpness": SHARPNESS,
            "theta_grid": list(THETA_GRID),
            "cosine_weight": COSINE_WEIGHT,
            "token_weight": TOKEN_WEIGHT,
        },
    }


def run_benchmark(engine, features, *, output_root="artifacts/benchmarks",
                  max_swipes=MAX_SWIPES, top_k=TOP_K,
                  model_label=MODEL_LABEL):
    """Run all five cases against the live stack and write artifacts."""
    config = benchmark_config(model_label, max_swipes, top_k)
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    benchmark_id = f"targeted-precision-{digest[:10]}"
    outdir = Path(output_root) / benchmark_id
    outdir.mkdir(parents=True, exist_ok=True)

    features_by_id = {f["variant_id"]: f for f in features}
    catalog_counts = case_match_counts(features)

    rows_by_case = {
        case.case_id: run_case(engine, features_by_id, case.predicate,
                               max_swipes=max_swipes, top_k=top_k)
        for case in CASES
    }
    case_summaries = {
        case.case_id: summarize_case(case, rows_by_case[case.case_id],
                                     catalog_counts[case.case_id])
        for case in CASES
    }
    summary = {
        "benchmark_id": benchmark_id,
        "config": config,
        "catalog_variants": len(features),
        "catalog_match_counts": catalog_counts,
        "success_definition": {
            "precision_at_5_target": PRECISION_TARGET,
            "first_80_swipe": "first swipe count (0 included) where at "
                              "least 4 of the live top-5 match the case",
            "stable_consecutive_swipes": STABLE_RUN,
            "stable_80_swipe": "earliest start of three consecutive "
                               "evaluated swipes at precision@5 >= 0.80; a "
                               "run may begin at swipe 0 only when swipes "
                               "0, 1, and 2 all qualify; None when not "
                               "achieved within max_swipes",
            "swipe_counting": "every thumbs-up or thumbs-down counts as "
                              "one swipe; swipe 0 is the cold start",
        },
        "cases": case_summaries,
        "overall": summarize_overall(case_summaries),
        "system": config["system"],
    }

    _write_atomic_text(outdir / "events.csv",
                       _events_csv(benchmark_id, rows_by_case))
    _write_atomic_text(outdir / "summary.json",
                       json.dumps(summary, indent=2) + "\n")
    _chart_precision(outdir / "precision.png", benchmark_id, rows_by_case,
                     max_swipes)
    return {
        "benchmark_id": benchmark_id,
        "output_dir": str(outdir),
        "summary": summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Broad-preference Precision@5 benchmark for Targeted "
                    "Learning — Current Relationships")
    parser.add_argument("--output-root", default="artifacts/benchmarks",
                        help="directory to write the content-addressed run "
                             "under (useful for tests and scratch runs)")
    args = parser.parse_args(argv)

    print("loading frozen catalog features…", flush=True)
    features = load_catalog_features()
    print(f"pretraining the content tower ({MODEL_LABEL})…", flush=True)
    model = pretrain(features, seed=0)
    engine = build_engine(model, features)
    result = run_benchmark(engine, features, output_root=args.output_root)

    summary = result["summary"]
    print(f"\nbenchmark: {result['benchmark_id']}")
    print(f"artifacts: {result['output_dir']}")
    print(f"catalog match counts: {summary['catalog_match_counts']}")
    print(f"{'case':<24}{'first80':>8}{'stable80':>9}{'final':>7}{'max':>6}"
          f"{'ups':>5}{'downs':>6}")
    for case_id, block in summary["cases"].items():
        first = block["first_80_swipe"]
        stable = block["stable_80_swipe"]
        print(f"{case_id:<24}"
              f"{'>20' if first is None else first:>8}"
              f"{'>20' if stable is None else stable:>9}"
              f"{block['final_precision']:>7.2f}"
              f"{block['max_precision']:>6.2f}"
              f"{block['ups']:>5}{block['downs']:>6}")
    print(f"overall: {json.dumps(summary['overall'], indent=2)}")


if __name__ == "__main__":
    main()
