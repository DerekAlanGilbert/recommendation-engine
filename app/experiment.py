"""Measured policy experiments with a permanent raw-results + charts contract.

The 12-target cohort in `app.simulate` is a contaminated development cohort:
SHARPNESS and the θ grid were hand-selected while inspecting it, so it is kept
only for regression/sanity reporting. Policy claims come from HELDOUT_COHORT —
36 targets frozen here, deterministically, before any treatment outcome was
examined: stratified across make, EPA vehicle class, fuel type, and model
year, excluding every development target's nameplate (make + baseModel, a
superset of excluding its model-year family).

Run `python -m app.experiment` for the full four-policy comparison (greedy,
passive Bayesian, frozen joint-EIG active baseline, targeted-EIG treatment).
Every run writes raw event rows, machine- and human-readable summaries, and
four charts under `artifacts/experiments/<experiment-id>/`:

    events.csv   summary.json   summary.csv
    progress.png   topk.png   information.png   tundra.png
"""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: charts render identically with no display

import matplotlib.pyplot as plt

from app.preference import COSINE_WEIGHT, SHARPNESS, THETA_GRID, TOKEN_WEIGHT
from app.simulate import (
    COHORT_TARGETS,
    DEFAULT_PERSONA,
    Persona,
    TUNDRA_PRO,
    rollout,
    surfaced_by,
)

# The four compared policies. "active" is the frozen joint-EIG baseline;
# "targeted" is the treatment (threshold integrated out as a nuisance).
EXPERIMENT_POLICIES = ("greedy", "passive", "active", "targeted")

BAYESIAN_POLICIES = ("passive", "active", "targeted")

# Labeled robustness personas: meaningfully different satisficing bars around
# the default (utility scale: exact ideal 9.5, close trim sibling ≈ 7.0,
# same-nameplate sibling ≈ 4.4). Never used for tuning.
PERSONAS = {
    "default": DEFAULT_PERSONA,
    "picky": Persona("picky", base_threshold=5.5, aspiration_margin=0.3),
    "easygoing": Persona("easygoing", base_threshold=3.25, aspiration_margin=0.05),
}

TOP_KS = (1, 3, 10)

EVENT_FIELDS = (
    "experiment_id", "cohort", "persona", "policy", "target_id", "loop",
    "probe_id", "thumb", "target_rank", "target_share", "top_id",
    "entropy_before_bits", "entropy_after_bits", "realized_bits",
    "first_up_loop", "surfaced",
)

SUMMARY_FIELDS = (
    "cohort", "persona", "policy", "loop", "median_rank",
    "top1", "top3", "top10", "surfaced_rate",
    "median_realized_bits", "median_cumulative_bits",
)

# Fixed policy→color assignment (validated categorical palette; color follows
# the entity, never its rank in any particular chart).
POLICY_COLORS = {
    "greedy": "#2a78d6",
    "passive": "#1baf7a",
    "active": "#4a3aa7",
    "targeted": "#e34948",
}


def select_heldout_cohort(features, dev_targets=COHORT_TARGETS, size=36):
    """Deterministic stratified max-coverage selection over the real catalog.

    Greedy: each step takes the candidate least represented so far by make,
    then vehicle class, then fuel type, then model year, breaking ties by
    variant id. No randomness, no dependence on any measured outcome.
    """
    by_id = {f["variant_id"]: f for f in features}
    dev_nameplates = {(by_id[t]["make"], by_id[t]["base_model"]) for t in dev_targets}
    candidates = [
        f for f in features
        if (f["make"], f["base_model"]) not in dev_nameplates
    ]
    make, vclass, fuel, year = Counter(), Counter(), Counter(), Counter()
    chosen, chosen_ids = [], set()
    for _ in range(size):
        best = min(
            (f for f in candidates if f["variant_id"] not in chosen_ids),
            key=lambda f: (
                make[f["make"]],
                vclass[f["vehicle_class"]],
                fuel[f["fuel_type"]],
                year[f["year"]],
                f["variant_id"],
            ),
        )
        chosen.append(best["variant_id"])
        chosen_ids.add(best["variant_id"])
        make[best["make"]] += 1
        vclass[best["vehicle_class"]] += 1
        fuel[best["fuel_type"]] += 1
        year[best["year"]] += 1
    return chosen


# Frozen output of select_heldout_cohort(build_variant_features(load_snapshot()))
# — pinned as a literal so the evaluation targets cannot shift after results.
HELDOUT_COHORT = (
    "2017|Acura|ILX",
    "2018|Audi|A3",
    "2019|BMW|740e xDrive",
    "2020|Chevrolet|Bolt EV",
    "2021|Ford|Escape FWD PHEV",
    "2022|Jeep|Gladiator EcoDiesel 4WD",
    "2023|Ram|1500 2WD",
    "2026|Honda|CR-V e-FCEV",
    "2017|Nissan|Frontier 2WD FFV",
    "2024|Buick|Enclave AWD",
    "2025|Cadillac|CT5",
    "2018|GMC|Terrain AWD",
    "2024|Kia|EV9 Long Range RWD",
    "2025|Mercedes-Benz|AMG GT 63 S E Performance (coupe)",
    "2019|Subaru|BRZ",
    "2020|Toyota|Sienna 2WD",
    "2025|Volkswagen|ID. Buzz 4motion",
    "2021|Hyundai|Nexo",
    "2022|Lexus|NX 450h Plus AWD",
    "2023|Mazda|3 4-Door 2WD",
    "2017|Ford|Transit Connect Van FFV",
    "2018|Chevrolet|Colorado ZR2 4WD",
    "2021|Cadillac|XT5 Hearse AWD",
    "2017|GMC|Savana 2500 2WD (Passenger)",
    "2019|BMW|i3 with Range Extender",
    "2026|Jeep|Grand Cherokee 4xe",
    "2020|Honda|Clarity FCV",
    "2022|Ram|1500 2WD",
    "2023|Kia|Niro Plug-in Hybrid",
    "2018|Nissan|Frontier 2WD FFV",
    "2024|Acura|ZDX AWD",
    "2026|Mercedes-Benz|AMG E53 Hybrid 4matic Plus (station wagon)",
    "2019|Audi|A8 L",
    "2025|Volkswagen|ID. Buzz",
    "2020|Subaru|BRZ",
    "2021|Toyota|Sienna Hybrid AWD",
)


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


def _first_up_loop(records):
    return next((r["loop"] for r in records if r["liked"]), None)


def _cumulative_bits(records):
    """Cumulative realized variant information; None for non-Bayesian rollouts."""
    if records and records[0]["realized_bits"] is None:
        return [None] * len(records)
    total, out = 0.0, []
    for r in records:
        total += r["realized_bits"]
        out.append(total)
    return out


def _experiment_id(name, config):
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{name}-{digest[:10]}"


def _summarize_group(runs_by_target, loops):
    """Aggregate one (cohort, persona, policy) group across its targets."""
    targets = sorted(runs_by_target)
    ranks_by_loop = [
        [runs_by_target[t][loop]["target_rank"] for t in targets]
        for loop in range(loops)
    ]
    cumulative = {t: _cumulative_bits(runs_by_target[t]) for t in targets}
    realized_defined = cumulative[targets[0]][0] is not None
    first_ups = [_first_up_loop(runs_by_target[t]) for t in targets]
    # Censored median: rollouts with no thumbs-up count as loops + 1.
    censored = sorted(loops + 1 if f is None else f for f in first_ups)
    return {
        "targets": targets,
        "median_rank_by_loop": [_median(ranks) for ranks in ranks_by_loop],
        "top_k_by_loop": {
            str(k): [
                sum(rank <= k for rank in ranks) / len(targets)
                for ranks in ranks_by_loop
            ]
            for k in TOP_KS
        },
        "surfaced_rate_by_loop": [
            sum(surfaced_by(runs_by_target[t], loop + 1) for t in targets) / len(targets)
            for loop in range(loops)
        ],
        "median_first_up_loop": _median(censored),
        "any_up_rate": sum(f is not None for f in first_ups) / len(targets),
        "median_realized_bits_by_loop": [
            _median([runs_by_target[t][loop]["realized_bits"] for t in targets])
            if realized_defined else None
            for loop in range(loops)
        ],
        "median_cumulative_bits_by_loop": [
            _median([cumulative[t][loop] for t in targets]) if realized_defined else None
            for loop in range(loops)
        ],
    }


def _at(block, key, loop):
    values = block[key]
    return values[min(loop, len(values)) - 1]


def _comparison(summaries, loops, cohort="heldout", persona="default",
                baseline="active", treatment="targeted"):
    """Go/no-go inputs: treatment vs frozen baseline on the primary cohort."""
    base = summaries.get(f"{cohort}|{persona}|{baseline}")
    treat = summaries.get(f"{cohort}|{persona}|{treatment}")
    if base is None or treat is None:
        return None
    def rank_change_pct(loop):
        before = _at(base, "median_rank_by_loop", loop)
        after = _at(treat, "median_rank_by_loop", loop)
        return 100.0 * (after - before) / before
    early = min(4, loops)
    return {
        "cohort": cohort,
        "persona": persona,
        "baseline": baseline,
        "treatment": treatment,
        "median_rank_at_5": {
            baseline: _at(base, "median_rank_by_loop", 5),
            treatment: _at(treat, "median_rank_by_loop", 5),
            "change_pct": rank_change_pct(5),
        },
        "median_rank_at_15": {
            baseline: _at(base, "median_rank_by_loop", 15),
            treatment: _at(treat, "median_rank_by_loop", 15),
            "change_pct": rank_change_pct(15),
        },
        "median_first_up_loop": {
            baseline: base["median_first_up_loop"],
            treatment: treat["median_first_up_loop"],
        },
        "top10_rate": {
            "at_5": {baseline: base["top_k_by_loop"]["10"][min(5, loops) - 1],
                     treatment: treat["top_k_by_loop"]["10"][min(5, loops) - 1]},
            "at_15": {baseline: base["top_k_by_loop"]["10"][min(15, loops) - 1],
                      treatment: treat["top_k_by_loop"]["10"][min(15, loops) - 1]},
        },
        "surfaced_rate": {
            "at_5": {baseline: _at(base, "surfaced_rate_by_loop", 5),
                     treatment: _at(treat, "surfaced_rate_by_loop", 5)},
            "at_15": {baseline: _at(base, "surfaced_rate_by_loop", 15),
                      treatment: _at(treat, "surfaced_rate_by_loop", 15)},
        },
        "median_cumulative_bits_through_loop_4": {
            baseline: _at(base, "median_cumulative_bits_by_loop", early),
            treatment: _at(treat, "median_cumulative_bits_by_loop", early),
        },
    }


def _write_events(path, experiment_id, results):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for (cohort, persona, policy, target_id), records in results:
            first_up = _first_up_loop(records)
            for r in records:
                writer.writerow({
                    "experiment_id": experiment_id,
                    "cohort": cohort,
                    "persona": persona,
                    "policy": policy,
                    "target_id": target_id,
                    "loop": r["loop"],
                    "probe_id": r["probe_id"],
                    "thumb": "up" if r["liked"] else "down",
                    "target_rank": r["target_rank"],
                    "target_share": f"{r['target_share']:.10g}",
                    "top_id": r["top_id"],
                    "entropy_before_bits": _number(r["entropy_before_bits"]),
                    "entropy_after_bits": _number(r["entropy_after_bits"]),
                    "realized_bits": _number(r["realized_bits"]),
                    "first_up_loop": "" if first_up is None else first_up,
                    "surfaced": "true" if r["surfaced"] else "false",
                })


def _number(value):
    return "" if value is None else f"{value:.10g}"


def _write_summary_csv(path, summaries, loops):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for key in sorted(summaries):
            cohort, persona, policy = key.split("|")
            block = summaries[key]
            for loop in range(1, loops + 1):
                writer.writerow({
                    "cohort": cohort,
                    "persona": persona,
                    "policy": policy,
                    "loop": loop,
                    "median_rank": block["median_rank_by_loop"][loop - 1],
                    "top1": block["top_k_by_loop"]["1"][loop - 1],
                    "top3": block["top_k_by_loop"]["3"][loop - 1],
                    "top10": block["top_k_by_loop"]["10"][loop - 1],
                    "surfaced_rate": block["surfaced_rate_by_loop"][loop - 1],
                    "median_realized_bits":
                        _number(block["median_realized_bits_by_loop"][loop - 1]),
                    "median_cumulative_bits":
                        _number(block["median_cumulative_bits_by_loop"][loop - 1]),
                })


COHORT_TITLES = {
    "heldout": "Held-out cohort (frozen pre-treatment; default persona)",
    "development": "Development cohort — CONTAMINATED, regression/sanity only",
}


def _style_axis(ax):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _policy_lines(ax, loops, series_by_policy, **plot_kwargs):
    for policy in EXPERIMENT_POLICIES:
        if policy not in series_by_policy:
            continue
        values = series_by_policy[policy]
        if all(v is None for v in values):
            continue
        ax.plot(range(1, loops + 1), values, label=policy,
                color=POLICY_COLORS[policy], linewidth=2, marker="o",
                markersize=4, **plot_kwargs)


def _chart_progress(path, experiment_id, summaries, loops, cohorts):
    fig, axes = plt.subplots(1, len(cohorts), figsize=(6 * len(cohorts), 4.5),
                             squeeze=False)
    for ax, cohort in zip(axes[0], cohorts):
        series = {
            policy: summaries[f"{cohort}|default|{policy}"]["median_rank_by_loop"]
            for policy in EXPERIMENT_POLICIES
            if f"{cohort}|default|{policy}" in summaries
        }
        _policy_lines(ax, loops, series)
        ax.set_yscale("log")
        ax.set_xlabel("loop (thumbs)")
        ax.set_ylabel("median target rank (log scale; lower is better)")
        ax.set_title(COHORT_TITLES.get(cohort, cohort), fontsize=10)
        _style_axis(ax)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Median target-rank trajectory by policy — run {experiment_id}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _chart_topk(path, experiment_id, summaries, loops, cohorts):
    fig, axes = plt.subplots(len(cohorts), len(TOP_KS),
                             figsize=(4.2 * len(TOP_KS), 3.6 * len(cohorts)),
                             squeeze=False)
    for row, cohort in enumerate(cohorts):
        for col, k in enumerate(TOP_KS):
            ax = axes[row][col]
            series = {
                policy: summaries[f"{cohort}|default|{policy}"]["top_k_by_loop"][str(k)]
                for policy in EXPERIMENT_POLICIES
                if f"{cohort}|default|{policy}" in summaries
            }
            _policy_lines(ax, loops, series)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("loop (thumbs)")
            if col == 0:
                ax.set_ylabel("fraction of targets")
            ax.set_title(f"top-{k} — {COHORT_TITLES.get(cohort, cohort)}", fontsize=9)
            _style_axis(ax)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Top-k retrieval progression — run {experiment_id}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _chart_information(path, experiment_id, summaries, loops, cohort):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    keys = ("median_realized_bits_by_loop", "median_cumulative_bits_by_loop")
    titles = ("median realized variant information per loop",
              "median cumulative variant information")
    for ax, key, title in zip(axes, keys, titles):
        series = {
            policy: summaries[f"{cohort}|default|{policy}"][key]
            for policy in BAYESIAN_POLICIES
            if f"{cohort}|default|{policy}" in summaries
        }
        _policy_lines(ax, loops, series)
        ax.set_xlabel("loop (thumbs)")
        ax.set_ylabel("bits")
        ax.set_title(title, fontsize=10)
        _style_axis(ax)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        f"Variant-marginal information — {COHORT_TITLES.get(cohort, cohort)} — "
        f"run {experiment_id}\n(greedy has no posterior and is omitted)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _chart_dev_case(path, experiment_id, dev_case, records_by_policy, loops):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = {
        policy: [r["target_rank"] for r in records]
        for policy, records in records_by_policy.items()
    }
    _policy_lines(ax, loops, series)
    ax.set_yscale("log")
    ax.set_xlabel("loop (thumbs)")
    ax.set_ylabel("target rank (log scale; lower is better)")
    _style_axis(ax)
    if series:
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "development case not part of this run's plan",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
    fig.suptitle(f"Development case: {dev_case} — run {experiment_id}\n"
                 "descriptive only — development cohort, NOT held-out evidence",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_experiment(model, engine, features, plan, *, loops=15,
                   policies=EXPERIMENT_POLICIES, name="experiment",
                   dev_case_target=TUNDRA_PRO, output_root="artifacts/experiments"):
    """Run the policy comparison described by `plan` and write the artifact set.

    `plan` is a tuple of (cohort_label, persona_name, target_ids). Every
    (cohort, persona, policy, target) rollout becomes `loops` rows in
    events.csv; summaries aggregate each (cohort, persona, policy) group. The
    experiment id is a content hash of the full configuration, so identical
    configurations land in the same directory with identical results.
    """
    config = {
        "name": name,
        "loops": loops,
        "policies": list(policies),
        "plan": [
            {"cohort": cohort, "persona": persona,
             "persona_parameters": list(PERSONAS[persona][1:]),
             "targets": list(targets)}
            for cohort, persona, targets in plan
        ],
        "dev_case_target": dev_case_target,
        "likelihood": {
            "sharpness": SHARPNESS,
            "theta_grid": list(THETA_GRID),
            "cosine_weight": COSINE_WEIGHT,
            "token_weight": TOKEN_WEIGHT,
        },
    }
    experiment_id = _experiment_id(name, config)
    outdir = Path(output_root) / experiment_id
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    results = []
    for cohort, persona_name, targets in plan:
        persona = PERSONAS[persona_name]
        for policy in policies:
            for target_id in targets:
                records = rollout(policy, model, engine, features, target_id,
                                  loops=loops, persona=persona)
                results.append(((cohort, persona_name, policy, target_id), records))

    groups = {}
    for (cohort, persona_name, policy, target_id), records in results:
        groups.setdefault((cohort, persona_name, policy), {})[target_id] = records
    summaries = {
        f"{cohort}|{persona_name}|{policy}": _summarize_group(runs, loops)
        for (cohort, persona_name, policy), runs in groups.items()
    }
    comparison = _comparison(summaries, loops)
    runtime = time.monotonic() - started

    _write_events(outdir / "events.csv", experiment_id, results)
    _write_summary_csv(outdir / "summary.csv", summaries, loops)

    cohorts = list(dict.fromkeys(cohort for cohort, persona, _ in plan
                                 if persona == "default"))
    primary = cohorts[0]
    _chart_progress(outdir / "progress.png", experiment_id, summaries, loops, cohorts)
    _chart_topk(outdir / "topk.png", experiment_id, summaries, loops, cohorts)
    _chart_information(outdir / "information.png", experiment_id, summaries, loops,
                       primary)
    dev_case_records = {
        policy: groups[(cohort, persona_name, policy)][dev_case_target]
        for cohort, persona_name, targets in plan
        for policy in policies
        if dev_case_target in targets and (cohort, persona_name, policy) in groups
        and dev_case_target in groups[(cohort, persona_name, policy)]
    }
    _chart_dev_case(outdir / "tundra.png", experiment_id, dev_case_target,
                    dev_case_records, loops)

    summary = {
        "experiment_id": experiment_id,
        "config": config,
        "summaries": summaries,
        "comparison_targeted_vs_joint": comparison,
        "runtime_seconds": round(runtime, 3),
        "artifacts": sorted(p.name for p in outdir.iterdir()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                         encoding="utf-8")
    metrics = {key: {
        "median_rank_by_loop": block["median_rank_by_loop"],
        "top_k_by_loop": block["top_k_by_loop"],
        "surfaced_rate_by_loop": block["surfaced_rate_by_loop"],
        "median_first_up_loop": block["median_first_up_loop"],
    } for key, block in summaries.items()}
    return {
        "experiment_id": experiment_id,
        "output_dir": str(outdir),
        "summaries": summaries,
        "comparison": comparison,
        "metrics": metrics,
        "runtime_seconds": runtime,
    }


RAW_ARTIFACTS = ("events.csv", "summary.json", "summary.csv")
CHART_ARTIFACTS = ("progress.png", "topk.png", "information.png", "tundra.png")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def validate_artifacts(output_dir):
    """Check an existing run's artifact set for completeness and integrity.

    Read-only: verifies the seven required files, the events.csv schema and
    row count implied by the recorded configuration, the summary schemas, and
    that every chart is a real, non-blank PNG — without recomputing anything.
    """
    outdir = Path(output_dir)
    for name in RAW_ARTIFACTS + CHART_ARTIFACTS:
        path = outdir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty artifact: {name}")
    for name in CHART_ARTIFACTS:
        data = (outdir / name).read_bytes()
        if data[:8] != PNG_MAGIC or len(data) < 5000:
            raise ValueError(f"corrupt or blank chart: {name}")
    summary = json.loads((outdir / "summary.json").read_text())
    for key in ("experiment_id", "config", "summaries"):
        if key not in summary:
            raise ValueError(f"summary.json missing key: {key}")
    experiment_id = summary["experiment_id"]
    if outdir.name != experiment_id:
        raise ValueError(
            f"directory {outdir.name} does not match experiment id {experiment_id}")
    config = summary["config"]
    expected_events = (config["loops"] * len(config["policies"])
                       * sum(len(entry["targets"]) for entry in config["plan"]))
    with (outdir / "events.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames) != EVENT_FIELDS:
            raise ValueError("events.csv header does not match EVENT_FIELDS")
        rows = list(reader)
    if len(rows) != expected_events:
        raise ValueError(
            f"events.csv has {len(rows)} rows; configuration implies {expected_events}")
    if any(row["experiment_id"] != experiment_id for row in rows):
        raise ValueError("events.csv contains rows from another experiment")
    with (outdir / "summary.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames) != SUMMARY_FIELDS:
            raise ValueError("summary.csv header does not match SUMMARY_FIELDS")
        summary_rows = list(reader)
    expected_summary = len(summary["summaries"]) * config["loops"]
    if len(summary_rows) != expected_summary:
        raise ValueError(
            f"summary.csv has {len(summary_rows)} rows; expected {expected_summary}")
    return {
        "experiment_id": experiment_id,
        "files": list(RAW_ARTIFACTS + CHART_ARTIFACTS),
        "event_rows": len(rows),
        "summary_rows": len(summary_rows),
    }


def replot(output_dir):
    """Regenerate the four charts from summary.json + events.csv alone.

    Charts are a pure function of the raw files, so a completed run can be
    re-rendered (style tweaks, matplotlib upgrades) without rerunning it.
    """
    outdir = Path(output_dir)
    summary = json.loads((outdir / "summary.json").read_text())
    experiment_id = summary["experiment_id"]
    config = summary["config"]
    loops = config["loops"]
    summaries = summary["summaries"]
    cohorts = list(dict.fromkeys(entry["cohort"] for entry in config["plan"]
                                 if entry["persona"] == "default"))
    _chart_progress(outdir / "progress.png", experiment_id, summaries, loops, cohorts)
    _chart_topk(outdir / "topk.png", experiment_id, summaries, loops, cohorts)
    _chart_information(outdir / "information.png", experiment_id, summaries, loops,
                       cohorts[0])
    dev_case = config["dev_case_target"]
    with (outdir / "events.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["target_id"] == dev_case and r["persona"] == "default"]
    records_by_policy = {}
    for policy in config["policies"]:
        records = sorted((r for r in rows if r["policy"] == policy),
                         key=lambda r: int(r["loop"]))
        if records:
            records_by_policy[policy] = [
                {"target_rank": int(r["target_rank"])} for r in records
            ]
    _chart_dev_case(outdir / "tundra.png", experiment_id, dev_case,
                    records_by_policy, loops)


def _print_report(result):
    outdir = Path(result["output_dir"])
    print(f"\nexperiment id: {result['experiment_id']}")
    print(f"output directory: {outdir}")
    print("raw results:")
    for name in ("events.csv", "summary.json", "summary.csv"):
        print(f"  {outdir / name}")
    print("charts:")
    for name in ("progress.png", "topk.png", "information.png", "tundra.png"):
        print(f"  {outdir / name}")
    print(f"runtime: {result['runtime_seconds']:.1f}s")
    comparison = result["comparison"]
    if comparison:
        print("\ntargeted vs frozen joint-EIG baseline (held-out, default persona):")
        print(json.dumps(comparison, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loops", type=int, default=15)
    parser.add_argument("--name", default="targeted-eig")
    parser.add_argument("--output-root", default="artifacts/experiments")
    parser.add_argument("--robustness-targets", type=int, default=12,
                        help="held-out targets per robustness persona (0 disables)")
    parser.add_argument("--validate", metavar="RUN_DIR",
                        help="validate an existing run's artifacts and exit")
    parser.add_argument("--replot", metavar="RUN_DIR",
                        help="regenerate an existing run's charts from its raw files and exit")
    args = parser.parse_args()

    if args.validate:
        report = validate_artifacts(args.validate)
        print(f"artifacts OK: {report['experiment_id']} "
              f"({report['event_rows']} event rows, {report['summary_rows']} summary rows)")
        for name in report["files"]:
            print(f"  {Path(args.validate) / name}")
        return
    if args.replot:
        replot(args.replot)
        print(f"charts regenerated from raw files in {args.replot}")
        return

    from app.data import load_snapshot
    from app.model import build_variant_features, pretrain
    from app.preference import build_engine

    print("pretraining content tower and building preference engine ...")
    features = build_variant_features(load_snapshot())
    model = pretrain(features, seed=0)
    engine = build_engine(model, features)

    plan = [
        ("heldout", "default", HELDOUT_COHORT),
        ("development", "default", COHORT_TARGETS),
    ]
    for persona_name in ("picky", "easygoing"):
        if args.robustness_targets > 0:
            plan.append(("heldout", persona_name,
                         HELDOUT_COHORT[:args.robustness_targets]))

    result = run_experiment(
        model, engine, features, tuple(plan),
        loops=args.loops, name=args.name, output_root=args.output_root,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
