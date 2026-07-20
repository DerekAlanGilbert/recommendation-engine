"""Experiment harness tests: frozen held-out cohort, personas, artifact contract.

The 12-target simulation cohort is a contaminated development cohort
(SHARPNESS and the θ grid were inspected against it), so policy comparisons
need a held-out cohort frozen before treatment outcomes are examined: 36
deterministic targets stratified across make, EPA vehicle class, and fuel
type, excluding every development target's nameplate (make + baseModel, a
superset of its model-year family).
"""

import csv
from pathlib import Path

import pytest

from app.data import import_source_rows, load_snapshot
from app.experiment import (
    EVENT_FIELDS,
    EXPERIMENT_POLICIES,
    HELDOUT_COHORT,
    replot,
    run_experiment,
    select_heldout_cohort,
    validate_artifacts,
)
from app.model import build_variant_features, untrained_model
from app.preference import build_engine
from app.simulate import COHORT_TARGETS, rollout, shopper_thumb, Persona, DEFAULT_PERSONA

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"


@pytest.fixture(scope="module")
def features():
    return build_variant_features(load_snapshot())


@pytest.fixture(scope="module")
def by_id(features):
    return {f["variant_id"]: f for f in features}


@pytest.fixture(scope="module")
def fixture_features():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        return build_variant_features(import_source_rows(list(csv.DictReader(f))))


# Held-out cohort: frozen before any treatment outcome was examined.


def test_heldout_cohort_is_pinned_and_reproducible(features):
    assert HELDOUT_COHORT == tuple(select_heldout_cohort(features))
    assert len(HELDOUT_COHORT) == 36
    assert len(set(HELDOUT_COHORT)) == 36


def test_heldout_cohort_targets_exist(by_id):
    for target in HELDOUT_COHORT:
        assert target in by_id, target


def test_heldout_cohort_excludes_development_targets_and_their_families(by_id):
    dev_families = {by_id[t]["family_id"] for t in COHORT_TARGETS}
    dev_nameplates = {(by_id[t]["make"], by_id[t]["base_model"]) for t in COHORT_TARGETS}
    for target in HELDOUT_COHORT:
        assert target not in COHORT_TARGETS
        assert by_id[target]["family_id"] not in dev_families, target
        # Stronger than required: no shared nameplate with any development
        # target, so no held-out result leans on a Tundra-adjacent sibling.
        assert (by_id[target]["make"], by_id[target]["base_model"]) not in dev_nameplates, target


def test_heldout_cohort_is_stratified_across_make_class_and_powertrain(by_id):
    targets = [by_id[t] for t in HELDOUT_COHORT]
    assert len({t["make"] for t in targets}) >= 15
    assert len({t["vehicle_class"] for t in targets}) >= 10
    assert len({t["fuel_type"] for t in targets}) >= 4
    assert len({t["year"] for t in targets}) >= 6


# Personas: the hidden shopper's satisficing bar as an explicit parameter.


def test_default_persona_reproduces_baseline_thumbs(by_id):
    tundra = by_id["2025|Toyota|Tundra 4WD PRO"]
    plain = by_id["2025|Toyota|Tundra 4WD"]
    assert shopper_thumb(tundra, plain) == shopper_thumb(
        tundra, plain,
        base_threshold=DEFAULT_PERSONA.base_threshold,
        aspiration_margin=DEFAULT_PERSONA.aspiration_margin,
    )


def test_picky_persona_rejects_what_the_default_persona_accepts(by_id):
    tundra = by_id["2025|Toyota|Tundra 4WD PRO"]
    plain = by_id["2025|Toyota|Tundra 4WD"]
    assert shopper_thumb(tundra, plain)  # default bar: worth considering
    assert not shopper_thumb(tundra, plain, base_threshold=8.0)
    picky = Persona("picky", base_threshold=8.0, aspiration_margin=0.3)
    assert picky.base_threshold == 8.0


def test_rollout_with_default_persona_matches_rollout_without(fixture_features):
    engine = build_engine(untrained_model(fixture_features, seed=0), fixture_features)
    target = fixture_features[0]["variant_id"]
    plain = rollout("passive", None, engine, fixture_features, target, loops=3)
    explicit = rollout("passive", None, engine, fixture_features, target, loops=3,
                       persona=DEFAULT_PERSONA)
    assert plain == explicit


# Rollout instrumentation for the experiment runner.


@pytest.fixture(scope="module")
def fixture_engine(fixture_features):
    return build_engine(untrained_model(fixture_features, seed=0), fixture_features)


def test_rollout_supports_the_targeted_policy(fixture_engine, fixture_features):
    target = fixture_features[0]["variant_id"]
    records = rollout("targeted", None, fixture_engine, fixture_features, target, loops=3)
    assert [r["loop"] for r in records] == [1, 2, 3]
    probes = [r["probe_id"] for r in records]
    assert len(set(probes)) == len(probes)
    assert records == rollout("targeted", None, fixture_engine, fixture_features,
                              target, loops=3)


def test_bayesian_rollouts_record_variant_entropy_and_realized_bits(
        fixture_engine, fixture_features):
    import math
    n = len(fixture_features)
    target = fixture_features[0]["variant_id"]
    for policy in ("passive", "targeted"):
        records = rollout(policy, None, fixture_engine, fixture_features, target, loops=3)
        assert records[0]["entropy_before_bits"] == pytest.approx(math.log2(n))
        for before, after in zip(records, records[1:]):
            # Chained: this loop's posterior is the next loop's starting point.
            assert after["entropy_before_bits"] == pytest.approx(before["entropy_after_bits"])
        for r in records:
            assert r["realized_bits"] == pytest.approx(
                r["entropy_before_bits"] - r["entropy_after_bits"])


def test_greedy_rollout_reports_no_posterior_entropy(fixture_engine, fixture_features):
    from app.model import pretrain
    model = pretrain(fixture_features, seed=0, epochs=5)
    target = fixture_features[0]["variant_id"]
    records = rollout("greedy", model, fixture_engine, fixture_features, target, loops=2)
    for r in records:
        assert r["entropy_before_bits"] is None
        assert r["entropy_after_bits"] is None
        assert r["realized_bits"] is None


# Permanent artifact contract: every experiment run emits raw results + charts.

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

ARTIFACTS = (
    "events.csv", "summary.json", "summary.csv",
    "progress.png", "topk.png", "information.png", "tundra.png",
)


@pytest.fixture(scope="module")
def tiny_run(fixture_engine, fixture_features, tmp_path_factory):
    from app.model import pretrain
    model = pretrain(fixture_features, seed=0, epochs=5)
    ids = [f["variant_id"] for f in fixture_features]
    plan = (
        ("heldout", "default", (ids[0], ids[1])),
        ("development", "default", (ids[2],)),
    )
    root = tmp_path_factory.mktemp("artifacts")
    result = run_experiment(
        model, fixture_engine, fixture_features, plan,
        loops=3, name="tiny-fixture", dev_case_target=ids[2], output_root=root,
    )
    return result, root


def test_run_experiment_creates_the_full_artifact_set(tiny_run):
    result, root = tiny_run
    outdir = Path(result["output_dir"])
    assert outdir.parent == root
    assert outdir.name == result["experiment_id"]
    for artifact in ARTIFACTS:
        path = outdir / artifact
        assert path.is_file(), artifact
        assert path.stat().st_size > 0, artifact
    for chart in ("progress.png", "topk.png", "information.png", "tundra.png"):
        assert (outdir / chart).read_bytes()[:8] == PNG_MAGIC, chart
        assert (outdir / chart).stat().st_size > 5000, chart  # not a blank canvas


def test_events_csv_has_the_stable_schema(tiny_run):
    result, _ = tiny_run
    with open(Path(result["output_dir"]) / "events.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == EVENT_FIELDS
        rows = list(reader)
    # (2 held-out + 1 development targets) x 4 policies x 3 loops
    assert len(rows) == 3 * len(EXPERIMENT_POLICIES) * 3
    for row in rows:
        assert row["experiment_id"] == result["experiment_id"]
        assert row["cohort"] in {"heldout", "development"}
        assert row["persona"] == "default"
        assert row["policy"] in EXPERIMENT_POLICIES
        assert int(row["loop"]) in (1, 2, 3)
        assert row["thumb"] in {"up", "down"}
        assert int(row["target_rank"]) >= 1
        assert row["surfaced"] in {"true", "false"}
        if row["policy"] == "greedy":
            assert row["realized_bits"] == ""
        else:
            float(row["realized_bits"])


def test_summary_json_is_complete_and_machine_readable(tiny_run):
    import json
    result, _ = tiny_run
    summary = json.loads((Path(result["output_dir"]) / "summary.json").read_text())
    assert summary["experiment_id"] == result["experiment_id"]
    assert summary["config"]["loops"] == 3
    assert summary["config"]["policies"] == list(EXPERIMENT_POLICIES)
    key = "heldout|default|targeted"
    assert key in summary["summaries"]
    block = summary["summaries"][key]
    assert len(block["median_rank_by_loop"]) == 3
    assert len(block["top_k_by_loop"]["10"]) == 3
    assert len(block["surfaced_rate_by_loop"]) == 3
    assert "median_first_up_loop" in block
    assert len(block["median_cumulative_bits_by_loop"]) == 3


def test_validate_artifacts_accepts_a_complete_run(tiny_run):
    result, _ = tiny_run
    report = validate_artifacts(result["output_dir"])
    assert report["experiment_id"] == result["experiment_id"]
    assert set(report["files"]) == set(ARTIFACTS)
    assert report["event_rows"] == 3 * len(EXPERIMENT_POLICIES) * 3


def test_validate_artifacts_rejects_missing_or_corrupt_files(tiny_run, tmp_path):
    import shutil
    result, _ = tiny_run
    source = Path(result["output_dir"])

    truncated = tmp_path / "truncated" / source.name
    shutil.copytree(source, truncated)
    (truncated / "progress.png").write_bytes(b"not a png")
    with pytest.raises(ValueError, match="progress.png"):
        validate_artifacts(truncated)

    missing = tmp_path / "missing" / source.name
    shutil.copytree(source, missing)
    (missing / "events.csv").unlink()
    with pytest.raises(ValueError, match="events.csv"):
        validate_artifacts(missing)


def test_replot_regenerates_charts_from_raw_files_alone(tiny_run, tmp_path):
    import shutil
    result, _ = tiny_run
    source = Path(result["output_dir"])
    copy = tmp_path / source.name
    shutil.copytree(source, copy)
    for chart in ("progress.png", "topk.png", "information.png", "tundra.png"):
        (copy / chart).unlink()
    replot(copy)
    for chart in ("progress.png", "topk.png", "information.png", "tundra.png"):
        data = (copy / chart).read_bytes()
        assert data[:8] == PNG_MAGIC, chart
        assert len(data) > 5000, chart
    validate_artifacts(copy)  # the regenerated run passes the same gate


def test_cli_validate_and_help_run_without_the_full_experiment(tiny_run):
    import subprocess
    import sys
    result, _ = tiny_run
    root = Path(__file__).resolve().parent.parent
    validate = subprocess.run(
        [sys.executable, "-m", "app.experiment", "--validate", result["output_dir"]],
        capture_output=True, text=True, cwd=root, timeout=120,
    )
    assert validate.returncode == 0, validate.stderr
    assert result["experiment_id"] in validate.stdout
    for module in ("app.experiment", "app.simulate"):
        helped = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True, text=True, cwd=root, timeout=120,
        )
        assert helped.returncode == 0, helped.stderr
        assert "usage" in helped.stdout.lower()


def test_run_experiment_is_deterministic(fixture_engine, fixture_features, tmp_path):
    from app.model import pretrain
    model = pretrain(fixture_features, seed=0, epochs=5)
    ids = [f["variant_id"] for f in fixture_features]
    plan = (("heldout", "default", (ids[0],)),)
    runs = []
    for sub in ("a", "b"):
        runs.append(run_experiment(
            model, fixture_engine, fixture_features, plan,
            loops=2, name="repeat", dev_case_target=ids[0], output_root=tmp_path / sub,
        ))
    assert runs[0]["experiment_id"] == runs[1]["experiment_id"]
    first = (Path(runs[0]["output_dir"]) / "events.csv").read_text()
    second = (Path(runs[1]["output_dir"]) / "events.csv").read_text()
    assert first == second
    assert runs[0]["metrics"] == runs[1]["metrics"]
