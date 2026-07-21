"""Focused benchmark tests: the only experiment surface in this repository.

The benchmark measures the one methodology (Targeted Learning — Current
Relationships) against five data-supported broad preference proxies, scoring
Precision@5 of the live ranking after every swipe. Tests run on the tiny EPA
fixture with an untrained tower injected, so they are fast and need no
database and no pretraining.
"""

import csv
import json
import tomllib
from pathlib import Path

import pytest

from app.benchmark import (
    CASES,
    MAX_SWIPES,
    STABLE_RUN,
    TARGET_MATCHES,
    TOP_K,
    first_reach,
    is_electric_passenger_car,
    is_passenger_car,
    is_performance_suv,
    is_pickup_truck,
    is_premium_brand,
    run_benchmark,
    stable_reach,
)
from app.data import import_source_rows
from app.methodology import METHODOLOGY_ID
from app.model import build_variant_features, untrained_model
from app.preference import build_engine

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"


@pytest.fixture(scope="module")
def features():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        return build_variant_features(import_source_rows(list(csv.DictReader(f))))


@pytest.fixture(scope="module")
def engine(features):
    return build_engine(untrained_model(features, seed=0), features)


@pytest.fixture(scope="module")
def result(engine, features, tmp_path_factory):
    root = tmp_path_factory.mktemp("benchmarks")
    return run_benchmark(engine, features, output_root=root,
                         max_swipes=2, top_k=3,
                         model_label="untrained fixture tower (test only)")


def test_production_constants_pin_the_product_objective():
    assert MAX_SWIPES == 20
    assert TOP_K == 5
    assert TARGET_MATCHES == 4  # >= 4 of 5 → Precision@5 >= 0.80
    assert STABLE_RUN == 3


def test_case_ids_are_the_five_honest_proxies():
    assert [case.case_id for case in CASES] == [
        "passenger_car", "pickup_truck", "premium_brand",
        "electric_passenger_car", "performance_suv",
    ]


def test_predicates_match_documented_epa_attribute_rules():
    car = {"vehicle_class": "Midsize Cars", "make": "Toyota",
           "model": "Camry LE", "electric_range": 0.0, "cylinders": 4.0}
    assert is_passenger_car(car)
    assert not is_pickup_truck(car)
    assert not is_electric_passenger_car(car)
    wagon = dict(car, vehicle_class="Small Station Wagons")
    two_seater = dict(car, vehicle_class="Two Seaters")
    assert not is_passenger_car(wagon)
    assert not is_passenger_car(two_seater)
    assert is_pickup_truck({**car, "vehicle_class": "Standard Pickup Trucks 4WD"})
    assert is_premium_brand(dict(car, make="BMW"))
    assert not is_premium_brand(car)
    electric = dict(car, electric_range=250.0, cylinders=None)
    assert is_electric_passenger_car(electric)
    assert not is_electric_passenger_car(dict(electric, electric_range=0.0))
    assert not is_electric_passenger_car(dict(electric, cylinders=4.0))  # PHEV
    suv = {"vehicle_class": "Small Sport Utility Vehicle 4WD",
           "model": "GLC 300 4matic", "cylinders": 4.0}
    assert not is_performance_suv(suv)
    assert is_performance_suv(dict(suv, cylinders=6.0))
    assert is_performance_suv(dict(suv, model="GLC 63 AMG"))
    assert not is_performance_suv(dict(suv, vehicle_class="Midsize Cars",
                                       cylinders=8.0))


def test_first_and_stable_reach_accounting():
    assert first_reach([0, 4, 3, 4, 4, 4]) == 1
    assert first_reach([5, 5, 5]) == 0
    assert first_reach([3, 2, 0]) is None
    assert stable_reach([0, 4, 3, 4, 4, 4]) == 3
    assert stable_reach([4, 4, 4, 0]) == 0
    assert stable_reach([0, 0, 4, 4]) is None  # qualifying tail too short
    assert stable_reach([0, 0, 0, 0]) is None


def test_events_cover_every_case_and_swipe_with_exact_shapes(result, features):
    events_path = Path(result["output_dir"]) / "events.csv"
    with events_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # swipe 0 plus max_swipes rows per case
    assert len(rows) == len(CASES) * 3
    for row in rows:
        top5 = json.loads(row["top5_json"])
        matches = json.loads(row["matches_json"])
        assert len(top5) == 3
        assert len(matches) == 3
        assert len(set(top5)) == 3
        assert int(row["match_count"]) == sum(matches)
        assert row["benchmark_id"] == result["benchmark_id"]
    first = [row for row in rows if row["case"] == "passenger_car"]
    assert [row["swipe"] for row in first] == ["0", "1", "2"]
    assert first[0]["probe_id"] == "" and first[0]["thumb"] == ""
    for row in first[1:]:
        assert row["probe_id"] in {f["variant_id"] for f in features}
        assert row["thumb"] in {"up", "down"}
    assert int(first[-1]["ups"]) + int(first[-1]["downs"]) == 2


def test_run_is_deterministic_and_content_addressed(engine, features, result,
                                                    tmp_path):
    again = run_benchmark(engine, features, output_root=tmp_path,
                          max_swipes=2, top_k=3,
                          model_label="untrained fixture tower (test only)")
    assert again["benchmark_id"] == result["benchmark_id"]
    assert again["benchmark_id"].startswith("targeted-precision-")
    for name in ("events.csv", "summary.json"):
        assert (Path(again["output_dir"]) / name).read_text(encoding="utf-8") \
            == (Path(result["output_dir"]) / name).read_text(encoding="utf-8")
    assert (Path(result["output_dir"]) / "precision.png").stat().st_size > 0


def test_summary_states_methodology_predicates_and_success_metric(result,
                                                                  features):
    summary = json.loads(
        (Path(result["output_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_id"] == result["benchmark_id"]
    assert summary["system"]["methodology_id"] == METHODOLOGY_ID
    assert summary["catalog_variants"] == len(features)
    for case in summary["config"]["cases"]:
        assert case["proxy"]  # exact predicate definition stated
        assert case["caveat"]
    definition = summary["success_definition"]
    assert definition["precision_at_5_target"] == 0.8
    assert definition["stable_consecutive_swipes"] == STABLE_RUN
    cases = summary["cases"]
    assert set(cases) == {case.case_id for case in CASES}
    # fixture ground truth: three passenger cars, one premium brand, no
    # pickups, no battery-electric passenger cars, no performance SUVs
    counts = summary["catalog_match_counts"]
    assert counts["passenger_car"] == 3
    assert counts["premium_brand"] == 1
    assert counts["pickup_truck"] == 0
    assert counts["electric_passenger_car"] == 0
    assert counts["performance_suv"] == 0
    for block in cases.values():
        assert 0.0 <= block["final_precision"] <= 1.0
        assert block["max_precision"] >= block["final_precision"]
    overall = summary["overall"]
    assert overall["cases"] == len(CASES)
    assert "median_first_80_swipe" in overall
    assert "worst_first_80_swipe" in overall
    assert "median_stable_80_swipe" in overall
    assert "worst_stable_80_swipe" in overall
    assert "weighted_score" not in overall


def test_benchmark_is_registered_as_the_only_experiment_command():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["tool"]["hatch"]["envs"]["default"]["scripts"]
    assert scripts["benchmark"] == "python -m app.benchmark {args}"
    for obsolete in ("simulate", "experiment", "validate-experiment"):
        assert obsolete not in scripts
