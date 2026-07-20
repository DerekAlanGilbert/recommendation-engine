"""Catalog tests: fixture-driven importer units plus frozen-snapshot integration.

The catalog has three levels, all derived deterministically from the frozen
EPA snapshot: 9,134 raw EPA configurations, 6,606 consumer-facing variants by
(year, make, model), and 2,138 model-year families by (year, make, baseModel).
"""

import csv
import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from app.data import (
    BRANDS,
    EXPECTED_CONFIGS,
    EXPECTED_CONSUMER_VARIANTS,
    EXPECTED_FAMILIES,
    MANIFEST_PATH,
    SNAPSHOT_COLUMNS,
    SNAPSHOT_PATH,
    SOURCE_URL,
    TRANSFORMATION_VERSION,
    YEAR_MAX,
    YEAR_MIN,
    group_consumer_variants,
    group_families,
    import_source_rows,
    load_manifest,
    load_snapshot,
    write_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"

ELECTRIC_FUEL_TYPES = {"Electricity", "Hydrogen", "Electricity and Hydrogen"}


def read_fixture_rows():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Fixture-driven importer unit tests


def test_fixture_filters_years_and_brands():
    configs = import_source_rows(read_fixture_rows())
    assert [c["epa_id"] for c in configs] == [1010, 1011, 1012, 1030, 1020]
    assert all(YEAR_MIN <= c["year"] <= YEAR_MAX for c in configs)
    assert all(c["make"] in BRANDS for c in configs)


def test_fixture_normalizes_types_without_fabrication():
    configs = {c["epa_id"]: c for c in import_source_rows(read_fixture_rows())}
    camry = configs[1010]
    assert camry["year"] == 2017
    assert camry["cylinders"] == 4
    assert camry["displacement"] == 2.5
    assert (camry["city_mpg"], camry["highway_mpg"], camry["combined_mpg"]) == (28, 39, 32)
    assert camry["co2_tailpipe_gpm"] == 278.0
    bolt = configs[1020]
    assert bolt["cylinders"] is None
    assert bolt["displacement"] is None
    assert bolt["electric_range"] == 247
    assert bolt["co2_tailpipe_gpm"] == 0.0
    assert bolt["model"] == "Bolt EUV"
    assert bolt["base_model"] == ""


def test_fixture_family_grouping():
    configs = import_source_rows(read_fixture_rows())
    families = group_families(configs)
    assert len(families) == 4
    by_id = {f["family_id"]: f for f in families}
    camry_2017 = by_id["2017|Toyota|Camry"]
    assert camry_2017["config_epa_ids"] == [1010, 1011]
    assert camry_2017["config_count"] == 2
    assert by_id["2018|Toyota|Camry"]["config_epa_ids"] == [1012]
    assert by_id["2019|Mercedes-Benz|GLC"]["config_count"] == 1


def test_fixture_family_base_model_fallback_uses_normalized_model():
    families = group_families(import_source_rows(read_fixture_rows()))
    bolt = [f for f in families if f["make"] == "Chevrolet"]
    assert len(bolt) == 1
    assert bolt[0]["base_model"] == "Bolt EUV"
    assert bolt[0]["family_id"] == "2026|Chevrolet|Bolt EUV"


def test_fixture_consumer_variant_grouping_splits_families_by_model():
    configs = import_source_rows(read_fixture_rows())
    variants = group_consumer_variants(configs)
    # The 2017 Camry family holds two consumer-facing variants: LE and XSE.
    assert [v["variant_id"] for v in variants] == [
        "2017|Toyota|Camry LE",
        "2017|Toyota|Camry XSE",
        "2018|Toyota|Camry LE",
        "2019|Mercedes-Benz|GLC 300 4matic",
        "2026|Chevrolet|Bolt EUV",
    ]
    by_id = {v["variant_id"]: v for v in variants}
    le = by_id["2017|Toyota|Camry LE"]
    assert le["family_id"] == "2017|Toyota|Camry"
    assert le["base_model"] == "Camry"
    assert le["model"] == "Camry LE"
    assert le["config_epa_ids"] == [1010]
    assert le["config_count"] == 1
    assert by_id["2017|Toyota|Camry XSE"]["family_id"] == "2017|Toyota|Camry"
    bolt = by_id["2026|Chevrolet|Bolt EUV"]
    assert bolt["family_id"] == "2026|Chevrolet|Bolt EUV"
    assert bolt["base_model"] == "Bolt EUV"


def test_consumer_variant_primary_family_is_modal_with_lexicographic_tie_break():
    """A source quirk can put one (year, make, model) across two baseModels; the
    variant keeps one deterministic primary family and both families survive."""
    def config(epa_id, base_model):
        return {
            "epa_id": epa_id, "year": 2020, "make": "Mercedes-Benz",
            "model": "Metris (Passenger Van)", "base_model": base_model,
            "vehicle_class": "Minivan - 2WD", "fuel_type": "Premium",
            "drive": "Rear-Wheel Drive", "transmission": "Automatic (S7)",
            "cylinders": 4, "displacement": 2.0, "city_mpg": 19,
            "highway_mpg": 23, "combined_mpg": 21, "electric_range": 0,
            "co2_tailpipe_gpm": 423.0,
        }

    configs = [config(1, "Metris Cargo"), config(2, "Metris Passenger")]
    variants = group_consumer_variants(configs)
    assert len(variants) == 1
    assert variants[0]["family_id"] == "2020|Mercedes-Benz|Metris Cargo"
    assert variants[0]["config_epa_ids"] == [1, 2]
    assert len(group_families(configs)) == 2
    # A clear majority beats the lexicographic tie-break.
    majority = configs + [config(3, "Metris Passenger")]
    assert group_consumer_variants(majority)[0]["family_id"] == "2020|Mercedes-Benz|Metris Passenger"


def test_fixture_import_is_deterministic_and_order_independent():
    rows = read_fixture_rows()
    forward = import_source_rows(rows)
    backward = import_source_rows(list(reversed(rows)))
    assert forward == backward
    assert group_families(forward) == group_families(backward)
    assert group_consumer_variants(forward) == group_consumer_variants(backward)


def test_fixture_snapshot_round_trip(tmp_path):
    configs = import_source_rows(read_fixture_rows())
    path = tmp_path / "vehicles.csv"
    write_snapshot(configs, path)
    assert load_snapshot(path) == configs
    write_snapshot(configs, tmp_path / "again.csv")
    assert (tmp_path / "again.csv").read_bytes() == path.read_bytes()


def test_import_script_rejects_count_mismatch(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/import_epa.py",
         "--source", str(FIXTURE_PATH), "--output-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert str(EXPECTED_CONFIGS) in result.stderr
    assert not (tmp_path / "vehicles.csv").exists()
    assert not (tmp_path / "catalog_manifest.json").exists()


# Frozen-snapshot integration tests


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot()


@pytest.fixture(scope="module")
def families(snapshot):
    return group_families(snapshot)


@pytest.fixture(scope="module")
def variants(snapshot):
    return group_consumer_variants(snapshot)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_snapshot_exact_counts(snapshot, families, variants):
    assert len(snapshot) == EXPECTED_CONFIGS
    assert len(variants) == EXPECTED_CONSUMER_VARIANTS
    assert len(families) == EXPECTED_FAMILIES


def test_snapshot_columns_are_exactly_the_contract(snapshot):
    with SNAPSHOT_PATH.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == SNAPSHOT_COLUMNS
    assert all(set(c) == set(SNAPSHOT_COLUMNS) for c in snapshot)


def test_snapshot_partitions_every_config_into_one_family(snapshot, families):
    epa_ids = [c["epa_id"] for c in snapshot]
    assert len(set(epa_ids)) == len(epa_ids)
    grouped_ids = [epa_id for f in families for epa_id in f["config_epa_ids"]]
    assert sorted(grouped_ids) == sorted(epa_ids)
    assert all(f["config_count"] == len(f["config_epa_ids"]) for f in families)
    assert len({f["family_id"] for f in families}) == len(families)


def test_snapshot_partitions_every_config_into_one_consumer_variant(snapshot, variants, families):
    epa_ids = [c["epa_id"] for c in snapshot]
    grouped_ids = [epa_id for v in variants for epa_id in v["config_epa_ids"]]
    assert sorted(grouped_ids) == sorted(epa_ids)
    assert all(v["config_count"] == len(v["config_epa_ids"]) for v in variants)
    assert len({v["variant_id"] for v in variants}) == len(variants)
    # Every variant links to an existing family.
    family_ids = {f["family_id"] for f in families}
    assert all(v["family_id"] in family_ids for v in variants)


def test_snapshot_known_source_quirk_is_the_only_family_ambiguity(snapshot, variants):
    """Exactly one consumer variant has configs under two baseModels; its primary
    family is deterministic. Every other variant nests cleanly in one family."""
    by_config = {c["epa_id"]: c for c in snapshot}
    ambiguous = []
    for v in variants:
        bases = {by_config[epa_id]["base_model"] or by_config[epa_id]["model"]
                 for epa_id in v["config_epa_ids"]}
        if len(bases) > 1:
            ambiguous.append((v["variant_id"], sorted(bases)))
    assert ambiguous == [(
        "2020|Mercedes-Benz|Metris (Passenger Van)",
        ["Metris Cargo", "Metris Passenger"],
    )]
    quirk = next(v for v in variants
                 if v["variant_id"] == "2020|Mercedes-Benz|Metris (Passenger Van)")
    assert quirk["family_id"] == "2020|Mercedes-Benz|Metris Cargo"


def test_snapshot_distinguishes_tundra_trims_as_consumer_variants(variants):
    """The redesign's target granularity: 2025 Tundra 2WD, 4WD, and 4WD PRO are
    distinct recommendation targets inside one family."""
    tundra = [v for v in variants if v["year"] == 2025 and v["make"] == "Toyota"
              and v["base_model"] == "Tundra"]
    assert {v["variant_id"] for v in tundra} >= {
        "2025|Toyota|Tundra 2WD",
        "2025|Toyota|Tundra 4WD",
        "2025|Toyota|Tundra 4WD PRO",
    }
    assert all(v["family_id"] == "2025|Toyota|Tundra" for v in tundra)


def test_snapshot_field_quality(snapshot):
    for c in snapshot:
        assert YEAR_MIN <= c["year"] <= YEAR_MAX
        assert c["make"] in BRANDS
        assert c["model"] and c["model"] == " ".join(c["model"].split())
        assert c["base_model"]
        assert c["vehicle_class"] and c["fuel_type"] and c["drive"] and c["transmission"]
        assert c["city_mpg"] > 0 and c["highway_mpg"] > 0 and c["combined_mpg"] > 0
        assert c["electric_range"] >= 0
        assert c["co2_tailpipe_gpm"] >= 0.0
        if c["fuel_type"] in ELECTRIC_FUEL_TYPES:
            assert c["cylinders"] is None and c["displacement"] is None
        else:
            assert c["cylinders"] > 0 and c["displacement"] > 0.0
    assert {c["make"] for c in snapshot} == set(BRANDS)
    assert {c["year"] for c in snapshot} == set(range(YEAR_MIN, YEAR_MAX + 1))


def test_snapshot_grouping_is_deterministic(snapshot, families, variants):
    assert load_snapshot() == snapshot
    assert group_families(snapshot) == families
    assert group_consumer_variants(snapshot) == variants


def test_manifest_records_provenance_and_counts(snapshot, families, variants, manifest):
    assert manifest["source_url"] == SOURCE_URL
    date.fromisoformat(manifest["retrieved_date"])
    assert len(manifest["source_sha256"]) == 64
    int(manifest["source_sha256"], 16)
    assert manifest["transformation_version"] == TRANSFORMATION_VERSION
    assert manifest["model_year_min"] == YEAR_MIN
    assert manifest["model_year_max"] == YEAR_MAX
    assert manifest["brands"] == list(BRANDS)
    assert manifest["config_count"] == len(snapshot) == EXPECTED_CONFIGS
    assert manifest["consumer_variant_count"] == len(variants) == EXPECTED_CONSUMER_VARIANTS
    assert manifest["family_count"] == len(families) == EXPECTED_FAMILIES
    snapshot_sha256 = hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest()
    assert manifest["snapshot_sha256"] == snapshot_sha256
    assert MANIFEST_PATH.exists()
