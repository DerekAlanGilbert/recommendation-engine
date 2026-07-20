"""Catalog tests: fixture-driven importer units plus frozen-snapshot integration."""

import csv
import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from app.data import (
    BRANDS,
    EXPECTED_CANONICAL,
    EXPECTED_VARIANTS,
    MANIFEST_PATH,
    SNAPSHOT_COLUMNS,
    SNAPSHOT_PATH,
    SOURCE_URL,
    TRANSFORMATION_VERSION,
    YEAR_MAX,
    YEAR_MIN,
    group_variants,
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
    variants = import_source_rows(read_fixture_rows())
    assert [v["epa_id"] for v in variants] == [1010, 1011, 1012, 1030, 1020]
    assert all(YEAR_MIN <= v["year"] <= YEAR_MAX for v in variants)
    assert all(v["make"] in BRANDS for v in variants)


def test_fixture_normalizes_types_without_fabrication():
    variants = {v["epa_id"]: v for v in import_source_rows(read_fixture_rows())}
    camry = variants[1010]
    assert camry["year"] == 2017
    assert camry["cylinders"] == 4
    assert camry["displacement"] == 2.5
    assert (camry["city_mpg"], camry["highway_mpg"], camry["combined_mpg"]) == (28, 39, 32)
    assert camry["co2_tailpipe_gpm"] == 278.0
    bolt = variants[1020]
    assert bolt["cylinders"] is None
    assert bolt["displacement"] is None
    assert bolt["electric_range"] == 247
    assert bolt["co2_tailpipe_gpm"] == 0.0
    assert bolt["model"] == "Bolt EUV"
    assert bolt["base_model"] == ""


def test_fixture_canonical_grouping():
    variants = import_source_rows(read_fixture_rows())
    items = group_variants(variants)
    assert len(items) == 4
    by_id = {i["item_id"]: i for i in items}
    camry_2017 = by_id["2017|Toyota|Camry"]
    assert camry_2017["variant_epa_ids"] == [1010, 1011]
    assert camry_2017["variant_count"] == 2
    assert by_id["2018|Toyota|Camry"]["variant_epa_ids"] == [1012]
    assert by_id["2019|Mercedes-Benz|GLC"]["variant_count"] == 1


def test_fixture_base_model_fallback_uses_normalized_model():
    items = group_variants(import_source_rows(read_fixture_rows()))
    bolt = [i for i in items if i["make"] == "Chevrolet"]
    assert len(bolt) == 1
    assert bolt[0]["base_model"] == "Bolt EUV"
    assert bolt[0]["item_id"] == "2026|Chevrolet|Bolt EUV"


def test_fixture_import_is_deterministic_and_order_independent():
    rows = read_fixture_rows()
    forward = import_source_rows(rows)
    backward = import_source_rows(list(reversed(rows)))
    assert forward == backward
    assert group_variants(forward) == group_variants(backward)


def test_fixture_snapshot_round_trip(tmp_path):
    variants = import_source_rows(read_fixture_rows())
    path = tmp_path / "vehicles.csv"
    write_snapshot(variants, path)
    assert load_snapshot(path) == variants
    write_snapshot(variants, tmp_path / "again.csv")
    assert (tmp_path / "again.csv").read_bytes() == path.read_bytes()


def test_import_script_rejects_count_mismatch(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/import_epa.py",
         "--source", str(FIXTURE_PATH), "--output-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert str(EXPECTED_VARIANTS) in result.stderr
    assert not (tmp_path / "vehicles.csv").exists()
    assert not (tmp_path / "catalog_manifest.json").exists()


# Frozen-snapshot integration tests


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot()


@pytest.fixture(scope="module")
def items(snapshot):
    return group_variants(snapshot)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_snapshot_exact_counts(snapshot, items):
    assert len(snapshot) == EXPECTED_VARIANTS
    assert len(items) == EXPECTED_CANONICAL


def test_snapshot_columns_are_exactly_the_contract(snapshot):
    with SNAPSHOT_PATH.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == SNAPSHOT_COLUMNS
    assert all(set(v) == set(SNAPSHOT_COLUMNS) for v in snapshot)


def test_snapshot_partitions_every_variant_into_one_item(snapshot, items):
    epa_ids = [v["epa_id"] for v in snapshot]
    assert len(set(epa_ids)) == len(epa_ids)
    grouped_ids = [epa_id for i in items for epa_id in i["variant_epa_ids"]]
    assert sorted(grouped_ids) == sorted(epa_ids)
    assert all(i["variant_count"] == len(i["variant_epa_ids"]) for i in items)
    assert len({i["item_id"] for i in items}) == len(items)


def test_snapshot_field_quality(snapshot):
    for v in snapshot:
        assert YEAR_MIN <= v["year"] <= YEAR_MAX
        assert v["make"] in BRANDS
        assert v["model"] and v["model"] == " ".join(v["model"].split())
        assert v["base_model"]
        assert v["vehicle_class"] and v["fuel_type"] and v["drive"] and v["transmission"]
        assert v["city_mpg"] > 0 and v["highway_mpg"] > 0 and v["combined_mpg"] > 0
        assert v["electric_range"] >= 0
        assert v["co2_tailpipe_gpm"] >= 0.0
        if v["fuel_type"] in ELECTRIC_FUEL_TYPES:
            assert v["cylinders"] is None and v["displacement"] is None
        else:
            assert v["cylinders"] > 0 and v["displacement"] > 0.0
    assert {v["make"] for v in snapshot} == set(BRANDS)
    assert {v["year"] for v in snapshot} == set(range(YEAR_MIN, YEAR_MAX + 1))


def test_snapshot_grouping_is_deterministic(snapshot, items):
    assert load_snapshot() == snapshot
    assert group_variants(snapshot) == items


def test_manifest_records_provenance_and_counts(snapshot, items, manifest):
    assert manifest["source_url"] == SOURCE_URL
    date.fromisoformat(manifest["retrieved_date"])
    assert len(manifest["source_sha256"]) == 64
    int(manifest["source_sha256"], 16)
    assert manifest["transformation_version"] == TRANSFORMATION_VERSION
    assert manifest["model_year_min"] == YEAR_MIN
    assert manifest["model_year_max"] == YEAR_MAX
    assert manifest["brands"] == list(BRANDS)
    assert manifest["variant_count"] == len(snapshot) == EXPECTED_VARIANTS
    assert manifest["canonical_count"] == len(items) == EXPECTED_CANONICAL
    snapshot_sha256 = hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest()
    assert manifest["snapshot_sha256"] == snapshot_sha256
    assert MANIFEST_PATH.exists()
