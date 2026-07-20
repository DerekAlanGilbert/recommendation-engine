"""Postgres store tests: fixture-driven behavior units plus frozen-snapshot round-trip integration.

These tests require the local pgvector database from compose.yaml: `docker compose up -d`.
The suite creates and removes a dedicated test database so development data is untouched.
"""

import csv
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import torch

from app.data import (
    EXPECTED_CANONICAL,
    EXPECTED_VARIANTS,
    group_variants,
    import_source_rows,
    load_manifest,
    load_snapshot,
)
from app.model import build_item_features, rank_items, refit_profile, untrained_model
from app.store import (
    DuplicateFeedbackError,
    UnknownItemError,
    add_feedback,
    clear_feedback,
    connect,
    database_url,
    import_catalog,
    load_feedback,
    load_items,
    load_meta,
    load_model,
    load_variants,
    save_model,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"

TABLES = {"feedback", "item_embeddings", "model_meta", "vehicle_items", "vehicle_variants"}


def read_fixture_variants():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        return import_source_rows(list(csv.DictReader(f)))


def fixture_manifest(variants):
    return {
        "snapshot_sha256": "f" * 64,
        "variant_count": len(variants),
        "canonical_count": len(group_variants(variants)),
    }


def expected_item_rows(variants):
    """The persisted canonical-item contract: grouping fields merged with aggregate features."""
    features = {f["item_id"]: f for f in build_item_features(variants)}
    return [
        {
            **features[item["item_id"]],
            "base_model": item["base_model"],
            "variant_count": item["variant_count"],
        }
        for item in group_variants(variants)
    ]


def public_tables(connection):
    rows = connection.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    return {name for (name,) in rows}


def _database_named(url, name):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{name}", parts.query, parts.fragment))


@pytest.fixture(scope="session", autouse=True)
def isolated_test_database():
    admin_url = database_url()
    test_url = _database_named(admin_url, "rec_test")
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as admin:
            admin.execute("DROP DATABASE IF EXISTS rec_test WITH (FORCE)")
            admin.execute("CREATE DATABASE rec_test")
    except psycopg.OperationalError as error:
        pytest.fail(f"PostgreSQL unavailable at {admin_url}; run `docker compose up -d` ({error})")
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = test_url
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as admin:
            admin.execute("DROP DATABASE IF EXISTS rec_test WITH (FORCE)")


@pytest.fixture()
def conn(isolated_test_database):
    with psycopg.connect(database_url(), autocommit=True, connect_timeout=5) as admin:
        admin.execute("DROP TABLE IF EXISTS " + ", ".join(sorted(TABLES)) + " CASCADE")
    connection = connect()
    yield connection
    connection.close()


# Fixture-driven behavior tests


def test_connect_applies_idempotent_schema_with_no_profile_table(conn):
    assert public_tables(conn) == TABLES
    again = connect()
    try:
        assert public_tables(again) == TABLES
    finally:
        again.close()


def test_fixture_catalog_round_trip(conn):
    variants = read_fixture_variants()
    assert import_catalog(conn, variants, fixture_manifest(variants)) is True
    assert load_variants(conn) == variants
    items = load_items(conn)
    assert items == expected_item_rows(variants)
    by_id = {item["item_id"]: item for item in items}
    camry = by_id["2017|Toyota|Camry"]
    assert camry["base_model"] == "Camry"
    assert camry["variant_count"] == 2
    assert camry["combined_mpg"] == 29.0
    bolt = by_id["2026|Chevrolet|Bolt EUV"]
    assert bolt["cylinders"] is None
    assert bolt["displacement"] is None
    assert bolt["variant_count"] == 1
    assert load_meta(conn) == {"snapshot_sha256": "f" * 64, "bias": None}


def test_import_is_transactional_when_a_variant_fails(conn):
    variants = read_fixture_variants()
    broken = variants + [dict(variants[-1])]  # duplicate EPA ID fails mid-load
    with pytest.raises(psycopg.errors.UniqueViolation):
        import_catalog(conn, broken, fixture_manifest(broken))
    assert load_variants(conn) == []
    assert load_items(conn) == []
    assert load_meta(conn) is None


def test_import_runs_once_and_rejects_a_changed_snapshot(conn):
    variants = read_fixture_variants()
    manifest = fixture_manifest(variants)
    assert import_catalog(conn, variants, manifest) is True
    assert import_catalog(conn, variants, manifest) is False
    assert load_variants(conn) == variants
    with pytest.raises(ValueError):
        import_catalog(conn, variants, {**manifest, "snapshot_sha256": "0" * 64})


def test_import_rejects_manifest_count_mismatch(conn):
    variants = read_fixture_variants()
    manifest = fixture_manifest(variants)
    with pytest.raises(ValueError):
        import_catalog(conn, variants, {**manifest, "variant_count": len(variants) + 1})
    with pytest.raises(ValueError):
        import_catalog(conn, variants, {**manifest, "canonical_count": 999})
    assert load_variants(conn) == []
    assert load_meta(conn) is None


def test_feedback_is_recorded_once_per_canonical_item(conn):
    variants = read_fixture_variants()
    import_catalog(conn, variants, fixture_manifest(variants))
    add_feedback(conn, "2017|Toyota|Camry", True)
    add_feedback(conn, "2026|Chevrolet|Bolt EUV", False)
    history = [("2017|Toyota|Camry", True), ("2026|Chevrolet|Bolt EUV", False)]
    assert load_feedback(conn) == history
    with pytest.raises(DuplicateFeedbackError):
        add_feedback(conn, "2017|Toyota|Camry", False)
    with pytest.raises(UnknownItemError):
        add_feedback(conn, "1999|Yugo|GV", True)
    # Rejected events leave the connection usable and the history unchanged.
    assert load_feedback(conn) == history
    clear_feedback(conn)
    assert load_feedback(conn) == []
    add_feedback(conn, "2017|Toyota|Camry", False)
    assert load_feedback(conn) == [("2017|Toyota|Camry", False)]


def test_model_round_trip_preserves_embeddings_and_bias(conn):
    variants = read_fixture_variants()
    model = untrained_model(build_item_features(variants), seed=0)._replace(bias=-0.25)
    with pytest.raises(ValueError):
        save_model(conn, model)  # the catalog must be imported first
    import_catalog(conn, variants, fixture_manifest(variants))
    assert load_model(conn) is None
    save_model(conn, model)
    loaded = load_model(conn)
    assert loaded.item_ids == model.item_ids
    assert loaded.index == model.index
    assert loaded.bias == model.bias
    assert torch.equal(loaded.embeddings, model.embeddings)
    assert load_meta(conn)["bias"] == model.bias
    save_model(conn, model)  # re-saving replaces embeddings rather than duplicating them
    assert torch.equal(load_model(conn).embeddings, model.embeddings)


# Frozen-snapshot integration tests


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot()


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def features(snapshot):
    return build_item_features(snapshot)


@pytest.fixture(scope="module")
def catalog_model(features):
    return untrained_model(features, seed=0)._replace(bias=-0.125)


def test_frozen_snapshot_round_trip_records_manifest_checksum(conn, snapshot, manifest):
    assert import_catalog(conn, snapshot, manifest) is True
    variants = load_variants(conn)
    assert len(variants) == EXPECTED_VARIANTS
    assert variants == snapshot
    items = load_items(conn)
    assert len(items) == EXPECTED_CANONICAL
    assert items == expected_item_rows(snapshot)
    assert load_meta(conn) == {"snapshot_sha256": manifest["snapshot_sha256"], "bias": None}


def test_restart_reconstructs_profile_and_ranking_from_feedback(
    conn, snapshot, manifest, features, catalog_model
):
    import_catalog(conn, snapshot, manifest)
    save_model(conn, catalog_model)
    liked = [f["item_id"] for f in features if f["fuel_type"] == "Electricity"][:8]
    disliked = [f["item_id"] for f in features if f["co2_tailpipe_gpm"] >= 450][:8]
    for item_id in liked:
        add_feedback(conn, item_id, True)
    for item_id in disliked:
        add_feedback(conn, item_id, False)
    events = load_feedback(conn)
    assert sorted(events) == sorted([(i, True) for i in liked] + [(i, False) for i in disliked])
    profile = refit_profile(catalog_model, events)
    rated = set(liked) | set(disliked)
    ranking = rank_items(catalog_model, profile, exclude=rated, limit=10)
    conn.close()

    reopened = connect()
    try:
        restored = load_model(reopened)
        assert restored.item_ids == catalog_model.item_ids
        assert restored.bias == catalog_model.bias
        assert torch.equal(restored.embeddings, catalog_model.embeddings)
        assert load_feedback(reopened) == events
        restored_profile = refit_profile(restored, load_feedback(reopened))
        assert torch.equal(restored_profile, profile)
        assert rank_items(restored, restored_profile, exclude=rated, limit=10) == ranking
    finally:
        reopened.close()
