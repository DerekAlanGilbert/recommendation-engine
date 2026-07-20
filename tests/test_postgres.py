"""Postgres store tests: fixture-driven behavior units plus frozen-snapshot round-trip integration.

Schema v2 persists the three-level catalog: raw EPA configs, consumer-facing
variants (the recommendation targets), and model-year families, plus variant
embeddings, model metadata, and chronologically ordered variant feedback.

These tests require the local pgvector database from compose.yaml:
`docker compose up -d`. The suite creates and removes a dedicated test
database so development data is untouched.
"""

import csv
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import torch

from app.data import (
    EXPECTED_CONFIGS,
    EXPECTED_CONSUMER_VARIANTS,
    EXPECTED_FAMILIES,
    group_consumer_variants,
    group_families,
    import_source_rows,
    load_manifest,
    load_snapshot,
)
from app.model import build_variant_features, untrained_model
from app.preference import (
    build_engine,
    posterior_from_feedback,
    rank_recommendations,
    select_probe,
)
from app.store import (
    DuplicateFeedbackError,
    LegacySchemaError,
    UnknownItemError,
    add_feedback,
    clear_feedback,
    connect,
    database_url,
    import_catalog,
    load_configs,
    load_families,
    load_feedback,
    load_meta,
    load_model,
    load_variants,
    save_model,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"

TABLES = {
    "consumer_variants", "epa_configs", "feedback",
    "model_meta", "variant_embeddings", "vehicle_families",
}
V1_TABLES = ("feedback", "item_embeddings", "vehicle_variants", "vehicle_items", "model_meta")


def read_fixture_configs():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        return import_source_rows(list(csv.DictReader(f)))


def fixture_manifest(configs):
    return {
        "snapshot_sha256": "f" * 64,
        "config_count": len(configs),
        "consumer_variant_count": len(group_consumer_variants(configs)),
        "family_count": len(group_families(configs)),
    }


def expected_variant_rows(configs):
    """The persisted consumer-variant contract: aggregate features plus config count."""
    counts = {v["variant_id"]: v["config_count"] for v in group_consumer_variants(configs)}
    return [
        {**feature, "config_count": counts[feature["variant_id"]]}
        for feature in build_variant_features(configs)
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
        admin.execute("DROP TABLE IF EXISTS " + ", ".join(V1_TABLES) + " CASCADE")
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


def test_connect_refuses_legacy_schema_until_explicit_reset(conn):
    """V1 feedback targets families, so normal startup must not destroy it."""
    for table in TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    conn.execute("CREATE TABLE vehicle_items (item_id text PRIMARY KEY)")
    conn.execute("CREATE TABLE vehicle_variants (epa_id integer PRIMARY KEY, item_id text)")
    conn.execute("CREATE TABLE item_embeddings (item_id text PRIMARY KEY)")
    conn.execute("CREATE TABLE model_meta (snapshot_sha256 text)")
    conn.execute("CREATE TABLE feedback (item_id text PRIMARY KEY, liked boolean)")
    conn.execute("INSERT INTO feedback VALUES ('2017|Toyota|Camry', true)")

    with pytest.raises(LegacySchemaError, match="explicit reset"):
        connect()
    assert conn.execute("SELECT count(*) FROM feedback").fetchone()[0] == 1
    assert "vehicle_items" in public_tables(conn)

    upgraded = connect(allow_legacy_reset=True)
    try:
        assert public_tables(upgraded) == TABLES
        assert load_meta(upgraded) is None
        assert load_feedback(upgraded) == []
        configs = read_fixture_configs()
        assert import_catalog(upgraded, configs, fixture_manifest(configs)) is True
    finally:
        upgraded.close()


def test_fixture_catalog_round_trip(conn):
    configs = read_fixture_configs()
    assert import_catalog(conn, configs, fixture_manifest(configs)) is True
    assert load_configs(conn) == configs
    assert load_families(conn) == group_families(configs)
    variants = load_variants(conn)
    assert variants == expected_variant_rows(configs)
    by_id = {v["variant_id"]: v for v in variants}
    le = by_id["2017|Toyota|Camry LE"]
    assert le["family_id"] == "2017|Toyota|Camry"
    assert le["base_model"] == "Camry"
    assert le["config_count"] == 1
    assert le["combined_mpg"] == 32.0
    bolt = by_id["2026|Chevrolet|Bolt EUV"]
    assert bolt["cylinders"] is None
    assert bolt["displacement"] is None
    assert load_meta(conn) == {"snapshot_sha256": "f" * 64, "bias": None}


def test_import_is_transactional_when_a_config_fails(conn):
    configs = read_fixture_configs()
    broken = configs + [dict(configs[-1])]  # duplicate EPA ID fails mid-load
    manifest = fixture_manifest(read_fixture_configs())
    manifest["config_count"] += 1
    with pytest.raises(psycopg.errors.UniqueViolation):
        import_catalog(conn, broken, manifest)
    assert load_configs(conn) == []
    assert load_variants(conn) == []
    assert load_families(conn) == []
    assert load_meta(conn) is None


def test_import_runs_once_and_rejects_a_changed_snapshot(conn):
    configs = read_fixture_configs()
    manifest = fixture_manifest(configs)
    assert import_catalog(conn, configs, manifest) is True
    assert import_catalog(conn, configs, manifest) is False
    assert load_configs(conn) == configs
    with pytest.raises(ValueError):
        import_catalog(conn, configs, {**manifest, "snapshot_sha256": "0" * 64})


def test_import_rejects_manifest_count_mismatch(conn):
    configs = read_fixture_configs()
    manifest = fixture_manifest(configs)
    for key in ("config_count", "consumer_variant_count", "family_count"):
        with pytest.raises(ValueError):
            import_catalog(conn, configs, {**manifest, key: 999})
    assert load_configs(conn) == []
    assert load_meta(conn) is None


def test_feedback_is_chronological_and_recorded_once_per_variant(conn):
    configs = read_fixture_configs()
    import_catalog(conn, configs, fixture_manifest(configs))
    add_feedback(conn, "2026|Chevrolet|Bolt EUV", False)
    add_feedback(conn, "2017|Toyota|Camry LE", True)
    add_feedback(conn, "2017|Toyota|Camry XSE", False)
    # Stable audit order is preserved even though the current posterior update
    # is mathematically order-independent.
    history = [
        ("2026|Chevrolet|Bolt EUV", False),
        ("2017|Toyota|Camry LE", True),
        ("2017|Toyota|Camry XSE", False),
    ]
    assert load_feedback(conn) == history
    with pytest.raises(DuplicateFeedbackError):
        add_feedback(conn, "2017|Toyota|Camry LE", False)
    with pytest.raises(UnknownItemError):
        add_feedback(conn, "1999|Yugo|GV", True)
    # Rejected events leave the connection usable and the history unchanged.
    assert load_feedback(conn) == history
    clear_feedback(conn)
    assert load_feedback(conn) == []
    add_feedback(conn, "2017|Toyota|Camry LE", False)
    assert load_feedback(conn) == [("2017|Toyota|Camry LE", False)]


def test_model_round_trip_preserves_embeddings_and_bias(conn):
    configs = read_fixture_configs()
    model = untrained_model(build_variant_features(configs), seed=0)._replace(bias=-0.25)
    with pytest.raises(ValueError):
        save_model(conn, model)  # the catalog must be imported first
    import_catalog(conn, configs, fixture_manifest(configs))
    assert load_model(conn) is None
    save_model(conn, model)
    loaded = load_model(conn)
    assert loaded.variant_ids == model.variant_ids
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
    return build_variant_features(snapshot)


@pytest.fixture(scope="module")
def catalog_model(features):
    return untrained_model(features, seed=0)._replace(bias=-0.125)


def test_frozen_snapshot_round_trip_records_manifest_checksum(conn, snapshot, manifest, features):
    assert import_catalog(conn, snapshot, manifest) is True
    assert len(load_configs(conn)) == EXPECTED_CONFIGS
    assert load_configs(conn) == snapshot
    variants = load_variants(conn)
    assert len(variants) == EXPECTED_CONSUMER_VARIANTS
    assert variants == expected_variant_rows(snapshot)
    assert len(load_families(conn)) == EXPECTED_FAMILIES
    assert load_meta(conn) == {"snapshot_sha256": manifest["snapshot_sha256"], "bias": None}


def test_restart_reconstructs_posterior_probe_and_ranking_from_feedback(
    conn, snapshot, manifest, features, catalog_model
):
    import_catalog(conn, snapshot, manifest)
    save_model(conn, catalog_model)
    engine = build_engine(catalog_model, features)
    events = [
        ("2024|Toyota|RAV4 AWD", False),
        ("2024|Ford|F150 Pickup 4WD", True),
        ("2021|Toyota|Tundra 4WD", True),
        ("2022|Toyota|Tundra 4WD", False),
    ]
    for variant_id, liked in events:
        add_feedback(conn, variant_id, liked)
    assert load_feedback(conn) == events
    posterior = posterior_from_feedback(engine, events)
    rated = {variant_id for variant_id, _ in events}
    ranking = rank_recommendations(engine, posterior, exclude=rated, limit=10)
    probe = select_probe(engine, posterior, events)
    conn.close()

    reopened = connect()
    try:
        restored_model = load_model(reopened)
        assert restored_model.variant_ids == catalog_model.variant_ids
        assert restored_model.bias == catalog_model.bias
        assert torch.equal(restored_model.embeddings, catalog_model.embeddings)
        restored_events = load_feedback(reopened)
        assert restored_events == events
        restored_engine = build_engine(restored_model, load_variants(reopened))
        restored_posterior = posterior_from_feedback(restored_engine, restored_events)
        assert torch.equal(restored_posterior, posterior)
        assert rank_recommendations(restored_engine, restored_posterior,
                                    exclude=rated, limit=10) == ranking
        assert select_probe(restored_engine, restored_posterior, restored_events) == probe
    finally:
        reopened.close()
