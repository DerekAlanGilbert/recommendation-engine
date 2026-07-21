"""Plain-SQL PostgreSQL persistence for the three-level catalog and feedback.

Feedback is the authoritative preference state; the posterior is never
persisted and is always replayed in memory from the complete history. Events
retain a stable audit order even though the current posterior is order-independent.
Model data (embeddings and bias) lives only in the models/ file artifact
(see app.artifact), never in PostgreSQL.
"""

import os
from pathlib import Path

import psycopg

from app.data import (
    SNAPSHOT_COLUMNS,
    consumer_key,
    family_id_of,
    group_consumer_variants,
    group_families,
)
from app.model import build_variant_features

DEFAULT_DATABASE_URL = "postgresql://rec:rec@localhost:5433/rec"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

FAMILY_COLUMNS = ("family_id", "year", "make", "base_model", "config_count")

VARIANT_COLUMNS = (
    "variant_id", "family_id", "year", "make", "model", "base_model",
    "vehicle_class", "fuel_type", "drive_family", "transmission_family",
    "city_mpg", "highway_mpg", "combined_mpg", "cylinders", "displacement",
    "electric_range", "co2_tailpipe_gpm", "config_count",
)

CONFIG_COLUMNS = ("epa_id", "variant_id", "family_id") + tuple(
    c for c in SNAPSHOT_COLUMNS if c != "epa_id"
)


class DuplicateFeedbackError(Exception):
    """The consumer variant already has recorded feedback."""


class UnknownItemError(Exception):
    """The consumer variant does not exist in the catalog."""


class LegacySchemaError(Exception):
    """A v1 family-level schema requires an explicit destructive reset."""


LEGACY_RESET_TABLES = (
    "feedback", "variant_embeddings", "epa_configs", "consumer_variants",
    "vehicle_families", "item_embeddings", "vehicle_variants", "vehicle_items",
    "model_meta",
)


def database_url():
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def _has_legacy_schema(conn):
    return conn.execute("SELECT to_regclass('public.vehicle_items')").fetchone()[0] is not None


def _drop_legacy_schema(conn):
    tables = ", ".join(LEGACY_RESET_TABLES)
    with conn.transaction():
        conn.execute(f"DROP TABLE IF EXISTS {tables} CASCADE")


def _drop_v2_model_persistence(conn):
    """Schema v3 keeps model data out of PostgreSQL. Migrate a v2 database in
    place by dropping only its model store; catalog and feedback are untouched."""
    with conn.transaction():
        conn.execute("DROP TABLE IF EXISTS variant_embeddings")
        conn.execute("ALTER TABLE IF EXISTS model_meta DROP COLUMN IF EXISTS bias")
        conn.execute("DROP EXTENSION IF EXISTS vector")


def connect(url=None, allow_legacy_reset=False):
    """Open a connection and apply schema v3.

    A v1 schema contains family-level feedback that cannot be mapped safely to
    consumer variants. Normal startup therefore fails closed. Only the explicit
    reset utility passes ``allow_legacy_reset=True`` to destroy that state.
    A v2 schema differs only in its model store and is migrated in place
    without touching catalog or feedback.
    """
    conn = psycopg.connect(url or database_url(), autocommit=True)
    if _has_legacy_schema(conn):
        if not allow_legacy_reset:
            conn.close()
            raise LegacySchemaError(
                "legacy v1 schema detected; explicit reset required: run "
                "scripts/reset_v1_schema.py with --confirm-destroy-feedback "
                "before starting the API"
            )
        _drop_legacy_schema(conn)
    _drop_v2_model_persistence(conn)
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def load_meta(conn):
    row = conn.execute("SELECT snapshot_sha256 FROM model_meta").fetchone()
    return None if row is None else {"snapshot_sha256": row[0]}


def _variant_rows(configs):
    """Merge deterministic aggregate features with the consumer grouping."""
    counts = {v["variant_id"]: v["config_count"] for v in group_consumer_variants(configs)}
    return [
        {**feature, "config_count": counts[feature["variant_id"]]}
        for feature in build_variant_features(configs)
    ]


def import_catalog(conn, configs, manifest):
    """Transactionally load the frozen catalog once; return False when already loaded."""
    meta = load_meta(conn)
    if meta is not None:
        if meta["snapshot_sha256"] != manifest["snapshot_sha256"]:
            raise ValueError(
                f"stored catalog checksum {meta['snapshot_sha256']} does not match "
                f"manifest checksum {manifest['snapshot_sha256']}"
            )
        return False
    families = group_families(configs)
    variants = _variant_rows(configs)
    expected = (
        ("config_count", len(configs)),
        ("consumer_variant_count", len(variants)),
        ("family_count", len(families)),
    )
    for key, actual in expected:
        if manifest[key] != actual:
            raise ValueError(f"expected {manifest[key]} for {key}, got {actual}")
    variant_id_by_key = {consumer_key(c): None for c in configs}
    for variant in variants:
        variant_id_by_key[(variant["year"], variant["make"], variant["model"])] = variant["variant_id"]
    with conn.transaction():
        with conn.cursor() as cur:
            with cur.copy(f"COPY vehicle_families ({', '.join(FAMILY_COLUMNS)}) FROM STDIN") as copy:
                for family in families:
                    copy.write_row(tuple(family[column] for column in FAMILY_COLUMNS))
            with cur.copy(f"COPY consumer_variants ({', '.join(VARIANT_COLUMNS)}) FROM STDIN") as copy:
                for variant in variants:
                    copy.write_row(tuple(variant[column] for column in VARIANT_COLUMNS))
            with cur.copy(f"COPY epa_configs ({', '.join(CONFIG_COLUMNS)}) FROM STDIN") as copy:
                for config in configs:
                    row = {
                        **config,
                        "variant_id": variant_id_by_key[consumer_key(config)],
                        "family_id": family_id_of(config),
                    }
                    copy.write_row(tuple(row[column] for column in CONFIG_COLUMNS))
        conn.execute(
            "INSERT INTO model_meta (snapshot_sha256) VALUES (%s)",
            (manifest["snapshot_sha256"],),
        )
    return True


def load_configs(conn):
    """Return all raw EPA configs in the canonical snapshot order."""
    rows = conn.execute(f"SELECT {', '.join(SNAPSHOT_COLUMNS)} FROM epa_configs").fetchall()
    configs = [dict(zip(SNAPSHOT_COLUMNS, row)) for row in rows]
    configs.sort(key=lambda c: (c["year"], c["make"], c["base_model"], c["model"], c["epa_id"]))
    return configs


def load_families(conn):
    """Return all model-year families with their config groupings."""
    rows = conn.execute(
        "SELECT f.family_id, f.year, f.make, f.base_model, f.config_count, "
        "array_agg(c.epa_id ORDER BY c.epa_id) "
        "FROM vehicle_families f JOIN epa_configs c ON c.family_id = f.family_id "
        "GROUP BY f.family_id"
    ).fetchall()
    families = [
        {
            "family_id": family_id, "year": year, "make": make, "base_model": base_model,
            "config_epa_ids": list(epa_ids), "config_count": config_count,
        }
        for family_id, year, make, base_model, config_count, epa_ids in rows
    ]
    families.sort(key=lambda f: (f["year"], f["make"], f["base_model"]))
    return families


def load_variants(conn):
    """Return all consumer variants in the deterministic (year, make, model) order."""
    rows = conn.execute(f"SELECT {', '.join(VARIANT_COLUMNS)} FROM consumer_variants").fetchall()
    variants = [dict(zip(VARIANT_COLUMNS, row)) for row in rows]
    variants.sort(key=lambda v: (v["year"], v["make"], v["model"]))
    return variants


def add_feedback(conn, variant_id, liked):
    """Record one binary response per consumer variant; reject duplicates and unknowns."""
    try:
        conn.execute(
            "INSERT INTO feedback (variant_id, liked) VALUES (%s, %s)", (variant_id, liked)
        )
    except psycopg.errors.UniqueViolation as error:
        raise DuplicateFeedbackError(variant_id) from error
    except psycopg.errors.ForeignKeyViolation as error:
        raise UnknownItemError(variant_id) from error


def load_feedback(conn):
    """Return the complete feedback history as (variant_id, liked), chronologically."""
    rows = conn.execute(
        "SELECT variant_id, liked FROM feedback ORDER BY event_order"
    ).fetchall()
    return [(variant_id, liked) for variant_id, liked in rows]


def clear_feedback(conn):
    conn.execute("DELETE FROM feedback")
