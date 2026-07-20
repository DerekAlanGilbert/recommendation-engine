"""Plain-SQL PostgreSQL persistence: frozen catalog, item embeddings, model meta, feedback.

Feedback is the authoritative profile state; the active profile vector is never
persisted and is always refit in memory from the complete feedback history.
"""

import os
from pathlib import Path

import numpy as np
import psycopg
import torch
from pgvector.psycopg import register_vector

from app.data import SNAPSHOT_COLUMNS, canonical_key, group_variants
from app.model import RecommenderModel, build_item_features

DEFAULT_DATABASE_URL = "postgresql://rec:rec@localhost:5433/rec"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

ITEM_COLUMNS = (
    "item_id", "year", "make", "base_model", "vehicle_class", "fuel_type",
    "drive_family", "transmission_family", "city_mpg", "highway_mpg",
    "combined_mpg", "cylinders", "displacement", "electric_range",
    "co2_tailpipe_gpm", "variant_count",
)

VARIANT_COLUMNS = ("epa_id", "item_id") + tuple(c for c in SNAPSHOT_COLUMNS if c != "epa_id")


class DuplicateFeedbackError(Exception):
    """The canonical item already has recorded feedback."""


class UnknownItemError(Exception):
    """The canonical item does not exist in the catalog."""


def database_url():
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def connect(url=None):
    """Open a connection, apply the idempotent schema, and register the vector adapter."""
    conn = psycopg.connect(url or database_url(), autocommit=True)
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    register_vector(conn)
    return conn


def _variant_item_id(variant):
    year, make, name = canonical_key(variant)
    return f"{year}|{make}|{name}"


def _item_rows(variants):
    """Merge canonical grouping with deterministic aggregate features, one row per item."""
    features = {f["item_id"]: f for f in build_item_features(variants)}
    return [
        {
            **features[item["item_id"]],
            "base_model": item["base_model"],
            "variant_count": item["variant_count"],
        }
        for item in group_variants(variants)
    ]


def load_meta(conn):
    row = conn.execute("SELECT snapshot_sha256, bias FROM model_meta").fetchone()
    return None if row is None else {"snapshot_sha256": row[0], "bias": row[1]}


def import_catalog(conn, variants, manifest):
    """Transactionally load the frozen catalog once; return False when already loaded."""
    meta = load_meta(conn)
    if meta is not None:
        if meta["snapshot_sha256"] != manifest["snapshot_sha256"]:
            raise ValueError(
                f"stored catalog checksum {meta['snapshot_sha256']} does not match "
                f"manifest checksum {manifest['snapshot_sha256']}"
            )
        return False
    if len(variants) != manifest["variant_count"]:
        raise ValueError(f"expected {manifest['variant_count']} variants, got {len(variants)}")
    items = _item_rows(variants)
    if len(items) != manifest["canonical_count"]:
        raise ValueError(f"expected {manifest['canonical_count']} canonical items, got {len(items)}")
    with conn.transaction():
        with conn.cursor() as cur:
            with cur.copy(f"COPY vehicle_items ({', '.join(ITEM_COLUMNS)}) FROM STDIN") as copy:
                for item in items:
                    copy.write_row(tuple(item[column] for column in ITEM_COLUMNS))
            with cur.copy(f"COPY vehicle_variants ({', '.join(VARIANT_COLUMNS)}) FROM STDIN") as copy:
                for variant in variants:
                    row = {**variant, "item_id": _variant_item_id(variant)}
                    copy.write_row(tuple(row[column] for column in VARIANT_COLUMNS))
        conn.execute(
            "INSERT INTO model_meta (snapshot_sha256) VALUES (%s)",
            (manifest["snapshot_sha256"],),
        )
    return True


def load_variants(conn):
    """Return all source variants in the canonical snapshot order."""
    rows = conn.execute(f"SELECT {', '.join(SNAPSHOT_COLUMNS)} FROM vehicle_variants").fetchall()
    variants = [dict(zip(SNAPSHOT_COLUMNS, row)) for row in rows]
    variants.sort(key=lambda v: (v["year"], v["make"], v["base_model"], v["model"], v["epa_id"]))
    return variants


def load_items(conn):
    """Return all canonical items in the deterministic (year, make, baseModel) order."""
    rows = conn.execute(f"SELECT {', '.join(ITEM_COLUMNS)} FROM vehicle_items").fetchall()
    items = [dict(zip(ITEM_COLUMNS, row)) for row in rows]
    items.sort(key=lambda i: (i["year"], i["make"], i["base_model"]))
    return items


def save_model(conn, model):
    """Persist frozen item embeddings and the model bias, replacing any prior model."""
    with conn.transaction():
        updated = conn.execute("UPDATE model_meta SET bias = %s", (model.bias,))
        if updated.rowcount != 1:
            raise ValueError("import the catalog before saving a model")
        conn.execute("DELETE FROM item_embeddings")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO item_embeddings (item_id, embedding) VALUES (%s, %s)",
                [
                    (item_id, model.embeddings[row].numpy())
                    for row, item_id in enumerate(model.item_ids)
                ],
            )


def load_model(conn):
    """Reconstruct the frozen model from PostgreSQL, or return None before pretraining."""
    meta = load_meta(conn)
    if meta is None or meta["bias"] is None:
        return None
    item_ids = tuple(item["item_id"] for item in load_items(conn))
    vectors = dict(conn.execute("SELECT item_id, embedding FROM item_embeddings").fetchall())
    embeddings = torch.from_numpy(np.stack([vectors[item_id].to_numpy() for item_id in item_ids]))
    index = {item_id: row for row, item_id in enumerate(item_ids)}
    return RecommenderModel(item_ids, embeddings, meta["bias"], index)


def add_feedback(conn, item_id, liked):
    """Record one thumbs-up/down per canonical item; reject duplicates and unknown items."""
    try:
        conn.execute("INSERT INTO feedback (item_id, liked) VALUES (%s, %s)", (item_id, liked))
    except psycopg.errors.UniqueViolation as error:
        raise DuplicateFeedbackError(item_id) from error
    except psycopg.errors.ForeignKeyViolation as error:
        raise UnknownItemError(item_id) from error


def load_feedback(conn):
    """Return the complete feedback history as (item_id, liked), sorted by item."""
    rows = conn.execute("SELECT item_id, liked FROM feedback").fetchall()
    return sorted((item_id, liked) for item_id, liked in rows)


def clear_feedback(conn):
    conn.execute("DELETE FROM feedback")
