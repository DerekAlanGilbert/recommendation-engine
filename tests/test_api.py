"""API tests: the single-profile FastAPI contract over an isolated fixture-seeded database.

Expected scores and rankings are recomputed independently with app.model, so the
API is held to exact in-memory scoring rather than approximate behavior. These
tests require the local pgvector database from compose.yaml: `docker compose up -d`.
"""

import csv
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from app import store
from app.data import group_variants, import_source_rows
from app.main import create_app
from app.model import (
    build_item_features,
    cold_start_profile,
    rank_items,
    refit_profile,
    untrained_model,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"

TABLES = ("feedback", "item_embeddings", "model_meta", "vehicle_items", "vehicle_variants")


@pytest.fixture(scope="module")
def variants():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        return import_source_rows(list(csv.DictReader(f)))


@pytest.fixture(scope="module")
def manifest(variants):
    return {
        "snapshot_sha256": "f" * 64,
        "variant_count": len(variants),
        "canonical_count": len(group_variants(variants)),
    }


@pytest.fixture(scope="module")
def features(variants):
    return build_item_features(variants)


@pytest.fixture(scope="module")
def model(features):
    return untrained_model(features, seed=0)._replace(bias=-0.125)


def drop_tables(url):
    with psycopg.connect(url, autocommit=True, connect_timeout=5) as raw:
        raw.execute("DROP TABLE IF EXISTS " + ", ".join(TABLES) + " CASCADE")


@pytest.fixture()
def database(api_database_url, variants, manifest, model):
    """A fresh schema seeded with the fixture catalog and a deterministic frozen model."""
    drop_tables(api_database_url)
    conn = store.connect(api_database_url)
    try:
        store.import_catalog(conn, variants, manifest)
        store.save_model(conn, model)
    finally:
        conn.close()
    return api_database_url


def make_client(url):
    return TestClient(create_app(database_url=url, connect_attempts=1))


@pytest.fixture()
def client(database):
    with make_client(database) as started:
        yield started


def label_by_id(variants):
    """The human label the reason field cites, from the canonical grouping contract."""
    return {
        item["item_id"]: f"{item['year']} {item['make']} {item['base_model']}"
        for item in group_variants(variants)
    }


def item_ids(body):
    return [rec["item_id"] for rec in body["recommendations"]]


def ranked_pairs(body):
    return [(rec["item_id"], rec["score"]) for rec in body["recommendations"]]


def test_health_reports_api_catalog_model_and_database_readiness(client, features):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "catalog_items": len(features),
        "model_loaded": True,
        "feedback_count": 0,
    }


def test_health_reports_503_when_the_database_is_unavailable(client):
    client.app.state.service.conn.close()
    assert client.get("/health").status_code == 503
    assert client.post("/feedback", json={"item_id": "2017|Toyota|Camry", "liked": True}).status_code == 503
    assert client.post("/reset").status_code == 503


def test_startup_fails_after_bounded_retries_when_the_database_is_unreachable():
    app = create_app(
        database_url="postgresql://rec:rec@127.0.0.1:9/rec?connect_timeout=1",
        connect_attempts=2,
        connect_delay=0.0,
    )
    with pytest.raises(psycopg.OperationalError):
        with TestClient(app):
            pass


def test_cold_start_recommendations_score_all_items_exactly(client, model, features, variants):
    expected = rank_items(model, cold_start_profile(), limit=3)
    response = client.get("/recommendations?limit=3")
    assert response.status_code == 200
    body = response.json()
    assert ranked_pairs(body) == expected
    grouped = {i["item_id"]: i for i in group_variants(variants)}
    by_id = {f["item_id"]: f for f in features}
    for rec in body["recommendations"]:
        assert rec["evidence_strength"] == 0.0
        assert rec["reason"] == "no feedback yet"
        assert rec["variant_count"] == grouped[rec["item_id"]]["variant_count"]
        assert rec["base_model"] == grouped[rec["item_id"]]["base_model"]
        for name in ("year", "make", "vehicle_class", "fuel_type", "combined_mpg"):
            assert rec[name] == by_id[rec["item_id"]][name]
    # deterministic: the same request returns the identical response
    assert client.get("/recommendations?limit=3").json() == body
    # the default limit returns every unrated item in this small catalog
    assert len(item_ids(client.get("/recommendations").json())) == len(features)


def test_recommendations_reject_an_invalid_limit(client):
    for invalid in ("0", "-2", "abc"):
        assert client.get(f"/recommendations?limit={invalid}").status_code == 422


def test_feedback_immediately_updates_the_profile_and_hides_rated_items(
    client, model, features, variants
):
    target = item_ids(client.get("/recommendations?limit=1").json())[0]
    response = client.post("/feedback", json={"item_id": target, "liked": True})
    assert response.status_code == 200
    assert response.json() == {
        "item_id": target,
        "liked": True,
        "feedback_count": 1,
        "evidence_strength": 1 / 6,
    }
    profile = refit_profile(model, [(target, True)])
    expected = rank_items(model, profile, exclude={target})
    body = client.get(f"/recommendations?limit={len(features)}").json()
    assert ranked_pairs(body) == expected
    assert target not in item_ids(body)
    label = label_by_id(variants)[target]
    for rec in body["recommendations"]:
        assert rec["evidence_strength"] == 1 / 6
        assert rec["reason"].startswith(f"most similar to liked {label} (cosine similarity ")


def test_disliked_only_feedback_cites_a_contrast_reason(client, model, features, variants):
    target = sorted(f["item_id"] for f in features)[0]
    assert client.post("/feedback", json={"item_id": target, "liked": False}).status_code == 200
    profile = refit_profile(model, [(target, False)])
    expected = rank_items(model, profile, exclude={target})
    body = client.get(f"/recommendations?limit={len(features)}").json()
    assert ranked_pairs(body) == expected
    label = label_by_id(variants)[target]
    for rec in body["recommendations"]:
        assert rec["reason"].startswith(f"least similar to disliked {label} (cosine similarity ")


def test_feedback_rejects_unknown_duplicate_and_invalid_input(client):
    assert client.post("/feedback", json={"item_id": "1999|Yugo|GV", "liked": True}).status_code == 404
    target = "2017|Toyota|Camry"
    assert client.post("/feedback", json={"item_id": target, "liked": True}).status_code == 200
    assert client.post("/feedback", json={"item_id": target, "liked": False}).status_code == 409
    for invalid in (
        {},
        {"item_id": target},
        {"liked": True},
        {"item_id": target, "liked": "yes"},
        {"item_id": 7, "liked": True},
    ):
        assert client.post("/feedback", json=invalid).status_code == 422
    # rejected events leave the recorded history unchanged
    assert client.get("/health").json()["feedback_count"] == 1


def test_rated_items_never_reappear_and_the_catalog_can_be_exhausted(client, features):
    ids = sorted(f["item_id"] for f in features)
    for index, item_id in enumerate(ids):
        response = client.post("/feedback", json={"item_id": item_id, "liked": index % 2 == 0})
        assert response.status_code == 200
        assert response.json()["evidence_strength"] == (index + 1) / (index + 6)
        remaining = item_ids(client.get(f"/recommendations?limit={len(ids)}").json())
        assert set(remaining).isdisjoint(ids[: index + 1])
        assert len(remaining) == len(ids) - index - 1
    assert client.get(f"/recommendations?limit={len(ids)}").json()["recommendations"] == []


def test_reset_restores_the_reproducible_cold_start(client, features):
    cold = client.get(f"/recommendations?limit={len(features)}").json()
    first = item_ids(cold)[0]
    assert client.post("/feedback", json={"item_id": first, "liked": True}).status_code == 200
    assert client.get(f"/recommendations?limit={len(features)}").json() != cold
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json() == {"feedback_count": 0, "evidence_strength": 0.0}
    assert client.get(f"/recommendations?limit={len(features)}").json() == cold
    # the previously rated item is recommendable and ratable again
    assert client.post("/feedback", json={"item_id": first, "liked": True}).status_code == 200


def test_restart_reconstructs_the_profile_from_persisted_feedback(database, model, features):
    ids = sorted(f["item_id"] for f in features)
    events = [(ids[0], True), (ids[-1], False)]
    with make_client(database) as first:
        for item_id, liked in events:
            assert first.post("/feedback", json={"item_id": item_id, "liked": liked}).status_code == 200
        before = first.get(f"/recommendations?limit={len(ids)}").json()
    profile = refit_profile(model, events)
    assert ranked_pairs(before) == rank_items(model, profile, exclude={i for i, _ in events})
    with make_client(database) as reopened:
        assert reopened.get("/health").json()["feedback_count"] == len(events)
        assert reopened.get(f"/recommendations?limit={len(ids)}").json() == before
        assert reopened.post("/feedback", json={"item_id": ids[0], "liked": True}).status_code == 409


def test_startup_pretrains_and_persists_a_model_when_absent(api_database_url, variants, manifest):
    drop_tables(api_database_url)
    conn = store.connect(api_database_url)
    try:
        store.import_catalog(conn, variants, manifest)
        assert store.load_model(conn) is None
    finally:
        conn.close()
    with make_client(api_database_url) as started:
        body = started.get("/recommendations?limit=2").json()
        assert len(body["recommendations"]) == 2
    conn = store.connect(api_database_url)
    try:
        assert store.load_model(conn) is not None
    finally:
        conn.close()
    # a second startup reuses the persisted model and serves identical recommendations
    with make_client(api_database_url) as reopened:
        assert reopened.get("/recommendations?limit=2").json() == body


def test_no_profile_endpoint_is_exposed(client):
    assert client.get("/profile").status_code == 404
