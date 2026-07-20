"""API tests: the single-profile FastAPI contract over an isolated fixture-seeded database.

The API separates two concepts: GET /probe returns the next package for
active preference elicitation, and GET /recommendations returns the current
best-fit exploitation ranking. Expected probes, posteriors, and rankings are
recomputed independently with app.preference, so the API is held to exact
in-memory inference. These tests require the local pgvector database from
compose.yaml: `docker compose up -d`.
"""

import csv
from pathlib import Path

import psycopg
import pytest
import torch
from fastapi.testclient import TestClient

from app import store
from app.data import group_consumer_variants, group_families, import_source_rows
from app.main import EVIDENCE_PRIOR, evidence_strength
from app.main import create_app
from app.model import build_variant_features, untrained_model
from app.preference import (
    MAX_EXPLORATION_WEIGHT,
    build_engine,
    posterior_from_feedback,
    rank_recommendations,
    select_probe,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"

TABLES = ("consumer_variants", "epa_configs", "feedback",
          "model_meta", "variant_embeddings", "vehicle_families")

VARIANT_FIELDS = {
    "variant_id", "family_id", "year", "make", "model", "base_model",
    "vehicle_class", "fuel_type", "drive_family", "transmission_family",
    "city_mpg", "highway_mpg", "combined_mpg", "cylinders", "displacement",
    "electric_range", "co2_tailpipe_gpm", "config_count",
}


@pytest.fixture(scope="module")
def configs():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        return import_source_rows(list(csv.DictReader(f)))


@pytest.fixture(scope="module")
def manifest(configs):
    return {
        "snapshot_sha256": "f" * 64,
        "config_count": len(configs),
        "consumer_variant_count": len(group_consumer_variants(configs)),
        "family_count": len(group_families(configs)),
    }


@pytest.fixture(scope="module")
def features(configs):
    return build_variant_features(configs)


@pytest.fixture(scope="module")
def model(features):
    return untrained_model(features, seed=0)._replace(bias=-0.125)


@pytest.fixture(scope="module")
def engine(model, features):
    return build_engine(model, features)


def drop_tables(url):
    with psycopg.connect(url, autocommit=True, connect_timeout=5) as raw:
        raw.execute("DROP TABLE IF EXISTS " + ", ".join(TABLES) + " CASCADE")


@pytest.fixture()
def database(api_database_url, configs, manifest, model):
    """A fresh schema seeded with the fixture catalog and a deterministic frozen model."""
    drop_tables(api_database_url)
    conn = store.connect(api_database_url)
    try:
        store.import_catalog(conn, configs, manifest)
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


def variant_label(features, variant_id):
    feature = next(f for f in features if f["variant_id"] == variant_id)
    return f"{feature['year']} {feature['make']} {feature['model']}"


def variant_ids(body):
    return [rec["variant_id"] for rec in body["recommendations"]]


def ranked_pairs(body):
    return [(rec["variant_id"], rec["score"]) for rec in body["recommendations"]]


def test_health_reports_catalog_model_and_database_readiness(client, features):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "consumer_variants": len(features),
        "families": 4,
        "model_loaded": True,
        "feedback_count": 0,
    }


def test_health_feedback_and_reset_report_503_when_the_database_is_unavailable(client):
    client.app.state.service.conn.close()
    assert client.get("/health").status_code == 503
    assert client.post(
        "/feedback", json={"variant_id": "2017|Toyota|Camry LE", "liked": True}
    ).status_code == 503
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


def test_cold_start_probe_is_the_active_elicitation_choice(client, engine):
    expected = select_probe(engine, posterior_from_feedback(engine, []))
    response = client.get("/probe")
    assert response.status_code == 200
    body = response.json()
    assert body["variant_id"] == expected["variant_id"]
    assert body["expected_information_gain"] == pytest.approx(expected["expected_information_gain"])
    assert body["expected_approval"] == pytest.approx(expected["expected_approval"])
    assert body["exploration_weight"] == pytest.approx(MAX_EXPLORATION_WEIGHT)
    assert body["evidence_strength"] == 0.0
    assert body["reason"] == "no feedback yet"
    assert VARIANT_FIELDS <= set(body)
    # deterministic: the same request returns the identical response
    assert client.get("/probe").json() == body


def test_cold_start_recommendations_rank_by_posterior(client, engine, features):
    posterior = posterior_from_feedback(engine, [])
    expected = rank_recommendations(engine, posterior, limit=3)
    response = client.get("/recommendations?limit=3")
    assert response.status_code == 200
    body = response.json()
    assert [(rec["variant_id"], pytest.approx(rec["score"])) for rec in body["recommendations"]] == [
        (variant_id, pytest.approx(score)) for variant_id, score in expected
    ]
    by_id = {f["variant_id"]: f for f in features}
    for rec in body["recommendations"]:
        assert rec["evidence_strength"] == 0.0
        assert rec["reason"] == "no feedback yet"
        assert VARIANT_FIELDS <= set(rec)
        for name in ("year", "make", "model", "base_model", "vehicle_class", "fuel_type"):
            assert rec[name] == by_id[rec["variant_id"]][name]
    assert client.get("/recommendations?limit=3").json() == body
    # the default limit returns every unrated variant in this small catalog
    assert len(variant_ids(client.get("/recommendations").json())) == len(features)


def test_recommendations_reject_an_invalid_limit(client):
    for invalid in ("0", "-2", "abc"):
        assert client.get(f"/recommendations?limit={invalid}").status_code == 422


def test_feedback_immediately_updates_probe_and_recommendations(client, engine, features):
    target = client.get("/probe").json()["variant_id"]
    response = client.post("/feedback", json={"variant_id": target, "liked": True})
    assert response.status_code == 200
    assert response.json() == {
        "variant_id": target,
        "liked": True,
        "feedback_count": 1,
        "evidence_strength": 1 / (1 + EVIDENCE_PRIOR),
    }
    events = [(target, True)]
    posterior = posterior_from_feedback(engine, events)
    expected = rank_recommendations(engine, posterior, exclude={target})
    body = client.get(f"/recommendations?limit={len(features)}").json()
    assert variant_ids(body) == [variant_id for variant_id, _ in expected]
    assert target not in variant_ids(body)
    label = variant_label(features, target)
    for rec in body["recommendations"]:
        assert rec["evidence_strength"] == 1 / (1 + EVIDENCE_PRIOR)
        assert rec["reason"].startswith(f"most similar to liked {label} (similarity ")
    probe = client.get("/probe").json()
    assert probe["variant_id"] == select_probe(engine, posterior, events)["variant_id"]
    assert probe["variant_id"] != target


def test_feedback_rolls_back_if_posterior_derivation_fails(client, monkeypatch):
    """A failed in-memory update must not leave a committed, unretryable thumb."""
    target = "2017|Toyota|Camry LE"
    service = client.app.state.service
    original_feedback = list(service.feedback)
    original_posterior = service.posterior.clone()

    def fail_derivation(*_args, **_kwargs):
        raise RuntimeError("injected posterior failure")

    monkeypatch.setattr("app.main.posterior_from_feedback", fail_derivation)
    with pytest.raises(RuntimeError, match="injected posterior failure"):
        client.post("/feedback", json={"variant_id": target, "liked": True})

    assert store.load_feedback(service.conn) == []
    assert service.feedback == original_feedback
    assert torch.equal(service.posterior, original_posterior)


def test_disliked_only_feedback_cites_a_contrast_reason(client, engine, features):
    target = sorted(f["variant_id"] for f in features)[0]
    assert client.post("/feedback", json={"variant_id": target, "liked": False}).status_code == 200
    posterior = posterior_from_feedback(engine, [(target, False)])
    expected = rank_recommendations(engine, posterior, exclude={target})
    body = client.get(f"/recommendations?limit={len(features)}").json()
    assert variant_ids(body) == [variant_id for variant_id, _ in expected]
    label = variant_label(features, target)
    for rec in body["recommendations"]:
        assert rec["reason"].startswith(f"least similar to disliked {label} (similarity ")


def test_feedback_rejects_unknown_duplicate_and_invalid_input(client):
    assert client.post(
        "/feedback", json={"variant_id": "1999|Yugo|GV", "liked": True}
    ).status_code == 404
    target = "2017|Toyota|Camry LE"
    assert client.post("/feedback", json={"variant_id": target, "liked": True}).status_code == 200
    assert client.post("/feedback", json={"variant_id": target, "liked": False}).status_code == 409
    for invalid in (
        {},
        {"variant_id": target},
        {"liked": True},
        {"variant_id": target, "liked": "yes"},
        {"variant_id": 7, "liked": True},
        {"item_id": target, "liked": True},
    ):
        assert client.post("/feedback", json=invalid).status_code == 422
    # rejected events leave the recorded history unchanged
    assert client.get("/health").json()["feedback_count"] == 1


def test_rated_variants_never_reappear_and_the_catalog_can_be_exhausted(client, features):
    ids = sorted(f["variant_id"] for f in features)
    for index, variant_id in enumerate(ids):
        response = client.post("/feedback", json={"variant_id": variant_id, "liked": index % 2 == 0})
        assert response.status_code == 200
        assert response.json()["evidence_strength"] == evidence_strength(index + 1)
        remaining = variant_ids(client.get(f"/recommendations?limit={len(ids)}").json())
        assert set(remaining).isdisjoint(ids[: index + 1])
        assert len(remaining) == len(ids) - index - 1
        probe = client.get("/probe")
        if index < len(ids) - 1:
            assert probe.status_code == 200
            assert probe.json()["variant_id"] not in ids[: index + 1]
        else:
            assert probe.status_code == 404
    assert client.get(f"/recommendations?limit={len(ids)}").json()["recommendations"] == []


def test_reset_restores_the_reproducible_cold_start(client, features):
    cold_recommendations = client.get(f"/recommendations?limit={len(features)}").json()
    cold_probe = client.get("/probe").json()
    first = cold_probe["variant_id"]
    assert client.post("/feedback", json={"variant_id": first, "liked": True}).status_code == 200
    assert client.get(f"/recommendations?limit={len(features)}").json() != cold_recommendations
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json() == {"feedback_count": 0, "evidence_strength": 0.0}
    assert client.get(f"/recommendations?limit={len(features)}").json() == cold_recommendations
    assert client.get("/probe").json() == cold_probe
    # the previously rated variant is recommendable and ratable again
    assert client.post("/feedback", json={"variant_id": first, "liked": True}).status_code == 200


def test_restart_reconstructs_the_posterior_from_ordered_feedback(database, features):
    # The stable event order is persisted for audit/replay; the current static
    # likelihood produces the same posterior regardless of that order.
    ids = sorted(f["variant_id"] for f in features)
    events = [(ids[2], False), (ids[0], True), (ids[4], False)]
    with make_client(database) as first:
        for variant_id, liked in events:
            assert first.post(
                "/feedback", json={"variant_id": variant_id, "liked": liked}
            ).status_code == 200
        before_recommendations = first.get(f"/recommendations?limit={len(ids)}").json()
        before_probe = first.get("/probe").json()
    with make_client(database) as reopened:
        assert reopened.get("/health").json()["feedback_count"] == len(events)
        assert reopened.get(f"/recommendations?limit={len(ids)}").json() == before_recommendations
        assert reopened.get("/probe").json() == before_probe
        assert reopened.post(
            "/feedback", json={"variant_id": ids[2], "liked": True}
        ).status_code == 409


def test_startup_pretrains_and_persists_a_model_when_absent(api_database_url, configs, manifest):
    drop_tables(api_database_url)
    conn = store.connect(api_database_url)
    try:
        store.import_catalog(conn, configs, manifest)
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
    # a second startup reuses the persisted model and serves identical responses
    with make_client(api_database_url) as reopened:
        assert reopened.get("/recommendations?limit=2").json() == body


def test_vehicle_session_frontend_is_served_as_one_self_contained_html_file(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "Vehicle preference session" in html
    assert 'fetch("/reset", { method: "POST" })' in html
    assert 'fetch("/probe")' in html
    assert 'fetch("/feedback"' in html
    assert 'fetch("/recommendations?limit=5")' in html
    assert "https://" not in html
    assert "<link" not in html


def test_no_profile_endpoint_is_exposed(client):
    assert client.get("/profile").status_code == 404
