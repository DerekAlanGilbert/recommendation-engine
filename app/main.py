"""Single-profile FastAPI service: reset, recommendations, feedback, health.

Every canonical item is scored exactly in process memory. Feedback in
PostgreSQL is the authoritative profile state: the in-memory profile vector is
refit from the complete persisted history at startup and after every event.
"""

import threading
import time
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, StrictBool, StrictStr

from app import store
from app.data import load_manifest, load_snapshot
from app.model import (
    build_item_features,
    cold_start_profile,
    pretrain,
    rank_items,
    refit_profile,
)

# Heuristic evidence measure n / (n + EVIDENCE_PRIOR), not calibrated probability.
EVIDENCE_PRIOR = 5

CONNECT_ATTEMPTS = 30
CONNECT_DELAY_SECONDS = 1.0


class FeedbackRequest(BaseModel):
    item_id: StrictStr
    liked: StrictBool


def evidence_strength(feedback_count):
    return feedback_count / (feedback_count + EVIDENCE_PRIOR)


def similarity_reason(model, items, feedback, item_id):
    """Cite the rated item by embedding similarity — similarity, not causal influence."""
    if not feedback:
        return "no feedback yet"
    candidate = model.embeddings[model.index[item_id]]
    similarity = {
        rated_id: float(candidate @ model.embeddings[model.index[rated_id]])
        for rated_id, _ in feedback
    }
    liked_ids = [rated_id for rated_id, liked in feedback if liked]
    if liked_ids:
        cited = max(liked_ids, key=lambda rated_id: (similarity[rated_id], rated_id))
        template = "most similar to liked {label} (cosine similarity {value:.2f})"
    else:
        cited = min(similarity, key=lambda rated_id: (similarity[rated_id], rated_id))
        template = "least similar to disliked {label} (cosine similarity {value:.2f})"
    item = items[cited]
    label = f"{item['year']} {item['make']} {item['base_model']}"
    return template.format(label=label, value=similarity[cited])


class Service:
    """Serving state for the one active profile; `lock` serializes feedback/refit/persist."""

    def __init__(self, database_url, connect_attempts, connect_delay):
        self.database_url = database_url
        self.connect_attempts = connect_attempts
        self.connect_delay = connect_delay
        self.lock = threading.Lock()
        self.conn = None
        self.model = None
        self.items = {}
        self.feedback = []
        self.profile = None

    def start(self):
        self.conn = self._connect_with_retries()
        try:
            if store.load_meta(self.conn) is None:
                store.import_catalog(self.conn, load_snapshot(), load_manifest())
            self.model = store.load_model(self.conn)
            if self.model is None:
                self.model = pretrain(build_item_features(store.load_variants(self.conn)), seed=0)
                store.save_model(self.conn, self.model)
            self.items = {item["item_id"]: item for item in store.load_items(self.conn)}
            self.feedback = store.load_feedback(self.conn)
            self.profile = refit_profile(self.model, self.feedback)
        except Exception:
            self.conn.close()
            raise

    def _connect_with_retries(self):
        for attempt in range(1, self.connect_attempts + 1):
            try:
                return store.connect(self.database_url)
            except psycopg.OperationalError:
                if attempt == self.connect_attempts:
                    raise
                time.sleep(self.connect_delay)


def create_app(database_url=None, connect_attempts=CONNECT_ATTEMPTS,
               connect_delay=CONNECT_DELAY_SECONDS):
    service = Service(database_url, connect_attempts, connect_delay)

    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:
            yield
        finally:
            service.conn.close()

    app = FastAPI(title="Live Recommendation Engine", lifespan=lifespan)
    app.state.service = service

    @app.post("/reset")
    def reset():
        with service.lock:
            try:
                store.clear_feedback(service.conn)
            except psycopg.Error as error:
                raise HTTPException(status_code=503, detail="database unavailable") from error
            service.feedback = []
            service.profile = cold_start_profile()
        return {"feedback_count": 0, "evidence_strength": 0.0}

    @app.get("/recommendations")
    def recommendations(limit: int = Query(default=10, ge=1)):
        with service.lock:
            rated = {item_id for item_id, _ in service.feedback}
            ranking = rank_items(service.model, service.profile, exclude=rated, limit=limit)
            evidence = evidence_strength(len(service.feedback))
            return {
                "recommendations": [
                    {
                        **service.items[item_id],
                        "score": score,
                        "evidence_strength": evidence,
                        "reason": similarity_reason(
                            service.model, service.items, service.feedback, item_id
                        ),
                    }
                    for item_id, score in ranking
                ]
            }

    @app.post("/feedback")
    def feedback(request: FeedbackRequest):
        with service.lock:
            if request.item_id not in service.items:
                raise HTTPException(status_code=404, detail=f"unknown item: {request.item_id}")
            try:
                store.add_feedback(service.conn, request.item_id, request.liked)
            except store.DuplicateFeedbackError as error:
                raise HTTPException(
                    status_code=409, detail=f"feedback already recorded for {request.item_id}"
                ) from error
            except store.UnknownItemError as error:
                raise HTTPException(
                    status_code=404, detail=f"unknown item: {request.item_id}"
                ) from error
            except psycopg.Error as error:
                raise HTTPException(status_code=503, detail="database unavailable") from error
            service.feedback = store.load_feedback(service.conn)
            service.profile = refit_profile(service.model, service.feedback)
            count = len(service.feedback)
        return {
            "item_id": request.item_id,
            "liked": request.liked,
            "feedback_count": count,
            "evidence_strength": evidence_strength(count),
        }

    @app.get("/health")
    def health():
        with service.lock:
            try:
                service.conn.execute("SELECT 1")
            except psycopg.Error as error:
                raise HTTPException(status_code=503, detail="database unavailable") from error
            return {
                "status": "ok",
                "database": "ok",
                "catalog_items": len(service.items),
                "model_loaded": service.model is not None,
                "feedback_count": len(service.feedback),
            }

    return app


app = create_app()
