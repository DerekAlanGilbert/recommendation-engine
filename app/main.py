"""Single-profile FastAPI service: reset, probe, recommendations, feedback, health.

Two distinct read surfaces serve the one active profile (Targeted Learning —
Current Relationships, see `app.methodology`):

- GET /probe: the next package to show for feedback — Targeted Learning
  active elicitation (`select_probe`: expected information gain about the
  ideal variant alone, threshold as nuisance, with a plausibility term).
- GET /recommendations: the current best-fit ranking — exploitation of the
  posterior over ideal-variant hypotheses, with family-aware diversity.

Feedback in PostgreSQL is the authoritative preference state: the in-memory
posterior is replayed from the complete persisted history at startup and after
every event. Every variant hypothesis is scored exactly in process memory.
The frozen model is served from the validated models/ file artifact; startup
retrains and atomically replaces it whenever validation fails (see app.artifact).
"""

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, StrictBool, StrictStr

from app import store
from app.artifact import artifact_path as default_artifact_path
from app.artifact import load_valid_artifact, save_artifact
from app.data import load_manifest, load_snapshot
from app.methodology import (
    METHODOLOGY_ID,
    METHODOLOGY_NAME,
    MODEL_ID,
    PROBE_OBJECTIVE,
)
from app.model import pretrain
from app.preference import (
    build_engine,
    posterior_from_feedback,
    rank_recommendations,
    select_probe,
    similarity_between,
)

# Heuristic evidence measure n / (n + EVIDENCE_PRIOR), not calibrated probability.
EVIDENCE_PRIOR = 5

CONNECT_ATTEMPTS = 30
CONNECT_DELAY_SECONDS = 1.0
SESSION_HTML_PATH = Path(__file__).with_name("vehicle-session.html")


class FeedbackRequest(BaseModel):
    variant_id: StrictStr
    liked: StrictBool


def evidence_strength(feedback_count):
    return feedback_count / (feedback_count + EVIDENCE_PRIOR)


def similarity_reason(engine, variants, feedback, variant_id):
    """Cite the rated variant by blended similarity — similarity, not causal influence."""
    if not feedback:
        return "no feedback yet"
    similarity = {
        rated_id: similarity_between(engine, variant_id, rated_id)
        for rated_id, _ in feedback
    }
    liked_ids = [rated_id for rated_id, liked in feedback if liked]
    if liked_ids:
        cited = max(liked_ids, key=lambda rated_id: (similarity[rated_id], rated_id))
        template = "most similar to liked {label} (similarity {value:.2f})"
    else:
        cited = min(similarity, key=lambda rated_id: (similarity[rated_id], rated_id))
        template = "least similar to disliked {label} (similarity {value:.2f})"
    variant = variants[cited]
    label = f"{variant['year']} {variant['make']} {variant['model']}"
    return template.format(label=label, value=similarity[cited])


class Service:
    """Serving state for the one active profile; `lock` serializes feedback/replay/persist."""

    def __init__(self, database_url, connect_attempts, connect_delay, artifact_path):
        self.database_url = database_url
        self.connect_attempts = connect_attempts
        self.connect_delay = connect_delay
        self.artifact_path = artifact_path
        self.lock = threading.Lock()
        self.conn = None
        self.model = None
        self.engine = None
        self.variants = {}
        self.family_count = 0
        self.feedback = []
        self.posterior = None

    def start(self):
        self.conn = self._connect_with_retries()
        try:
            meta = store.load_meta(self.conn)
            if meta is None:
                store.import_catalog(self.conn, load_snapshot(), load_manifest())
                meta = store.load_meta(self.conn)
            variant_rows = store.load_variants(self.conn)
            variant_ids = tuple(variant["variant_id"] for variant in variant_rows)
            self.model = load_valid_artifact(
                self.artifact_path, meta["snapshot_sha256"], variant_ids
            )
            if self.model is None:
                self.model = pretrain(variant_rows, seed=0)
                save_artifact(self.model, meta["snapshot_sha256"], self.artifact_path)
            self.engine = build_engine(self.model, variant_rows)
            self.variants = {variant["variant_id"]: variant for variant in variant_rows}
            self.family_count = len(store.load_families(self.conn))
            self.feedback = store.load_feedback(self.conn)
            self.posterior = posterior_from_feedback(self.engine, self.feedback)
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
               connect_delay=CONNECT_DELAY_SECONDS, artifact_path=None):
    service = Service(
        database_url, connect_attempts, connect_delay,
        Path(artifact_path) if artifact_path is not None else default_artifact_path(),
    )

    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:
            yield
        finally:
            service.conn.close()

    app = FastAPI(title="Live Recommendation Engine", lifespan=lifespan)
    app.state.service = service

    @app.get("/", include_in_schema=False)
    def vehicle_session():
        return FileResponse(SESSION_HTML_PATH, media_type="text/html")

    @app.post("/reset")
    def reset():
        with service.lock:
            try:
                new_posterior = posterior_from_feedback(service.engine, [])
                with service.conn.transaction():
                    store.clear_feedback(service.conn)
            except psycopg.Error as error:
                raise HTTPException(status_code=503, detail="database unavailable") from error
            service.feedback = []
            service.posterior = new_posterior
        return {"feedback_count": 0, "evidence_strength": 0.0}

    @app.get("/probe")
    def probe():
        with service.lock:
            choice = select_probe(service.engine, service.posterior, service.feedback)
            if choice is None:
                raise HTTPException(status_code=404, detail="no unrated variants remain")
            return {
                **service.variants[choice["variant_id"]],
                "expected_information_gain": choice["expected_information_gain"],
                "expected_approval": choice["expected_approval"],
                "exploration_weight": choice["exploration_weight"],
                "evidence_strength": evidence_strength(len(service.feedback)),
                "reason": similarity_reason(
                    service.engine, service.variants, service.feedback, choice["variant_id"]
                ),
            }

    @app.get("/recommendations")
    def recommendations(limit: int = Query(default=10, ge=1)):
        with service.lock:
            rated = {variant_id for variant_id, _ in service.feedback}
            ranking = rank_recommendations(
                service.engine, service.posterior, exclude=rated, limit=limit
            )
            evidence = evidence_strength(len(service.feedback))
            return {
                "recommendations": [
                    {
                        **service.variants[variant_id],
                        "score": score,
                        "evidence_strength": evidence,
                        "reason": similarity_reason(
                            service.engine, service.variants, service.feedback, variant_id
                        ),
                    }
                    for variant_id, score in ranking
                ]
            }

    @app.post("/feedback")
    def feedback(request: FeedbackRequest):
        with service.lock:
            if request.variant_id not in service.variants:
                raise HTTPException(
                    status_code=404, detail=f"unknown variant: {request.variant_id}"
                )
            try:
                with service.conn.transaction():
                    store.add_feedback(service.conn, request.variant_id, request.liked)
                    new_feedback = store.load_feedback(service.conn)
                    new_posterior = posterior_from_feedback(service.engine, new_feedback)
            except store.DuplicateFeedbackError as error:
                raise HTTPException(
                    status_code=409, detail=f"feedback already recorded for {request.variant_id}"
                ) from error
            except store.UnknownItemError as error:
                raise HTTPException(
                    status_code=404, detail=f"unknown variant: {request.variant_id}"
                ) from error
            except psycopg.Error as error:
                raise HTTPException(status_code=503, detail="database unavailable") from error
            service.feedback = new_feedback
            service.posterior = new_posterior
            count = len(service.feedback)
        return {
            "variant_id": request.variant_id,
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
                "consumer_variants": len(service.variants),
                "families": service.family_count,
                "model_loaded": service.model is not None,
                "feedback_count": len(service.feedback),
                "methodology_id": METHODOLOGY_ID,
                "methodology_name": METHODOLOGY_NAME,
                "model_id": MODEL_ID,
                "probe_objective": PROBE_OBJECTIVE,
            }

    return app


app = create_app()
