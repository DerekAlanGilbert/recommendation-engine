"""The one methodology this repository demonstrates, pinned end to end:
Targeted Learning (probe acquisition) over the Current Relationships content
tower, ranked by the baseline marginal-posterior recommendation ranking.
"""

import csv
from pathlib import Path

import pytest

from app import methodology
from app.data import import_source_rows
from app.model import build_variant_features, untrained_model
from app.preference import (
    build_engine,
    exploration_weight,
    posterior_from_feedback,
    select_probe,
    targeted_information_gain,
    variant_marginal,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"


@pytest.fixture(scope="module")
def engine():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        features = build_variant_features(import_source_rows(list(csv.DictReader(f))))
    return build_engine(untrained_model(features, seed=0), features)


def test_methodology_metadata_is_exact():
    assert methodology.METHODOLOGY_ID == "targeted_learning_current_relationships"
    assert methodology.METHODOLOGY_NAME == "Targeted Learning — Current Relationships"
    assert methodology.MODEL_ID == "current"
    assert methodology.PROBE_OBJECTIVE == "targeted"


def test_probe_maximizes_the_documented_targeted_acquisition(engine):
    """Independent recomputation: the selected probe is the acquisition argmax
    (targeted gain blended with approval and marginal exploitation) over every
    unrated variant of this tiny fixture, where the candidate pool is total."""
    for feedback in ([], [("2017|Toyota|Camry LE", True)]):
        posterior = posterior_from_feedback(engine, feedback)
        rated = {variant_id for variant_id, _ in feedback}
        rows = [row for row, variant_id in enumerate(engine.variant_ids)
                if variant_id not in rated]
        gain = targeted_information_gain(engine, posterior, rows)
        marginal = variant_marginal(posterior)
        weight = exploration_weight(posterior)
        scores = {}
        for i, row in enumerate(rows):
            similarity_row = engine.similarity[row]
            approval = 0.0
            for level, theta in enumerate(engine.thetas):
                likelihood = 1.0 / (1.0 + (-(10.0) * (similarity_row - theta)).exp())
                approval += float((likelihood * posterior[level].to(likelihood.dtype)).sum())
            exploit = float(marginal[row] / marginal.max())
            scores[engine.variant_ids[row]] = (
                weight * float(gain[i]) + (1.0 - weight) * 0.5 * (approval + exploit)
            )
        expected = min(scores, key=lambda variant_id: (-scores[variant_id], variant_id))
        probe = select_probe(engine, posterior, feedback)
        assert probe["variant_id"] == expected
