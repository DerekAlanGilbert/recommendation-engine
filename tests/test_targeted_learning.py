"""Targeted expected information gain: bits about the ideal variant alone.

The probe objective is I(response; T): the mutual information between one
binary thumb and the ideal-variant hypothesis T, with the approval threshold
Θ integrated out per target as a nuisance parameter:

    p_t(x)   = Σ_θ π(t, θ) · l(x, t, θ) / m(t)      (m(t) = Σ_θ π(t, θ))
    p̄(x)    = Σ_t m(t) · p_t(x)
    IG_T(x)  = H_b(p̄(x)) − Σ_t m(t) · H_b(p_t(x))

Bits that would only teach the shopper's pickiness threshold do not count.
`select_probe` has exactly one acquisition behavior — this one.
"""

import csv
import inspect
import math
from pathlib import Path

import pytest
import torch

from app.data import import_source_rows
from app.model import build_variant_features, untrained_model
from app.preference import (
    SHARPNESS,
    build_engine,
    posterior_from_feedback,
    select_probe,
    targeted_information_gain,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"

LE17, XSE17 = "2017|Toyota|Camry LE", "2017|Toyota|Camry XSE"
BOLT26 = "2026|Chevrolet|Bolt EUV"


@pytest.fixture(scope="module")
def engine():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        features = build_variant_features(import_source_rows(list(csv.DictReader(f))))
    return build_engine(untrained_model(features, seed=0), features)


def _binary_entropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _independent_targeted_gain(engine, posterior, row):
    """Pure-Python reference: marginalize θ per target, then take I(response; T)."""
    k, n = posterior.shape
    p_up_overall = 0.0
    conditional = 0.0
    for t in range(n):
        target_mass = sum(float(posterior[level, t]) for level in range(k))
        joint_up = sum(
            float(posterior[level, t])
            * (1.0 / (1.0 + math.exp(-SHARPNESS * (float(engine.similarity[row, t])
                                                   - float(engine.thetas[level])))))
            for level in range(k)
        )
        p_up_overall += joint_up
        if target_mass > 0.0:
            conditional += target_mass * _binary_entropy(joint_up / target_mass)
    return _binary_entropy(p_up_overall) - conditional


def test_targeted_gain_matches_independent_tiny_example(engine):
    posterior = posterior_from_feedback(engine, [(LE17, True), (BOLT26, False)])
    rows = list(range(len(engine.variant_ids)))
    gain = targeted_information_gain(engine, posterior, rows)
    assert gain.shape == (len(rows),)
    for i, row in enumerate(rows):
        expected = _independent_targeted_gain(engine, posterior, row)
        assert float(gain[i]) == pytest.approx(expected, rel=1e-4, abs=1e-5)
        assert float(gain[i]) >= 0.0


def test_targeted_gain_is_finite_and_nonnegative_across_histories(engine):
    rows = list(range(len(engine.variant_ids)))
    histories = (
        [],
        [(LE17, True)],
        [(LE17, False)],
        [(LE17, True), (XSE17, False)],
        [(LE17, True), (BOLT26, False)],
    )
    for history in histories:
        posterior = posterior_from_feedback(engine, history)
        gain = targeted_information_gain(engine, posterior, rows)
        assert not torch.isnan(gain).any(), history
        assert (gain >= 0.0).all(), history


def test_targeted_gain_is_zero_when_a_response_teaches_only_threshold(engine):
    """Identity known, threshold unknown: no probe carries target information,
    so the acquisition must not spend a thumb learning only pickiness."""
    k, n = len(engine.thetas), len(engine.variant_ids)
    posterior = torch.zeros((k, n), dtype=torch.float64)
    posterior[:, engine.index[LE17]] = 1.0 / k  # T certain, Θ uniform
    gain = targeted_information_gain(engine, posterior, list(range(n)))
    assert not torch.isnan(gain).any()  # zero-mass targets stay safe
    assert float(gain.abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_select_probe_has_no_objective_selector():
    """One methodology: probe acquisition is targeted information gain only."""
    assert "objective" not in inspect.signature(select_probe).parameters


def test_select_probe_is_deterministic_and_excludes_rated(engine):
    probe = select_probe(engine, posterior_from_feedback(engine, []))
    again = select_probe(engine, posterior_from_feedback(engine, []))
    assert probe == again
    assert probe["expected_information_gain"] >= 0.0
    assert 0.0 <= probe["expected_approval"] <= 1.0
    feedback = [(probe["variant_id"], False)]
    following = select_probe(engine, posterior_from_feedback(engine, feedback), feedback)
    assert following["variant_id"] != probe["variant_id"]
    everything = [(variant_id, False) for variant_id in engine.variant_ids]
    assert select_probe(
        engine, posterior_from_feedback(engine, everything), everything
    ) is None
