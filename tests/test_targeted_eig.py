"""Targeted expected information gain: bits about the ideal variant alone.

The joint objective I(response; T, Θ) rewards a probe for anything it teaches,
including bits about the nuisance threshold Θ. The targeted objective
I(response; T) treats Θ as a nuisance parameter and integrates it out:

    p_t(x)   = Σ_θ π(t, θ) · l(x, t, θ) / m(t)      (m(t) = Σ_θ π(t, θ))
    p̄(x)    = Σ_t m(t) · p_t(x)
    IG_T(x)  = H_b(p̄(x)) − Σ_t m(t) · H_b(p_t(x))

By the chain rule I(response; T, Θ) = I(response; T) + I(response; Θ | T),
so the targeted gain can never exceed the joint gain, and is strictly below
it whenever a response teaches threshold but not target identity.
"""

import csv
import math
from pathlib import Path

import pytest
import torch

from app.data import import_source_rows
from app.model import build_variant_features, untrained_model
from app.preference import (
    SHARPNESS,
    build_engine,
    joint_information_gain,
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


def test_targeted_gain_never_exceeds_joint_gain(engine):
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
        targeted = targeted_information_gain(engine, posterior, rows)
        joint = joint_information_gain(engine, posterior, rows)
        assert not torch.isnan(targeted).any()
        assert (targeted <= joint + 1e-4).all(), history


def test_targeted_gain_is_zero_when_a_response_teaches_only_threshold(engine):
    """Identity known, threshold unknown: the joint objective still values a
    probe (it would learn pickiness); the targeted objective must not."""
    k, n = len(engine.thetas), len(engine.variant_ids)
    posterior = torch.zeros((k, n), dtype=torch.float64)
    posterior[:, engine.index[LE17]] = 1.0 / k  # T certain, Θ uniform
    rows = list(range(n))
    targeted = targeted_information_gain(engine, posterior, rows)
    joint = joint_information_gain(engine, posterior, rows)
    assert not torch.isnan(targeted).any()  # zero-mass targets stay safe
    assert float(targeted.abs().max()) == pytest.approx(0.0, abs=1e-5)
    # Probing the known ideal itself would still teach threshold: strict gap.
    assert float(joint[engine.index[LE17]]) > 0.01


def test_select_probe_targeted_is_deterministic_and_excludes_rated(engine):
    probe = select_probe(engine, posterior_from_feedback(engine, []), objective="targeted")
    again = select_probe(engine, posterior_from_feedback(engine, []), objective="targeted")
    assert probe == again
    assert probe["expected_information_gain"] >= 0.0
    assert 0.0 <= probe["expected_approval"] <= 1.0
    feedback = [(probe["variant_id"], False)]
    following = select_probe(
        engine, posterior_from_feedback(engine, feedback), feedback, objective="targeted"
    )
    assert following["variant_id"] != probe["variant_id"]
    everything = [(variant_id, False) for variant_id in engine.variant_ids]
    assert select_probe(
        engine, posterior_from_feedback(engine, everything), everything, objective="targeted"
    ) is None


def test_select_probe_rejects_unknown_objective(engine):
    with pytest.raises(ValueError):
        select_probe(engine, posterior_from_feedback(engine, []), objective="oracle")


def test_select_probe_defaults_to_the_frozen_joint_objective(engine):
    """The measured no-go decision: joint EIG stays the product/API policy.

    `app.main` calls select_probe without an objective, so pinning the default
    parameter and its behavior pins the API probe surface. Targeted EIG exists
    only as the explicitly named experimental comparison arm."""
    import inspect

    assert inspect.signature(select_probe).parameters["objective"].default == "joint"
    for history in ([], [(LE17, True)], [(LE17, True), (XSE17, False)]):
        posterior = posterior_from_feedback(engine, history)
        assert select_probe(engine, posterior, history) == select_probe(
            engine, posterior, history, objective="joint"
        )
