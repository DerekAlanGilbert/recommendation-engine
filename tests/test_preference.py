"""Preference-engine tests: exact joint posterior over (ideal variant, threshold).

The engine is an ideal-point model with a latent approval threshold: the
shopper has one unknown ideal variant and one unknown pickiness level, and a
package-level thumb is a Bernoulli observation whose probability rises with
similarity to the ideal relative to that threshold. The joint posterior over
(variant, threshold) is computed exactly on a discrete threshold grid; the
variant ranking is the threshold-marginalized posterior.

Marginalizing the threshold is what preserves ambiguity: a single rejection is
explained partly by pickiness and partly by dissimilarity, so no hypothesis is
crushed, and there are no ad-hoc down-weighting constants.
"""

import csv
import math
from pathlib import Path

import pytest
import torch

from app.model import build_variant_features, untrained_model
from app.data import import_source_rows
from app.preference import (
    FAMILY_CAP,
    MAX_EXPLORATION_WEIGHT,
    THETA_GRID,
    build_engine,
    exploration_weight,
    model_tokens,
    posterior_from_feedback,
    rank_recommendations,
    select_probe,
    similarity_between,
    threshold_marginal,
    variant_marginal,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"

LE17, XSE17 = "2017|Toyota|Camry LE", "2017|Toyota|Camry XSE"
LE18 = "2018|Toyota|Camry LE"
GLC19 = "2019|Mercedes-Benz|GLC 300 4matic"
BOLT26 = "2026|Chevrolet|Bolt EUV"


@pytest.fixture(scope="module")
def features():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        return build_variant_features(import_source_rows(list(csv.DictReader(f))))


@pytest.fixture(scope="module")
def engine(features):
    return build_engine(untrained_model(features, seed=0), features)


def mass(engine, posterior, variant_id):
    return float(variant_marginal(posterior)[engine.index[variant_id]])


def test_model_tokens_split_and_normalize_case():
    assert model_tokens("Tundra 4WD PRO") == frozenset({"TUNDRA", "4WD", "PRO"})
    assert model_tokens("Camry LE") == frozenset({"CAMRY", "LE"})
    assert model_tokens("Bolt EUV") == frozenset({"BOLT", "EUV"})


def test_engine_similarity_is_maximal_on_self_and_symmetric(engine):
    for variant_id in engine.variant_ids:
        assert similarity_between(engine, variant_id, variant_id) == pytest.approx(1.0)
    for a in engine.variant_ids:
        for b in engine.variant_ids:
            s = similarity_between(engine, a, b)
            assert 0.0 <= s <= 1.0
            assert s == pytest.approx(similarity_between(engine, b, a))
            if a != b:
                assert s < 1.0


def test_same_nameplate_is_more_similar_than_cross_nameplate(engine):
    assert similarity_between(engine, LE17, LE18) > similarity_between(engine, LE17, GLC19)
    assert similarity_between(engine, LE17, XSE17) > similarity_between(engine, LE17, BOLT26)


def test_cold_start_posterior_is_uniform_over_variants_and_thresholds(engine):
    posterior = posterior_from_feedback(engine, [])
    n = len(engine.variant_ids)
    k = len(THETA_GRID)
    assert posterior.shape == (k, n)
    assert float(posterior.sum()) == pytest.approx(1.0)
    marginal = variant_marginal(posterior)
    assert torch.allclose(marginal, torch.full((n,), 1.0 / n, dtype=marginal.dtype))
    assert torch.allclose(
        threshold_marginal(posterior),
        torch.full((k,), 1.0 / k, dtype=posterior.dtype),
    )
    assert exploration_weight(posterior) == pytest.approx(MAX_EXPLORATION_WEIGHT)


def test_one_rejection_preserves_ambiguity_while_penalizing_similar_hypotheses(engine):
    posterior = posterior_from_feedback(engine, [(LE17, False)])
    n = len(engine.variant_ids)
    uniform = 1.0 / n
    # The rejected package itself loses mass, siblings lose more than
    # dissimilar packages, and dissimilar packages gain relative mass.
    assert mass(engine, posterior, LE17) < uniform
    assert mass(engine, posterior, LE18) < mass(engine, posterior, BOLT26)
    assert mass(engine, posterior, BOLT26) > uniform
    # Ambiguity is preserved where it matters: the rejected package itself is
    # demoted (it was literally rejected) but never zeroed, and a moderately
    # similar unrated neighbor keeps substantial mass because the rejection is
    # partly explained by a high threshold (a picky shopper), not by labeling
    # the neighbor's shared attributes as disliked.
    assert mass(engine, posterior, LE17) > uniform / 20
    assert mass(engine, posterior, XSE17) > uniform / 2
    assert (variant_marginal(posterior) > 0).all()
    # The rejection shifts threshold belief upward (pickier explanations gain).
    cold_thresholds = threshold_marginal(posterior_from_feedback(engine, []))
    after = threshold_marginal(posterior)
    expected_cold = sum(t * float(p) for t, p in zip(THETA_GRID, cold_thresholds))
    expected_after = sum(t * float(p) for t, p in zip(THETA_GRID, after))
    assert expected_after > expected_cold


def test_endorsement_shifts_mass_toward_similar_hypotheses(engine):
    posterior = posterior_from_feedback(engine, [(LE17, True)])
    uniform = 1.0 / len(engine.variant_ids)
    assert mass(engine, posterior, LE17) > uniform
    assert mass(engine, posterior, LE18) > mass(engine, posterior, BOLT26)
    assert (variant_marginal(posterior) > 0).all()
    assert float(posterior.sum()) == pytest.approx(1.0)


def test_posterior_is_deterministic_and_order_independent(engine):
    events = [(LE17, True), (BOLT26, False), (GLC19, False)]
    first = posterior_from_feedback(engine, events)
    assert torch.equal(first, posterior_from_feedback(engine, events))
    second = posterior_from_feedback(engine, list(reversed(events)))
    assert torch.allclose(first, second)


def test_rejecting_a_distinguishable_neighbor_of_an_endorsed_package_cannot_crush_it(engine):
    """The endgame property that motivated the threshold marginalization: after
    the shopper endorses a package, rejecting a similar-but-distinguishable
    sibling must not collapse the endorsed hypothesis — the joint posterior
    explains the pair with a threshold between the two similarities. (For
    exact similarity clones no model can separate the pair; that limit is
    documented rather than asserted away.)"""
    posterior = posterior_from_feedback(engine, [(LE17, True), (XSE17, False)])
    marginal = variant_marginal(posterior)
    ranking = sorted(range(len(engine.variant_ids)), key=lambda r: -float(marginal[r]))
    assert engine.variant_ids[ranking[0]] == LE17


def test_negative_only_feedback_preserves_broad_uncertainty(engine):
    posterior = posterior_from_feedback(engine, [(LE17, False), (GLC19, False)])
    marginal = variant_marginal(posterior)
    assert (marginal > 0).all()
    positive = marginal[marginal > 0]
    entropy = float(-(positive * torch.log(positive)).sum())
    # Two of the five fixture variants are rated, so entropy necessarily drops
    # well below uniform; the requirement is that broad uncertainty survives.
    assert entropy > 0.6 * math.log(len(engine.variant_ids))
    assert exploration_weight(posterior) > 0.5


def test_exploration_weight_falls_as_evidence_accumulates(engine):
    histories = [
        [],
        [(LE17, True)],
        [(LE17, True), (LE18, True)],
        [(LE17, True), (LE18, True), (BOLT26, False)],
    ]
    weights = [exploration_weight(posterior_from_feedback(engine, h)) for h in histories]
    assert weights[0] == max(weights)
    assert weights[-1] < weights[0]
    assert all(0.0 <= w <= MAX_EXPLORATION_WEIGHT for w in weights)


def test_select_probe_is_deterministic_and_excludes_rated(engine):
    probe = select_probe(engine, posterior_from_feedback(engine, []))
    again = select_probe(engine, posterior_from_feedback(engine, []))
    assert probe == again
    assert set(probe) >= {
        "variant_id", "expected_information_gain", "expected_approval", "exploration_weight",
    }
    assert probe["expected_information_gain"] >= 0.0
    assert 0.0 <= probe["expected_approval"] <= 1.0
    assert probe["exploration_weight"] == pytest.approx(MAX_EXPLORATION_WEIGHT)
    feedback = [(probe["variant_id"], False)]
    following = select_probe(engine, posterior_from_feedback(engine, feedback), feedback)
    assert following["variant_id"] != probe["variant_id"]
    everything = [(variant_id, False) for variant_id in engine.variant_ids]
    assert select_probe(engine, posterior_from_feedback(engine, everything), everything) is None


def test_probe_refines_an_endorsed_nameplate(engine):
    # After two Camry endorsements the posterior concentrates on the nameplate
    # and the probe asks about the remaining sibling — coarse-to-fine.
    endorsed = [(LE17, True), (LE18, True)]
    probe = select_probe(engine, posterior_from_feedback(engine, endorsed), endorsed)
    assert probe["variant_id"] == XSE17


def test_rank_recommendations_orders_by_marginal_posterior_and_excludes_rated(engine):
    events = [(LE17, True)]
    posterior = posterior_from_feedback(engine, events)
    ranking = rank_recommendations(engine, posterior, exclude={LE17})
    ids = [variant_id for variant_id, _ in ranking]
    assert LE17 not in ids
    assert sorted(ids) == sorted(set(engine.variant_ids) - {LE17})
    scores = dict(ranking)
    for variant_id in ids:
        assert scores[variant_id] == pytest.approx(mass(engine, posterior, variant_id))
    assert rank_recommendations(engine, posterior, exclude={LE17}, limit=2) == ranking[:2]


def test_rank_recommendations_defers_family_siblings_beyond_the_cap(engine, features):
    posterior = posterior_from_feedback(engine, [(LE17, True), (LE18, True)])
    ranking = rank_recommendations(engine, posterior, family_cap=1)
    ids = [variant_id for variant_id, _ in ranking]
    assert sorted(ids) == sorted(engine.variant_ids)
    family_by_id = {f["variant_id"]: f["family_id"] for f in features}
    families = {family_by_id[v] for v in ids}
    top_families = [family_by_id[v] for v in ids[: len(families)]]
    assert len(top_families) == len(set(top_families))
    assert FAMILY_CAP == 2
    uncapped = rank_recommendations(engine, posterior)
    assert sorted(v for v, _ in uncapped) == sorted(engine.variant_ids)
