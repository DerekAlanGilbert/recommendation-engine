"""Offline simulation proof: active package-level elicitation on the frozen catalog.

A deterministic hidden shopper answers package-level thumbs from a raw-attribute
utility with a satisficing, rising-aspiration bar — functionally independent of
the engine's ideal-point likelihood. Three policies run the same loop protocol:
the greedy profile-vector baseline, passive Bayesian inference (best-fit
probes, no active selection), and active Bayesian elicitation
(information-gain probes over the hierarchical nameplate pool).

These tests assert the earned, deterministic baseline of the first
implementation — they are measurements, not aspirations. Surfacing the exact
2025 Toyota Tundra 4WD PRO within five thumbs is the north-star metric for
iteration; the current baseline reaches roughly the top 2% of the catalog by
loop 5 and the top 5 by loop 25, and five binary answers cannot isolate one
of 6,606 hypotheses in general (2^5 distinguishable outcomes). PLAN.md
records the limitation and the iteration levers.
"""

import pytest

from app.data import load_snapshot
from app.model import build_variant_features, pretrain
from app.preference import build_engine
from app.simulate import (
    COHORT_TARGETS,
    TUNDRA_PRO,
    POLICIES,
    rollout,
    shopper_thumb,
    shopper_utility,
    surfaced_by,
    top_k_hits,
)

COHORT_LOOPS = 15


@pytest.fixture(scope="module")
def features():
    return build_variant_features(load_snapshot())


@pytest.fixture(scope="module")
def model(features):
    return pretrain(features, seed=0)


@pytest.fixture(scope="module")
def engine(model, features):
    return build_engine(model, features)


@pytest.fixture(scope="module")
def by_id(features):
    return {f["variant_id"]: f for f in features}


@pytest.fixture(scope="module")
def cohort_runs(model, engine, features):
    """Rollouts for every policy and cohort target."""
    return {
        policy: {
            target: rollout(policy, model, engine, features, target, loops=COHORT_LOOPS)
            for target in COHORT_TARGETS
        }
        for policy in POLICIES
    }


def median_rank(runs_by_target, loop):
    ranks = sorted(runs_by_target[t][loop - 1]["target_rank"] for t in runs_by_target)
    middle = len(ranks) // 2
    return (ranks[middle - 1] + ranks[middle]) / 2 if len(ranks) % 2 == 0 else ranks[middle]


# Hidden shopper


def test_cohort_exists_and_spans_makes_classes_and_powertrains(by_id):
    assert TUNDRA_PRO == "2025|Toyota|Tundra 4WD PRO"
    assert TUNDRA_PRO in COHORT_TARGETS
    assert len(COHORT_TARGETS) >= 10
    targets = [by_id[target] for target in COHORT_TARGETS]
    assert len({t["make"] for t in targets}) >= 8
    assert len({t["vehicle_class"] for t in targets}) >= 6
    assert len({t["fuel_type"] for t in targets}) >= 3


def test_shopper_approves_the_ideal_and_rejects_distant_packages(by_id):
    for target in COHORT_TARGETS:
        ideal = by_id[target]
        assert shopper_thumb(ideal, ideal)
    tundra = by_id[TUNDRA_PRO]
    assert not shopper_thumb(tundra, by_id["2023|Mazda|MX-5"])
    assert not shopper_thumb(tundra, by_id["2024|Hyundai|Ioniq 5 Long range AWD"])


def test_shopper_aspiration_rises_with_endorsements(by_id):
    """Fresh sessions consider close packages; once something better has been
    endorsed, near-misses (like the plain 4WD against the PRO ideal) are
    rejected. This is what makes trim identity observable through thumbs."""
    tundra = by_id[TUNDRA_PRO]
    plain = by_id["2025|Toyota|Tundra 4WD"]
    assert shopper_thumb(tundra, plain)  # worth considering, cold
    best = shopper_utility(tundra, plain)
    assert not shopper_thumb(tundra, plain, best_liked_utility=best)
    assert shopper_thumb(tundra, tundra, best_liked_utility=best)
    assert not shopper_thumb(tundra, by_id["2025|Toyota|Tundra 2WD"], best_liked_utility=best)


def test_shopper_utility_is_not_a_rebranding_of_engine_similarity(engine, by_id):
    """Anti-tautology: the hidden utility must order candidates differently from
    the inference similarity for a substantial share of pairs."""
    ideal = by_id[TUNDRA_PRO]
    ids = sorted(by_id)[::13]  # deterministic sample across the catalog
    utilities = {v: shopper_utility(ideal, by_id[v]) for v in ids}
    similarities = {
        v: float(engine.similarity[engine.index[TUNDRA_PRO], engine.index[v]]) for v in ids
    }
    discordant, comparable = 0, 0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if utilities[a] == utilities[b] or similarities[a] == similarities[b]:
                continue
            comparable += 1
            if (utilities[a] > utilities[b]) != (similarities[a] > similarities[b]):
                discordant += 1
    assert comparable > 1000
    assert discordant / comparable > 0.05


# Rollout mechanics


def test_rollout_is_deterministic(model, engine, features):
    first = rollout("active", model, engine, features, TUNDRA_PRO, loops=5)
    second = rollout("active", model, engine, features, TUNDRA_PRO, loops=5)
    assert first == second


def test_rollout_records_probe_thumb_rank_and_posterior_movement(cohort_runs):
    for policy in POLICIES:
        for target, records in cohort_runs[policy].items():
            assert [r["loop"] for r in records] == list(range(1, COHORT_LOOPS + 1))
            for record in records:
                assert record["probe_id"]
                assert isinstance(record["liked"], bool)
                assert record["target_rank"] >= 1
                assert 0.0 < record["target_share"] <= 1.0
                assert record["top_id"]
                assert isinstance(record["surfaced"], bool)


def test_rollout_never_repeats_probes(cohort_runs):
    for policy in POLICIES:
        for target, records in cohort_runs[policy].items():
            probes = [r["probe_id"] for r in records]
            assert len(set(probes)) == len(probes), (policy, target)


def test_active_probes_span_multiple_families_early(cohort_runs, by_id):
    for target, records in cohort_runs["active"].items():
        families = {by_id[r["probe_id"]]["family_id"] for r in records[:5]}
        assert len(families) >= 3, target


def test_cohort_exercises_negative_only_and_mixed_trajectories(cohort_runs):
    signatures = set()
    for target, records in cohort_runs["active"].items():
        thumbs = tuple(r["liked"] for r in records[:5])
        if not any(thumbs):
            signatures.add("negative-only")
        elif not all(thumbs):
            signatures.add("mixed")
    assert signatures >= {"negative-only", "mixed"}


# Learning quality: the objective loop comparisons (earned baseline)


def test_tundra_pro_case_active_trajectory(model, engine, features):
    """The target case, measured honestly: from a cold 6,606-variant start the
    active policy reaches roughly the top 2% by loop 5, the top ~25 by loop
    15, and the top 5 by loop 25, first showing the shopper a Tundra-family
    package around loop 23. It does not reach rank 1 or surface the exact
    variant within five thumbs — the north-star for iteration, not a claim."""
    records = rollout("active", model, engine, features, TUNDRA_PRO, loops=25)
    ranks = [r["target_rank"] for r in records]
    assert ranks[4] <= 250
    assert ranks[14] <= 50
    assert ranks[-1] <= 10
    assert min(ranks) <= 5
    assert surfaced_by(records, 25)
    # The convergence is earned through package thumbs on the Tundra nameplate.
    endorsed = [r["probe_id"] for r in records if r["liked"]]
    assert any("Tundra" in variant_id for variant_id in endorsed)
    # Documented baseline gap: not surfaced within five thumbs yet.
    assert not surfaced_by(records, 5)


def test_active_elicitation_beats_passive_and_greedy_on_cohort_median_rank(cohort_runs):
    for loop in (5, COHORT_LOOPS):
        active = median_rank(cohort_runs["active"], loop)
        passive = median_rank(cohort_runs["passive"], loop)
        greedy = median_rank(cohort_runs["greedy"], loop)
        assert active < passive, loop
        assert active < greedy, loop
    # By loop 15 the separation is decisive.
    assert median_rank(cohort_runs["active"], COHORT_LOOPS) * 2 < median_rank(
        cohort_runs["passive"], COHORT_LOOPS
    )
    assert median_rank(cohort_runs["active"], COHORT_LOOPS) * 2 < median_rank(
        cohort_runs["greedy"], COHORT_LOOPS
    )


def test_active_ranks_improve_over_loops_for_most_targets(cohort_runs):
    improved = 0
    for target, records in cohort_runs["active"].items():
        if records[COHORT_LOOPS - 1]["target_rank"] < records[0]["target_rank"]:
            improved += 1
    assert improved >= len(COHORT_TARGETS) * 0.6


def test_top_k_hits_reports_all_required_loops(cohort_runs):
    hits = top_k_hits(cohort_runs["active"])
    assert set(hits) == {1, 3, 10}
    for k, fractions in hits.items():
        assert len(fractions) == COHORT_LOOPS
        assert all(0.0 <= fraction <= 1.0 for fraction in fractions)
