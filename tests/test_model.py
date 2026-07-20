"""Model tests: feature-building units plus pretraining/profile integration on the frozen catalog."""

import csv
import random
import time
from pathlib import Path

import pytest
import torch

from app.data import EXPECTED_CANONICAL, group_variants, import_source_rows, load_snapshot
from app.model import (
    CATEGORICAL_FEATURES,
    EMBEDDING_DIM,
    HELDOUT_RULES,
    NUMERIC_FEATURES,
    build_feature_space,
    build_item_features,
    cold_start_profile,
    drive_family,
    encode_features,
    generate_interactions,
    pretrain,
    pretrain_rules,
    rank_items,
    refit_profile,
    score_items,
    transmission_family,
    untrained_model,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"


def read_fixture_variants():
    with FIXTURE_PATH.open(newline="", encoding="utf-8") as f:
        return import_source_rows(list(csv.DictReader(f)))


# Fixture-driven feature unit tests


def test_drive_family_mapping():
    assert drive_family("Front-Wheel Drive") == "front"
    assert drive_family("Rear-Wheel Drive") == "rear"
    assert drive_family("All-Wheel Drive") == "all"
    assert drive_family("4-Wheel Drive") == "four"
    assert drive_family("Part-time 4-Wheel Drive") == "four"


def test_transmission_family_mapping():
    assert transmission_family("Manual 6-spd") == "manual"
    assert transmission_family("Automatic (variable gear ratios)") == "cvt"
    assert transmission_family("Automatic (AV-S8)") == "cvt"
    assert transmission_family("Automatic (AM7)") == "automated manual"
    assert transmission_family("Automatic (AM-S7)") == "automated manual"
    assert transmission_family("Automatic (S8)") == "automatic"
    assert transmission_family("Automatic 9-spd") == "automatic"
    assert transmission_family("Automatic (A1)") == "automatic"


def test_fixture_features_aggregate_variants_deterministically():
    variants = read_fixture_variants()
    features = build_item_features(variants)
    assert [f["item_id"] for f in features] == [i["item_id"] for i in group_variants(variants)]
    by_id = {f["item_id"]: f for f in features}
    camry = by_id["2017|Toyota|Camry"]
    assert camry["make"] == "Toyota"
    assert camry["vehicle_class"] == "Midsize Cars"
    assert camry["fuel_type"] == "Regular"
    assert camry["drive_family"] == "front"
    assert camry["transmission_family"] == "automatic"
    assert camry["year"] == 2017
    assert camry["cylinders"] == 5.0
    assert camry["displacement"] == 3.0
    assert camry["city_mpg"] == 25.0
    assert camry["highway_mpg"] == 35.5
    assert camry["combined_mpg"] == 29.0
    assert camry["electric_range"] == 0.0
    assert camry["co2_tailpipe_gpm"] == 310.5
    assert build_item_features(list(reversed(variants))) == features


def test_fixture_features_keep_missing_numerics_missing():
    features = build_item_features(read_fixture_variants())
    bolt = {f["item_id"]: f for f in features}["2026|Chevrolet|Bolt EUV"]
    assert bolt["cylinders"] is None
    assert bolt["displacement"] is None
    assert bolt["electric_range"] == 247.0
    assert bolt["fuel_type"] == "Electricity"


def test_modal_aggregation_breaks_ties_lexicographically():
    variants = read_fixture_variants()
    camry_variants = [dict(v) for v in variants if v["epa_id"] in (1010, 1011)]
    camry_variants[0]["fuel_type"] = "Regular"
    camry_variants[1]["fuel_type"] = "Premium"
    features = build_item_features(camry_variants)
    assert features[0]["fuel_type"] == "Premium"


def test_encoding_standardizes_and_imputes_missing_as_zero():
    features = build_item_features(read_fixture_variants())
    space = build_feature_space(features)
    categorical, numeric = encode_features(features, space)
    assert categorical.shape == (len(features), len(CATEGORICAL_FEATURES))
    assert numeric.shape == (len(features), len(NUMERIC_FEATURES))
    assert torch.isfinite(numeric).all()
    for column, (name, _) in enumerate(CATEGORICAL_FEATURES):
        assert categorical[:, column].max().item() < len(space.vocabs[name])
    bolt_row = [f["item_id"] for f in features].index("2026|Chevrolet|Bolt EUV")
    assert numeric[bolt_row, NUMERIC_FEATURES.index("cylinders")].item() == 0.0
    assert numeric[bolt_row, NUMERIC_FEATURES.index("displacement")].item() == 0.0


# Frozen-catalog integration tests


@pytest.fixture(scope="module")
def features():
    return build_item_features(load_snapshot())


@pytest.fixture(scope="module")
def trained(features):
    return pretrain(features, seed=0)


@pytest.fixture(scope="module")
def untrained(features):
    return untrained_model(features, seed=0)


def simulate_feedback(features, predicate, liked_count, disliked_count, seed):
    rng = random.Random(seed)
    positives = [f["item_id"] for f in features if predicate(f)]
    negatives = [f["item_id"] for f in features if not predicate(f)]
    liked = rng.sample(positives, liked_count)
    disliked = rng.sample(negatives, disliked_count)
    return [(item_id, True) for item_id in liked] + [(item_id, False) for item_id in disliked]


def evaluate_heldout_rule(model, features, predicate, feedback):
    """Precision@10 and ranking AUC for rule positives over unrated items."""
    profile = refit_profile(model, feedback)
    rated = {item_id for item_id, _ in feedback}
    ranking = rank_items(model, profile, exclude=rated)
    by_id = {f["item_id"]: f for f in features}
    precision = sum(bool(predicate(by_id[item_id])) for item_id, _ in ranking[:10]) / 10
    positive_ranks = [r for r, (item_id, _) in enumerate(ranking) if predicate(by_id[item_id])]
    n_pos, n = len(positive_ranks), len(ranking)
    pairs_won = sum(n - 1 - r for r in positive_ranks) - n_pos * (n_pos - 1) / 2
    return precision, pairs_won / (n_pos * (n - n_pos))


def test_catalog_features_cover_all_canonical_items(features):
    assert len(features) == EXPECTED_CANONICAL
    assert [f["item_id"] for f in features] == [i["item_id"] for i in group_variants(load_snapshot())]
    space = build_feature_space(features)
    assert len(space.vocabs["make"]) == 20
    assert set(space.vocabs["drive_family"]) == {"all", "four", "front", "rear"}
    assert set(space.vocabs["transmission_family"]) == {"automated manual", "automatic", "cvt", "manual"}


def test_pretraining_is_deterministic_with_practical_runtime(features, trained):
    started = time.perf_counter()
    again = pretrain(features, seed=0)
    elapsed = time.perf_counter() - started
    assert torch.equal(again.embeddings, trained.embeddings)
    assert again.bias == trained.bias
    assert trained.embeddings.shape == (EXPECTED_CANONICAL, EMBEDDING_DIM)
    assert torch.allclose(trained.embeddings.norm(dim=1), torch.ones(EXPECTED_CANONICAL))
    assert elapsed < 60.0


def test_heldout_rules_are_structurally_excluded_from_pretraining(features):
    assert len(HELDOUT_RULES) >= 3
    rules = pretrain_rules(features)
    assert {name for name, _ in rules}.isdisjoint({name for name, _ in HELDOUT_RULES})
    interactions = generate_interactions(features, rules, seed=0)
    assert max(profile for profile, _, _ in interactions) == len(rules) - 1
    pretrain_labelings = {tuple(bool(predicate(f)) for f in features) for _, predicate in rules}
    for _, predicate in HELDOUT_RULES:
        assert tuple(bool(predicate(f)) for f in features) not in pretrain_labelings


def test_trained_beats_untrained_on_heldout_rules(features, trained, untrained):
    trained_precisions = []
    untrained_precisions = []
    for rule_index, (name, predicate) in enumerate(HELDOUT_RULES):
        base_rate = sum(bool(predicate(f)) for f in features) / len(features)
        rule_trained, rule_untrained = [], []
        for sample in range(5):
            feedback = simulate_feedback(features, predicate, 15, 15, seed=1000 * rule_index + sample)
            rule_trained.append(evaluate_heldout_rule(trained, features, predicate, feedback))
            rule_untrained.append(evaluate_heldout_rule(untrained, features, predicate, feedback))
        trained_p10 = sum(p for p, _ in rule_trained) / 5
        untrained_p10 = sum(p for p, _ in rule_untrained) / 5
        trained_auc = sum(a for _, a in rule_trained) / 5
        untrained_auc = sum(a for _, a in rule_untrained) / 5
        assert trained_p10 > base_rate, name
        assert trained_p10 >= 0.8, name
        assert trained_p10 > untrained_p10, name
        assert trained_auc > untrained_auc, name
        trained_precisions.append(trained_p10)
        untrained_precisions.append(untrained_p10)
    assert sum(trained_precisions) / len(HELDOUT_RULES) >= 0.9
    assert sum(untrained_precisions) / len(HELDOUT_RULES) < 0.7


def test_feedback_moves_scores_toward_liked_attributes(features, trained):
    electric = [f["item_id"] for f in features if f["fuel_type"] == "Electricity"]
    guzzlers = [
        f["item_id"]
        for f in sorted(features, key=lambda f: -f["co2_tailpipe_gpm"])
        if f["fuel_type"] != "Electricity"
    ]
    feedback = [(item_id, True) for item_id in electric[:5]]
    feedback += [(item_id, False) for item_id in guzzlers[:5]]
    cold_scores = score_items(trained, cold_start_profile())
    warm_scores = score_items(trained, refit_profile(trained, feedback))
    rows = {item_id: row for row, item_id in enumerate(trained.item_ids)}
    unrated_electric = [rows[item_id] for item_id in electric[5:]]
    unrated_guzzlers = [rows[item_id] for item_id in guzzlers[5:105]]
    assert warm_scores[unrated_electric].mean() > cold_scores[unrated_electric].mean()
    assert warm_scores[unrated_guzzlers].mean() < cold_scores[unrated_guzzlers].mean()


def test_rank_items_never_repeats_rated_items(features, trained):
    predicate = dict(HELDOUT_RULES)["likes efficient small SUVs"]
    feedback = simulate_feedback(features, predicate, 15, 15, seed=99)
    rated = {item_id for item_id, _ in feedback}
    profile = refit_profile(trained, feedback)
    ranking = rank_items(trained, profile, exclude=rated)
    ranked_ids = [item_id for item_id, _ in ranking]
    assert rated.isdisjoint(ranked_ids)
    assert len(ranking) == EXPECTED_CANONICAL - len(rated)
    assert len(set(ranked_ids)) == len(ranked_ids)
    scores = [score for _, score in ranking]
    assert scores == sorted(scores, reverse=True)
    assert rank_items(trained, profile, exclude=rated) == ranking
    assert rank_items(trained, profile, exclude=rated, limit=10) == ranking[:10]


def test_reset_restores_cold_start_ranking(features, trained):
    cold_ranking = rank_items(trained, cold_start_profile(), limit=10)
    predicate = dict(HELDOUT_RULES)["likes premium six-cylinder sedans"]
    feedback = simulate_feedback(features, predicate, 10, 10, seed=7)
    warm_ranking = rank_items(trained, refit_profile(trained, feedback), limit=10)
    assert warm_ranking != cold_ranking
    reset_profile = refit_profile(trained, [])
    assert torch.equal(reset_profile, cold_start_profile())
    assert rank_items(trained, reset_profile, limit=10) == cold_ranking


def test_refit_is_deterministic_order_independent_and_fast(features, trained):
    predicate = dict(HELDOUT_RULES)["likes recent AWD family cars"]
    feedback = simulate_feedback(features, predicate, 15, 15, seed=3)
    first = refit_profile(trained, feedback)
    second = refit_profile(trained, list(reversed(feedback)))
    assert torch.equal(first, second)
    started = time.perf_counter()
    profile = refit_profile(trained, feedback)
    ranking = rank_items(trained, profile, exclude={item_id for item_id, _ in feedback})
    elapsed = time.perf_counter() - started
    assert len(ranking) == EXPECTED_CANONICAL - len(feedback)
    assert elapsed < 1.0
