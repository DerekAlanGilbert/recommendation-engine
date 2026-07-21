"""Model tests: feature-building units plus tower quality on the frozen catalog.

The Current Relationships content tower encodes the 6,606 consumer-facing
variants into frozen embeddings. Embedding quality is proven on held-out
synthetic rules with a logistic evaluation probe defined in this test file —
a measurement harness, not a recommendation policy.
"""

import csv
import random
import time
from pathlib import Path

import pytest
import torch

from app.data import EXPECTED_CONSUMER_VARIANTS, group_consumer_variants, import_source_rows, load_snapshot
from app.model import (
    CATEGORICAL_FEATURES,
    EMBEDDING_DIM,
    HELDOUT_RULES,
    NUMERIC_FEATURES,
    build_feature_space,
    build_variant_features,
    drive_family,
    encode_features,
    generate_interactions,
    pretrain,
    pretrain_rules,
    transmission_family,
    untrained_model,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "epa_sample.csv"


def read_fixture_configs():
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


def test_fixture_features_are_per_consumer_variant():
    configs = read_fixture_configs()
    features = build_variant_features(configs)
    grouped = group_consumer_variants(configs)
    assert [f["variant_id"] for f in features] == [v["variant_id"] for v in grouped]
    by_id = {f["variant_id"]: f for f in features}
    le = by_id["2017|Toyota|Camry LE"]
    assert le["family_id"] == "2017|Toyota|Camry"
    assert le["base_model"] == "Camry"
    assert le["model"] == "Camry LE"
    assert le["make"] == "Toyota"
    assert le["vehicle_class"] == "Midsize Cars"
    assert le["fuel_type"] == "Regular"
    assert le["drive_family"] == "front"
    assert le["transmission_family"] == "automatic"
    assert le["year"] == 2017
    assert le["cylinders"] == 4.0
    assert le["displacement"] == 2.5
    assert le["city_mpg"] == 28.0
    assert le["highway_mpg"] == 39.0
    assert le["combined_mpg"] == 32.0
    assert le["electric_range"] == 0.0
    assert le["co2_tailpipe_gpm"] == 278.0
    # The XSE is a separate consumer variant inside the same family.
    xse = by_id["2017|Toyota|Camry XSE"]
    assert xse["family_id"] == "2017|Toyota|Camry"
    assert xse["cylinders"] == 6.0
    assert build_variant_features(list(reversed(configs))) == features


def test_fixture_features_aggregate_same_model_configs_deterministically():
    configs = read_fixture_configs()
    le_2017 = next(c for c in configs if c["epa_id"] == 1010)
    second = {**le_2017, "epa_id": 9999, "combined_mpg": 30, "fuel_type": "Premium"}
    features = build_variant_features([le_2017, second])
    assert len(features) == 1
    assert features[0]["combined_mpg"] == 31.0
    # Modal tie-break is lexicographic, matching the family tie-break rule.
    assert features[0]["fuel_type"] == "Premium"


def test_fixture_features_keep_missing_numerics_missing():
    features = build_variant_features(read_fixture_configs())
    bolt = {f["variant_id"]: f for f in features}["2026|Chevrolet|Bolt EUV"]
    assert bolt["cylinders"] is None
    assert bolt["displacement"] is None
    assert bolt["electric_range"] == 247.0
    assert bolt["fuel_type"] == "Electricity"


def test_encoding_standardizes_and_imputes_missing_as_zero():
    features = build_variant_features(read_fixture_configs())
    space = build_feature_space(features)
    categorical, numeric = encode_features(features, space)
    assert categorical.shape == (len(features), len(CATEGORICAL_FEATURES))
    assert numeric.shape == (len(features), len(NUMERIC_FEATURES))
    assert torch.isfinite(numeric).all()
    for column, (name, _) in enumerate(CATEGORICAL_FEATURES):
        assert categorical[:, column].max().item() < len(space.vocabs[name])
    bolt_row = [f["variant_id"] for f in features].index("2026|Chevrolet|Bolt EUV")
    assert numeric[bolt_row, NUMERIC_FEATURES.index("cylinders")].item() == 0.0
    assert numeric[bolt_row, NUMERIC_FEATURES.index("displacement")].item() == 0.0


# Frozen-catalog integration tests


@pytest.fixture(scope="module")
def features():
    return build_variant_features(load_snapshot())


@pytest.fixture(scope="module")
def trained(features):
    return pretrain(features, seed=0)


@pytest.fixture(scope="module")
def untrained(features):
    return untrained_model(features, seed=0)


def simulate_feedback(features, predicate, liked_count, disliked_count, seed):
    rng = random.Random(seed)
    positives = [f["variant_id"] for f in features if predicate(f)]
    negatives = [f["variant_id"] for f in features if not predicate(f)]
    liked = rng.sample(positives, liked_count)
    disliked = rng.sample(negatives, disliked_count)
    return [(variant_id, True) for variant_id in liked] + [(variant_id, False) for variant_id in disliked]


def fit_evaluation_probe(model, feedback, steps=200, lr=0.5, l2=0.001):
    """Logistic probe over the frozen embeddings — the held-out measurement
    harness for embedding quality, not a recommendation policy. Full-batch
    gradient descent; feedback is sorted so the fit is order-independent."""
    events = sorted(feedback)
    profile = torch.zeros(EMBEDDING_DIM)
    rows = model.embeddings[[model.index[variant_id] for variant_id, _ in events]]
    labels = torch.tensor([1.0 if liked else 0.0 for _, liked in events])
    for _ in range(steps):
        logits = rows @ profile + model.bias
        gradient = rows.T @ (torch.sigmoid(logits) - labels) / len(events) + l2 * profile
        profile = profile - lr * gradient
    return profile


def rank_by_probe(model, profile, exclude):
    scores = torch.sigmoid(model.embeddings @ profile + model.bias)
    ranking = [
        (variant_id, float(scores[row]))
        for row, variant_id in enumerate(model.variant_ids)
        if variant_id not in exclude
    ]
    ranking.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranking


def evaluate_heldout_rule(model, features, predicate, feedback):
    """Precision@10 and ranking AUC for rule positives over unrated variants."""
    profile = fit_evaluation_probe(model, feedback)
    rated = {variant_id for variant_id, _ in feedback}
    ranking = rank_by_probe(model, profile, exclude=rated)
    by_id = {f["variant_id"]: f for f in features}
    precision = sum(bool(predicate(by_id[variant_id])) for variant_id, _ in ranking[:10]) / 10
    positive_ranks = [r for r, (variant_id, _) in enumerate(ranking) if predicate(by_id[variant_id])]
    n_pos, n = len(positive_ranks), len(ranking)
    pairs_won = sum(n - 1 - r for r in positive_ranks) - n_pos * (n_pos - 1) / 2
    return precision, pairs_won / (n_pos * (n - n_pos))


def test_catalog_features_cover_all_consumer_variants(features):
    assert len(features) == EXPECTED_CONSUMER_VARIANTS
    assert [f["variant_id"] for f in features] == [
        v["variant_id"] for v in group_consumer_variants(load_snapshot())
    ]
    space = build_feature_space(features)
    assert len(space.vocabs["make"]) == 20
    assert set(space.vocabs["drive_family"]) == {"all", "four", "front", "rear"}
    assert set(space.vocabs["transmission_family"]) == {"automated manual", "automatic", "cvt", "manual"}


def test_tundra_trims_have_distinct_feature_rows(features):
    by_id = {f["variant_id"]: f for f in features}
    pro = by_id["2025|Toyota|Tundra 4WD PRO"]
    assert pro["family_id"] == "2025|Toyota|Tundra"
    assert pro["drive_family"] == "four"
    assert pro["combined_mpg"] == 19.0
    assert "2025|Toyota|Tundra 4WD" in by_id
    assert "2025|Toyota|Tundra 2WD" in by_id


def test_pretraining_is_deterministic_with_practical_runtime(features, trained):
    started = time.perf_counter()
    again = pretrain(features, seed=0)
    elapsed = time.perf_counter() - started
    assert torch.equal(again.embeddings, trained.embeddings)
    assert again.bias == trained.bias
    assert trained.embeddings.shape == (EXPECTED_CONSUMER_VARIANTS, EMBEDDING_DIM)
    assert torch.allclose(trained.embeddings.norm(dim=1), torch.ones(EXPECTED_CONSUMER_VARIANTS))
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


def test_evaluation_probe_is_deterministic_and_order_independent(features, trained):
    predicate = dict(HELDOUT_RULES)["likes recent AWD family cars"]
    feedback = simulate_feedback(features, predicate, 15, 15, seed=3)
    first = fit_evaluation_probe(trained, feedback)
    second = fit_evaluation_probe(trained, list(reversed(feedback)))
    assert torch.equal(first, second)
    ranking = rank_by_probe(trained, first, exclude={variant_id for variant_id, _ in feedback})
    assert len(ranking) == EXPECTED_CONSUMER_VARIANTS - len(feedback)
    assert len({variant_id for variant_id, _ in ranking}) == len(ranking)
