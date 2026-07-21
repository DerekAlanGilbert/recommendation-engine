"""Current Relationships content tower over consumer variants.

The tower encodes each of the 6,606 consumer-facing variants into a frozen
32-dimensional embedding from structured EPA attributes; the embeddings are
the catalog similarity prior for the Bayesian preference engine. Pretraining
uses deterministic synthetic single-attribute rules over real catalog
attributes, with held-out combination rules structurally excluded so
embedding quality can be proven on interactions the training never saw
(see tests/test_model.py).
"""

import random
from collections import Counter
from typing import NamedTuple

import torch

from app.data import consumer_key, family_id_of, load_snapshot

# Single-threaded torch keeps every reduction order, and therefore every embedding,
# score, and ranking, bit-reproducible across runs and restarts. The model is far
# too small for parallelism to matter.
torch.set_num_threads(1)

EMBEDDING_DIM = 32
HIDDEN_DIM = 64

# (feature name, learned embedding dimensions)
CATEGORICAL_FEATURES = (
    ("make", 8),
    ("vehicle_class", 6),
    ("fuel_type", 4),
    ("drive_family", 4),
    ("transmission_family", 4),
)

NUMERIC_FEATURES = (
    "year", "city_mpg", "highway_mpg", "combined_mpg",
    "cylinders", "displacement", "electric_range", "co2_tailpipe_gpm",
)


def drive_family(drive):
    if "All-Wheel" in drive:
        return "all"
    if "4-Wheel" in drive:
        return "four"
    if "Front" in drive:
        return "front"
    if "Rear" in drive:
        return "rear"
    return "other"


def transmission_family(transmission):
    if transmission.startswith("Manual"):
        return "manual"
    if "variable gear ratios" in transmission or "(AV" in transmission:
        return "cvt"
    if "(AM" in transmission:
        return "automated manual"
    return "automatic"


def _modal(values):
    counts = Counter(values)
    best = max(counts.values())
    return min(value for value, count in counts.items() if count == best)


def _mean(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _primary_family_id(members):
    counts = Counter(family_id_of(c) for c in members)
    best = max(counts.values())
    return min(family_id for family_id, count in counts.items() if count == best)


def build_variant_features(configs):
    """Aggregate EPA configs into one deterministic feature dict per consumer variant."""
    groups = {}
    for config in configs:
        groups.setdefault(consumer_key(config), []).append(config)
    features = []
    for year, make, model in sorted(groups):
        members = groups[(year, make, model)]
        family_id = _primary_family_id(members)
        feature = {
            "variant_id": f"{year}|{make}|{model}",
            "family_id": family_id,
            "model": model,
            "base_model": family_id.split("|")[2],
            "year": year,
            "make": make,
            "vehicle_class": _modal([m["vehicle_class"] for m in members]),
            "fuel_type": _modal([m["fuel_type"] for m in members]),
            "drive_family": _modal([drive_family(m["drive"]) for m in members]),
            "transmission_family": _modal([transmission_family(m["transmission"]) for m in members]),
        }
        for name in NUMERIC_FEATURES:
            if name != "year":
                feature[name] = _mean([m[name] for m in members])
        features.append(feature)
    return features


def load_catalog_features(path=None):
    configs = load_snapshot(path) if path is not None else load_snapshot()
    return build_variant_features(configs)


class FeatureSpace(NamedTuple):
    vocabs: dict
    means: dict
    stds: dict


def build_feature_space(features):
    vocabs = {
        name: {value: index for index, value in enumerate(sorted({f[name] for f in features}))}
        for name, _ in CATEGORICAL_FEATURES
    }
    means, stds = {}, {}
    for name in NUMERIC_FEATURES:
        present = [f[name] for f in features if f[name] is not None]
        mean = sum(present) / len(present)
        variance = sum((value - mean) ** 2 for value in present) / len(present)
        means[name] = mean
        stds[name] = variance ** 0.5 or 1.0
    return FeatureSpace(vocabs, means, stds)


def encode_features(features, space):
    """Encode variant features as index and standardized tensors; missing numerics become 0."""
    categorical = torch.tensor(
        [[space.vocabs[name][f[name]] for name, _ in CATEGORICAL_FEATURES] for f in features],
        dtype=torch.long,
    )
    numeric = torch.tensor(
        [
            [
                0.0 if f[name] is None else (f[name] - space.means[name]) / space.stds[name]
                for name in NUMERIC_FEATURES
            ]
            for f in features
        ],
        dtype=torch.float32,
    )
    return categorical, numeric


class ContentTower(torch.nn.Module):
    def __init__(self, space):
        super().__init__()
        self.embeddings = torch.nn.ModuleList(
            torch.nn.Embedding(len(space.vocabs[name]), dimensions)
            for name, dimensions in CATEGORICAL_FEATURES
        )
        input_dim = sum(d for _, d in CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, EMBEDDING_DIM),
        )

    def forward(self, categorical, numeric):
        parts = [embed(categorical[:, i]) for i, embed in enumerate(self.embeddings)]
        return self.mlp(torch.cat(parts + [numeric], dim=1))


# Synthetic preference rules over real variant attributes. Pretraining sees only
# pretrain_rules(): dense single-attribute rules derived from the catalog. The
# HELDOUT_RULES are novel attribute combinations whose labelings match no
# pretraining rule; they exist solely for held-out evaluation and are never passed
# to interaction generation by pretrain().

MIN_RULE_POSITIVES = 8

NUMERIC_PRETRAIN_RULES = (
    ("likes combined mpg at most 17", lambda f: f["combined_mpg"] <= 17),
    ("likes combined mpg at most 22", lambda f: f["combined_mpg"] <= 22),
    ("likes combined mpg at least 25", lambda f: f["combined_mpg"] >= 25),
    ("likes combined mpg at least 30", lambda f: f["combined_mpg"] >= 30),
    ("likes four or fewer cylinders", lambda f: f["cylinders"] is not None and f["cylinders"] <= 4),
    ("likes six or more cylinders", lambda f: (f["cylinders"] or 0) >= 6),
    ("likes eight or more cylinders", lambda f: (f["cylinders"] or 0) >= 8),
    ("likes displacement at least 3.5", lambda f: (f["displacement"] or 0) >= 3.5),
    ("likes displacement at least 5.0", lambda f: (f["displacement"] or 0) >= 5.0),
    ("likes model year at most 2019", lambda f: f["year"] <= 2019),
    ("likes model year at least 2022", lambda f: f["year"] >= 2022),
    ("likes model year at least 2024", lambda f: f["year"] >= 2024),
    ("likes electric range at least 100", lambda f: f["electric_range"] >= 100),
    ("likes co2 at most 100", lambda f: f["co2_tailpipe_gpm"] <= 100),
    ("likes co2 at least 450", lambda f: f["co2_tailpipe_gpm"] >= 450),
)


def pretrain_rules(features):
    """One rule per sufficiently common categorical value, plus fixed numeric thresholds."""
    rules = []
    for name, _ in CATEGORICAL_FEATURES:
        for value in sorted({f[name] for f in features}):
            if sum(f[name] == value for f in features) >= MIN_RULE_POSITIVES:
                rules.append(
                    (f"likes {name} {value}", lambda f, name=name, value=value: f[name] == value)
                )
    return tuple(rules) + NUMERIC_PRETRAIN_RULES


HELDOUT_RULES = (
    ("likes efficient small SUVs",
     lambda f: f["vehicle_class"].startswith("Small Sport Utility") and f["combined_mpg"] >= 27),
    ("likes premium six-cylinder sedans",
     lambda f: f["fuel_type"] == "Premium" and (f["cylinders"] or 0) >= 6
     and f["vehicle_class"] in ("Compact Cars", "Midsize Cars", "Large Cars")),
    ("likes recent AWD family cars",
     lambda f: f["drive_family"] == "all" and f["year"] >= 2022
     and f["vehicle_class"] in ("Midsize Cars", "Small Station Wagons", "Midsize Station Wagons")),
)


def generate_interactions(features, rules, per_rule=120, seed=0):
    """Deterministic balanced (profile, variant, label) interactions for the given rules."""
    interactions = []
    for profile_index, (_, predicate) in enumerate(rules):
        rng = random.Random(seed * 100003 + profile_index)
        positives = [i for i, f in enumerate(features) if predicate(f)]
        negatives = [i for i, f in enumerate(features) if not predicate(f)]
        count = min(per_rule, len(positives), len(negatives))
        for variant_index in sorted(rng.sample(positives, count)):
            interactions.append((profile_index, variant_index, 1.0))
        for variant_index in sorted(rng.sample(negatives, count)):
            interactions.append((profile_index, variant_index, 0.0))
    return interactions


class RecommenderModel(NamedTuple):
    variant_ids: tuple
    embeddings: torch.Tensor  # frozen unit-norm (len(variant_ids), EMBEDDING_DIM)
    bias: float
    index: dict


def _normalize(embeddings):
    return embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-12)


def _freeze_model(features, embeddings, bias):
    variant_ids = tuple(f["variant_id"] for f in features)
    index = {variant_id: row for row, variant_id in enumerate(variant_ids)}
    return RecommenderModel(variant_ids, _normalize(embeddings).detach().clone(), float(bias), index)


def untrained_model(features, seed=0):
    """Randomly initialized content tower with no pretraining; the comparison baseline."""
    torch.manual_seed(seed)
    space = build_feature_space(features)
    tower = ContentTower(space)
    categorical, numeric = encode_features(features, space)
    with torch.no_grad():
        embeddings = tower(categorical, numeric)
    return _freeze_model(features, embeddings, 0.0)


def pretrain(features, seed=0, epochs=500, lr=0.02, per_rule=120):
    """Train the content tower on pretrain_rules() interactions and freeze variant embeddings.

    Variant embeddings live on the unit sphere during training and after freezing so
    similarity and profile refits work in the same well-conditioned geometry.
    """
    torch.manual_seed(seed)
    space = build_feature_space(features)
    tower = ContentTower(space)
    rules = pretrain_rules(features)
    profiles = torch.nn.Embedding(len(rules), EMBEDDING_DIM)
    bias = torch.nn.Parameter(torch.zeros(1))
    interactions = generate_interactions(features, rules, per_rule=per_rule, seed=seed)
    profile_index = torch.tensor([p for p, _, _ in interactions])
    variant_index = torch.tensor([i for _, i, _ in interactions])
    labels = torch.tensor([label for _, _, label in interactions])
    categorical, numeric = encode_features(features, space)
    parameters = list(tower.parameters()) + list(profiles.parameters()) + [bias]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        variant_embeddings = _normalize(tower(categorical, numeric))
        logits = (profiles(profile_index) * variant_embeddings[variant_index]).sum(dim=1) + bias
        loss_fn(logits, labels).backward()
        optimizer.step()
    with torch.no_grad():
        embeddings = tower(categorical, numeric)
    return _freeze_model(features, embeddings, bias.item())
