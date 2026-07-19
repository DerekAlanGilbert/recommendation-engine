"""Deterministic synthetic cars, training personas, and labeled interactions."""

import random

import torch

CARS_SEED = 42
PERSONAS_SEED = 7

MAKES = ["Toyota", "Honda", "Ford", "Chevy", "BMW", "Audi", "Kia", "Subaru", "Tesla", "Volvo"]
BODIES = ["sedan", "suv", "truck", "coupe", "hatchback", "wagon"]
FUELS = ["gas", "hybrid", "electric"]
NUMERIC_FIELDS = ["year", "price", "horsepower", "mpg"]


def make_cars(n=1000, seed=CARS_SEED):
    rng = random.Random(seed)
    cars = []
    for i in range(n):
        cars.append({
            "id": i,
            "make": rng.choice(MAKES),
            "body": rng.choice(BODIES),
            "fuel": rng.choice(FUELS),
            "year": rng.randint(2010, 2025),
            "price": rng.randint(10_000, 120_000),
            "horsepower": rng.randint(70, 600),
            "mpg": rng.randint(15, 60),
        })
    return cars


def car_features(cars):
    """One-hot categoricals plus standardized numerics."""
    means = {f: sum(c[f] for c in cars) / len(cars) for f in NUMERIC_FIELDS}
    stds = {
        f: (sum((c[f] - means[f]) ** 2 for c in cars) / len(cars)) ** 0.5
        for f in NUMERIC_FIELDS
    }
    rows = []
    for c in cars:
        row = [1.0 if c["make"] == m else 0.0 for m in MAKES]
        row += [1.0 if c["body"] == b else 0.0 for b in BODIES]
        row += [1.0 if c["fuel"] == f else 0.0 for f in FUELS]
        row += [(c[f] - means[f]) / stds[f] for f in NUMERIC_FIELDS]
        rows.append(row)
    return torch.tensor(rows)


def quantile(cars, field, frac):
    vals = sorted(c[field] for c in cars)
    return vals[int(frac * len(vals))]


def persona_likes(persona, car):
    """A persona likes a car iff all of its conditions hold."""
    for field, op, value in persona["conditions"]:
        actual = car[field]
        ok = actual == value if op == "eq" else actual < value if op == "lt" else actual > value
        if not ok:
            return False
    return True


def eval_personas(cars):
    """Held-out personas for the simulation; never used in pretraining."""
    return [
        {"name": "affordable electric",
         "conditions": (("fuel", "eq", "electric"), ("price", "lt", quantile(cars, "price", 0.5)))},
        {"name": "recent suv",
         "conditions": (("body", "eq", "suv"), ("year", "gt", quantile(cars, "year", 0.25) - 1))},
        {"name": "efficient hybrid",
         "conditions": (("fuel", "eq", "hybrid"), ("mpg", "gt", quantile(cars, "mpg", 0.5)))},
    ]


def _condition_pool(cars):
    pool = [("make", "eq", m) for m in MAKES]
    pool += [("body", "eq", b) for b in BODIES]
    pool += [("fuel", "eq", f) for f in FUELS]
    for field in NUMERIC_FIELDS:
        for frac in (0.25, 0.5, 0.75):
            cut = quantile(cars, field, frac)
            pool.append((field, "lt", cut))
            pool.append((field, "gt", cut))
    return pool


def make_personas(cars, n=64, seed=PERSONAS_SEED):
    rng = random.Random(seed)
    pool = _condition_pool(cars)
    held_out = {frozenset(p["conditions"]) for p in eval_personas(cars)}
    personas, used = [], set()
    while len(personas) < n:
        conditions = tuple(sorted(rng.sample(pool, 2)))
        key = frozenset(conditions)
        if key in used or key in held_out:
            continue
        persona = {"name": f"persona-{len(personas)}", "conditions": conditions}
        liked = sum(persona_likes(persona, c) for c in cars)
        if 20 <= liked <= 500:
            used.add(key)
            personas.append(persona)
    return personas


def make_interactions(cars, personas, seed=PERSONAS_SEED, per_side=100):
    """Balanced (persona_idx, car_id, label) triples, sampled with replacement."""
    rng = random.Random(seed + 1)
    out = []
    for i, persona in enumerate(personas):
        liked = [c["id"] for c in cars if persona_likes(persona, c)]
        disliked = [c["id"] for c in cars if not persona_likes(persona, c)]
        out += [(i, rng.choice(liked), 1) for _ in range(per_side)]
        out += [(i, rng.choice(disliked), 0) for _ in range(per_side)]
    return out
