from app.data import (
    MAKES,
    BODIES,
    FUELS,
    car_features,
    eval_personas,
    make_cars,
    make_interactions,
    make_personas,
    persona_likes,
)


def test_exactly_1000_cars():
    cars = make_cars()
    assert len(cars) == 1000
    assert sorted(c["id"] for c in cars) == list(range(1000))


def test_cars_deterministic():
    assert make_cars() == make_cars()


def test_car_fields():
    for car in make_cars():
        assert car["make"] in MAKES
        assert car["body"] in BODIES
        assert car["fuel"] in FUELS
        assert 2010 <= car["year"] <= 2025
        assert 10_000 <= car["price"] <= 120_000
        assert 70 <= car["horsepower"] <= 600
        assert 15 <= car["mpg"] <= 60


def test_car_features():
    cars = make_cars()
    feats = car_features(cars)
    assert feats.shape[0] == 1000
    assert feats.isfinite().all()
    assert (feats == car_features(cars)).all()


def test_personas_deterministic():
    cars = make_cars()
    a, b = make_personas(cars), make_personas(cars)
    assert a == b
    assert len(a) == 64


def test_persona_liked_counts_reasonable():
    cars = make_cars()
    for p in make_personas(cars):
        liked = sum(persona_likes(p, c) for c in cars)
        assert 20 <= liked <= 500


def test_training_personas_disjoint_from_eval():
    cars = make_cars()
    held_out = {frozenset(p["conditions"]) for p in eval_personas(cars)}
    for p in make_personas(cars):
        assert frozenset(p["conditions"]) not in held_out


def test_eval_persona_base_rates():
    cars = make_cars()
    personas = eval_personas(cars)
    assert len(personas) == 3
    for p in personas:
        liked = sum(persona_likes(p, c) for c in cars)
        assert 100 <= liked <= 200


def test_interactions():
    cars = make_cars()
    personas = make_personas(cars)
    inter = make_interactions(cars, personas)
    assert inter == make_interactions(cars, personas)
    for p_idx, car_id, label in inter:
        assert label == int(persona_likes(personas[p_idx], cars[car_id]))
    for i in range(len(personas)):
        pos = sum(1 for p, _, l in inter if p == i and l == 1)
        neg = sum(1 for p, _, l in inter if p == i and l == 0)
        assert pos == 100 and neg == 100
