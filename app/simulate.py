"""Deterministic offline simulator for the active preference-elicitation proof.

The hidden shopper is a raw-attribute threshold utility over the ideal and the
shown package. It is deliberately a different functional family from the
engine's inference likelihood: hard threshold instead of sigmoid, directional
trim-token containment instead of symmetric jaccard, hand-set attribute
weights over raw EPA fields instead of learned embedding cosine. The shopper
answers only package-level thumbs; no attribute labels ever leak.

Three policies run the same loop protocol so active query selection can be
isolated: `greedy` (the previous product's profile-vector top score),
`passive` (Bayesian posterior, best-fit probes, no active selection), and
`active` (Bayesian posterior with information-gain probe selection).

Run `python -m app.simulate` for the full per-loop report.
"""

from typing import NamedTuple

from app.model import rank_items, refit_profile, score_items
from app.preference import (
    model_tokens,
    posterior_from_feedback,
    rank_recommendations,
    select_probe,
    variant_entropy_bits,
    variant_marginal,
)

# Hidden shopper weights. Interpretation: a shopper with one exact ideal
# package who recognizes the trim tokens printed on the label and judges shown
# packages by raw-attribute closeness to that ideal.
IDENTITY_WEIGHT = 2.0
TRIM_WEIGHT = 1.5
NAMEPLATE_WEIGHT = 1.0
DRIVE_WEIGHT = 1.0
CLASS_WEIGHT = 1.0
FUEL_WEIGHT = 0.75
MAKE_WEIGHT = 0.75
MPG_WEIGHT = 0.75
YEAR_WEIGHT = 0.75
MPG_SCALE = 15.0
YEAR_SCALE = 2.0

# Satisficing with rising aspiration: early in a session the shopper endorses
# any package clearing the base bar ("worth considering"), but once something
# better has been endorsed, only a clear improvement earns the next thumbs-up.
# This is a standard aspiration-adaptation account of sequential choice; it is
# what lets binary thumbs both steer discovery and separate trims.
BASE_THRESHOLD = 4.0
ASPIRATION_MARGIN = 0.1


class Persona(NamedTuple):
    """One hidden-shopper temperament: the satisficing bar as explicit data.

    The default persona reproduces the original constants exactly; alternative
    personas exist only for labeled robustness checks, never for tuning.
    """

    name: str
    base_threshold: float
    aspiration_margin: float


DEFAULT_PERSONA = Persona("default", BASE_THRESHOLD, ASPIRATION_MARGIN)

TUNDRA_PRO = "2025|Toyota|Tundra 4WD PRO"

# Deterministic multi-target cohort spanning makes, vehicle classes, and
# powertrains, so five-loop results cannot come from target-specific tuning.
COHORT_TARGETS = (
    TUNDRA_PRO,
    "2024|Hyundai|Ioniq 5 Long range AWD",
    "2024|Toyota|Prius Prime",
    "2023|Honda|Civic 4Dr",
    "2023|Chevrolet|Corvette Z06",
    "2024|Jeep|Wrangler 4dr 4xe",
    "2024|Subaru|Outback Wilderness AWD",
    "2023|Mazda|MX-5",
    "2024|BMW|M340i xDrive Sedan",
    "2020|Honda|Odyssey",
    "2024|Ford|F-150 Lightning PRO 4WD Extended Range",
    "2023|Mercedes-Benz|EQS 450 4matic (SUV)",
)

POLICIES = ("greedy", "passive", "active")

# The Bayesian active policies differ only in the probe-acquisition objective:
# "active" is the frozen joint-EIG baseline I(response; T, Θ); "targeted" is
# the treatment I(response; T) with the threshold integrated out as a nuisance.
PROBE_OBJECTIVES = {"active": "joint", "targeted": "targeted"}


def _closeness(a, b, scale):
    if a is None or b is None:
        return 0.0
    return max(0.0, 1.0 - abs(a - b) / scale)


def shopper_utility(ideal, candidate):
    """Hidden package utility from raw attributes, trim tokens, and identity."""
    ideal_tokens = model_tokens(ideal["model"])
    containment = len(ideal_tokens & model_tokens(candidate["model"])) / len(ideal_tokens)
    return (
        IDENTITY_WEIGHT * (candidate["variant_id"] == ideal["variant_id"])
        + TRIM_WEIGHT * containment
        + NAMEPLATE_WEIGHT * (
            candidate["make"] == ideal["make"]
            and candidate["base_model"] == ideal["base_model"]
        )
        + DRIVE_WEIGHT * (candidate["drive_family"] == ideal["drive_family"])
        + CLASS_WEIGHT * (candidate["vehicle_class"] == ideal["vehicle_class"])
        + FUEL_WEIGHT * (candidate["fuel_type"] == ideal["fuel_type"])
        + MAKE_WEIGHT * (candidate["make"] == ideal["make"])
        + MPG_WEIGHT * _closeness(candidate["combined_mpg"], ideal["combined_mpg"], MPG_SCALE)
        + YEAR_WEIGHT * _closeness(candidate["year"], ideal["year"], YEAR_SCALE)
    )


def shopper_thumb(ideal, candidate, best_liked_utility=None,
                  base_threshold=BASE_THRESHOLD, aspiration_margin=ASPIRATION_MARGIN):
    """The only observable signal: one thumb on the complete package.

    `best_liked_utility` is the shopper's own best already-endorsed utility;
    passing None models the start of a session. The bar parameters default to
    the original constants (the default persona).
    """
    bar = base_threshold
    if best_liked_utility is not None:
        bar = max(bar, best_liked_utility + aspiration_margin)
    return shopper_utility(ideal, candidate) >= bar


def _bayes_order(engine, posterior):
    """Full identification ranking over every variant, rated ones included."""
    marginal = variant_marginal(posterior)
    return sorted(
        range(len(engine.variant_ids)),
        key=lambda row: (
            -float(marginal[row]),
            -float(engine.cold_approval[row]),
            engine.variant_ids[row],
        ),
    )


def _greedy_rank_and_score(model, profile, target_id):
    ranking = rank_items(model, profile)
    rank = next(i for i, (variant_id, _) in enumerate(ranking) if variant_id == target_id) + 1
    return rank, float(score_items(model, profile)[model.index[target_id]])


def rollout(policy, model, engine, features, target_id, loops=5, persona=None):
    """Run one shopper against one policy; report per-loop identification state.

    `target_rank` is the target's position in the policy's full ranking over
    all variants (rated included), so rating the target itself still counts as
    identification. `target_share` is marginal posterior mass for the Bayesian
    policies and the sigmoid score for the greedy baseline. `surfaced` is the
    product-level outcome for the loop: the shopper was shown the target,
    either as the probe or inside the top-10 recommendations they would see
    after their thumb was recorded.
    """
    persona = persona or DEFAULT_PERSONA
    by_id = {f["variant_id"]: f for f in features}
    ideal = by_id[target_id]
    feedback = []
    records = []
    best_liked_utility = None
    for loop in range(1, loops + 1):
        rated = {variant_id for variant_id, _ in feedback}
        entropy_before = None
        if policy == "greedy":
            profile = refit_profile(model, feedback)
            probe_id = rank_items(model, profile, exclude=rated, limit=1)[0][0]
        elif policy in ("passive",) or policy in PROBE_OBJECTIVES:
            posterior_before = posterior_from_feedback(engine, feedback)
            entropy_before = variant_entropy_bits(posterior_before)
            if policy == "passive":
                order = _bayes_order(engine, posterior_before)
                probe_id = next(engine.variant_ids[row] for row in order
                                if engine.variant_ids[row] not in rated)
            else:
                probe_id = select_probe(engine, posterior_before, feedback,
                                        objective=PROBE_OBJECTIVES[policy])["variant_id"]
        else:
            raise ValueError(f"unknown policy: {policy}")
        liked = shopper_thumb(ideal, by_id[probe_id], best_liked_utility,
                              base_threshold=persona.base_threshold,
                              aspiration_margin=persona.aspiration_margin)
        if liked:
            utility = shopper_utility(ideal, by_id[probe_id])
            if best_liked_utility is None or utility > best_liked_utility:
                best_liked_utility = utility
        feedback.append((probe_id, liked))
        rated_now = {variant_id for variant_id, _ in feedback}
        entropy_after = None
        if policy == "greedy":
            profile = refit_profile(model, feedback)
            target_rank, target_share = _greedy_rank_and_score(model, profile, target_id)
            top_id = rank_items(model, profile, limit=1)[0][0]
            shown = [v for v, _ in rank_items(model, profile, exclude=rated_now, limit=10)]
        else:
            posterior = posterior_from_feedback(engine, feedback)
            entropy_after = variant_entropy_bits(posterior)
            order = _bayes_order(engine, posterior)
            target_rank = order.index(engine.index[target_id]) + 1
            target_share = float(variant_marginal(posterior)[engine.index[target_id]])
            top_id = engine.variant_ids[order[0]]
            shown = [v for v, _ in rank_recommendations(engine, posterior,
                                                        exclude=rated_now, limit=10)]
        records.append({
            "loop": loop,
            "probe_id": probe_id,
            "liked": liked,
            "target_rank": target_rank,
            "target_share": target_share,
            "top_id": top_id,
            "surfaced": probe_id == target_id or target_id in shown,
            "entropy_before_bits": entropy_before,
            "entropy_after_bits": entropy_after,
            "realized_bits": (None if entropy_before is None
                              else entropy_before - entropy_after),
        })
    return records


def surfaced_by(records, loop):
    """Whether the shopper had seen the target (probe or top-10) by this loop."""
    return any(r["surfaced"] for r in records[:loop])


def top_k_hits(runs_by_target, ks=(1, 3, 10)):
    """hit fraction per k and loop: hits[k][loop-1] over the given rollouts."""
    targets = sorted(runs_by_target)
    loops = len(runs_by_target[targets[0]])
    return {
        k: [
            sum(runs_by_target[t][loop]["target_rank"] <= k for t in targets) / len(targets)
            for loop in range(loops)
        ]
        for k in ks
    }


def _print_rollout(policy, target_id, records):
    print(f"\n  policy={policy}")
    for r in records:
        thumb = "up  " if r["liked"] else "down"
        print(f"    loop {r['loop']}: probe={r['probe_id']:<50} thumb={thumb} "
              f"target_rank={r['target_rank']:<5} share={r['target_share']:.5f} "
              f"top={r['top_id']}")


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (ordered[middle - 1] + ordered[middle]) / 2 if len(ordered) % 2 == 0 else ordered[middle]


def main():
    import argparse

    argparse.ArgumentParser(
        description="Deterministic offline simulation report: greedy, passive, and "
                    "active joint-EIG policies over the development cohort."
    ).parse_args()

    from app.data import load_snapshot
    from app.model import build_variant_features, pretrain
    from app.preference import build_engine

    print("pretraining content tower and building preference engine ...")
    features = build_variant_features(load_snapshot())
    model = pretrain(features, seed=0)
    engine = build_engine(model, features)

    print(f"\n=== Target case: {TUNDRA_PRO}, first five loops ===")
    for policy in POLICIES:
        _print_rollout(policy, TUNDRA_PRO, rollout(policy, model, engine, features, TUNDRA_PRO))

    print(f"\n=== Target case: {TUNDRA_PRO}, active policy to 25 loops ===")
    records = rollout("active", model, engine, features, TUNDRA_PRO, loops=25)
    _print_rollout("active", TUNDRA_PRO, records)
    first_top1 = next((r["loop"] for r in records if r["target_rank"] == 1), None)
    first_surfaced = next((r["loop"] for r in records if r["surfaced"]), None)
    print(f"\n  exact variant first reaches rank 1 at loop {first_top1}; "
          f"first surfaced to the shopper (probe or top-10) at loop {first_surfaced}; "
          f"min rank {min(r['target_rank'] for r in records)}; "
          f"final rank {records[-1]['target_rank']}")

    print("\n=== Cohort: top-k retrieval after each of the first five loops ===")
    runs_by_policy = {
        policy: {t: rollout(policy, model, engine, features, t, loops=15) for t in COHORT_TARGETS}
        for policy in POLICIES
    }
    for policy in POLICIES:
        runs = runs_by_policy[policy]
        hits = top_k_hits(runs)
        print(f"\n  policy={policy}")
        for k, fractions in hits.items():
            print(f"    top-{k}:  " + "  ".join(f"loop{i+1}={f:.2f}" for i, f in enumerate(fractions[:5])))
        for loop in (5, 15):
            print(f"    median target rank after loop {loop}: "
                  f"{_median([runs[t][loop - 1]['target_rank'] for t in COHORT_TARGETS]):.0f}")
            surfaced = sum(surfaced_by(runs[t], loop) for t in COHORT_TARGETS)
            print(f"    targets surfaced (probe or top-10) by loop {loop}: "
                  f"{surfaced}/{len(COHORT_TARGETS)}")
        final = {t: runs[t][-1]["target_rank"] for t in COHORT_TARGETS}
        print("    ranks after loop 15: "
              + ", ".join(f"{t.split('|')[2]}={r}" for t, r in sorted(final.items())))

    negative_only = [t.split("|")[2] for t in COHORT_TARGETS
                     if not any(r["liked"] for r in runs_by_policy["active"][t][:5])]
    mixed = [t.split("|")[2] for t in COHORT_TARGETS
             if any(r["liked"] for r in runs_by_policy["active"][t][:5])
             and not all(r["liked"] for r in runs_by_policy["active"][t][:5])]
    print(f"\n  negative-only first-five trajectories (active): {', '.join(negative_only)}")
    print(f"  mixed first-five trajectories (active): {', '.join(mixed)}")


if __name__ == "__main__":
    main()
