"""Bayesian package-level preference engine with hierarchical active probing.

Ideal-point model with a latent approval threshold. The shopper has one
unknown ideal consumer variant t* and one unknown pickiness threshold θ; a
package-level thumb on shown variant x is a Bernoulli observation

    P(up | x, t, θ) = sigmoid(SHARPNESS · (sim(x, t) − θ))

where sim blends the frozen content-tower embedding cosine (structured EPA
attributes) with jaccard overlap of EPA model-string tokens (the only honest
handle on trim identity such as "Tundra 4WD PRO"). The posterior over the
joint hypothesis space (every variant × a discrete threshold grid) is
computed exactly, in log space, from the complete feedback history, and the
recommendation ranking marginalizes the threshold out.

The threshold marginalization is what makes binary package feedback usable:

- One rejection is explained partly by pickiness (high θ) and partly by
  dissimilarity (t far from x), so ambiguity is preserved and no hypothesis
  is crushed — with no ad-hoc down-weighting constants.
- Early responses update beliefs with broad reach (marginal over θ), so
  probes bisect the catalog by package character; as evidence accumulates
  the threshold belief concentrates and updates sharpen automatically —
  coarse-to-fine without a schedule.
- Rejecting a near-clone of an endorsed package is explained by a threshold
  between the two similarities instead of collapsing the endorsed
  neighborhood, which is exactly how a satisficing shopper behaves.

Probe selection maximizes expected information gain (bits about the joint
hypothesis from one binary response) blended with an exploitation term as
posterior entropy falls. Candidates come from a hierarchical pool — one
representative per nameplate plus the current posterior leaders — so
elicitation sweeps the catalog's structure instead of drilling one
nameplate's near-identical year siblings. Everything is deterministic given
the feedback history: no sampling anywhere.
"""

import math
from typing import NamedTuple

import numpy as np
import torch

# Similarity blend over [0, 1]: embedding cosine rescaled to [0, 1] plus
# model-token jaccard, equally weighted so trim identity and structured
# attributes both matter.
COSINE_WEIGHT = 0.5
TOKEN_WEIGHT = 0.5

# Response acuity: how sharply approval probability rises as similarity
# crosses the shopper's threshold. One interpretable constant, selected on
# the whole simulation cohort (never on a single target).
SHARPNESS = 10.0

# Discrete grid for the latent approval threshold, integrated exactly. The
# 0.99 level is the "nothing but an exact match" shopper, so endorsing a
# package and rejecting its close siblings stays explainable.
THETA_GRID = tuple(round(0.30 + 0.05 * k, 2) for k in range(14)) + (0.99,)

# Probe acquisition: weight on information gain versus exploitation. The
# exploration weight tracks normalized posterior entropy but is capped so a
# plausibility term always survives and probes never chase pure novelty.
MAX_EXPLORATION_WEIGHT = 0.8

# At most this many variants of one model-year family among the top-ranked
# recommendations; further siblings are deferred below, never dropped.
FAMILY_CAP = 2

# Hierarchical probe candidate pool: one representative per nameplate
# (coarse sweep) plus the leading variants by marginal posterior (fine
# refinement), capped per nameplate so year siblings cannot flood the pool.
POOL_LEADERS = 50
POOL_LEADERS_PER_NAMEPLATE = 3

_LN2 = math.log(2.0)


def model_tokens(model):
    return frozenset(model.upper().split())


class PreferenceEngine(NamedTuple):
    variant_ids: tuple
    index: dict
    family_ids: tuple  # primary family aligned with variant_ids
    nameplates: tuple  # (make, base_model) aligned with variant_ids
    similarity: torch.Tensor  # (n, n) blended similarity in [0, 1]
    thetas: torch.Tensor  # (k,) threshold grid
    cold_approval: torch.Tensor  # (n,) predictive approval under the uniform prior


def _token_jaccard_matrix(models):
    """Exact pairwise jaccard of model-token sets via 0/1 incidence matmuls.

    Counts are small integers, exactly representable in float32, so the result
    is deterministic regardless of BLAS reduction order.
    """
    token_sets = [model_tokens(model) for model in models]
    vocabulary = sorted(set().union(*token_sets))
    column = {token: i for i, token in enumerate(vocabulary)}
    incidence = np.zeros((len(models), len(vocabulary)), dtype=np.float32)
    for row, tokens in enumerate(token_sets):
        for token in tokens:
            incidence[row, column[token]] = 1.0
    intersection = incidence @ incidence.T
    sizes = incidence.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - intersection
    return intersection / union


def _likelihood_rows(similarity_rows, thetas):
    """P(up | shown x, ideal t, θ) for the given similarity rows, per θ level."""
    return torch.sigmoid(
        SHARPNESS * (similarity_rows.unsqueeze(0) - thetas.view(-1, 1, 1))
    )


def build_engine(model, features):
    """Precompute the similarity matrix and cold-start predictive approval."""
    variant_ids = tuple(f["variant_id"] for f in features)
    if variant_ids != model.variant_ids:
        raise ValueError("features and model must cover the same variants in the same order")
    cosine01 = (model.embeddings @ model.embeddings.T + 1.0) / 2.0
    jaccard = torch.from_numpy(_token_jaccard_matrix([f["model"] for f in features]))
    similarity = COSINE_WEIGHT * cosine01 + TOKEN_WEIGHT * jaccard
    thetas = torch.tensor(THETA_GRID, dtype=torch.float32)
    n = len(variant_ids)
    cold_approval = torch.zeros(n)
    for theta in thetas:  # one pass per level keeps peak memory at one matrix
        cold_approval += torch.sigmoid(SHARPNESS * (similarity - theta)).mean(dim=1)
    cold_approval /= len(thetas)
    return PreferenceEngine(
        variant_ids=variant_ids,
        index={variant_id: row for row, variant_id in enumerate(variant_ids)},
        family_ids=tuple(f["family_id"] for f in features),
        nameplates=tuple((f["make"], f["base_model"]) for f in features),
        similarity=similarity,
        thetas=thetas,
        cold_approval=cold_approval,
    )


def similarity_between(engine, a, b):
    return float(engine.similarity[engine.index[a], engine.index[b]])


def posterior_from_feedback(engine, feedback):
    """Exact joint posterior over (variant, threshold) from the complete history.

    Log-space accumulation of static per-event likelihoods; the result is
    deterministic and independent of feedback order. Uniform joint prior
    encodes cold start.
    """
    k, n = len(engine.thetas), len(engine.variant_ids)
    log_posterior = torch.zeros((k, n), dtype=torch.float64)
    thetas = engine.thetas.to(torch.float64).view(-1, 1)
    for variant_id, liked in sorted(feedback):
        row = engine.similarity[engine.index[variant_id]].to(torch.float64)
        likelihood = torch.sigmoid(SHARPNESS * (row.unsqueeze(0) - thetas))
        log_posterior += torch.log(likelihood) if liked else torch.log1p(-likelihood)
    log_posterior -= log_posterior.max()
    posterior = torch.exp(log_posterior)
    return posterior / posterior.sum()


def variant_marginal(posterior):
    """Posterior probability per variant, threshold marginalized out."""
    return posterior.sum(dim=0)


def threshold_marginal(posterior):
    """Posterior probability per threshold level, variants marginalized out."""
    return posterior.sum(dim=1)


def variant_entropy_bits(posterior):
    """Shannon entropy (bits) of the variant-marginal posterior."""
    marginal = variant_marginal(posterior)
    positive = marginal[marginal > 0]
    return float(-(positive * torch.log(positive)).sum()) / _LN2


def exploration_weight(posterior):
    """Normalized variant-marginal entropy, capped so exploitation always keeps weight."""
    marginal = variant_marginal(posterior)
    positive = marginal[marginal > 0]
    entropy = float(-(positive * torch.log(positive)).sum())
    return min(entropy / math.log(len(marginal)), MAX_EXPLORATION_WEIGHT)


def _candidate_pool(engine, marginal, rated):
    """Hierarchical probe candidates: nameplate representatives plus leaders."""
    representative = {}
    for row, variant_id in enumerate(engine.variant_ids):
        if variant_id in rated:
            continue
        nameplate = engine.nameplates[row]
        key = (-float(marginal[row]), variant_id)
        if nameplate not in representative or key < representative[nameplate][0]:
            representative[nameplate] = (key, row)
    pool = {row for _, row in representative.values()}
    ordered = sorted(
        (row for row, variant_id in enumerate(engine.variant_ids) if variant_id not in rated),
        key=lambda row: (-float(marginal[row]), engine.variant_ids[row]),
    )
    per_nameplate = {}
    leaders = 0
    for row in ordered:
        if leaders >= POOL_LEADERS:
            break
        nameplate = engine.nameplates[row]
        if per_nameplate.get(nameplate, 0) >= POOL_LEADERS_PER_NAMEPLATE:
            continue
        per_nameplate[nameplate] = per_nameplate.get(nameplate, 0) + 1
        pool.add(row)
        leaders += 1
    return sorted(pool)


def _binary_entropy_bits(p):
    return -(
        torch.special.xlogy(p, p) + torch.special.xlogy(1.0 - p, 1.0 - p)
    ) / _LN2


def _joint_gain_and_approval(engine, posterior, rows):
    """I(response; T, Θ) in bits, plus predictive approval, per candidate row."""
    weights = posterior.to(torch.float32)  # (k, n)
    approval = torch.zeros(len(rows))
    conditional_entropy = torch.zeros(len(rows))
    similarity_rows = engine.similarity[rows]  # (m, n)
    for level, theta in enumerate(engine.thetas):
        likelihood = torch.sigmoid(SHARPNESS * (similarity_rows - theta))
        approval += likelihood @ weights[level]
        conditional_entropy += _binary_entropy_bits(likelihood) @ weights[level]
    gain = (_binary_entropy_bits(approval) - conditional_entropy).clamp(min=0.0)
    return gain, approval


def _targeted_gain_and_approval(engine, posterior, rows):
    """I(response; T) in bits — threshold integrated out per target hypothesis.

    For each candidate x and target t, the θ-marginalized approval predictive
    is p_t(x) = Σ_θ π(t, θ)·l(x, t, θ) / m(t); the gain is the mutual
    information between the binary response and T alone, so bits spent
    learning only the nuisance threshold no longer count. Targets with zero
    marginal mass contribute nothing (their joint term is exactly zero).
    """
    weights = posterior.to(torch.float32)  # (k, n)
    similarity_rows = engine.similarity[rows]  # (m, n)
    joint_up = torch.zeros(similarity_rows.shape)  # Σ_θ π(t, θ)·l(x, t, θ)
    for level, theta in enumerate(engine.thetas):
        joint_up += torch.sigmoid(SHARPNESS * (similarity_rows - theta)) * weights[level]
    approval = joint_up.sum(dim=1)
    marginal = weights.sum(dim=0)  # m(t)
    per_target = (joint_up / torch.where(marginal > 0, marginal, 1.0)).clamp(0.0, 1.0)
    conditional_entropy = _binary_entropy_bits(per_target) @ marginal
    gain = (_binary_entropy_bits(approval) - conditional_entropy).clamp(min=0.0)
    return gain, approval


def joint_information_gain(engine, posterior, rows):
    return _joint_gain_and_approval(engine, posterior, rows)[0]


def targeted_information_gain(engine, posterior, rows):
    return _targeted_gain_and_approval(engine, posterior, rows)[0]


GAIN_OBJECTIVES = {
    "joint": _joint_gain_and_approval,
    "targeted": _targeted_gain_and_approval,
}


def select_probe(engine, posterior, feedback=(), objective="joint"):
    """Pick the pool candidate maximizing the explore/exploit acquisition.

    Expected information gain is mutual information (bits) between one binary
    response and either the joint (variant, threshold) hypothesis
    (`objective="joint"`, the frozen baseline) or the ideal variant alone with
    the threshold treated as a nuisance parameter (`objective="targeted"`).
    The exploitation side blends predictive approval (plausibility: never
    probe absurd packages for information alone) with the variant-marginal
    posterior (as uncertainty falls, the probe becomes "is the current best
    guess your ideal?").
    """
    if objective not in GAIN_OBJECTIVES:
        raise ValueError(f"unknown probe objective: {objective}")
    rated = {variant_id for variant_id, _ in feedback}
    marginal = variant_marginal(posterior)
    pool = _candidate_pool(engine, marginal, rated)
    if not pool:
        return None
    gain, approval = GAIN_OBJECTIVES[objective](engine, posterior, pool)
    weight = exploration_weight(posterior)
    exploit = (marginal[pool] / marginal.max()).to(torch.float32)
    acquisition = weight * gain + (1.0 - weight) * 0.5 * (approval + exploit)
    best = min(
        range(len(pool)),
        key=lambda i: (-float(acquisition[i]), engine.variant_ids[pool[i]]),
    )
    return {
        "variant_id": engine.variant_ids[pool[best]],
        "expected_information_gain": float(gain[best]),
        "expected_approval": float(approval[best]),
        "exploration_weight": weight,
    }


def rank_recommendations(engine, posterior, exclude=frozenset(), limit=None,
                         family_cap=FAMILY_CAP):
    """Rank unrated variants by marginal posterior, deferring family siblings.

    The full ordering is by (marginal posterior, cold-start approval,
    variant_id). At most `family_cap` variants per model-year family keep
    their position at the top; further siblings are deferred to the end of
    the list rather than dropped.
    """
    marginal = variant_marginal(posterior)
    ordered = sorted(
        (
            (row, variant_id)
            for row, variant_id in enumerate(engine.variant_ids)
            if variant_id not in exclude
        ),
        key=lambda pair: (
            -float(marginal[pair[0]]),
            -float(engine.cold_approval[pair[0]]),
            pair[1],
        ),
    )
    leading, deferred, family_counts = [], [], {}
    for row, variant_id in ordered:
        family_id = engine.family_ids[row]
        count = family_counts.get(family_id, 0)
        if count < family_cap:
            family_counts[family_id] = count + 1
            leading.append((variant_id, float(marginal[row])))
        else:
            deferred.append((variant_id, float(marginal[row])))
    ranking = leading + deferred
    return ranking if limit is None else ranking[:limit]
