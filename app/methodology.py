"""Methodology identity reported by the API.

This repository demonstrates exactly one methodology: the Current
Relationships content tower (``app.model.pretrain``) as the similarity
prior, Bayesian ideal-point inference with the approval threshold
marginalized as a nuisance, Targeted Learning probe acquisition
(``app.preference.select_probe``: expected information gain about the ideal
variant alone), and the baseline marginal-posterior recommendation ranking
(``app.preference.rank_recommendations``).
"""

METHODOLOGY_ID = "targeted_learning_current_relationships"
METHODOLOGY_NAME = "Targeted Learning — Current Relationships"
MODEL_ID = "current"
PROBE_OBJECTIVE = "targeted"
