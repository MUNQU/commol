"""Scores of a predictive against observations: ``from commol import scores``.

See `commol.api.inference.scores`.
"""

from commol.api.inference.scores import (
    coverage,
    crps,
    interval_score,
    log_score,
    pit,
    weighted_interval_score,
)

__all__ = [
    "coverage",
    "crps",
    "interval_score",
    "log_score",
    "pit",
    "weighted_interval_score",
]
