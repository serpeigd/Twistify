"""Shared statistics helpers for the calibration scripts.

A point estimate like "recall=0.089" implies more precision than a
180-review sample actually supports. The Wilson score interval (Wilson,
1927) is the standard fix for binomial proportions like precision/recall,
and unlike the naive normal approximation it stays inside [0, 1] and
doesn't collapse to a zero-width interval when k=0 (which the naive
`p +/- z*sqrt(p(1-p)/n)` formula does, wrongly implying certainty).
"""

from __future__ import annotations

import math


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% (default z=1.96) Wilson score interval for k successes out of n
    trials. Returns None if n == 0 (undefined, not zero-width)."""
    if n == 0:
        return None
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return max(0.0, lo), min(1.0, hi)
