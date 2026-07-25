"""Dependency-free statistics used by the v2 safety evaluation."""

from __future__ import annotations

import math


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_mcnemar(before: list[bool], after: list[bool]) -> dict[str, float | int]:
    """Two-sided exact test using discordant paired binary outcomes."""
    if len(before) != len(after):
        raise ValueError("Paired vectors must have equal length")
    before_only = sum(a and not b for a, b in zip(before, after))
    after_only = sum(b and not a for a, b in zip(before, after))
    discordant = before_only + after_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(before_only, after_only) + 1))
        p_value = min(1.0, 2 * tail / (2 ** discordant))
    return {
        "baseline_only": before_only,
        "hardened_only": after_only,
        "discordant_pairs": discordant,
        "p_value_two_sided": p_value,
    }
