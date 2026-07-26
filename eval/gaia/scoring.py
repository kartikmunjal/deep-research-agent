"""Official-style exact answer matching and uncertainty estimates."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any


def _normalize_number(value: str) -> float:
    for char in ["$", "%", ","]:
        value = value.replace(char, "")
    try:
        return float(value)
    except ValueError:
        return float("inf")


def _normalize_string(value: str, remove_punctuation: bool = True) -> str:
    value = re.sub(r"\s", "", str(value)).lower()
    return value.translate(str.maketrans("", "", string.punctuation)) if remove_punctuation else value


def exact_match(prediction: str, reference: str) -> bool:
    """Mirror gaia-benchmark/leaderboard's public question_scorer."""
    prediction, reference = str(prediction), str(reference)
    try:
        float(reference)
        numeric_reference = True
    except ValueError:
        numeric_reference = False
    if numeric_reference:
        return _normalize_number(prediction) == float(reference)
    if any(char in reference for char in [",", ";"]):
        expected = re.split(r"[,;]", reference)
        actual = re.split(r"[,;]", prediction)
        if len(expected) != len(actual):
            return False
        for actual_item, expected_item in zip(actual, expected):
            try:
                numeric = float(expected_item)
            except ValueError:
                if _normalize_string(actual_item, False) != _normalize_string(expected_item, False):
                    return False
            else:
                if _normalize_number(actual_item) != numeric:
                    return False
        return True
    return _normalize_string(prediction) == _normalize_string(reference)


def wilson(events: int, n_trials: int, z: float = 1.959963984540054) -> list[float]:
    if n_trials == 0:
        return [0.0, 0.0]
    p = events / n_trials
    denominator = 1 + z * z / n_trials
    center = (p + z * z / (2 * n_trials)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n_trials)) / n_trials) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    failures = Counter(row["failure_mode"] for row in rows if not row["correct"])
    return {
        "accuracy": {
            "events": correct,
            "n_trials": n,
            "rate": correct / n if n else 0.0,
            "ci_95": wilson(correct, n),
            "ci_method": "Wilson score",
        },
        "failure_modes": dict(sorted(failures.items())),
    }
