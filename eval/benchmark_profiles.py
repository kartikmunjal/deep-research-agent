"""Named task profiles for reproducible eval runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    description: str
    task_ids: tuple[str, ...]


PROFILES: dict[str, BenchmarkProfile] = {
    "core_live_28": BenchmarkProfile(
        name="core_live_28",
        description=(
            "Budget-bounded live benchmark for portfolio reporting: "
            "10 factual + 10 multi-hop + 8 unanswerable tasks."
        ),
        task_ids=(
            "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
            "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10",
            "U01", "U02", "U03", "U04", "U05", "U06", "U07", "U08",
        ),
    ),
    "internal_full_51": BenchmarkProfile(
        name="internal_full_51",
        description=(
            "All hand-authored internal tasks except the local GAIA subset."
        ),
        task_ids=tuple(
            [f"F{i:02d}" for i in range(1, 11)]
            + [f"M{i:02d}" for i in range(1, 18)]
            + [f"U{i:02d}" for i in range(1, 12)]
            + [f"C{i:02d}" for i in range(1, 14)]
        ),
    ),
    "gaia_local_12": BenchmarkProfile(
        name="gaia_local_12",
        description=(
            "Local 12-task GAIA Level-1 subset for sanity checks only; "
            "not the official GAIA validation benchmark."
        ),
        task_ids=tuple(f"G{i:02d}" for i in range(1, 13)),
    ),
}


def get_profile(name: str) -> BenchmarkProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown benchmark profile: {name}. "
            f"Available: {', '.join(sorted(PROFILES))}"
        ) from exc
