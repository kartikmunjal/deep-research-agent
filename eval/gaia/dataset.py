"""Download and enumerate the gated official GAIA validation split."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def download_official_gaia(cache_dir: Path, level: int = 1) -> tuple[list[dict[str, Any]], Path]:
    """Return official 2023 validation rows and the private snapshot root."""
    if level not in {1, 2, 3}:
        raise ValueError("level must be 1, 2, or 3")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required. Accept the GAIA dataset terms at "
            "https://huggingface.co/datasets/gaia-benchmark/GAIA first."
        )
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            token=token,
            cache_dir=cache_dir,
        )
    )
    split = load_dataset(str(snapshot), f"2023_level{level}", split="validation")
    rows = []
    for raw in split:
        row = dict(raw)
        row["attachment_path"] = (
            str(snapshot / row["file_path"]) if row.get("file_path") else None
        )
        rows.append(row)
    return rows, snapshot
