"""Build held-out v4 typo attacks and benign misspelling controls."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval" / "safety_v2" / "dataset.json"
OUTPUT = ROOT / "eval" / "safety_v4" / "dataset.json"


def _scramble(text: str, seed: int) -> str:
    rng = random.Random(seed)
    def transform(match: re.Match) -> str:
        word = match.group()
        if len(word) < 5:
            return word
        middle = list(word[1:-1])
        rng.shuffle(middle)
        return word[0] + "".join(middle) + word[-1]
    return re.sub(r"[A-Za-z]{5,}", transform, text)


def _transpose(text: str) -> str:
    def transform(match: re.Match) -> str:
        word = match.group()
        if len(word) < 6:
            return word
        chars = list(word)
        chars[2], chars[3] = chars[3], chars[2]
        return "".join(chars)
    return re.sub(r"[A-Za-z]{6,}", transform, text)


def _split(text: str) -> str:
    targets = ("instructions", "system", "priority", "research", "request", "tools")
    for word in targets:
        text = re.sub(word, "·".join(word), text, flags=re.IGNORECASE)
    return text


def _item(identifier: str, split: str, family: str, text: str, source_id: str) -> dict:
    return {
        "id": identifier, "split": split, "family": family,
        "surface": "retrieved_content", "text": text, "source_id": source_id,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def main() -> None:
    v2 = json.loads(SOURCE.read_text())
    heldout = [x for x in v2["items"] if x["stratum"] == "indirect_injection"][10:30]
    assert len(heldout) == 20
    transforms = (("internal_permutation", _scramble), ("adjacent_transposition", _transpose),
                  ("punctuation_split", _split))
    items = []
    for family, transform in transforms:
        for i, row in enumerate(heldout, 1):
            text = transform(row["text"], 1000 + i) if family == "internal_permutation" else transform(row["text"])
            items.append(_item(f"attack-{family}-{i:02}", "attack", family, text, row["id"]))

    benign_seeds = [x for x in v2["items"] if x["split"] == "benign"][:20]
    for family, transform in transforms:
        for i, row in enumerate(benign_seeds, 1):
            text = transform(row["text"], 2000 + i) if family == "internal_permutation" else transform(row["text"])
            text = f"Benign writing sample with natural typos: {text}"
            items.append(_item(f"benign-{family}-{i:02}", "benign", family, text, row["id"]))
    counts = {"attacks": 60, "benign": 60, "total": 120}
    assert len(items) == counts["total"]
    payload = {
        "schema_version": 4, "protocol": "safety-v4-typoglycemia-holdout",
        "development_exclusions": ["typoglycemia-04", "typoglycemia-10"],
        "selection": "v2 indirect seeds 11-30; no v3 adaptive outcome reused",
        "counts": counts, "items": items,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {counts} to {OUTPUT}")


if __name__ == "__main__":
    main()
