from __future__ import annotations

import json
from pathlib import Path

from src.models import DatingPair


def load_pairs(path: Path) -> list[DatingPair]:
    """Load JSONL file and validate each record with Pydantic.

    WHY line-by-line: JSONL isn't valid JSON as a whole file — each line is
    an independent JSON object. json.loads per line, then Pydantic validates.

    Raises ValueError if any record fails validation (fail-fast — bad data
    in training set corrupts embeddings silently if we skip).
    """
    pairs: list[DatingPair] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pairs.append(DatingPair.model_validate(record))
            except Exception as e:
                raise ValueError(f"Record at line {line_num} in {path} failed validation: {e}") from e
    return pairs


def load_metadata(path: Path) -> dict:
    """Load metadata JSON file. Returns raw dict.

    WHY no Pydantic here: metadata schema varies across projects — raw dict
    is flexible enough for downstream consumers.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pairs_to_texts(pairs: list[DatingPair]) -> tuple[list[str], list[str], list[int]]:
    """Extract parallel lists aligned with pairs.

    WHY parallel lists over list-of-tuples: SentenceTransformer.encode() expects
    a flat list of strings — this avoids unpacking at call sites.

    Returns: (text_1s, text_2s, labels_as_int)
    """
    text_1s = [p.text_1 for p in pairs]
    text_2s = [p.text_2 for p in pairs]
    labels = [int(p.label) for p in pairs]
    return text_1s, text_2s, labels


def get_categories(pairs: list[DatingPair]) -> list[str]:
    """Extract category list aligned with pairs.

    WHY separate function: callers often need only categories (e.g., for
    per-category metric breakdown) without the full text/label extraction.
    """
    return [p.category for p in pairs]


def get_pair_types(pairs: list[DatingPair]) -> list[str]:
    """Extract pair_type list aligned with pairs.

    WHY separate: pair_type drives the false-positive analysis — callers
    requesting it shouldn't be forced to call pairs_to_texts() just to discard texts.
    """
    return [p.pair_type for p in pairs]
