from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data_loader import get_categories, get_pair_types, load_pairs, pairs_to_texts
from src.models import CompatibilityLabel, DatingPair

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
TRAIN_PATH = DATA_DIR / "dating_pairs.jsonl"
EVAL_PATH = DATA_DIR / "eval_pairs.jsonl"


def test_load_pairs_valid():
    """Training file must load exactly 1195 valid pairs."""
    pairs = load_pairs(TRAIN_PATH)
    assert len(pairs) == 1195


def test_load_pairs_eval():
    """Eval file must load exactly 295 valid pairs."""
    pairs = load_pairs(EVAL_PATH)
    assert len(pairs) == 295


def test_dating_pair_valid_record():
    """Happy-path: valid record constructs without error."""
    pair = DatingPair(
        text_1="boy: I love hiking on weekends.",
        text_2="girl: Outdoor adventures are my favorite.",
        label=CompatibilityLabel.COMPATIBLE,
        category="hobbies",
        subcategory="outdoor",
        pair_type="compatible_match",
    )
    assert pair.label == CompatibilityLabel.COMPATIBLE


def test_dating_pair_invalid_gender():
    """Non-boy/girl gender prefix must raise ValidationError."""
    with pytest.raises(ValidationError):
        DatingPair(
            text_1="cat: hello",
            text_2="girl: hi there",
            label=0,
            category="test",
            subcategory="test",
            pair_type="test",
        )


def test_dating_pair_no_colon():
    """Text without colon must raise ValidationError."""
    with pytest.raises(ValidationError):
        DatingPair(
            text_1="no colon here",
            text_2="girl: hi there",
            label=0,
            category="test",
            subcategory="test",
            pair_type="test",
        )


def test_dating_pair_invalid_label():
    """Label value outside {0, 1} must raise ValidationError."""
    with pytest.raises(ValidationError):
        DatingPair(
            text_1="boy: hello",
            text_2="girl: hi",
            label=2,
            category="test",
            subcategory="test",
            pair_type="test",
        )


def test_pairs_to_texts():
    """pairs_to_texts returns three aligned parallel lists of correct length."""
    pairs = load_pairs(TRAIN_PATH)
    text_1s, text_2s, labels = pairs_to_texts(pairs)

    assert len(text_1s) == len(pairs)
    assert len(text_2s) == len(pairs)
    assert len(labels) == len(pairs)

    # WHY spot check first pair: ensures alignment, not just length
    assert text_1s[0] == pairs[0].text_1
    assert text_2s[0] == pairs[0].text_2
    assert labels[0] == int(pairs[0].label)

    # Labels must all be 0 or 1 (CompatibilityLabel enforces this, but verify int cast)
    assert all(lbl in (0, 1) for lbl in labels)


def test_all_records_have_gender_prefix():
    """Every text in the training set must start with boy: or girl:."""
    pairs = load_pairs(TRAIN_PATH)
    for i, pair in enumerate(pairs):
        for field_name, text in (("text_1", pair.text_1), ("text_2", pair.text_2)):
            gender = text.split(":", 1)[0].strip()
            assert gender in ("boy", "girl"), (
                f"Pair {i} {field_name} has invalid gender prefix '{gender}': {text[:60]}"
            )


def test_get_categories():
    """get_categories returns list aligned with pairs."""
    pairs = load_pairs(TRAIN_PATH)[:10]
    cats = get_categories(pairs)
    assert len(cats) == 10
    assert cats[0] == pairs[0].category


def test_get_pair_types():
    """get_pair_types returns list aligned with pairs."""
    pairs = load_pairs(TRAIN_PATH)[:10]
    ptypes = get_pair_types(pairs)
    assert len(ptypes) == 10
    assert ptypes[0] == pairs[0].pair_type
