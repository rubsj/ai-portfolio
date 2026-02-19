from __future__ import annotations

from pathlib import Path

from src.data_evaluator import SyntheticDataEvaluator
from src.data_loader import load_pairs
from src.models import CompatibilityLabel, DatingPair

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
TRAIN_PATH = DATA_DIR / "dating_pairs.jsonl"


def _make_pair(text_1: str = "boy: hello there", text_2: str = "girl: hi there", label: int = 1) -> DatingPair:
    """Helper to build a minimal valid DatingPair."""
    return DatingPair(
        text_1=text_1,
        text_2=text_2,
        label=CompatibilityLabel(label),
        category="test",
        subcategory="test",
        pair_type="compatible_match",
    )


def test_evaluate_returns_report():
    """Evaluator on first 50 real pairs returns a DataQualityReport with correct structure."""
    pairs = load_pairs(TRAIN_PATH)[:50]
    evaluator = SyntheticDataEvaluator(pairs)
    report = evaluator.evaluate()

    assert report.record_count == 50
    assert report.timestamp  # Non-empty ISO string

    # All 4 dimensions present in details
    assert set(report.details.keys()) == {"data_quality", "diversity", "bias", "linguistic_quality"}

    # Each detail has sub_scores dict and a dimension_score
    for detail in report.details.values():
        assert isinstance(detail.sub_scores, dict)
        assert len(detail.sub_scores) > 0
        assert isinstance(detail.dimension_score, float)


def test_all_dimensions_in_range():
    """All scores (sub-scores and overall) must be in [0, 100]."""
    pairs = load_pairs(TRAIN_PATH)[:50]
    report = SyntheticDataEvaluator(pairs).evaluate()

    assert 0 <= report.scores.data_quality <= 100
    assert 0 <= report.scores.diversity <= 100
    assert 0 <= report.scores.bias <= 100
    assert 0 <= report.scores.linguistic_quality <= 100
    assert 0 <= report.scores.overall <= 100

    for detail in report.details.values():
        assert 0 <= detail.dimension_score <= 100
        for sub_score in detail.sub_scores.values():
            assert 0 <= sub_score <= 100, f"Sub-score out of range: {sub_score}"


def test_overall_is_average():
    """Overall == mean of the 4 dimension scores (within rounding tolerance)."""
    pairs = load_pairs(TRAIN_PATH)[:50]
    report = SyntheticDataEvaluator(pairs).evaluate()

    expected = (
        report.scores.data_quality
        + report.scores.diversity
        + report.scores.bias
        + report.scores.linguistic_quality
    ) / 4

    assert abs(report.scores.overall - expected) < 0.1, (
        f"Overall {report.scores.overall} differs from mean {expected}"
    )


def test_perfect_completeness():
    """All fully-populated records → completeness sub-score == 100.0."""
    pairs = [_make_pair() for _ in range(20)]
    evaluator = SyntheticDataEvaluator(pairs)
    detail = evaluator._evaluate_data_quality()
    assert detail.sub_scores["completeness"] == 100.0


def test_syllable_counter():
    """Known words → known syllable counts."""
    evaluator = SyntheticDataEvaluator([_make_pair()])

    # "hello" = hel-lo → 2 vowel groups (e, o)
    assert evaluator._count_syllables("hello") == 2

    # "a" = 1 vowel → 1 syllable
    assert evaluator._count_syllables("a") == 1

    # "beautiful" = beau-ti-ful → 3 vowel groups (eau=e+a+u→3 transitions, i, u)
    # Our heuristic: b-EAU-t-I-f-U-l → counts vowel groups: "eau"=1 run, "i"=1 run, "u"=1 run → 3
    assert evaluator._count_syllables("beautiful") == 3

    # "the" = th-E → 1 vowel group
    assert evaluator._count_syllables("the") == 1

    # Minimum 1 syllable even for consonant-only words (stripped punctuation)
    assert evaluator._count_syllables("xyz") >= 1


def test_duplicates_score_in_real_data():
    """Duplicate score is computed and in valid range (real data has some duplicates)."""
    pairs = load_pairs(TRAIN_PATH)
    evaluator = SyntheticDataEvaluator(pairs)
    detail = evaluator._evaluate_data_quality()
    # Score must be in range — we don't assert 100 since real data has duplicate pairs
    assert 0 <= detail.sub_scores["duplicates"] <= 100


def test_label_balance_with_synthetic_data():
    """Perfect 50/50 split → label_balance = 100."""
    pairs = [_make_pair(label=i % 2) for i in range(20)]
    evaluator = SyntheticDataEvaluator(pairs)
    detail = evaluator._evaluate_diversity()
    assert detail.sub_scores["label_balance"] == 100.0
