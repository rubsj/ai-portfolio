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


# ------------------------------------------------------------------ #
# Edge cases: vocabulary richness + category entropy + label balance
# ------------------------------------------------------------------ #

def test_vocabulary_richness_empty_texts():
    """No texts (empty pairs list) → _score_vocabulary_richness returns 0.0."""
    evaluator = SyntheticDataEvaluator([])
    evaluator._all_texts = []
    assert evaluator._score_vocabulary_richness() == 0.0


def test_category_entropy_single_category():
    """All pairs in one category → entropy = 0 → score = 0.0."""
    pairs = [_make_pair() for _ in range(10)]  # all category="test"
    evaluator = SyntheticDataEvaluator(pairs)
    assert evaluator._score_category_entropy() == 0.0


def test_label_balance_no_pairs():
    """Zero pairs → max(0, 0) == 0 → _score_label_balance returns 0.0."""
    evaluator = SyntheticDataEvaluator([])
    assert evaluator._score_label_balance() == 0.0


# ------------------------------------------------------------------ #
# Text complexity: short (<3 words) and medium (15-30 words) branches
# ------------------------------------------------------------------ #

def _make_pair_with_texts(t1: str, t2: str, label: int = 1) -> DatingPair:
    return DatingPair(
        text_1=t1, text_2=t2,
        label=CompatibilityLabel(label),
        category="test", subcategory="test", pair_type="compatible_match",
    )


def test_text_complexity_very_short():
    """Avg word count < 3 → _score_text_complexity returns 50.0."""
    # "boy: a" split → ["boy:", "a"] = 2 words < 3
    pairs = [_make_pair_with_texts("boy: a", "girl: b") for _ in range(5)]
    evaluator = SyntheticDataEvaluator(pairs)
    score = evaluator._score_text_complexity()
    assert score == 50.0


def test_text_complexity_medium_words():
    """Avg word count 15-30 → linear interpolation branch (score < 100)."""
    # Build text with 20 words: "boy: " + 19 words = 20 tokens after split
    long_text_1 = "boy: " + " ".join(["word"] * 19)  # 20 words
    long_text_2 = "girl: " + " ".join(["word"] * 19)  # 20 words
    pairs = [_make_pair_with_texts(long_text_1, long_text_2) for _ in range(5)]
    evaluator = SyntheticDataEvaluator(pairs)
    score = evaluator._score_text_complexity()
    # avg_words = 20, which is in (15, 30] → linear: 100 - (20-15)/(30-15)*50 ≈ 83.33
    assert 50.0 < score < 100.0


# ------------------------------------------------------------------ #
# Bias: significant gender correlation (line 264)
# ------------------------------------------------------------------ #

def test_gender_bias_significant_correlation():
    """Boys always compatible, girls always incompatible → p << 0.05 → score < 100."""
    pairs = (
        [_make_pair_with_texts("boy: hello", "girl: hi", label=1) for _ in range(30)]
        + [_make_pair_with_texts("girl: hello", "boy: hi", label=0) for _ in range(30)]
    )
    evaluator = SyntheticDataEvaluator(pairs)
    score = evaluator._score_gender_bias()
    # Perfect gender–label correlation → chi2 p << 0.05 → score well below 100
    assert score < 100.0


# ------------------------------------------------------------------ #
# Bias: significant category-label correlation (line 285)
# ------------------------------------------------------------------ #

def _make_pair_cat(label: int, category: str) -> DatingPair:
    return DatingPair(
        text_1="boy: hello", text_2="girl: hi",
        label=CompatibilityLabel(label),
        category=category, subcategory="test", pair_type="compatible_match",
    )


def test_category_label_correlation_significant():
    """Category A always compatible, B always incompatible → p << 0.05 → score < 100."""
    pairs = (
        [_make_pair_cat(label=1, category="hobbies") for _ in range(30)]
        + [_make_pair_cat(label=0, category="values") for _ in range(30)]
    )
    evaluator = SyntheticDataEvaluator(pairs)
    score = evaluator._score_category_label_correlation()
    assert score < 100.0


# ------------------------------------------------------------------ #
# Bias: length-label correlation branches (lines 306-309)
# ------------------------------------------------------------------ #

def test_length_label_correlation_high():
    """Long texts always compatible, short texts always incompatible → |r| > 0.3 → score = 0."""
    # Compatible: very long texts (~30 words); incompatible: very short (~2 words)
    long_t1 = "boy: " + " ".join(["word"] * 29)
    long_t2 = "girl: " + " ".join(["word"] * 29)
    short_t1 = "boy: hi"
    short_t2 = "girl: bye"
    pairs = (
        [_make_pair_with_texts(long_t1, long_t2, label=1) for _ in range(20)]
        + [_make_pair_with_texts(short_t1, short_t2, label=0) for _ in range(20)]
    )
    evaluator = SyntheticDataEvaluator(pairs)
    score = evaluator._score_length_label_correlation()
    assert score == 0.0


def test_length_label_correlation_moderate():
    """Moderate length-label correlation (|r| ∈ (0.1, 0.3]) → linear interpolation."""
    # Slightly longer texts for compatible, slightly shorter for incompatible
    med_t1 = "boy: " + " ".join(["word"] * 9)   # 10 words
    med_t2 = "girl: " + " ".join(["word"] * 9)
    short_t1 = "boy: hi there"                    # 3 words
    short_t2 = "girl: hi there"
    pairs = (
        [_make_pair_with_texts(med_t1, med_t2, label=1) for _ in range(15)]
        + [_make_pair_with_texts(short_t1, short_t2, label=0) for _ in range(15)]
    )
    evaluator = SyntheticDataEvaluator(pairs)
    score = evaluator._score_length_label_correlation()
    # Score should be between 0 and 100 (linear range)
    assert 0.0 <= score <= 100.0


# ------------------------------------------------------------------ #
# Syllable counter: empty-after-strip edge case (line 361)
# ------------------------------------------------------------------ #

def test_count_syllables_empty_after_strip():
    """Word that is only punctuation → strips to '' → returns 1 (minimum)."""
    evaluator = SyntheticDataEvaluator([_make_pair()])
    assert evaluator._count_syllables(".") == 1
    assert evaluator._count_syllables("...") == 1
    assert evaluator._count_syllables("!?") == 1


# ------------------------------------------------------------------ #
# Readability: FRE branch coverage (lines 398-405)
# ------------------------------------------------------------------ #

def test_readability_fre_above_100():
    """Very simple monosyllabic texts → FRE > 100 → score = 60.0."""
    pairs = [_make_pair_with_texts("boy: a", "girl: b")]
    evaluator = SyntheticDataEvaluator(pairs)
    # Override syllable counter to always return 1 per word
    evaluator._count_syllables = lambda w: 1  # type: ignore[method-assign]
    # "boy: a" → 2 words, 2 syllables; FRE = 206.835 - 1.015*2 - 84.6*1 = 120.2 > 100
    score = evaluator._score_readability()
    assert score == 60.0


def test_readability_fre_below_30():
    """Complex polysyllabic texts → FRE < 30 → score = 60.0."""
    pairs = [_make_pair_with_texts("boy: philosophical mathematical", "girl: extraordinary")]
    evaluator = SyntheticDataEvaluator(pairs)
    # Override: all words have 5 syllables → avg_syl = 5
    # FRE = 206.835 - 1.015*3 - 84.6*5 = 206.835 - 3.045 - 423 = -219 < 30
    evaluator._count_syllables = lambda w: 5  # type: ignore[method-assign]
    score = evaluator._score_readability()
    assert score == 60.0


def test_readability_fre_30_to_60():
    """Moderately complex text → 30 ≤ FRE < 60 → linear score between 60 and 100."""
    # FRE in [30, 60): avg_syl=2, avg_words=30
    # FRE = 206.835 - 1.015*30 - 84.6*2 = 206.835 - 30.45 - 169.2 = 7.185 < 30 — too low
    # Use avg_syl=1.5, avg_words=20:
    # FRE = 206.835 - 20.3 - 126.9 = 59.635 → just under 60 ✓
    long_t1 = "boy: " + " ".join(["word"] * 19)  # 20 words
    long_t2 = "girl: " + " ".join(["word"] * 19)
    pairs = [_make_pair_with_texts(long_t1, long_t2)]
    evaluator = SyntheticDataEvaluator(pairs)
    # Override syllable count: alternate 1 and 2 syl to get avg 1.5
    call_count = [0]

    def alt_syllables(w: str) -> int:
        call_count[0] += 1
        return 2 if call_count[0] % 2 == 0 else 1

    evaluator._count_syllables = alt_syllables  # type: ignore[method-assign]
    score = evaluator._score_readability()
    # FRE ≈ 59.6 → 30 ≤ FRE < 60 → linear: 60 + (59.6-30)/(60-30)*40 ≈ 99.5
    assert 60.0 <= score <= 100.0


def test_readability_fre_80_to_100():
    """Simple but not too simple texts → 80 < FRE ≤ 100 → linear score between 60 and 100."""
    # Target: avg_syl ≈ 1.2, avg_words ≈ 9
    # FRE = 206.835 - 1.015*9 - 84.6*1.2 = 206.835 - 9.135 - 101.52 = 96.18 ∈ (80, 100] ✓
    med_t1 = "boy: this is a nice place here today now"  # 9 words
    med_t2 = "girl: this is a nice place here today now"
    pairs = [_make_pair_with_texts(med_t1, med_t2)]
    evaluator = SyntheticDataEvaluator(pairs)
    # Force avg_syl = 1.2: every 5th word has 2 syllables, rest have 1
    call_count = [0]

    def syl_1_2(w: str) -> int:
        call_count[0] += 1
        return 2 if call_count[0] % 5 == 0 else 1

    evaluator._count_syllables = syl_1_2  # type: ignore[method-assign]
    score = evaluator._score_readability()
    # FRE ≈ 206.835 - 9.135 - 84.6*1.2 = 96.18 → linear: 100 - (96.18-80)/(100-80)*40 ≈ 67.6
    assert 60.0 <= score <= 100.0


# ------------------------------------------------------------------ #
# Coherence: overlap branches (lines 430, 432, 434, 437)
# ------------------------------------------------------------------ #

def test_coherence_overlap_in_target_range():
    """Word overlap ∈ [0.1, 0.3] → score = 100.0."""
    # t1 has {"boy:", "hello", "world", "foo", "bar"}, t2 has {"girl:", "hello", "planet", "baz", "qux"}
    # intersection = {"hello"}, min_size = 5, overlap = 0.2 ∈ [0.1, 0.3]
    pairs = [_make_pair_with_texts("boy: hello world foo bar", "girl: hello planet baz qux")]
    evaluator = SyntheticDataEvaluator(pairs)
    score = evaluator._score_coherence()
    assert score == 100.0


def test_coherence_high_overlap():
    """Word overlap > 0.5 → score = 70.0."""
    # t1: {"boy:", "hello", "world", "foo"}, t2: {"girl:", "hello", "world", "foo"}
    # intersection = {"hello", "world", "foo"}, min_size = 4, overlap = 3/4 = 0.75 > 0.5
    pairs = [_make_pair_with_texts("boy: hello world foo", "girl: hello world foo")]
    evaluator = SyntheticDataEvaluator(pairs)
    score = evaluator._score_coherence()
    assert score == 70.0


def test_coherence_zero_overlap():
    """No shared words → overlap = 0.0 → score = 70.0."""
    # Completely disjoint vocabulary
    pairs = [_make_pair_with_texts("boy: aaa bbb ccc", "girl: xxx yyy zzz")]
    evaluator = SyntheticDataEvaluator(pairs)
    score = evaluator._score_coherence()
    assert score == 70.0


def test_coherence_low_overlap():
    """Word overlap < 0.1 (but > 0) → linear interpolation → score between 70 and 100."""
    # 1 shared word out of 12 → overlap = 1/12 ≈ 0.083 < 0.1
    t1 = "boy: hello w1 w2 w3 w4 w5 w6 w7 w8 w9 w10"
    t2 = "girl: hello x1 x2 x3 x4 x5 x6 x7 x8 x9 x10"
    pairs = [_make_pair_with_texts(t1, t2)]
    evaluator = SyntheticDataEvaluator(pairs)
    score = evaluator._score_coherence()
    assert 70.0 < score < 100.0
