from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table
from scipy import stats

from src.models import DataQualityReport, DataQualityScore, DimensionDetail, DatingPair

EVALUATION_DIR = Path("data/evaluation")


class SyntheticDataEvaluator:
    """Scores a list of DatingPair records across 4 quality dimensions.

    WHY 4 dimensions: mirrors standard ML dataset auditing practice —
    completeness checks that data is usable, diversity ensures generalization,
    bias flags systematic skew that would corrupt training, and linguistic
    quality confirms human-like text for embedding models trained on natural language.
    """

    def __init__(self, pairs: list[DatingPair]) -> None:
        self.pairs = pairs
        self._all_texts: list[str] = []
        for p in pairs:
            self._all_texts.append(p.text_1)
            self._all_texts.append(p.text_2)

    def evaluate(self) -> DataQualityReport:
        """Run all 4 dimensions, compute overall, save outputs, return report."""
        dq_detail = self._evaluate_data_quality()
        div_detail = self._evaluate_diversity()
        bias_detail = self._evaluate_bias()
        lq_detail = self._evaluate_linguistic_quality()

        scores = DataQualityScore(
            data_quality=dq_detail.dimension_score,
            diversity=div_detail.dimension_score,
            bias=bias_detail.dimension_score,
            linguistic_quality=lq_detail.dimension_score,
            overall=round(
                (
                    dq_detail.dimension_score
                    + div_detail.dimension_score
                    + bias_detail.dimension_score
                    + lq_detail.dimension_score
                )
                / 4,
                2,
            ),
        )

        report = DataQualityReport(
            scores=scores,
            details={
                "data_quality": dq_detail,
                "diversity": div_detail,
                "bias": bias_detail,
                "linguistic_quality": lq_detail,
            },
            record_count=len(self.pairs),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._save_report(report)
        self._print_rich_table(report)
        return report

    # ------------------------------------------------------------------ #
    # Dimension 1: Data Quality
    # ------------------------------------------------------------------ #

    def _evaluate_data_quality(self) -> DimensionDetail:
        sub_scores = {
            "completeness": self._score_completeness(),
            "consistency": self._score_consistency(),
            "duplicates": self._score_duplicates(),
            "format": self._score_format(),
        }
        return DimensionDetail(
            sub_scores=sub_scores,
            dimension_score=round(sum(sub_scores.values()) / len(sub_scores), 2),
        )

    def _score_completeness(self) -> float:
        """% of records where all 6 required string fields are non-empty."""
        complete = sum(
            1
            for p in self.pairs
            if all([p.text_1, p.text_2, p.category, p.subcategory, p.pair_type])
        )
        return round(complete / len(self.pairs) * 100, 2)

    def _score_consistency(self) -> float:
        """% of records where label is a valid CompatibilityLabel (0 or 1).

        WHY re-check here: Pydantic enforces this at load time, but we verify
        the raw IntEnum value to confirm no data corruption post-validation.
        """
        valid = sum(1 for p in self.pairs if int(p.label) in (0, 1))
        return round(valid / len(self.pairs) * 100, 2)

    def _score_duplicates(self) -> float:
        """Score based on absence of duplicate (text_1, text_2) pairs.

        WHY count exact duplicates: identical pairs with different labels would
        confuse contrastive loss — the model sees two identical pairs and must
        produce opposite similarity scores, which is impossible.
        """
        seen: set[tuple[str, str]] = set()
        duplicates = 0
        for p in self.pairs:
            key = (p.text_1, p.text_2)
            if key in seen:
                duplicates += 1
            seen.add(key)
        return round((1 - duplicates / len(self.pairs)) * 100, 2)

    def _score_format(self) -> float:
        """% of texts matching the 'boy/girl: <non-empty>' pattern."""
        # WHY both text_1 and text_2: both fields must be well-formed for a usable pair
        pattern = re.compile(r"^(boy|girl):\s+\S")
        total_texts = len(self.pairs) * 2
        valid = sum(
            1
            for p in self.pairs
            for text in (p.text_1, p.text_2)
            if pattern.match(text)
        )
        return round(valid / total_texts * 100, 2)

    # ------------------------------------------------------------------ #
    # Dimension 2: Diversity
    # ------------------------------------------------------------------ #

    def _evaluate_diversity(self) -> DimensionDetail:
        sub_scores = {
            "vocabulary_richness": self._score_vocabulary_richness(),
            "category_entropy": self._score_category_entropy(),
            "label_balance": self._score_label_balance(),
            "text_complexity": self._score_text_complexity(),
        }
        return DimensionDetail(
            sub_scores=sub_scores,
            dimension_score=round(sum(sub_scores.values()) / len(sub_scores), 2),
        )

    def _score_vocabulary_richness(self) -> float:
        """Type-token ratio across all texts, scaled to 0-100.

        WHY TTR: measures lexical diversity — a dataset of 1190 texts should
        have far more unique words than any single text, catching templates that
        reuse identical phrasing across all records.
        """
        all_words = [
            w.lower()
            for text in self._all_texts
            for w in text.split()
        ]
        if not all_words:
            return 0.0
        unique_words = len(set(all_words))
        ttr = unique_words / len(all_words)
        return round(min(ttr * 100, 100), 2)

    def _score_category_entropy(self) -> float:
        """Shannon entropy of category distribution, normalized to 0-100.

        WHY normalized entropy: raw entropy depends on n_categories — normalizing
        by log2(n_categories) gives a [0,1] value regardless of category count.
        Score of 100 means perfectly uniform distribution.
        """
        counts = Counter(p.category for p in self.pairs)
        n = len(self.pairs)
        n_categories = len(counts)
        if n_categories <= 1:
            return 0.0

        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
        max_entropy = math.log2(n_categories)
        return round(entropy / max_entropy * 100, 2)

    def _score_label_balance(self) -> float:
        """Ratio of minority to majority class count × 100.

        WHY this metric: contrastive learning needs roughly equal positive/negative
        pairs — severe imbalance biases the model toward always predicting the
        majority class.
        """
        labels = [int(p.label) for p in self.pairs]
        n_compatible = sum(labels)
        n_incompatible = len(labels) - n_compatible
        if max(n_compatible, n_incompatible) == 0:
            return 0.0
        return round(min(n_compatible, n_incompatible) / max(n_compatible, n_incompatible) * 100, 2)

    def _score_text_complexity(self) -> float:
        """Average word count across all texts, mapped to a 0-100 score.

        WHY target 5-15 words: MiniLM was trained on short sentences — very short
        texts (<3 words) give sparse embeddings, very long ones (>30 words) get
        truncated and lose tail content.
        """
        avg_words = sum(len(t.split()) for t in self._all_texts) / len(self._all_texts)
        if 5 <= avg_words <= 15:
            return 100.0
        elif avg_words < 3 or avg_words > 30:
            return 50.0
        elif avg_words < 5:
            # Linear interpolation 3→50, 5→100
            return round(50 + (avg_words - 3) / (5 - 3) * 50, 2)
        else:
            # avg_words 15-30: linear interpolation 15→100, 30→50
            return round(100 - (avg_words - 15) / (30 - 15) * 50, 2)

    # ------------------------------------------------------------------ #
    # Dimension 3: Bias
    # ------------------------------------------------------------------ #

    def _evaluate_bias(self) -> DimensionDetail:
        sub_scores = {
            "gender_bias": self._score_gender_bias(),
            "category_label_correlation": self._score_category_label_correlation(),
            "length_label_correlation": self._score_length_label_correlation(),
            "vocabulary_label_bias": self._score_vocabulary_label_bias(),
        }
        return DimensionDetail(
            sub_scores=sub_scores,
            dimension_score=round(sum(sub_scores.values()) / len(sub_scores), 2),
        )

    def _score_gender_bias(self) -> float:
        """Chi-squared test for gender × label independence.

        WHY chi-squared: tests whether gender of text_1's speaker predicts
        the compatibility label — a significant association means the model
        might learn "girl-led pairs are always compatible" rather than content.
        """
        genders = [p.text_1.split(":", 1)[0].strip() for p in self.pairs]
        labels = [int(p.label) for p in self.pairs]

        gender_set = sorted(set(genders))
        label_set = sorted(set(labels))

        # Build contingency table
        table = [
            [sum(1 for g, lbl in zip(genders, labels) if g == gender and lbl == label)
             for label in label_set]
            for gender in gender_set
        ]

        # WHY check for empty cells: chi2_contingency requires non-zero expected frequencies
        if any(sum(row) == 0 for row in table):
            return 100.0  # Only one gender present — no bias detectable

        _, p_value, _, _ = stats.chi2_contingency(table)
        if p_value > 0.05:
            return 100.0
        return round(p_value / 0.05 * 100, 2)

    def _score_category_label_correlation(self) -> float:
        """Chi-squared test for category × label independence."""
        categories = [p.category for p in self.pairs]
        labels = [int(p.label) for p in self.pairs]

        cat_set = sorted(set(categories))
        label_set = sorted(set(labels))

        table = [
            [sum(1 for c, lbl in zip(categories, labels) if c == cat and lbl == label)
             for label in label_set]
            for cat in cat_set
        ]

        if any(sum(row) == 0 for row in table):
            return 100.0

        _, p_value, _, _ = stats.chi2_contingency(table)
        if p_value > 0.05:
            return 100.0
        return round(p_value / 0.05 * 100, 2)

    def _score_length_label_correlation(self) -> float:
        """Point-biserial correlation between text length and label.

        WHY |r| thresholds: |r| < 0.1 is negligible correlation (score 100),
        |r| > 0.3 is moderate — concerning enough to score 0 since the model
        could learn length as a proxy for compatibility.
        """
        lengths = [
            (len(p.text_1.split()) + len(p.text_2.split())) / 2
            for p in self.pairs
        ]
        labels = [int(p.label) for p in self.pairs]

        r, _ = stats.pointbiserialr(labels, lengths)
        abs_r = abs(r)

        if abs_r < 0.1:
            return 100.0
        elif abs_r > 0.3:
            return 0.0
        # Linear interpolation: 0.1→100, 0.3→0
        return round((0.3 - abs_r) / (0.3 - 0.1) * 100, 2)

    def _score_vocabulary_label_bias(self) -> float:
        """Jaccard similarity between word sets of compatible and incompatible pairs.

        WHY Jaccard: if compatible/incompatible pairs share almost no vocabulary,
        the model can game the benchmark by memorizing topic words rather than
        learning compatibility semantics. High overlap (score ~100) is good.
        """
        labels = [int(p.label) for p in self.pairs]
        compat_words: set[str] = set()
        incompat_words: set[str] = set()

        for pair, label in zip(self.pairs, labels):
            words = set(pair.text_1.lower().split() + pair.text_2.lower().split())
            if label == 1:
                compat_words |= words
            else:
                incompat_words |= words

        union = compat_words | incompat_words
        if not union:
            return 0.0
        intersection = compat_words & incompat_words
        jaccard = len(intersection) / len(union)
        return round(jaccard * 100, 2)

    # ------------------------------------------------------------------ #
    # Dimension 4: Linguistic Quality
    # ------------------------------------------------------------------ #

    def _evaluate_linguistic_quality(self) -> DimensionDetail:
        sub_scores = {
            "readability": self._score_readability(),
            "coherence": self._score_coherence(),
            "naturalness": self._score_naturalness(),
            "repetition": self._score_repetition(),
        }
        return DimensionDetail(
            sub_scores=sub_scores,
            dimension_score=round(sum(sub_scores.values()) / len(sub_scores), 2),
        )

    def _count_syllables(self, word: str) -> int:
        """Count syllables by vowel group heuristic.

        WHY vowel groups: English syllables roughly correspond to vowel nuclei.
        This is approximate but fast — we don't need perfect readability scores,
        just a directional signal of text complexity.
        """
        word = word.lower().strip(".,!?;:'\"")
        if not word:
            return 1
        vowels = "aeiou"
        count = 0
        prev_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        return max(count, 1)  # Every word has at least one syllable

    def _score_readability(self) -> float:
        """Flesch Reading Ease approximation, mapped to 0-100.

        FRE = 206.835 - 1.015*(total_words/total_sentences) - 84.6*(total_syllables/total_words)
        WHY FRE 60-80 as target: that range corresponds to "plain English" — ideal
        for a dating profile style dataset. Below 30 is academic jargon, above 100
        is monosyllabic fragments.
        """
        total_words = 0
        total_syllables = 0
        total_sentences = len(self._all_texts)  # WHY 1 sentence per text: no sentence boundary detection

        for text in self._all_texts:
            words = text.split()
            total_words += len(words)
            total_syllables += sum(self._count_syllables(w) for w in words)

        if total_words == 0:
            return 0.0

        avg_sentence_length = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words
        fre = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

        if 60 <= fre <= 80:
            return 100.0
        elif fre < 30 or fre > 100:
            return 60.0
        elif fre < 60:
            # Linear: 30→60, 60→100
            return round(60 + (fre - 30) / (60 - 30) * 40, 2)
        else:
            # fre 80-100: linear: 80→100, 100→60
            return round(100 - (fre - 80) / (100 - 80) * 40, 2)

    def _score_coherence(self) -> float:
        """Word overlap between text_1 and text_2 within each pair, averaged.

        WHY 0.1-0.3 is ideal: some overlap shows the texts are topically related
        (e.g., both mention "hiking"), which is realistic for dating profiles.
        Zero overlap suggests random pairing; 0.5+ suggests near-identical texts.
        """
        overlaps = []
        for p in self.pairs:
            w1 = set(p.text_1.lower().split())
            w2 = set(p.text_2.lower().split())
            min_size = min(len(w1), len(w2))
            if min_size == 0:
                continue
            overlap = len(w1 & w2) / min_size
            overlaps.append(overlap)

        if not overlaps:
            return 70.0

        avg_overlap = sum(overlaps) / len(overlaps)

        if 0.1 <= avg_overlap <= 0.3:
            return 100.0
        elif avg_overlap > 0.5:
            return 70.0
        elif avg_overlap == 0:
            return 70.0
        elif avg_overlap < 0.1:
            # Linear: 0→70, 0.1→100
            return round(70 + avg_overlap / 0.1 * 30, 2)
        else:
            # avg_overlap 0.3-0.5: linear: 0.3→100, 0.5→70
            return round(100 - (avg_overlap - 0.3) / (0.5 - 0.3) * 30, 2)

    def _score_naturalness(self) -> float:
        """Bigram type-token ratio across all texts, scaled to 0-100.

        WHY bigrams not unigrams: adjacent word pairs reveal repetitive phrasing
        better than single words — template datasets often reuse phrases like
        "I love" or "I enjoy" everywhere even when word TTR looks acceptable.
        """
        all_bigrams: list[tuple[str, str]] = []
        for text in self._all_texts:
            words = text.lower().split()
            all_bigrams.extend(zip(words[:-1], words[1:]))

        if not all_bigrams:
            return 0.0

        unique_bigrams = len(set(all_bigrams))
        ttr = unique_bigrams / len(all_bigrams)
        return round(min(ttr * 100, 100), 2)

    def _score_repetition(self) -> float:
        """Score based on how dominant the most common bigram is.

        WHY most-frequent-bigram: if one bigram like ("I", "love") appears in
        30% of all texts, the dataset is templated — a red flag for diversity.
        Score = (1 - dominance_ratio) * 100.
        """
        all_bigrams: list[tuple[str, str]] = []
        for text in self._all_texts:
            words = text.lower().split()
            all_bigrams.extend(zip(words[:-1], words[1:]))

        if not all_bigrams:
            return 100.0

        most_common_count = Counter(all_bigrams).most_common(1)[0][1]
        dominance = most_common_count / len(all_bigrams)
        return round((1 - dominance) * 100, 2)

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #

    def _save_report(self, report: DataQualityReport) -> None:
        """Save JSON report and human-readable summary text."""
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

        json_path = EVALUATION_DIR / "data_quality_report.json"
        json_path.write_text(report.model_dump_json(indent=2))

        summary_path = EVALUATION_DIR / "data_quality_summary.txt"
        lines = [
            "Data Quality Evaluation Summary",
            "=" * 40,
            f"Records evaluated: {report.record_count}",
            f"Timestamp: {report.timestamp}",
            "",
            "Dimension Scores:",
            f"  Data Quality:      {report.scores.data_quality:.1f}",
            f"  Diversity:         {report.scores.diversity:.1f}",
            f"  Bias:              {report.scores.bias:.1f}",
            f"  Linguistic Quality:{report.scores.linguistic_quality:.1f}",
            f"  Overall:           {report.scores.overall:.1f}",
            "",
            "Sub-scores:",
        ]
        for dim_name, detail in report.details.items():
            lines.append(f"  [{dim_name}]")
            for sub_name, sub_score in detail.sub_scores.items():
                status = "PASS" if sub_score >= 60 else "FAIL"
                lines.append(f"    {sub_name:<30} {sub_score:6.1f}  [{status}]")
        summary_path.write_text("\n".join(lines))

    def _print_rich_table(self, report: DataQualityReport) -> None:
        """Print a Rich table with all scores and pass/fail indicators."""
        console = Console()
        table = Table(title="Data Quality Report", show_header=True, header_style="bold cyan")
        table.add_column("Dimension", style="bold")
        table.add_column("Sub-score", style="dim")
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="center")

        for dim_name, detail in report.details.items():
            for sub_name, sub_score in detail.sub_scores.items():
                status = "[green]PASS[/green]" if sub_score >= 60 else "[red]FAIL[/red]"
                table.add_row(dim_name, sub_name, f"{sub_score:.1f}", status)
            table.add_row(
                f"[bold]{dim_name} total[/bold]",
                "",
                f"[bold]{detail.dimension_score:.1f}[/bold]",
                "[green]PASS[/green]" if detail.dimension_score >= 60 else "[red]FAIL[/red]",
            )
            table.add_section()

        console.print(table)
        console.print(f"\n[bold]Overall score: {report.scores.overall:.1f}[/bold]")
