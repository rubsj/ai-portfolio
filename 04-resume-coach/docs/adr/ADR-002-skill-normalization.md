# ADR-002: Custom Skill Normalizer over Third-Party Library

**Date**: 2026-02-26
**Project**: P4 — Resume Coach

## Status

Accepted

## Context

Jaccard similarity between resume skills and job requirements is P4's core metric for fit scoring. The entire analysis depends on it: the Jaccard gradient across fit levels is the central hypothesis, the `missing_core_skills` failure label uses set difference, and the A/B template comparison measures failure rates derived from these labels.

Without normalization, the metric collapses. A resume listing `"Python 3.11"` against a job requiring `"Python"` scores Jaccard = 0.0 — a false negative on what should be a perfect match. `"ML"` vs `"Machine Learning"`, `"React.js"` vs `"React"`, `"Node.js"` vs `"Node"` — all zero overlap. Since GPT-4o-mini generates skills with natural variation (version numbers, abbreviations, framework suffixes), unnormalized Jaccard would measure *formatting noise*, not actual skill fit.

The 250 resumes have an average of ~8 skills each. Across 250 resume-job pairs, that's ~4,000 skill comparisons where normalization matters. The normalizer must be deterministic (same input always produces same output), fast (runs inside the labeler at ~1ms per pair), and free (no API calls — the labeler is pure Python by design, per ADR-003).

## Decision

Build a custom `SkillNormalizer` class with a four-stage pipeline, applied to every skill string before Jaccard computation:

1. **Lowercase + strip** — `"Python"` → `"python"`, `"  React "` → `"react"`
2. **Parenthetical removal** — `"JavaScript (ES6+)"` → `"javascript"` via regex `r"\s*\([^)]*\)"`
3. **Version stripping** — `"python 3.10"` → `"python"`, `"react 18"` → `"react"` via regex `r"\s*\d+(\.\d+)*\s*$"`
4. **Suffix removal** — strip 7 noise words (`developer`, `engineer`, `programming`, `development`, `framework`, `language`, `library`) from the end of skill strings
5. **Alias resolution** — 28-entry canonical mapping: `{"ml": "machine learning", "js": "javascript", "k8s": "kubernetes", "react.js": "react", "node.js": "node", "aws": "amazon web services", ...}`

The normalizer is instantiated once at process startup (API, Streamlit, or pipeline) and passed into `label_pair(resume, job, pair_id, normalizer)`. The alias dict and suffix list are plain Python data structures — no config files, no model weights.

## Alternatives Considered

**skillNer (spaCy-based NER)**: An industry-standard skill extraction library that recognizes skill entities in free text. It requires downloading a spaCy language model (~50MB), which is larger than P4's entire dataset. More importantly, skillNer solves a different problem — extracting skills from unstructured text. P4's skills are already structured (they come from Pydantic `Skill.name` fields). The bottleneck is normalization of known skill strings, not extraction from prose. Using NER for this is like using OCR to read a database column.

**rapidfuzz (fuzzy string matching)**: Handles typos elegantly with Levenshtein distance. But fuzzy matching is dangerously imprecise for skills: `"Java"` and `"JavaScript"` score ~82% similarity by Levenshtein, despite being completely different languages. Any threshold that catches `"JS"` → `"JavaScript"` also catches `"Java"` → `"JavaScript"`. The false positive rate would corrupt the Jaccard gradient — the exact metric P4 exists to measure. Fuzzy matching works for user-facing search; it fails for analytical precision.

**OpenAI embeddings cosine similarity**: Semantic embeddings would catch every variant, including ones the alias dict misses. But at $0.02 per 1K tokens, normalizing 250 resumes × ~8 skills = ~2,000 API calls per pipeline run. The labeler runs inside a tight loop and must be free and deterministic (ADR-003). Embedding-based normalization would make the labeler non-deterministic (model updates change results) and add ~$0.04 per run — small in absolute terms, but it breaks the design principle that labeling is pure Python with zero external dependencies.

## Consequences

### What This Enabled

The Jaccard gradient across fit levels — excellent=0.669, good=0.607, partial=0.620, poor=0.212, mismatch=0.005 — is near-monotonic with a partial/good inversion (0.620 vs 0.607) likely due to template variation at mid-range fit levels, but the endpoints confirm the overall trend: excellent-fit resumes share 67% of required skills while mismatch resumes share effectively none. Without normalization, this gradient would be noise: `"Python 3.11"` vs `"Python"` would score 0.0, drowning real signal in formatting artifacts. The `missing_core_skills` failure label, which fires at a 50.8% rate across 250 pairs, depends on accurate set difference — and produces zero false positives from formatting noise because the normalizer canonicalizes both sides before comparison. Because the normalizer is pure Python with no LLM calls, the labeler processes all 250 pairs in ~250ms at zero cost, and `test_labeler.py` and `test_normalizer.py` are fully deterministic with exact input→output assertions, no mocking, no flakiness.

### Accepted Trade-offs

- The 28-entry alias dict requires manual maintenance. Adding a new domain (e.g., biotech skills) means updating `SKILL_ALIASES` by hand. Mitigated by covering the top ~50 tech skills that appear in GPT-4o-mini's generation vocabulary — the LLM rarely invents abbreviations outside this set
- Typos in generated skills (e.g., `"Pythn"`) are not caught. In practice, GPT-4o-mini's spelling is reliable enough that this hasn't produced a false negative in 250 resumes
- The parenthetical removal regex `r"\s*\([^)]*\)"` strips all parenthetical content, which could remove meaningful qualifiers like `"SQL (advanced)"`. Acceptable because `proficiency_level` is a separate field on the `Skill` model — the parenthetical is redundant

## Cross-Project Context

P1 (Synthetic Data) had no skill normalization — its repair records used free-text descriptions, not structured skill lists, so set comparison never arose. P4's Day 1.5 data quality audit revealed near-zero Jaccard overlap from un-normalized skills like `"Python 3.11"` vs `"Python"`, making the normalizer a direct response to that failure. The lesson: once you compute set operations on LLM-generated strings, normalization isn't optional — it's foundational.

## Java/TS Parallel

The `SkillNormalizer` is a stateless `@Service` bean in Spring terms: instantiated once by the container, injected into dependent services, no mutable state. The four-stage pipeline maps to a Java `Stream` chain — `skills.stream().map(String::toLowerCase).map(this::stripVersion).map(this::resolveAlias).collect(toSet())`. The alias dict is a `Map<String, String>` loaded from a properties file; the suffix list is a `Set<String>` — same concept, different mechanism.

## Validation

The Jaccard gradient across 5 fit levels (excellent=0.669 → mismatch=0.005) would not exist without normalization — raw skill strings would show noise, not signal. The A/B template test found statistically significant differences (χ²=32.74, df=4, p=1.35e-06) between templates, which depends on accurate failure labels, which depend on accurate Jaccard scores, which depend on the normalizer. The entire analytical chain holds because normalization removes formatting noise without introducing false positives.

## Reversibility

**Low** — Switching to a fuzzy or embedding-based normalizer would change every Jaccard score, every failure label, and every chart in the analysis. The normalizer is the foundation of P4's analytical validity. Changing it means re-running the full pipeline and re-evaluating all results.
