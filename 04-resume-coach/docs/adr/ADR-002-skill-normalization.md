# ADR-002: Custom Skill Normalizer over Third-Party Library

**Date**: 2026-02-26
**Status**: Accepted
**Project**: P4 — Resume Coach

## Context

Jaccard similarity between resume skills and job requirements is the core signal for fit scoring. Without normalization, surface variants of the same skill produce false negatives: `"Python 3.10"` vs `"Python"`, `"ML"` vs `"Machine Learning"`, `"Javascript"` vs `"JavaScript"`.

If we skip normalization:
- A resume with `"Python 3.11"` and a job requiring `"Python"` score Jaccard = 0.0 despite identical skills
- The Jaccard gradient across fit levels (the central P4 hypothesis) would be confounded by noise, not signal
- The `missing_core_skills` failure label would fire falsely on every version-suffixed skill

We evaluated two normalization approaches: off-the-shelf libraries vs a custom pipeline.

## Decision

Build a **custom `SkillNormalizer`** class with a four-stage normalization pipeline:

1. **Lowercase** — `"Python"` → `"python"`, `"JavaScript"` → `"javascript"`
2. **Version stripping** — `"python 3.10"` → `"python"`, `"react 18"` → `"react"` (regex: `r'\s+\d[\d.]*$'`)
3. **Suffix removal** — `"machine learning algorithms"` → `"machine learning"` (strip common noise words: `"algorithms"`, `"framework"`, `"library"`, `"tools"`, `"techniques"`)
4. **Alias resolution** — canonical mapping: `{"ml": "machine learning", "ai": "artificial intelligence", "js": "javascript", "k8s": "kubernetes", ...}`

The normalizer is instantiated once per API process and passed into `label_pair()` — avoids reconstructing the alias table on every request.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| `skillNer` (spaCy-based NER) | Industry-standard entity recognition | Requires spaCy model download (~50MB); slow inference per skill; overkill for this use case |
| `fuzzy` / `rapidfuzz` | Handles typos | Fuzzy matching produces false positives (`"Java"` ≈ `"JavaScript"`); threshold tuning is brittle |
| OpenAI embeddings cosine similarity | Semantic, catches all variants | $0.02 per 1K calls; 250 resumes × avg 8 skills = 2000 calls; adds latency; overkill |
| **Custom normalizer** (chosen) | Deterministic; zero latency; zero cost; fully testable; specific to resume domain | Manual alias list maintenance; won't catch rare abbreviations not in the list |

## Consequences

**Easier**:
- `test_normalizer.py` can enumerate exact input→output pairs — fully deterministic
- Jaccard scores are meaningful signal (confirmed by gradient: excellent=0.669 → mismatch=0.005)
- Adding new aliases is a one-line dict update, not a model retrain
- No external dependencies — works offline

**Harder**:
- Domain-specific abbreviations not in the alias table will still mismatch (mitigated by covering top-50 tech skills)
- Case-sensitivity edge cases in proper nouns need manual audit (e.g., `"AWS"` → `"aws"` is fine since comparison is symmetric)

## Java/TS Parallel

The `SkillNormalizer` is a **stateless Spring Service** (`@Service`) with a single public method, initialized once in the application context and injected into controllers. The four-stage pipeline mirrors a **chain-of-responsibility pattern**: each stage is a pure transformation function, composable and independently testable. In TypeScript, this maps to a `normalize(skill: string): string` utility with the same pipeline — no class overhead needed, but the instantiation-once constraint still applies.
