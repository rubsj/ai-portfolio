Read PRD.md thoroughly. For every requirement, acceptance criterion, and deliverable listed, verify whether it is actually implemented and working in the current codebase.

Produce a gap analysis report with 3 sections:

1. **PASS** — requirement is fully implemented and verifiable (cite the file + function/line)
2. **PARTIAL** — requirement is implemented but incomplete or deviates from PRD (explain the gap)
3. **MISSING** — requirement is not implemented at all

Check these specifically:

- Section 3 (Data Schema): All 7 fields on DIYRepairRecord, all validators from the table, GeneratedRecord metadata fields
- Section 4 (Generation): 5 templates, 30 records (5 categories × 2 × 3 difficulties), JSON cache, Instructor integration
- Section 5 (Validation): Success rate tracking, first-attempt rate, per-field error frequency, validated_records.json + rejected_records.json
- Section 6 (Failure Labeling): All 6 failure modes implemented, manual labels for 10 records, LLM labels for all 30, agreement analysis with per-mode rate + Cohen's Kappa
- Section 7 (Analysis): Heatmap, correlation matrix, all charts listed, per-category and per-difficulty breakdowns
- Section 8 (Correction): Strategy A (individual correction), Strategy B (template v2), full 4-stage pipeline (36→12→8→0), >80% improvement metric
- Section 9 (File Structure): All directories and files exist as specified
- Section 12 (ADRs): ADR-001 through ADR-004 exist with meaningful content
- Streamlit app exists and runs
- README exists with problem, architecture, results, demo link
- All tests pass: `uv run pytest tests/ -v`

Output the report as a markdown file at `results/prd_gap_analysis.md`. Do NOT fix anything — just report.