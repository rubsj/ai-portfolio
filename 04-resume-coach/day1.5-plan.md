# Plan: Fix 3 Data Quality Issues + 1 Normalizer Gap

## Context

Day 1 generation produced 50 jobs + 250 resumes + 250 pairs, but an audit of actual data revealed that **skills are sentences instead of tokens** (40/50 jobs, 42% of resumes), company sizes are inconsistent (18 values), and the normalizer doesn't strip parentheticals. These issues cause 0% Jaccard on affected pairs and break the entire labeling/analysis pipeline. We fix schemas + prompts + normalizer, clear cache, and regenerate all data in one pass.

---

## Changes

### 1. `src/schemas.py` — Add `Field(description=...)` to 4 fields

**JobRequirements** (line ~206):
- `required_skills`: Add `Field(description="List of individual skill/technology names as short tokens. Each skill must be 1-3 words max — a specific tool, language, or technology name. GOOD: ['Python', 'AWS', 'Docker', 'PostgreSQL', 'React', 'Git'] BAD: ['Experience with Python programming', 'Knowledge of cloud platforms (e.g., AWS)']")`
- `preferred_skills`: Add `Field(description="List of individual skill/technology names as short tokens. Each skill must be 1-3 words max — a specific tool, language, or technology name. GOOD: ['TypeScript', 'Docker', 'CI/CD', 'GraphQL'] BAD: ['Experience with containerization tools', 'Knowledge of CI/CD pipelines (e.g., Jenkins)']")`

**Skill** (line ~151):
- `name`: Add `Field(description="Individual skill or technology name as a short token (e.g., 'Python', 'React', 'PostgreSQL'). Must be 1-3 words. NOT a description. NOT 'Proficient in Python'. NOT 'Cloud Computing (AWS, Azure)' — instead list 'AWS' and 'Azure' as separate skills.")`

**CompanyInfo** (line ~199):
- `size`: Add `Field(description="Company size. Use EXACTLY one of these values: 'Startup (1-50)', 'Small (51-200)', 'Mid-size (201-500)', 'Enterprise (500+)'")`

Import `Field` from pydantic if not already imported.

### 2. `src/templates.py` — Add format rules to system prompts

**Job generation system prompt** (`get_job_prompt`, line ~68):
Append after the existing "Requirements:" block:
```
CRITICAL FORMAT RULE: required_skills and preferred_skills must be individual
technology/tool tokens (1-3 words each). NOT descriptions, NOT sentences,
NOT 'experience with X', NOT 'knowledge of Y'.
GOOD: ['Python', 'React', 'PostgreSQL', 'Docker', 'AWS', 'Git', 'REST APIs']
BAD: ['Experience with Python programming language', 'Knowledge of cloud platforms (e.g., AWS, GCP)']
Each item = one specific skill name only.
```

**Resume generation system prompt** (`get_resume_prompt`, line ~103):
Append after the existing "IMPORTANT:" block:
```
CRITICAL: Each skill name must be a single technology/tool token (1-3 words).
GOOD: 'Python', 'AWS', 'Docker', 'React', 'PostgreSQL'
BAD: 'Cloud Computing (AWS, Azure)', 'RESTful API design and development'
If a skill has sub-items, list each as its own separate skill entry.
```

### 3. `src/normalizer.py` — Add parenthetical stripping step

In `SkillNormalizer.normalize()`, add **before** the version removal step (new Step 0):
```python
# Step 0: Remove parentheticals — "JavaScript (ES6+)" -> "JavaScript"
# Handles nested/multiple parens by matching non-closing-paren chars
s = re.sub(r'\s*\([^)]*\)', '', s).strip()
```

### 4. `tests/test_normalizer.py` — Add parenthetical test cases

Add a new parametrized test method:
```python
@pytest.mark.parametrize("input_skill,expected", [
    ("JavaScript (ES6+)", "javascript"),
    ("Cloud Computing (AWS, Azure)", "cloud computing"),
    ("Tools (e.g., AWS) and more (GCP)", "tools and more"),
    ("React", "react"),  # no change for clean inputs
])
def test_parenthetical_stripping(self, normalizer, input_skill, expected):
    assert normalizer.normalize(input_skill) == expected
```

---

## Execution Steps (post-implementation)

1. Run tests: `uv run pytest tests/ -x` — ensure no regressions
2. Clear data: `rm -rf data/cache/* data/generated/* data/validated/*`
3. Dry run: `uv run python -m src.run_generation --dry-run`
4. Spot-check dry run output:
   - Job `required_skills`: all 1-3 word tokens? No sentences? No parentheticals?
   - Resume `Skill.name`: all 1-3 word tokens?
   - `CompanyInfo.size`: consistent format from the 4 allowed values?
5. Quick Jaccard pre-check: After dry-run spot-check passes, manually compute Jaccard on 1-2 dry-run pairs to confirm non-zero overlap BEFORE committing to the full generation run. Fail fast — if Jaccard is still 0%, the prompt fix didn't work.
6. If good → full run: `uv run python -m src.run_generation`
7. Sanity check: `uv run python -m src.sanity_check`
   - Excellent fit Jaccard should be >40%
   - Mismatch should be <15%
   - Clear gradient across fit levels

## Verification

- `pytest -x` passes (no regressions from schema/normalizer changes)
- Dry-run output shows short skill tokens, no sentences
- Sanity check shows Jaccard gradient: Excellent > Good > Partial > Poor > Mismatch

## Commit

```
fix(p4): enforce skill tokens, normalize parentheticals, standardize company sizes
```
