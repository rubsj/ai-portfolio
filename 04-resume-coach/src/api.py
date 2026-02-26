from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from src.data_paths import FEEDBACK_DIR, DataStore
from src.labeler import calculate_total_experience, label_pair
from src.normalizer import SkillNormalizer
from src.schemas import (
    FailureRateResponse,
    FeedbackRequest,
    FeedbackResponse,
    JobListResponse,
    JobSummary,
    MultiHopRequest,
    MultiHopResponse,
    PairDetailResponse,
    ReviewRequest,
    ReviewResponse,
    SimilarCandidate,
    SimilarCandidatesResponse,
    TemplateComparisonResponse,
    TemplateStats,
)
from src.vector_store import get_collection, search_similar

app = FastAPI(
    title="Resume Coach API",
    description="AI-powered resume coaching: failure labeling, LLM judging, vector search.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once at startup (Spring @Bean equivalent)
# ---------------------------------------------------------------------------

# WHY module-level: FastAPI runs in a single process; constructing DataStore
# once avoids re-reading ~1MB of JSONL on every request.
_store = DataStore()
_normalizer = SkillNormalizer()

# WHY try/except: vector index may not exist if pipeline hasn't been run yet.
# API is still useful without it — search endpoint returns 503 gracefully.
try:
    _collection = get_collection()
except Exception:
    _collection = None


# ---------------------------------------------------------------------------
# Endpoint 1: Health check
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "jobs": _store.job_count,
        "resumes": _store.resume_count,
        "pairs": _store.pair_count,
        "vector_store_ready": _collection is not None,
    }


# ---------------------------------------------------------------------------
# Endpoint 2: Review resume
# ---------------------------------------------------------------------------


@app.post("/review-resume", response_model=ReviewResponse)
def review_resume(
    request: ReviewRequest,
    use_judge: Annotated[bool, Query(description="Also run LLM judge (slower, costs $)")] = False,
) -> ReviewResponse:
    start = time.perf_counter()
    pair_id = str(uuid.uuid4())

    labels = label_pair(request.resume, request.job_description, pair_id, _normalizer)

    judge_result = None
    if use_judge:
        from src.judge import _create_judge_client, judge_pair

        client = _create_judge_client()
        judge_result = judge_pair(client, request.job_description, request.resume, pair_id)

    return ReviewResponse(
        pair_id=pair_id,
        failure_labels=labels,
        judge_result=judge_result,
        processing_time_seconds=round(time.perf_counter() - start, 3),
    )


# ---------------------------------------------------------------------------
# Endpoint 3: Failure rates
# ---------------------------------------------------------------------------


@app.get("/analysis/failure-rates", response_model=FailureRateResponse)
def failure_rates() -> FailureRateResponse:
    summary = _store.pipeline_summary
    if not summary:
        raise HTTPException(status_code=503, detail="No pipeline run found. Run src/pipeline.py first.")

    labeling = summary.get("labeling", {})
    correction = summary.get("correction", {})
    generation = summary.get("generation", {})

    return FailureRateResponse(
        total_pairs=generation.get("pairs_generated", 0),
        validation_success_rate=generation.get("validation_rate", 0.0),
        failure_mode_rates=labeling.get("failure_mode_rates", {}),
        correction_success_rate=correction.get("correction_rate", 0.0),
        avg_jaccard_by_fit_level=labeling.get("avg_jaccard_by_fit_level", {}),
        last_run_timestamp=summary.get("day", "unknown"),
    )


# ---------------------------------------------------------------------------
# Endpoint 4: Template comparison
# ---------------------------------------------------------------------------


@app.get("/analysis/template-comparison", response_model=TemplateComparisonResponse)
def template_comparison() -> TemplateComparisonResponse:
    summary = _store.pipeline_summary
    ab = summary.get("ab_testing", {}) if summary else {}
    if not ab:
        raise HTTPException(status_code=503, detail="No A/B test data. Run src/pipeline.py first.")

    failure_rates_by_template: dict[str, float] = ab.get("failure_rates_by_template", {})

    # WHY 50 per template: 250 resumes / 5 templates = 50 each (balanced design).
    template_results: dict[str, TemplateStats] = {}
    for template_id, fail_rate in failure_rates_by_template.items():
        template_results[template_id] = TemplateStats(
            template_id=template_id,
            total_generated=50,
            validation_success_rate=1.0 - fail_rate,
            failure_mode_rates={"any_failure": fail_rate},
            avg_jaccard=0.0,  # Not broken out per template in pipeline_summary
            avg_judge_quality_score=None,
        )

    best = ab.get("best_template", "")
    worst = ab.get("worst_template", "")
    return TemplateComparisonResponse(
        template_results=template_results,
        chi_squared_statistic=ab.get("chi_squared_statistic", 0.0),
        chi_squared_p_value=ab.get("chi_squared_p_value", 1.0),
        significant=ab.get("significant", False),
        best_template=best,
        worst_template=worst,
        recommendation=f"Use '{best}' template; avoid '{worst}' (χ²={ab.get('chi_squared_statistic', 0):.2f}, p={ab.get('chi_squared_p_value', 1):.2e})",
    )


# ---------------------------------------------------------------------------
# Endpoint 5: Multi-hop evaluation
# ---------------------------------------------------------------------------


@app.post("/evaluate/multi-hop", response_model=MultiHopResponse)
def evaluate_multi_hop(request: MultiHopRequest) -> MultiHopResponse:
    from src.multi_hop import generate_multi_hop_questions

    start = time.perf_counter()
    pair_id = str(uuid.uuid4())

    labels = label_pair(request.resume, request.job_description, pair_id, _normalizer)
    result = generate_multi_hop_questions(request.resume, request.job_description, labels, pair_id)

    # Attach processing time (multi_hop returns MultiHopResponse with pair_id + questions)
    return MultiHopResponse(
        pair_id=result.pair_id,
        questions=result.questions,
        processing_time_seconds=round(time.perf_counter() - start, 3),
    )


# ---------------------------------------------------------------------------
# Endpoint 6: Similar candidate search
# ---------------------------------------------------------------------------


@app.get("/search/similar-candidates", response_model=SimilarCandidatesResponse)
def similar_candidates(
    query: Annotated[str, Query(description="Free-text job description or skill requirements")],
    top_k: Annotated[int, Query(ge=1, le=50)] = 5,
    fit_level: Annotated[
        str | None,
        Query(description="Filter by fit level: excellent, good, partial, poor, mismatch"),
    ] = None,
) -> SimilarCandidatesResponse:
    if _collection is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not ready. Run build_resume_index() first.",
        )

    start = time.perf_counter()
    hits = search_similar(_collection, query, top_k=top_k, fit_level=fit_level)

    candidates: list[SimilarCandidate] = []
    for hit in hits:
        trace_id = hit["trace_id"]
        gr = _store.resumes.get(trace_id)
        if gr is None:
            continue
        exp_years = calculate_total_experience(gr.resume.experience)
        candidates.append(
            SimilarCandidate(
                resume_trace_id=trace_id,
                similarity_score=hit["score"],
                name=gr.resume.contact_info.name,
                skills=[s.name for s in gr.resume.skills],
                experience_years=round(exp_years, 1),
                fit_level=hit["metadata"]["fit_level"],
            )
        )

    total_in_index = _collection.count()
    return SimilarCandidatesResponse(
        query=query,
        results=candidates,
        total_in_index=total_in_index,
        filter_applied=fit_level,
        processing_time_seconds=round(time.perf_counter() - start, 3),
    )


# ---------------------------------------------------------------------------
# Endpoint 7: Feedback
# ---------------------------------------------------------------------------


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Persist to disk
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_path = FEEDBACK_DIR / "feedback.jsonl"
    entry = {
        "feedback_id": feedback_id,
        "pair_id": request.pair_id,
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": timestamp,
    }
    with feedback_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    # Update in-memory store so subsequent /pairs/{id} calls see it
    _store.feedback.setdefault(request.pair_id, []).append(request)

    return FeedbackResponse(
        feedback_id=feedback_id,
        logged_to=["disk", "memory"],
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Endpoint 8: Jobs list (paginated + filtered)
# ---------------------------------------------------------------------------


@app.get("/jobs", response_model=JobListResponse)
def list_jobs(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=50)] = 20,
    industry: Annotated[str | None, Query(description="Filter by industry (case-insensitive)")] = None,
    is_niche: Annotated[bool | None, Query(description="Filter niche roles")] = None,
) -> JobListResponse:
    jobs = list(_store.jobs.values())

    # Apply filters
    if industry is not None:
        jobs = [gj for gj in jobs if gj.job.company.industry.lower() == industry.lower()]
    if is_niche is not None:
        jobs = [gj for gj in jobs if gj.is_niche_role == is_niche]

    total = len(jobs)
    total_pages = max(1, math.ceil(total / page_size))
    offset = (page - 1) * page_size
    page_jobs = jobs[offset : offset + page_size]

    summaries = [
        JobSummary(
            trace_id=gj.trace_id,
            title=gj.job.title,
            company_name=gj.job.company.name,
            industry=gj.job.company.industry,
            experience_level=gj.job.requirements.experience_level.value,
            is_niche=gj.is_niche_role,
            required_skills_count=len(gj.job.requirements.required_skills),
        )
        for gj in page_jobs
    ]

    return JobListResponse(
        jobs=summaries,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


# ---------------------------------------------------------------------------
# Endpoint 9: Pair detail
# ---------------------------------------------------------------------------


@app.get("/pairs/{pair_id}", response_model=PairDetailResponse)
def pair_detail(pair_id: str) -> PairDetailResponse:
    # Find the pair
    pair = next((p for p in _store.pairs if p.pair_id == pair_id), None)
    if pair is None:
        raise HTTPException(status_code=404, detail=f"Pair '{pair_id}' not found.")

    resume = _store.resumes.get(pair.resume_trace_id)
    job = _store.jobs.get(pair.job_trace_id)
    if resume is None or job is None:
        raise HTTPException(status_code=404, detail="Pair found but resume/job missing from store.")

    return PairDetailResponse(
        pair_id=pair_id,
        resume=resume,
        job=job,
        failure_labels=_store.failure_labels.get(pair_id),
        judge_result=_store.judge_results.get(pair_id),
        correction_history=_store.correction_results.get(pair_id, []),
        feedback=_store.feedback.get(pair_id, []),
    )
