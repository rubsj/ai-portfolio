"""P4 Resume Coach — Interactive Streamlit Demo.

Standalone mode: reads pipeline artifacts from disk via data_paths.py.
No API server required.

Run: uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from src.data_paths import (
    FEEDBACK_DIR,
    RESULTS_DIR,
    DataStore,
)
from src.labeler import calculate_total_experience

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Resume Coach",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# ChromaDB — optional (search page degrades gracefully if missing)
# ---------------------------------------------------------------------------

try:
    from src.vector_store import get_collection, search_similar

    _collection = get_collection()
    _VECTOR_STORE_READY = True
except Exception:
    _collection = None
    _VECTOR_STORE_READY = False

# ---------------------------------------------------------------------------
# Cached data loading — @st.cache_data memoizes on first call for the
# process lifetime. WHY: DataStore reads ~1MB of JSONL; loading once avoids
# repeated disk I/O on every page navigation / widget interaction.
# ---------------------------------------------------------------------------


@st.cache_data
def load_store() -> DataStore:
    return DataStore()


@st.cache_data
def load_pipeline_summary() -> dict:
    path = RESULTS_DIR / "pipeline_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def list_chart_paths() -> list[Path]:
    charts_dir = RESULTS_DIR / "charts"
    if not charts_dir.exists():
        return []
    return sorted(charts_dir.glob("*.png"))


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = [
    "Browse Jobs",
    "Review Resume",
    "Analysis Dashboard",
    "Search Similar",
    "Feedback",
]

st.sidebar.title("📄 Resume Coach")
st.sidebar.caption("P4 — AI Portfolio Demo")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Data loaded from disk — no API server required.")


# ---------------------------------------------------------------------------
# Page 1: Browse Jobs
# ---------------------------------------------------------------------------


def page_browse_jobs() -> None:
    store = load_store()

    st.header("Browse Jobs")
    st.caption(f"{store.job_count} jobs loaded from pipeline artifacts.")

    # Filters
    col_a, col_b = st.columns([2, 1])
    with col_a:
        industries = sorted({gj.job.company.industry for gj in store.jobs.values()})
        industry_filter = st.selectbox("Filter by industry", ["All"] + industries)
    with col_b:
        niche_filter = st.selectbox("Niche roles", ["All", "Niche only", "Standard only"])

    # Apply filters
    jobs = list(store.jobs.values())
    if industry_filter != "All":
        jobs = [gj for gj in jobs if gj.job.company.industry == industry_filter]
    if niche_filter == "Niche only":
        jobs = [gj for gj in jobs if gj.is_niche_role]
    elif niche_filter == "Standard only":
        jobs = [gj for gj in jobs if not gj.is_niche_role]

    st.caption(f"Showing {len(jobs)} jobs.")

    # Table
    import pandas as pd

    rows = [
        {
            "Title": gj.job.title,
            "Company": gj.job.company.name,
            "Industry": gj.job.company.industry,
            "Level": gj.job.requirements.experience_level.value,
            "Required Skills": len(gj.job.requirements.required_skills),
            "Niche": "✓" if gj.is_niche_role else "",
            "trace_id": gj.trace_id,
        }
        for gj in jobs
    ]
    df = pd.DataFrame(rows)

    # Show table without internal trace_id column
    st.dataframe(
        df.drop(columns=["trace_id"]),
        use_container_width=True,
        hide_index=True,
    )

    # Detail expander — pick a job to inspect
    st.divider()
    st.subheader("Job Detail")
    if jobs:
        job_labels = [f"{gj.job.title} @ {gj.job.company.name}" for gj in jobs]
        selected_idx = st.selectbox("Select a job", range(len(jobs)), format_func=lambda i: job_labels[i])
        gj = jobs[selected_idx]
        j = gj.job

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Title**: {j.title}")
            st.markdown(f"**Company**: {j.company.name} ({j.company.industry})")
            st.markdown(f"**Size**: {j.company.size}")
            st.markdown(f"**Location**: {j.company.location}")
            st.markdown(f"**Level**: {j.requirements.experience_level.value}")
            st.markdown(f"**Experience required**: {j.requirements.experience_years} years")
        with col2:
            st.markdown("**Required skills:**")
            for s in j.requirements.required_skills:
                st.markdown(f"- {s}")
            if j.requirements.preferred_skills:
                st.markdown("**Preferred skills:**")
                for s in j.requirements.preferred_skills:
                    st.markdown(f"- {s}")

        with st.expander("Full description"):
            st.write(j.description)


# ---------------------------------------------------------------------------
# Page 2: Review Resume
# ---------------------------------------------------------------------------


def page_review_resume() -> None:
    store = load_store()

    st.header("Review Resume")
    st.caption("Select a pair from the pipeline to inspect failure labels and judge results.")

    if not store.pairs:
        st.warning("No pairs found. Run the pipeline first.")
        return

    # Pair selector
    pair_labels = [
        f"{p.pair_id[:8]}… — {p.fit_level.value}"
        for p in store.pairs[:100]  # cap at 100 for UI performance
    ]
    selected_idx = st.selectbox("Select a resume-job pair", range(len(pair_labels)), format_func=lambda i: pair_labels[i])
    pair = store.pairs[selected_idx]

    resume_obj = store.resumes.get(pair.resume_trace_id)
    job_obj = store.jobs.get(pair.job_trace_id)
    labels = store.failure_labels.get(pair.pair_id)
    judge = store.judge_results.get(pair.pair_id)

    if resume_obj is None or job_obj is None:
        st.error("Resume or job data missing for this pair.")
        return

    r = resume_obj.resume
    j = job_obj.job

    st.divider()

    # --- Key metrics row ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fit Level", pair.fit_level.value.title())
    col2.metric("Jaccard Overlap", f"{labels.skills_overlap:.2f}" if labels else "N/A")
    exp_years = calculate_total_experience(r.experience)
    col3.metric("Resume Experience", f"{exp_years:.1f} yrs")
    col4.metric("Job Requires", f"{j.requirements.experience_years} yrs")

    st.divider()

    # --- Resume + Job side by side ---
    left, right = st.columns(2)

    with left:
        st.subheader(f"Resume — {r.contact_info.name}")
        st.markdown(f"*{resume_obj.writing_style.value.title()} style · {resume_obj.template_version}*")
        if r.summary:
            st.write(r.summary)
        st.markdown("**Skills:**")
        st.write(", ".join(s.name for s in r.skills))
        for exp in r.experience:
            st.markdown(f"**{exp.title}** @ {exp.company} ({exp.start_date}–{exp.end_date or 'present'})")
            for resp in exp.responsibilities[:3]:
                st.markdown(f"  - {resp}")

    with right:
        st.subheader(f"Job — {j.title}")
        st.markdown(f"*{j.company.name} · {j.company.industry}*")
        st.write(j.description[:400] + "…" if len(j.description) > 400 else j.description)
        st.markdown("**Required skills:**")
        st.write(", ".join(j.requirements.required_skills))

    st.divider()

    # --- Failure labels ---
    if labels:
        st.subheader("Failure Labels (Rule-based)")
        flag_col, score_col = st.columns(2)
        with flag_col:
            def flag(val: bool, label: str) -> str:
                return f"{'🔴' if val else '🟢'} {label}"

            st.markdown(flag(labels.experience_mismatch, "Experience mismatch"))
            st.markdown(flag(labels.seniority_mismatch, "Seniority mismatch"))
            st.markdown(flag(labels.missing_core_skills, "Missing core skills"))
            st.markdown(flag(labels.has_hallucinations, "Hallucinations"))
            st.markdown(flag(labels.has_awkward_language, "Awkward language"))

        with score_col:
            st.metric("Skills Jaccard", f"{labels.skills_overlap:.3f}")
            st.metric("Seniority (resume)", labels.seniority_level_resume)
            st.metric("Seniority (job)", labels.seniority_level_job)

        if labels.missing_skills:
            st.markdown("**Missing skills:** " + ", ".join(labels.missing_skills))
    else:
        st.info("No failure labels for this pair.")

    # --- Judge results ---
    st.divider()
    if judge:
        st.subheader("LLM Judge Results (GPT-4o)")
        j_col1, j_col2, j_col3 = st.columns(3)
        j_col1.metric("Quality Score", f"{judge.overall_quality_score:.2f}")
        j_col2.metric("Hallucinations", "Yes" if judge.has_hallucinations else "No")
        j_col3.metric("Awkward Language", "Yes" if judge.has_awkward_language else "No")
        st.markdown(f"**Assessment:** {judge.fit_assessment}")
        if judge.recommendations:
            st.markdown("**Recommendations:**")
            for rec in judge.recommendations:
                st.markdown(f"- {rec}")
        if judge.red_flags:
            st.markdown("**Red flags:**")
            for flag in judge.red_flags:
                st.markdown(f"- {flag}")
    else:
        st.info("No judge result for this pair (judge may not have been run).")


# ---------------------------------------------------------------------------
# Page 3: Analysis Dashboard
# ---------------------------------------------------------------------------


def page_analysis_dashboard() -> None:
    summary = load_pipeline_summary()
    charts = list_chart_paths()

    st.header("Analysis Dashboard")

    if not summary:
        st.warning("No pipeline_summary.json found. Run the pipeline first.")
        return

    gen = summary.get("generation", {})
    lab = summary.get("labeling", {})
    jdg = summary.get("judge", {})
    cor = summary.get("correction", {})
    ab = summary.get("ab_testing", {})

    # --- KPI row ---
    st.subheader("Key Metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Pairs Generated", gen.get("pairs_generated", 0))
    k2.metric("Validation Rate", f"{gen.get('validation_rate', 0):.0%}")
    k3.metric("Avg Judge Quality", f"{jdg.get('avg_quality_score', 0):.3f}")
    k4.metric("Correction Rate", f"{cor.get('correction_rate', 0):.0%}")
    k5.metric("A/B Significant", "Yes ✓" if ab.get("significant") else "No")

    st.divider()

    # --- Jaccard by fit level ---
    st.subheader("Jaccard Overlap by Fit Level")
    jaccard = lab.get("avg_jaccard_by_fit_level", {})
    if jaccard:
        import pandas as pd

        order = ["excellent", "good", "partial", "poor", "mismatch"]
        df_j = pd.DataFrame(
            [(k, v) for k, v in jaccard.items() if k in order],
            columns=["Fit Level", "Avg Jaccard"],
        )
        df_j["Fit Level"] = pd.Categorical(df_j["Fit Level"], categories=order, ordered=True)
        df_j = df_j.sort_values("Fit Level")
        st.bar_chart(df_j.set_index("Fit Level"))

    # --- Failure mode rates ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Failure Mode Rates")
        failure_rates = lab.get("failure_mode_rates", {})
        if failure_rates:
            import pandas as pd

            df_f = pd.DataFrame(failure_rates.items(), columns=["Mode", "Rate"])
            df_f["Rate"] = (df_f["Rate"] * 100).round(1)
            st.dataframe(df_f, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("A/B Template Results")
        tmpl_rates = ab.get("failure_rates_by_template", {})
        if tmpl_rates:
            import pandas as pd

            df_t = pd.DataFrame(tmpl_rates.items(), columns=["Template", "Failure Rate"])
            df_t["Failure Rate"] = (df_t["Failure Rate"] * 100).round(1)
            df_t = df_t.sort_values("Failure Rate")
            st.dataframe(df_t, use_container_width=True, hide_index=True)
            st.caption(
                f"Best: **{ab.get('best_template', '—')}** | "
                f"Worst: **{ab.get('worst_template', '—')}** | "
                f"χ²={ab.get('chi_squared_statistic', 0):.2f} p={ab.get('chi_squared_p_value', 1):.2e}"
            )

    st.divider()

    # --- Charts grid (3-column) ---
    if charts:
        st.subheader(f"Pipeline Charts ({len(charts)})")
        cols = st.columns(3)
        for i, chart_path in enumerate(charts):
            with cols[i % 3]:
                # Pretty-print filename: "failure_by_fit_level.png" → "Failure By Fit Level"
                title = chart_path.stem.replace("_", " ").title()
                st.caption(title)
                st.image(str(chart_path), use_container_width=True)
    else:
        st.info("No charts found in results/charts/. Run the pipeline first.")


# ---------------------------------------------------------------------------
# Page 4: Search Similar Candidates
# ---------------------------------------------------------------------------


def page_search_similar() -> None:
    st.header("Search Similar Candidates")

    if not _VECTOR_STORE_READY:
        st.warning(
            "ChromaDB vector index not ready. Run `python -m src.pipeline --skip-generation --skip-judge` "
            "or `python -c \"from src.vector_store import build_resume_index; build_resume_index()\"`"
        )
        return

    store = load_store()

    # Controls
    query = st.text_input(
        "Job description or skill query",
        placeholder="e.g. Senior Python engineer with FastAPI and PostgreSQL experience",
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        top_k = st.slider("Results", min_value=1, max_value=20, value=5)
    with col2:
        fit_options = ["Any", "excellent", "good", "partial", "poor", "mismatch"]
        fit_filter = st.selectbox("Filter by fit level", fit_options)
        fit_level = None if fit_filter == "Any" else fit_filter

    if not query:
        st.info("Enter a query above to search the resume index.")
        return

    with st.spinner("Searching..."):
        hits = search_similar(_collection, query, top_k=top_k, fit_level=fit_level)

    if not hits:
        st.warning("No results found. Try a broader query or different fit_level filter.")
        return

    st.caption(f"Top {len(hits)} results from {_collection.count()} indexed resumes.")
    st.divider()

    for i, hit in enumerate(hits, 1):
        trace_id = hit["trace_id"]
        score = hit["score"]
        meta = hit["metadata"]
        gr = store.resumes.get(trace_id)

        with st.expander(f"#{i} — Score: {score:.4f} | Fit: {meta.get('fit_level', '?')} | {meta.get('name', trace_id)}", expanded=(i == 1)):
            col_left, col_right = st.columns([2, 1])
            with col_left:
                if gr:
                    r = gr.resume
                    exp_years = calculate_total_experience(r.experience)
                    st.markdown(f"**{r.contact_info.name}** — {exp_years:.1f} years experience")
                    st.markdown(f"**Skills:** {', '.join(s.name for s in r.skills[:10])}")
                    if r.summary:
                        st.write(r.summary[:300] + "…" if len(r.summary) > 300 else r.summary)
                else:
                    st.markdown(f"**Skills:** {meta.get('skills', '—')}")
            with col_right:
                st.metric("Similarity", f"{score:.4f}")
                st.metric("Fit Level", meta.get("fit_level", "—").title())
                st.caption(f"trace_id: {trace_id[:16]}…")


# ---------------------------------------------------------------------------
# Page 5: Feedback
# ---------------------------------------------------------------------------


def page_feedback() -> None:
    store = load_store()

    st.header("Submit Feedback")
    st.caption("Rate the resume-job match quality. Saved to data/feedback/feedback.jsonl.")

    if not store.pairs:
        st.warning("No pairs found. Run the pipeline first.")
        return

    # Pair selector (show first 100)
    pair_labels = [
        f"{p.pair_id[:8]}… — {p.fit_level.value}"
        for p in store.pairs[:100]
    ]
    selected_idx = st.selectbox(
        "Select a pair to rate", range(len(pair_labels)), format_func=lambda i: pair_labels[i]
    )
    pair = store.pairs[selected_idx]

    # Show a quick summary
    resume_obj = store.resumes.get(pair.resume_trace_id)
    job_obj = store.jobs.get(pair.job_trace_id)
    if resume_obj and job_obj:
        st.markdown(
            f"**Resume**: {resume_obj.resume.contact_info.name} | "
            f"**Job**: {job_obj.job.title} @ {job_obj.job.company.name} | "
            f"**Fit**: {pair.fit_level.value}"
        )

    # Show existing feedback count
    existing = store.feedback.get(pair.pair_id, [])
    if existing:
        st.info(f"This pair has {len(existing)} existing feedback entries.")

    st.divider()

    # Feedback form
    with st.form("feedback_form"):
        st.markdown("**How accurate is the AI's fit assessment for this pair?**")
        rating_col1, rating_col2, rating_col3 = st.columns(3)
        with rating_col1:
            thumbs_up = st.form_submit_button("👍 Good assessment", use_container_width=True)
        with rating_col2:
            thumbs_neutral = st.form_submit_button("😐 Partially agree", use_container_width=True)
        with rating_col3:
            thumbs_down = st.form_submit_button("👎 Disagree", use_container_width=True)

        comment = st.text_area("Optional comment", placeholder="e.g. The Jaccard score seems too low given the resume depth.")

    # Determine rating from which button was clicked
    rating: str | None = None
    if thumbs_up:
        rating = "5"
    elif thumbs_neutral:
        rating = "3"
    elif thumbs_down:
        rating = "1"

    if rating is not None:
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = {
            "feedback_id": feedback_id,
            "pair_id": pair.pair_id,
            "rating": rating,
            "comment": comment,
            "timestamp": timestamp,
        }

        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        feedback_path = FEEDBACK_DIR / "feedback.jsonl"
        with feedback_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        label = {"5": "positive", "3": "neutral", "1": "negative"}[rating]
        st.success(f"Feedback saved! ({label}) — ID: {feedback_id[:8]}…")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Browse Jobs":
    page_browse_jobs()
elif page == "Review Resume":
    page_review_resume()
elif page == "Analysis Dashboard":
    page_analysis_dashboard()
elif page == "Search Similar":
    page_search_similar()
elif page == "Feedback":
    page_feedback()
