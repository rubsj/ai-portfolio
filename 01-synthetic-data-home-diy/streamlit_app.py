"""P1: Synthetic Data — Home DIY Repair — Streamlit Demo App.

WHY Streamlit: Fastest path from data artifacts to interactive portfolio demo.
Streamlit re-runs the entire script on every widget interaction, so we use
@st.cache_data on all file-loading functions to avoid redundant I/O.
This is similar to React's useMemo but at the function level.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="P1: Synthetic Data — Home DIY Repair",
    page_icon="🔧",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants — all paths relative to this file (project root)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

CATEGORIES = [
    "appliance_repair",
    "electrical_repair",
    "general_home_repair",
    "hvac_maintenance",
    "plumbing_repair",
]

DIFFICULTIES = ["beginner", "intermediate", "advanced"]

FAILURE_MODES = [
    "incomplete_answer",
    "safety_violations",
    "unrealistic_tools",
    "overcomplicated_solution",
    "missing_context",
    "poor_quality_tips",
]

# WHY human-readable labels: raw snake_case keys aren't portfolio-friendly
FAILURE_MODE_LABELS: dict[str, str] = {
    "incomplete_answer": "Incomplete Answer",
    "safety_violations": "Safety Violations",
    "unrealistic_tools": "Unrealistic Tools",
    "overcomplicated_solution": "Overcomplicated Solution",
    "missing_context": "Missing Context",
    "poor_quality_tips": "Poor Quality Tips",
}

CATEGORY_LABELS: dict[str, str] = {
    "appliance_repair": "Appliance Repair",
    "electrical_repair": "Electrical Repair",
    "general_home_repair": "General Home Repair",
    "hvac_maintenance": "HVAC Maintenance",
    "plumbing_repair": "Plumbing Repair",
}


# ---------------------------------------------------------------------------
# Data loading — cached to survive Streamlit re-runs
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(path: Path) -> list | dict | None:
    """Load a JSON file, returning None if missing."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


@st.cache_data
def load_records(version: str) -> list[dict]:
    """Load generated records for v1 or v2."""
    filename = "batch_v1.json" if version == "v1" else "batch_v2.json"
    data = load_json(DATA_DIR / "generated" / filename)
    return data if data else []


@st.cache_data
def load_labels(version: str) -> list[dict]:
    """Load LLM judge labels for a pipeline stage.

    WHY separate function: each label file covers a different pipeline stage,
    and the caller picks which stage to inspect.
    """
    filenames: dict[str, str] = {
        "v1": "llm_labels.json",
        "v1_corrected": "llm_labels_corrected.json",
        "v2": "llm_labels_v2.json",
        "v2_corrected": "v2_corrected_llm_labels.json",
    }
    data = load_json(DATA_DIR / "labels" / filenames.get(version, "llm_labels.json"))
    return data if data else []


@st.cache_data
def load_corrected_records(version: str) -> list[dict]:
    """Load corrected records (v1_corrected or v2_corrected)."""
    filename = (
        "corrected_records.json" if version == "v1" else "v2_corrected_records.json"
    )
    data = load_json(DATA_DIR / "corrected" / filename)
    return data if data else []


@st.cache_data
def load_metrics() -> dict | None:
    return load_json(RESULTS_DIR / "metrics.json")


@st.cache_data
def load_correction_comparison() -> dict | None:
    return load_json(RESULTS_DIR / "correction_comparison.json")


@st.cache_data
def load_agreement_report() -> dict | None:
    return load_json(DATA_DIR / "labels" / "agreement_report.json")


# ---------------------------------------------------------------------------
# Helper: build a lookup from trace_id → label dict
# ---------------------------------------------------------------------------
def build_label_lookup(labels: list[dict]) -> dict[str, dict]:
    """Index labels by trace_id for O(1) lookup."""
    return {item["trace_id"]: item for item in labels}


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Browse Records",
        "Judge Evaluations",
        "Failure Analysis",
        "Correction Pipeline",
        "Metrics Deep Dive",
    ],
    # WHY captions: gives context at a glance without cluttering the radio labels
    captions=[
        "Key results overview",
        "Explore generated records",
        "LLM-as-Judge results",
        "Charts & patterns",
        "The 36 → 0 story",
        "All numbers",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Built as part of a 9-project AI portfolio sprint | "
    "[GitHub](https://github.com/rubsj/ai-portfolio)"
)


# ===================================================================
# SECTION 1: Dashboard
# ===================================================================
def render_dashboard() -> None:
    st.title("P1: Synthetic Data — Home DIY Repair")
    st.markdown(
        "Generated **30 synthetic DIY repair records** using GPT-4o-mini "
        "→ Evaluated with **LLM-as-Judge** (GPT-4o) → Identified failure patterns "
        "→ **Improved templates** → Corrected remaining failures → **0 failures**."
    )

    # --- Metric cards ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records Generated", "30")
    col2.metric("V1 Failure Rate", "20.0%", delta=None)
    col3.metric("V2 Failure Rate", "4.4%", delta="-77.8%")
    col4.metric("Final (V2+Correction)", "0.0%", delta="-100%")

    st.divider()

    # --- Correction improvement chart ---
    chart_path = CHARTS_DIR / "correction_improvement.png"
    if chart_path.exists():
        st.subheader("Correction Pipeline Results")
        st.image(str(chart_path), use_container_width=True)
    else:
        st.warning("Chart not found: correction_improvement.png")

    # --- Pipeline summary ---
    st.divider()
    st.subheader("How It Works")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**1. Generate & Validate**")
        st.markdown(
            "GPT-4o-mini generates Q&A records via Instructor. "
            "Pydantic validates structure. 30/30 passed (100%)."
        )
    with cols[1]:
        st.markdown("**2. Evaluate & Analyze**")
        st.markdown(
            "GPT-4o judges each record against 6 failure modes. "
            "V1 had 36 failures (20%). Dominant: *incomplete_answer* (50%), "
            "*poor_quality_tips* (43%)."
        )
    with cols[2]:
        st.markdown("**3. Correct & Re-evaluate**")
        st.markdown(
            "Improved prompt templates cut failures to 8 (-78%). "
            "Targeted correction of remaining 8 brought total to **0** (-100%)."
        )


# ===================================================================
# SECTION 2: Browse Records
# ===================================================================
def render_browse_records() -> None:
    st.title("Browse Records")

    # --- Filters ---
    col1, col2, col3 = st.columns(3)
    with col1:
        version = st.radio("Version", ["V1 (Original)", "V2 (Improved Templates)"])
    with col2:
        category_filter = st.selectbox(
            "Category",
            ["All"] + [CATEGORY_LABELS[c] for c in CATEGORIES],
        )
    with col3:
        difficulty_filter = st.selectbox(
            "Difficulty",
            ["All"] + [d.capitalize() for d in DIFFICULTIES],
        )

    version_key = "v1" if "V1" in version else "v2"
    records = load_records(version_key)

    if not records:
        st.warning(f"No records found for {version_key}.")
        return

    # WHY reverse lookup: filter uses human-readable labels, data uses snake_case keys
    label_to_key = {v: k for k, v in CATEGORY_LABELS.items()}

    filtered = records
    if category_filter != "All":
        cat_key = label_to_key[category_filter]
        filtered = [r for r in filtered if r["category"] == cat_key]
    if difficulty_filter != "All":
        diff_key = difficulty_filter.lower()
        filtered = [r for r in filtered if r["difficulty"] == diff_key]

    st.markdown(f"**Showing {len(filtered)} of {len(records)} records**")

    for rec in filtered:
        inner = rec["record"]
        header = (
            f"{CATEGORY_LABELS.get(rec['category'], rec['category'])} | "
            f"{rec['difficulty'].capitalize()} | "
            f"`{rec['trace_id'][:8]}`"
        )
        with st.expander(header):
            st.markdown(f"**Question:** {inner['question']}")
            st.markdown(f"**Equipment Problem:** {inner['equipment_problem']}")
            st.markdown(f"**Answer:** {inner['answer']}")

            st.markdown("**Tools Required:**")
            # WHY inline badges: visually distinct from body text
            st.markdown(
                " ".join(f"`{tool}`" for tool in inner["tools_required"])
            )

            st.markdown("**Steps:**")
            for i, step in enumerate(inner["steps"], 1):
                st.markdown(f"{i}. {step}")

            st.markdown(f"**Safety Info:** {inner['safety_info']}")
            st.markdown(f"**Tips:** {inner['tips']}")


# ===================================================================
# SECTION 3: Judge Evaluations
# ===================================================================
def render_judge_evaluations() -> None:
    st.title("Judge Evaluations")

    labels_v1 = load_labels("v1")
    if not labels_v1:
        st.warning("No V1 labels found.")
        return

    lookup = build_label_lookup(labels_v1)
    trace_ids = list(lookup.keys())

    selected_id = st.selectbox(
        "Select a record (trace_id)",
        trace_ids,
        format_func=lambda tid: tid[:8] + "...",
    )
    if not selected_id:
        return

    eval_data = lookup[selected_id]

    # --- Overall quality score ---
    score = eval_data["overall_quality_score"]
    st.metric("Overall Quality Score", f"{score} / 5")

    st.divider()

    # --- Per-mode results ---
    st.subheader("Failure Mode Labels")
    cols = st.columns(3)
    for i, label_entry in enumerate(eval_data["labels"]):
        mode = label_entry["mode"]
        passed = label_entry["label"] == 0
        reason = label_entry["reason"]
        with cols[i % 3]:
            icon = "✅" if passed else "❌"
            status = "Pass" if passed else "Fail"
            st.markdown(
                f"**{icon} {FAILURE_MODE_LABELS.get(mode, mode)}** — {status}"
            )
            st.caption(reason)

    # --- Also show the actual record for context ---
    st.divider()
    st.subheader("Record Preview")
    records = load_records("v1")
    rec_lookup = {r["trace_id"]: r for r in records}
    rec = rec_lookup.get(selected_id)
    if rec:
        inner = rec["record"]
        st.markdown(f"**Question:** {inner['question']}")
        st.markdown(f"**Answer:** {inner['answer'][:300]}...")

    # --- Inter-rater agreement ---
    st.divider()
    st.subheader("Inter-Rater Agreement (Manual vs LLM)")
    agreement = load_agreement_report()
    if agreement:
        st.markdown(
            f"Based on **{agreement['matched_records']} records** labeled by both "
            f"human and LLM judge. Overall agreement: **{agreement['overall_agreement']}**"
        )
        # WHY columns layout: makes per-mode agreement scannable at a glance
        cols = st.columns(3)
        for i, (mode, rate) in enumerate(agreement["per_mode_agreement"].items()):
            with cols[i % 3]:
                st.metric(FAILURE_MODE_LABELS.get(mode, mode), rate)
    else:
        st.warning("Agreement report not found.")


# ===================================================================
# SECTION 4: Failure Analysis
# ===================================================================
def render_failure_analysis() -> None:
    st.title("Failure Analysis")
    st.markdown("Analysis of V1 generation failures across 30 records and 6 failure modes.")

    charts: list[tuple[str, str, str]] = [
        (
            "failure_heatmap.png",
            "Failure Heatmap — Records x Failure Modes",
            "Each row is a record, each column a failure mode. "
            "Red cells indicate failures. **21 of 30 records** had at least one failure.",
        ),
        (
            "failure_frequency.png",
            "Failure Frequency by Mode",
            "**incomplete_answer** (50%) and **poor_quality_tips** (43.3%) were the "
            "dominant failure modes. overcomplicated_solution and missing_context had 0 failures.",
        ),
        (
            "failure_correlation.png",
            "Failure Mode Co-occurrence",
            "Shows which failure modes tend to appear together. Useful for identifying "
            "root causes — if two modes always co-occur, they may share the same fix.",
        ),
        (
            "category_failures.png",
            "Failures by Repair Category",
            "**electrical_repair had 0 failures** — its template was already well-crafted. "
            "general_home_repair (30.6%) and appliance_repair (27.8%) had the highest rates.",
        ),
        (
            "difficulty_failures.png",
            "Failures by Difficulty Level",
            "intermediate (23.3%) had slightly more failures than beginner (16.7%) "
            "or advanced (20.0%). Difficulty level was not a strong predictor of quality.",
        ),
    ]

    for filename, title, insight in charts:
        chart_path = CHARTS_DIR / filename
        st.subheader(title)
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)
            st.markdown(f"*{insight}*")
        else:
            st.warning(f"Chart not found: {filename}")
        st.divider()


# ===================================================================
# SECTION 5: Correction Pipeline
# ===================================================================
def render_correction_pipeline() -> None:
    st.title("Correction Pipeline")
    st.markdown("The story of how we went from **36 failures → 8 → 0**.")

    # --- Three-stage progression ---
    col1, col2, col3 = st.columns(3)

    comparison = load_correction_comparison()
    if not comparison:
        st.warning("Correction comparison data not found.")
        return

    with col1:
        st.markdown("### Stage 1: V1 Original")
        v1 = comparison["v1_original"]
        st.metric("Failures", v1["total_failures"])
        st.metric("Failure Rate", v1["failure_rate"])
        st.markdown("**Top failure modes:**")
        # WHY sorted by count: highlights the most impactful modes first
        sorted_modes = sorted(
            v1["per_mode"].items(), key=lambda x: x[1], reverse=True
        )
        for mode, count in sorted_modes:
            if count > 0:
                st.markdown(f"- {FAILURE_MODE_LABELS.get(mode, mode)}: **{count}**")

    with col2:
        st.markdown("### Stage 2: V2 Templates")
        v2 = comparison["v2_generated"]
        st.metric("Failures", v2["total_failures"], delta="-77.8%")
        st.metric("Failure Rate", v2["failure_rate"])
        st.markdown("**What improved templates fixed:**")
        st.markdown(
            "- Added explicit instructions for comprehensive answers\n"
            "- Required actionable, specific tips\n"
            "- Eliminated safety_violations and unrealistic_tools entirely"
        )

    with col3:
        st.markdown("### Stage 3: V2 + Correction")
        v2c = comparison["v2_corrected"]
        st.metric("Failures", v2c["total_failures"], delta="-100%")
        st.metric("Failure Rate", v2c["failure_rate"])
        st.markdown("**Correction strategy:**")
        st.markdown(
            "- Fed judge reasoning back to GPT-4o-mini\n"
            "- Targeted only the 8 remaining failures\n"
            "- All failure modes resolved to **0**"
        )

    st.divider()

    # --- Before/After comparison ---
    st.subheader("Before/After Comparison")
    st.markdown("Pick a corrected record to see the original vs corrected version side by side.")

    # WHY v2 corrected records: they represent the final pipeline output
    corrected = load_corrected_records("v2")
    originals = load_records("v2")
    v2_labels = load_labels("v2")

    if not corrected or not originals:
        st.warning("Corrected or original records not found.")
    else:
        # Only show records that actually had failures (were corrected)
        label_lookup = build_label_lookup(v2_labels) if v2_labels else {}
        failed_ids = [
            tid
            for tid, lbl in label_lookup.items()
            if any(entry["label"] == 1 for entry in lbl["labels"])
        ]

        if not failed_ids:
            st.info("All V2 records passed — showing a sample comparison instead.")
            failed_ids = [originals[0]["trace_id"]] if originals else []

        if failed_ids:
            orig_lookup = {r["trace_id"]: r for r in originals}
            corr_lookup = {r["trace_id"]: r for r in corrected}

            selected = st.selectbox(
                "Select a corrected record",
                failed_ids,
                format_func=lambda tid: (
                    f"{tid[:8]}... | "
                    f"{CATEGORY_LABELS.get(orig_lookup.get(tid, {}).get('category', ''), '')} | "
                    f"{orig_lookup.get(tid, {}).get('difficulty', '')}"
                ),
            )

            if selected:
                orig = orig_lookup.get(selected)
                corr = corr_lookup.get(selected)

                if orig and corr:
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**Original (V2)**")
                        _render_record_card(orig)
                    with right:
                        st.markdown("**Corrected (V2 + Correction)**")
                        _render_record_card(corr)
                else:
                    st.warning("Could not find matching original/corrected pair.")

    st.divider()

    # --- Correction chart ---
    chart_path = CHARTS_DIR / "correction_improvement.png"
    if chart_path.exists():
        st.subheader("Correction Improvement Chart")
        st.image(str(chart_path), use_container_width=True)


def _render_record_card(rec: dict) -> None:
    """Render a compact record card for side-by-side comparison."""
    inner = rec["record"]
    st.markdown(f"**Q:** {inner['question']}")
    st.markdown(f"**A:** {inner['answer'][:500]}{'...' if len(inner['answer']) > 500 else ''}")
    st.markdown(f"**Safety:** {inner['safety_info'][:200]}")
    st.markdown(f"**Tips:** {inner['tips'][:200]}")


# ===================================================================
# SECTION 6: Metrics Deep Dive
# ===================================================================
def render_metrics_deep_dive() -> None:
    st.title("Metrics Deep Dive")

    metrics = load_metrics()
    if not metrics:
        st.warning("Metrics file not found.")
        return

    # --- Dataset summary ---
    st.subheader("Dataset Summary")
    ds = metrics["dataset_summary"]
    cols = st.columns(4)
    cols[0].metric("Total Records", ds["total_records"])
    cols[1].metric("Total Evaluations", ds["total_possible_failures"])
    cols[2].metric("Total Failures", ds["total_failures"])
    cols[3].metric("Avg Quality Score", ds["avg_quality_score"])

    st.divider()

    # --- Per-mode failure rates (V1) ---
    st.subheader("Per-Mode Failure Rates (V1)")
    mode_data = metrics["per_mode_failures"]
    cols = st.columns(3)
    for i, (mode, info) in enumerate(mode_data.items()):
        with cols[i % 3]:
            st.metric(
                FAILURE_MODE_LABELS.get(mode, mode),
                info["rate"],
                help=f"{info['count']} out of 30 records",
            )

    st.divider()

    # --- Pipeline comparison ---
    st.subheader("Pipeline Comparison (V1 vs V2 vs V2+Correction)")
    pipeline = metrics["combined_pipeline"]

    pipe_cols = st.columns(3)
    with pipe_cols[0]:
        st.markdown("**V1 Original**")
        st.metric("Failures", pipeline["v1_original"]["total_failures"])
        st.metric("Rate", pipeline["v1_original"]["failure_rate"])
    with pipe_cols[1]:
        st.markdown("**V2 Templates**")
        st.metric("Failures", pipeline["v2_generated"]["total_failures"])
        st.metric("Rate", pipeline["v2_generated"]["failure_rate"])
        st.metric("Reduction", pipeline["v2_generated"]["reduction_vs_v1"])
    with pipe_cols[2]:
        st.markdown("**V2 + Correction**")
        st.metric("Failures", pipeline["v2_corrected"]["total_failures"])
        st.metric("Rate", pipeline["v2_corrected"]["failure_rate"])
        st.metric("Reduction", pipeline["v2_corrected"]["reduction_vs_v1"])

    st.divider()

    # --- Per-category breakdown ---
    st.subheader("Failure Rate by Category")
    cat_data = metrics["per_category"]
    cols = st.columns(len(cat_data))
    for i, (cat, info) in enumerate(cat_data.items()):
        with cols[i]:
            st.metric(
                CATEGORY_LABELS.get(cat, cat),
                info["failure_rate"],
                help=f"{info['total_failures']} failures across {info['records']} records",
            )

    st.divider()

    # --- Per-difficulty breakdown ---
    st.subheader("Failure Rate by Difficulty")
    diff_data = metrics["per_difficulty"]
    cols = st.columns(len(diff_data))
    for i, (diff, info) in enumerate(diff_data.items()):
        with cols[i]:
            st.metric(
                diff.capitalize(),
                info["failure_rate"],
                help=f"{info['total_failures']} failures across {info['records']} records",
            )

    st.divider()

    # --- Inter-rater agreement ---
    st.subheader("Inter-Rater Agreement")
    agreement = metrics["inter_rater_agreement"]
    st.markdown(
        f"**{agreement['matched_records']} records** labeled by both human and LLM. "
        f"Overall agreement: **{agreement['overall_agreement']}**"
    )
    cols = st.columns(3)
    for i, (mode, rate) in enumerate(agreement["per_mode_agreement"].items()):
        with cols[i % 3]:
            st.metric(FAILURE_MODE_LABELS.get(mode, mode), rate)

    # --- Agreement chart ---
    chart_path = CHARTS_DIR / "agreement_matrix.png"
    if chart_path.exists():
        st.divider()
        st.subheader("Agreement Matrix")
        st.image(str(chart_path), use_container_width=True)

    # --- Correction comparison detail ---
    st.divider()
    st.subheader("Correction Comparison — Per Mode")
    comparison = load_correction_comparison()
    if comparison:
        # WHY table format: makes per-mode numbers scannable across pipeline stages
        header_cols = st.columns([2, 1, 1, 1, 1])
        header_cols[0].markdown("**Failure Mode**")
        header_cols[1].markdown("**V1**")
        header_cols[2].markdown("**V1 Corrected**")
        header_cols[3].markdown("**V2**")
        header_cols[4].markdown("**V2 Corrected**")

        for mode in FAILURE_MODES:
            row = st.columns([2, 1, 1, 1, 1])
            row[0].markdown(FAILURE_MODE_LABELS.get(mode, mode))
            row[1].markdown(str(comparison["v1_original"]["per_mode"].get(mode, 0)))
            row[2].markdown(str(comparison["corrected"]["per_mode"].get(mode, 0)))
            row[3].markdown(str(comparison["v2_generated"]["per_mode"].get(mode, 0)))
            row[4].markdown(str(comparison["v2_corrected"]["per_mode"].get(mode, 0)))


# ===================================================================
# Router — dispatch to selected section
# ===================================================================
SECTIONS: dict[str, callable] = {
    "Dashboard": render_dashboard,
    "Browse Records": render_browse_records,
    "Judge Evaluations": render_judge_evaluations,
    "Failure Analysis": render_failure_analysis,
    "Correction Pipeline": render_correction_pipeline,
    "Metrics Deep Dive": render_metrics_deep_dive,
}

SECTIONS[section]()
