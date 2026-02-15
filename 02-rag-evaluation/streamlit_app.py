"""P2: RAG Evaluation — Streamlit Interactive Dashboard.

WHY Streamlit: Fastest path to interactive portfolio demo. Plotly for
interactive charts (zoom, hover, filter). Deployed to Streamlit Cloud.

7 pages:
1. Dashboard — metrics overview + interactive heatmap
2. Chunk Strategy Analysis — size/overlap/semantic comparisons
3. Embedding Models — quality vs cost
4. Reranking Impact — before/after improvements
5. RAGAS Generation Quality — 4 metrics radar chart
6. LLM Judge Analysis — Bloom taxonomy + correctness
7. Charts Gallery — all 12 static PNGs
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="P2: RAG Evaluation Benchmark",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRICS_DIR = Path(__file__).parent / "results" / "metrics"
CHARTS_DIR = Path(__file__).parent / "results" / "charts"
REPORTS_DIR = Path(__file__).parent / "results" / "reports"

METRIC_COLS = [
    "avg_recall_at_1",
    "avg_recall_at_3",
    "avg_recall_at_5",
    "avg_precision_at_1",
    "avg_precision_at_3",
    "avg_precision_at_5",
    "avg_mrr_at_5",
]

METRIC_LABELS = {
    "avg_recall_at_1": "R@1",
    "avg_recall_at_3": "R@3",
    "avg_recall_at_5": "R@5",
    "avg_precision_at_1": "P@1",
    "avg_precision_at_3": "P@3",
    "avg_precision_at_5": "P@5",
    "avg_mrr_at_5": "MRR@5",
}

# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_grid_search_results() -> list[dict]:
    """Load all 16 config evaluations.

    WHY cache: 38K-line JSON file, expensive to parse on every re-render.
    """
    path = METRICS_DIR / "grid_search_results.json"
    return json.loads(path.read_text())


@st.cache_data
def load_reranking_results() -> list[dict]:
    """Load reranking before/after comparisons (3 configs)."""
    path = METRICS_DIR / "reranking_results.json"
    return json.loads(path.read_text())


@st.cache_data
def load_ragas_results() -> dict:
    """Load RAGAS generation metrics (E-openai only)."""
    path = METRICS_DIR / "ragas_results.json"
    return json.loads(path.read_text())


@st.cache_data
def load_judge_results() -> list[dict]:
    """Load LLM-as-Judge verdicts (56 questions)."""
    path = METRICS_DIR / "judge_results.json"
    return json.loads(path.read_text())


@st.cache_data
def load_qa_report() -> dict:
    """Load QA dataset quality report."""
    path = REPORTS_DIR / "qa_dataset_report.json"
    return json.loads(path.read_text())


@st.cache_data
def evals_to_dataframe(evaluations: list[dict]) -> pd.DataFrame:
    """Convert evaluations to DataFrame for Plotly.

    WHY DataFrame: Plotly requires tabular data. Easier to filter/sort/pivot.
    """
    rows = []
    for ev in evaluations:
        row = {
            "config_id": ev["config_id"],
            "chunk_config": ev["chunk_config"],
            "embedding_model": ev["embedding_model"],
            "retrieval_method": ev["retrieval_method"],
            **{col: ev[col] for col in METRIC_COLS},
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🔍 P2: RAG Evaluation")
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Dashboard",
        "📐 Chunk Strategy Analysis",
        "🔤 Embedding Models",
        "🎯 Reranking Impact",
        "🤖 RAGAS Generation Quality",
        "⚖️ LLM Judge Analysis",
        "🖼️ Charts Gallery",
    ],
)

# Load data once
grid_data = load_grid_search_results()
df = evals_to_dataframe(grid_data)

# ---------------------------------------------------------------------------
# Page 1: Dashboard
# ---------------------------------------------------------------------------

if page == "📊 Dashboard":
    st.title("📊 RAG Evaluation Dashboard")
    st.markdown("**Overview of 16 configurations across 5 evaluation layers**")

    # Metric cards for top-3 configs
    st.subheader("Top-3 Configurations (by Recall@5)")

    top_3 = df.nlargest(3, "avg_recall_at_5")

    cols = st.columns(3)
    for i, (_, row) in enumerate(top_3.iterrows()):
        with cols[i]:
            st.metric(
                label=f"#{i+1}: {row['config_id']}",
                value=f"R@5: {row['avg_recall_at_5']:.3f}",
                delta=f"P@5: {row['avg_precision_at_5']:.3f}",
            )

    # Interactive heatmap
    st.subheader("Config × Metric Heatmap (Interactive)")

    # Filter by embedding model
    embedding_filter = st.multiselect(
        "Filter by embedding model",
        options=df["embedding_model"].unique(),
        default=df["embedding_model"].unique(),
    )

    filtered_df = df[df["embedding_model"].isin(embedding_filter)]

    # Prepare heatmap data
    heatmap_data = filtered_df.set_index("config_id")[list(METRIC_LABELS.keys())]
    heatmap_data.columns = [METRIC_LABELS[col] for col in heatmap_data.columns]

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Metric", y="Configuration", color="Score"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

    # Config selector
    st.subheader("Detailed Metrics for Selected Config")
    selected_config = st.selectbox("Select configuration", df["config_id"])

    config_row = df[df["config_id"] == selected_config].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Retrieval Metrics**")
        st.write(f"- Recall@1: {config_row['avg_recall_at_1']:.3f}")
        st.write(f"- Recall@3: {config_row['avg_recall_at_3']:.3f}")
        st.write(f"- Recall@5: {config_row['avg_recall_at_5']:.3f}")

    with col2:
        st.write("**Precision & MRR**")
        st.write(f"- Precision@5: {config_row['avg_precision_at_5']:.3f}")
        st.write(f"- MRR@5: {config_row['avg_mrr_at_5']:.3f}")

# ---------------------------------------------------------------------------
# Page 2: Chunk Strategy Analysis
# ---------------------------------------------------------------------------

elif page == "📐 Chunk Strategy Analysis":
    st.title("📐 Chunk Strategy Analysis")

    # Chunk size effect
    st.subheader("Chunk Size Impact (Configs A/B/C)")

    chunk_size_configs = ["A-openai", "B-openai", "C-openai"]
    chunk_size_data = df[df["config_id"].isin(chunk_size_configs)].copy()
    chunk_size_data["chunk_size"] = chunk_size_data["config_id"].map({
        "A-openai": 128,
        "B-openai": 256,
        "C-openai": 512,
    })
    chunk_size_data = chunk_size_data.sort_values("chunk_size")

    fig = px.line(
        chunk_size_data,
        x="chunk_size",
        y="avg_recall_at_5",
        markers=True,
        title="Chunk Size vs Recall@5 (25% Overlap, OpenAI Embeddings)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("**Finding:** 256 tokens (Config B) achieves best balance between granularity and context.")

    # Overlap effect
    st.subheader("Overlap Impact (Configs B vs D)")

    overlap_configs = ["B-openai", "D-openai"]
    overlap_data = df[df["config_id"].isin(overlap_configs)]

    fig = go.Figure()
    for metric in ["avg_recall_at_5", "avg_precision_at_5", "avg_mrr_at_5"]:
        fig.add_trace(go.Bar(
            name=METRIC_LABELS.get(metric, metric),
            x=overlap_data["config_id"],
            y=overlap_data[metric],
        ))

    fig.update_layout(
        title="Config B (25% overlap) vs Config D (50% overlap)",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning("**Surprise Finding:** 50% overlap UNDERPERFORMS 25% by 13%. Redundancy dilutes ranking quality.")

    # Semantic vs fixed
    st.subheader("Semantic (Config E) vs Fixed-Size (Config B)")

    semantic_configs = ["E-openai", "B-openai"]
    semantic_data = df[df["config_id"].isin(semantic_configs)]

    fig = go.Figure()
    for metric in ["avg_recall_at_5", "avg_precision_at_5"]:
        fig.add_trace(go.Bar(
            name=METRIC_LABELS.get(metric, metric),
            x=semantic_data["config_id"],
            y=semantic_data[metric],
        ))

    fig.update_layout(
        title="Semantic Chunking (E) vs Fixed-Size (B), OpenAI Embeddings",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("**Key Finding:** Semantic chunking (E) wins! Splitting on document structure beats fixed token counts.")

# ---------------------------------------------------------------------------
# Page 3: Embedding Models
# ---------------------------------------------------------------------------

elif page == "🔤 Embedding Models":
    st.title("🔤 Embedding Model Comparison")

    st.markdown("**Comparing local (MiniLM, MPnet) vs API (OpenAI) embeddings on Config B (256/64)**")

    embedding_configs = ["B-minilm", "B-mpnet", "B-openai"]
    embedding_data = df[df["config_id"].isin(embedding_configs)]

    # Quality comparison
    fig = go.Figure()
    for metric in ["avg_recall_at_5", "avg_precision_at_5", "avg_mrr_at_5"]:
        fig.add_trace(go.Bar(
            name=METRIC_LABELS.get(metric, metric),
            x=embedding_data["config_id"],
            y=embedding_data[metric],
        ))

    fig.update_layout(
        title="Embedding Quality Comparison (Config B)",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cost vs quality table
    st.subheader("Cost vs Quality Tradeoff")

    cost_data = pd.DataFrame({
        "Model": ["MiniLM-L6-v2", "MPnet-base-v2", "OpenAI text-embedding-3-small"],
        "Dimensions": [384, 768, 1536],
        "Location": ["Local (M2)", "Local (M2)", "OpenAI API"],
        "Recall@5": [0.481, 0.467, 0.607],  # FIXED: actual values from data
        "Cost": ["Free", "Free", "$0.02/1M tokens"],
        "Winner": ["❌", "❌", "✅"],
    })

    st.dataframe(cost_data, use_container_width=True)

    st.info("**Recommendation:** OpenAI embeddings outperform local by 26% for ~$0.02/1M tokens. Worth it for production.")

# ---------------------------------------------------------------------------
# Page 4: Reranking Impact
# ---------------------------------------------------------------------------

elif page == "🎯 Reranking Impact":
    st.title("🎯 Reranking Impact (Cohere Rerank API)")

    rerank_data = load_reranking_results()

    # Before/after comparison
    # WHY flat field names: JSON structure uses flat fields, not nested original_metrics/reranked_metrics
    configs = [item["config_id"] for item in rerank_data]
    before = [item["recall_at_5_before"] for item in rerank_data]  # FIXED: was "before_recall_at_5"
    after = [item["recall_at_5_after"] for item in rerank_data]    # FIXED: was "after_recall_at_5"
    improvements = [item["recall_improvement_pct"] for item in rerank_data]  # FIXED: was "improvement_percent_recall_at_5"

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before Reranking", x=configs, y=before, marker_color="lightcoral"))
    fig.add_trace(go.Bar(name="After Reranking", x=configs, y=after, marker_color="seagreen"))

    fig.update_layout(
        title="Recall@5 Before vs After Reranking",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Improvements table
    st.subheader("Improvement Breakdown")

    improvement_df = pd.DataFrame({
        "Config": configs,
        "Before R@5": before,
        "After R@5": after,
        "Improvement %": [f"+{imp:.1f}%" for imp in improvements],
    })

    st.dataframe(improvement_df, use_container_width=True)

    st.success("**Key Finding:** ~20% average improvement. Reranking is non-negotiable for production RAG.")

# ---------------------------------------------------------------------------
# Page 5: RAGAS Generation Quality
# ---------------------------------------------------------------------------

elif page == "🤖 RAGAS Generation Quality":
    st.title("🤖 RAGAS Generation Quality (E-openai)")

    st.info("Note: RAGAS scores computed via manual implementation due to Pydantic V1/V2 incompatibility in the ragas library.")

    ragas_data = load_ragas_results()

    # Radar chart
    categories = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]
    values = [
        ragas_data["faithfulness"],
        ragas_data["answer_relevancy"],
        ragas_data["context_recall"],
        ragas_data["context_precision"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name="E-openai",
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="RAGAS Metrics (Best Config: E-openai)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation guide
    st.subheader("Metric Definitions")
    st.write("- **Faithfulness:** % of answer grounded in retrieved context (0.511 = 51% grounded)")
    st.write("- **Answer Relevancy:** % of answer directly addressing question (0.563 = 56%)")
    st.write("- **Context Recall:** % of gold chunks successfully retrieved (0.713 = 71%)")
    st.write("- **Context Precision:** % of retrieved chunks that were relevant (0.734 = 73%)")

# ---------------------------------------------------------------------------
# Page 6: LLM Judge Analysis
# ---------------------------------------------------------------------------

elif page == "⚖️ LLM Judge Analysis":
    st.title("⚖️ LLM-as-Judge Analysis")

    judge_data = load_judge_results()

    # Correctness breakdown
    correct_count = sum(1 for item in judge_data if item.get("correctness_score") is True)
    hallucination_count = sum(1 for item in judge_data if item.get("has_hallucination") is True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Correct Answers", f"{correct_count} / 56", f"{correct_count/56*100:.1f}%")
    with col2:
        st.metric("Hallucinations", f"{hallucination_count} / 56", f"{hallucination_count/56*100:.1f}%")

    st.warning("**Calibration Issue:** 22 refusals ('I don't have enough context') flagged as hallucinations. True rate: 59%.")

    # Bloom distribution
    st.subheader("Bloom Taxonomy Distribution")

    bloom_counts = {}
    for item in judge_data:
        bloom_level = item.get("bloom_level")
        if bloom_level and bloom_level != "None":
            bloom_counts[bloom_level] = bloom_counts.get(bloom_level, 0) + 1

    fig = go.Figure(go.Bar(
        x=list(bloom_counts.values()),
        y=list(bloom_counts.keys()),
        orientation="h",
        marker_color="steelblue",
    ))

    fig.update_layout(
        title="Question Distribution by Cognitive Complexity",
        xaxis_title="Number of Questions",
        yaxis_title="Bloom Level",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 7: Charts Gallery
# ---------------------------------------------------------------------------

elif page == "🖼️ Charts Gallery":
    st.title("🖼️ All Visualizations")

    chart_files = [
        ("Config × Metric Heatmap", "config_heatmap.png"),
        ("Metric Comparison", "metric_comparison.png"),
        ("BM25 vs Vector Search", "bm25_comparison.png"),
        ("Semantic vs Fixed-Size", "semantic_vs_fixed.png"),
        ("Chunk Size Effect", "chunk_size_effect.png"),
        ("Overlap Effect", "overlap_effect.png"),
        ("Embedding Comparison", "embedding_comparison.png"),
        ("Question Type Breakdown", "question_type_breakdown.png"),
        ("Reranking Impact", "reranking_impact.png"),
        ("RAGAS Radar Chart", "ragas_radar.png"),
        ("Bloom Distribution", "bloom_distribution.png"),
        ("QA Dataset Quality", "qa_quality.png"),
    ]

    for i in range(0, len(chart_files), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(chart_files):
                title, filename = chart_files[i + j]
                path = CHARTS_DIR / filename
                if path.exists():
                    with col:
                        st.subheader(title)
                        img = Image.open(path)
                        st.image(img, use_container_width=True)  # FIXED: deprecated API (was use_column_width)
                        with open(path, "rb") as f:
                            st.download_button(
                                label=f"Download {filename}",
                                data=f,
                                file_name=filename,
                                mime="image/png",
                            )
