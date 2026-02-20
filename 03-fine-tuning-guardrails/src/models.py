from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CompatibilityLabel(IntEnum):
    INCOMPATIBLE = 0
    COMPATIBLE = 1


class DatingPair(BaseModel):
    text_1: str = Field(..., description="First person's statement (boy/girl: ...)")
    text_2: str = Field(..., description="Second person's statement (boy/girl: ...)")
    label: CompatibilityLabel
    category: str
    subcategory: str
    pair_type: str

    @field_validator("text_1", "text_2")
    @classmethod
    def validate_gender_prefix(cls, v: str) -> str:
        # WHY: every text must identify speaker gender — downstream bias analysis depends on this
        if ":" not in v:
            raise ValueError(f"Text must follow 'gender: statement' format, got: {v}")
        gender = v.split(":", 1)[0].strip()
        if gender not in ("boy", "girl"):
            raise ValueError(f"Gender prefix must be 'boy' or 'girl', got: {gender}")
        return v


class DataQualityScore(BaseModel):
    data_quality: float = Field(..., ge=0, le=100)
    diversity: float = Field(..., ge=0, le=100)
    bias: float = Field(..., ge=0, le=100)
    linguistic_quality: float = Field(..., ge=0, le=100)
    overall: float = Field(..., ge=0, le=100)


class DimensionDetail(BaseModel):
    """Per-dimension breakdown with sub-scores."""

    # WHY dict: number of sub-scores varies per dimension (3-4), avoids rigid schema
    sub_scores: dict[str, float]
    dimension_score: float


class DataQualityReport(BaseModel):
    scores: DataQualityScore
    # WHY dict keyed by dimension name: allows dynamic lookup without index gymnastics
    details: dict[str, DimensionDetail]
    record_count: int
    timestamp: str  # ISO format


class CategoryMetrics(BaseModel):
    category: str
    compatible_mean: float
    incompatible_mean: float
    margin: float
    count: int
    # WHY Optional: Cohen's d undefined when a group has <2 members
    cohens_d: float | None = None


class BaselineMetrics(BaseModel):
    compatible_mean_cosine: float
    incompatible_mean_cosine: float
    compatibility_margin: float
    cohens_d: float
    t_statistic: float
    p_value: float
    auc_roc: float
    best_threshold: float
    best_f1: float
    accuracy_at_best_threshold: float
    precision_at_best_threshold: float
    recall_at_best_threshold: float
    cluster_purity: float
    n_clusters: int
    noise_ratio: float
    spearman_correlation: float
    # WHY default_factory=dict: field is optional output — may be empty if no FPs exist
    false_positive_counts: dict[str, int] = Field(default_factory=dict)
    category_metrics: list[CategoryMetrics] = Field(default_factory=list)
    pair_type_metrics: list[CategoryMetrics] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    baseline: BaselineMetrics
    standard_finetuned: BaselineMetrics
    # WHY Optional: LoRA model may not be trained in every experimental run
    lora_finetuned: BaselineMetrics | None = None
    margin_improvement: float
    margin_improvement_pct: float
    cohens_d_improvement: float
    spearman_improvement: float


class TrainingResult(BaseModel):
    """Training hyperparameters, timing, and Spearman progression."""

    # WHY Literal: Pydantic validates at construction, prevents typos like "standrd"
    model_type: Literal["standard", "lora"]
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    evaluation_steps: int
    training_time_seconds: float
    final_spearman: float
    # WHY list of tuples: preserves step order, CSV has (step, spearman) pairs
    spearman_history: list[tuple[int, float]] = Field(default_factory=list)
    output_path: str
    trainable_parameters: int
    total_parameters: int
    # LoRA-specific (None for standard)
    # WHY Optional: standard training doesn't use LoRA, fields must be nullable
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_target_modules: list[str] | None = None
