"""Validation tracking and rejection logging for generated records.

Sits between generation and evaluation. Re-validates generated records,
tracks per-field error frequencies, computes success rates, and separates
valid records from rejected ones.

Java/TS parallel: like a validation interceptor in a Spring pipeline —
catches and categorizes errors before they reach downstream consumers.

WHY this exists separately from Instructor's built-in validation:
Instructor retries on validation failure and only returns successes.
But we need to TRACK how many retries happened and which fields caused
problems. This module re-validates saved records (e.g., after schema
changes) and logs rejections for analysis.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from src.schemas import DIYRepairRecord, GeneratedRecord

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VALIDATED_DIR = _PROJECT_ROOT / "data" / "validated"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation result tracking
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating a single record.

    Java/TS parallel: like a Result<T, E> type — wraps either a success
    value or error details, never both.
    """

    trace_id: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    # Which fields caused validation errors (for per-field frequency tracking)
    failed_fields: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregate validation metrics for a batch of records.

    PRD Section 5b: track generation success rate, first-attempt success rate,
    and per-field error frequency.
    """

    total_records: int = 0
    valid_count: int = 0
    rejected_count: int = 0
    # Counter of field_name → how many records had errors on that field
    field_error_counts: Counter = field(default_factory=Counter)

    @property
    def success_rate(self) -> float:
        """Fraction of records that passed validation (0.0 to 1.0)."""
        if self.total_records == 0:
            return 0.0
        return self.valid_count / self.total_records

    @property
    def success_rate_pct(self) -> str:
        """Human-readable success rate as percentage."""
        return f"{self.success_rate * 100:.1f}%"

    def summary(self) -> dict:
        """Return a dict summary suitable for JSON serialization."""
        return {
            "total_records": self.total_records,
            "valid_count": self.valid_count,
            "rejected_count": self.rejected_count,
            "success_rate": self.success_rate_pct,
            "field_error_frequency": dict(self.field_error_counts.most_common()),
        }


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_record(record_data: dict) -> ValidationResult:
    """Validate a single record dict against the DIYRepairRecord schema.

    Attempts to construct a DIYRepairRecord from the dict. If validation
    fails, extracts field names and error messages from the ValidationError.

    Args:
        record_data: Dict with the 7 DIYRepairRecord fields.

    Returns:
        ValidationResult with is_valid=True on success, or error details.
    """
    trace_id = record_data.get("trace_id", "unknown")

    try:
        DIYRepairRecord.model_validate(record_data)
        return ValidationResult(trace_id=trace_id, is_valid=True)
    except ValidationError as exc:
        errors = []
        failed_fields = []
        for error in exc.errors():
            # Pydantic v2 error dict has "loc" (field path tuple) and "msg"
            field_path = ".".join(str(loc) for loc in error["loc"])
            errors.append(f"{field_path}: {error['msg']}")
            # Top-level field name for frequency tracking
            if error["loc"]:
                failed_fields.append(str(error["loc"][0]))

        return ValidationResult(
            trace_id=trace_id,
            is_valid=False,
            errors=errors,
            failed_fields=failed_fields,
        )


def validate_batch(records: list[GeneratedRecord]) -> tuple[
    list[GeneratedRecord], list[GeneratedRecord], ValidationReport
]:
    """Validate a batch of GeneratedRecords and produce a report.

    Separates valid records from rejected ones and tracks per-field errors.

    Args:
        records: List of GeneratedRecord objects to validate.

    Returns:
        Tuple of (valid_records, rejected_records, report).
    """
    report = ValidationReport()
    valid: list[GeneratedRecord] = []
    rejected: list[GeneratedRecord] = []

    for gen_record in records:
        report.total_records += 1

        # Re-validate the inner DIYRepairRecord
        result = validate_record(gen_record.record.model_dump())

        if result.is_valid:
            report.valid_count += 1
            valid.append(gen_record)
        else:
            report.rejected_count += 1
            rejected.append(gen_record)
            for field_name in result.failed_fields:
                report.field_error_counts[field_name] += 1
            logger.warning(
                "Record %s rejected: %s",
                gen_record.trace_id,
                "; ".join(result.errors),
            )

    logger.info(
        "Validation complete: %d valid, %d rejected (success rate: %s)",
        report.valid_count,
        report.rejected_count,
        report.success_rate_pct,
    )
    return valid, rejected, report


# ---------------------------------------------------------------------------
# Persistence — save validated/rejected records (PRD Section 5a)
# ---------------------------------------------------------------------------

def save_validation_results(
    valid: list[GeneratedRecord],
    rejected: list[GeneratedRecord],
    report: ValidationReport,
) -> tuple[Path, Path, Path]:
    """Save validated records, rejected records, and report to data/validated/.

    Returns:
        Tuple of (valid_path, rejected_path, report_path).
    """
    _VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

    valid_path = _VALIDATED_DIR / "validated_records.json"
    valid_path.write_text(
        json.dumps([r.model_dump() for r in valid], indent=2)
    )

    rejected_path = _VALIDATED_DIR / "rejected_records.json"
    rejected_path.write_text(
        json.dumps([r.model_dump() for r in rejected], indent=2)
    )

    report_path = _VALIDATED_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report.summary(), indent=2))

    logger.info("Saved validation results to %s", _VALIDATED_DIR)
    return valid_path, rejected_path, report_path
