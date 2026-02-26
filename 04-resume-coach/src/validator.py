from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ValidationTracker:
    """
    Tracks validation outcomes across the pipeline.
    Reuse of P1's ValidationReport pattern with per-model-type granularity.
    """

    # WHY: Separate dicts per model type lets us diagnose which model (Job vs Resume) fails more
    _successes: dict[str, list[str]] = field(default_factory=dict)
    _failures: dict[str, list[dict]] = field(default_factory=dict)
    _error_counts: Counter = field(default_factory=Counter)

    def record_success(self, model_type: str, trace_id: str) -> None:
        """Record a successful validation."""
        self._successes.setdefault(model_type, []).append(trace_id)

    def record_failure(
        self,
        model_type: str,
        trace_id: str,
        errors: list[dict],
    ) -> None:
        """
        Record a validation failure with structured error info.
        errors: list of dicts from Pydantic's ValidationError.errors()
        """
        self._failures.setdefault(model_type, []).append(
            {
                "trace_id": trace_id,
                "errors": errors,
            }
        )
        # WHY: Track field-level error frequency for debugging — tells us which fields LLMs mangle most
        for error in errors:
            field_path = ".".join(str(loc) for loc in error.get("loc", []))
            self._error_counts[field_path] += 1

    def get_stats(self) -> dict:
        """Return aggregate validation statistics."""
        total_success = sum(len(v) for v in self._successes.values())
        total_failure = sum(len(v) for v in self._failures.values())
        total = total_success + total_failure

        return {
            "total": total,
            "success_count": total_success,
            "failure_count": total_failure,
            "success_rate": total_success / total if total > 0 else 0.0,
            "by_model_type": {
                model_type: {
                    "successes": len(self._successes.get(model_type, [])),
                    "failures": len(self._failures.get(model_type, [])),
                }
                for model_type in set(
                    list(self._successes.keys()) + list(self._failures.keys())
                )
            },
            "errors_by_field": dict(self._error_counts.most_common()),
        }

    def save_stats(self, output_path: Path) -> None:
        """Save validation stats to JSON file.

        Writes all internal keys plus the PRD-required aliases:
          valid / invalid / field_errors / timestamp
        WHY aliases: downstream consumers (analyzer, tests) expect the PRD schema;
        keeping the original keys avoids breaking existing tests.
        """
        stats = self.get_stats()
        # WHY: add PRD-schema aliases so downstream files match spec
        output = {
            **stats,
            "valid": stats["success_count"],
            "invalid": stats["failure_count"],
            "field_errors": stats["errors_by_field"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))
        logger.info("Saved validation stats to %s", output_path)

    @staticmethod
    def save_valid(record: BaseModel, output_dir: Path) -> None:
        """Append a valid record to the validated JSONL file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "validated.jsonl"
        with open(filepath, "a") as f:
            f.write(json.dumps(record.model_dump()) + "\n")

    @staticmethod
    def save_invalid(
        record_json: str | dict,
        errors: list[dict],
        output_dir: Path,
    ) -> None:
        """Append an invalid record with errors to the invalid JSONL file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "invalid.jsonl"
        entry = {
            "raw_data": json.loads(record_json) if isinstance(record_json, str) else record_json,
            "errors": errors,
        }
        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")
