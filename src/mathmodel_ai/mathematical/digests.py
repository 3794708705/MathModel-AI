from __future__ import annotations

import hashlib
import json
from typing import Any

from mathmodel_ai.schemas.mathematical import MathematicalModel

_NON_SEMANTIC_KEYS = {
    "canonical_name",
    "confidence",
    "created_at",
    "derivation",
    "description",
    "display",
    "equation_id",
    "equation_ref",
    "expected_outputs",
    "first_definition",
    "latex",
    "limitations",
    "meaning",
    "name",
    "normalized_expression",
    "objective_id",
    "output_id",
    "parameter_id",
    "reason",
    "source_evidence",
    "source_ref",
    "source_refs",
    "status",
    "support_reason",
    "validation_requirements",
    "variable_id",
}


def _semantic_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _semantic_value(item)
            for key, item in sorted(value.items())
            if key not in _NON_SEMANTIC_KEYS
        }
    if isinstance(value, list):
        return [_semantic_value(item) for item in value]
    return value


def mathematical_model_digest(model: MathematicalModel) -> str:
    """Hash the executable mathematical meaning, excluding identity/audit prose."""

    payload = model.model_dump(
        mode="json",
        exclude={
            "model_id",
            "project_id",
            "problem_id",
            "version",
            "source_selected_model_id",
            "status",
        },
    )
    encoded = json.dumps(
        _semantic_value(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
