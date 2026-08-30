from __future__ import annotations

from typing import ClassVar

from mathmodel_ai.schemas.submission import CorrectionPlan, CorrectionScope, InvalidationResult


class InvalidationEngine:
    _SAFE_FORMAT_COMPONENTS: ClassVar[set[str]] = {
        "spacing",
        "font",
        "line_break",
        "pagination",
        "filename",
        "reference_ordering",
    }
    _MODEL_COMPONENT_MARKERS: ClassVar[set[str]] = {
        "equation",
        "parameter",
        "constraint",
        "model",
        "objective",
    }
    _STAGES: ClassVar[dict[CorrectionScope, list[str]]] = {
        CorrectionScope.FORMAT_ONLY: ["PAPER", "FINAL_JURY", "SUBMISSION"],
        CorrectionScope.PAPER_ONLY: ["PAPER", "FINAL_JURY", "SUBMISSION"],
        CorrectionScope.CITATION: ["PAPER", "FINAL_JURY", "SUBMISSION"],
        CorrectionScope.ARTIFACT: ["PAPER", "FINAL_JURY", "SUBMISSION"],
        CorrectionScope.CODE: ["SOLVE", "VALIDATE", "PAPER", "FINAL_JURY", "SUBMISSION"],
        CorrectionScope.MODEL: [
            "MODEL",
            "SOLVE",
            "VALIDATE",
            "SENSITIVITY",
            "ROBUSTNESS",
            "RED_TEAM",
            "PAPER",
            "FINAL_JURY",
            "SUBMISSION",
        ],
        CorrectionScope.DATA: [
            "DATA",
            "MODEL",
            "SOLVE",
            "VALIDATE",
            "PAPER",
            "FINAL_JURY",
            "SUBMISSION",
        ],
    }

    def evaluate(self, plan: CorrectionPlan) -> InvalidationResult:
        stages = self._STAGES[plan.scope]
        components = {
            item.strip().casefold().replace("-", "_") for item in plan.affected_components
        }
        if (
            plan.scope is CorrectionScope.FORMAT_ONLY
            and not components <= self._SAFE_FORMAT_COMPONENTS
        ):
            raise ValueError(
                "HUMAN_REVIEW: FORMAT_ONLY classification is not deterministically safe"
            )
        if plan.scope is not CorrectionScope.MODEL and any(
            any(marker in component for marker in self._MODEL_COMPONENT_MARKERS)
            for component in components
        ):
            raise ValueError("model-semantic correction must invalidate MODEL")
        if plan.requires_model_change and "MODEL" not in stages:
            raise ValueError("model-changing correction must invalidate MODEL")
        if plan.requires_result_change and "SOLVE" not in stages:
            raise ValueError("result-changing correction must invalidate SOLVE")
        if plan.requires_paper_change and "PAPER" not in stages:
            raise ValueError("paper-changing correction must invalidate PAPER")
        if not set(plan.required_revalidation) <= set(stages):
            raise ValueError("correction revalidation exceeds its deterministic scope")
        return InvalidationResult(
            invalidated_stages=stages,
            jury_recheck_required=True,
            submission_dirty=True,
        )


class CorrectionWorkflow:
    _AUTO_FIXABLE: ClassVar[set[CorrectionScope]] = {CorrectionScope.FORMAT_ONLY}

    def validate_auto_fix(self, plan: CorrectionPlan) -> bool:
        return (
            plan.scope in self._AUTO_FIXABLE
            and not plan.requires_model_change
            and not plan.requires_result_change
            and {item.strip().casefold().replace("-", "_") for item in plan.affected_components}
            <= InvalidationEngine._SAFE_FORMAT_COMPONENTS
        )
