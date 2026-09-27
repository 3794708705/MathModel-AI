from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from mathmodel_ai.files.storage import LocalFileStore
from mathmodel_ai.paper.assets import FigureAgent
from mathmodel_ai.paper.hashing import sha256_json
from mathmodel_ai.paper.integrity import AssetSemanticIntegrityValidator
from mathmodel_ai.paper.workflow import PaperWorkflow, PaperWorkflowError
from mathmodel_ai.schemas.paper import EvidenceRecord, EvidenceType, FigureType
from tests.paper.helpers import result_evidence


def _bundle():
    result_id = uuid4()
    replay_ids = {"s1": uuid4(), "s2": uuid4()}
    values = {"s1_response": 2.0, "s2_response": 3.0}
    scenarios = []
    for name, parameter in (("s1", "alpha"), ("s2", "beta")):
        scenarios.append(
            SimpleNamespace(
                scenario_id=name,
                parameter_values={parameter: 1.01},
                decision_values={"E": 0.5},
                metrics=[
                    SimpleNamespace(
                        metric_id=f"{name}_response",
                        required=True,
                        key=SimpleNamespace(value="final_value"),
                        series_key="J",
                        value_symbol=None,
                        quantity="final population",
                        unit="count",
                    )
                ],
            )
        )
    return SimpleNamespace(
        reviewed_plan=SimpleNamespace(result_id=result_id, scenarios=scenarios),
        context=SimpleNamespace(result=SimpleNamespace(result_id=result_id)),
        sensitivity=SimpleNamespace(reviewed_replay_ids=replay_ids, reviewed_metric_values=values),
    )


def _evidence(payload):
    base = result_evidence()
    data = base.model_dump(mode="json")
    data.update(
        evidence_id=str(uuid4()),
        evidence_type=EvidenceType.SENSITIVITY.value,
        source_type="sensitivity",
        source_id="reviewed-fixture",
        structured_payload=payload,
    )
    data["provenance"]["source_hash"] = sha256_json(payload)
    return EvidenceRecord.model_validate(data)


def test_reviewed_objective_free_figure_has_exact_replay_provenance(tmp_path):
    bundle = _bundle()
    payload = PaperWorkflow._reviewed_sensitivity_payload(bundle)
    evidence = _evidence(
        {
            "reviewed_replay_ids": {
                key: str(value) for key, value in bundle.sensitivity.reviewed_replay_ids.items()
            },
            "reviewed_metric_values": bundle.sensitivity.reviewed_metric_values,
            "experiments": [],
        }
    )
    figure, _ = FigureAgent(LocalFileStore(tmp_path)).generate(
        project_id=evidence.project_id,
        problem_id=evidence.problem_id,
        paper_id=uuid4(),
        paper_version=1,
        figure_id="FIG-001",
        title="Reviewed response",
        caption="Categorical scenarios",
        figure_type=FigureType.SENSITIVITY_CURVE,
        data_payload=payload,
        source_evidence_refs=[evidence.evidence_id],
        evidence=[evidence],
    )
    validator = AssetSemanticIntegrityValidator()
    assert validator.validate([figure], [], [evidence]) == []
    tampered = figure.model_copy(update={"data_payload": {**payload, "y": [2.0, 9.0]}})
    assert validator.validate([tampered], [], [evidence])


def test_reviewed_figure_fails_closed_without_comparable_metrics():
    bundle = _bundle()
    bundle.reviewed_plan.scenarios[1].metrics[0].quantity = "different quantity"
    with pytest.raises(PaperWorkflowError, match="no comparable reviewed metric"):
        PaperWorkflow._reviewed_sensitivity_payload(bundle)


def test_small_magnitude_figure_uses_its_data_range() -> None:
    rendered = FigureAgent._render_png(
        {
            "x": [1, 2, 3],
            "y": [-0.037, -0.036, -0.035],
            "x_label": "scenario index",
            "y_label": "dominant real part",
        }
    )
    image = Image.open(BytesIO(rendered)).convert("RGB")
    red_y = [
        row
        for row in range(image.height)
        for column in range(image.width)
        if image.getpixel((column, row)) == (214, 39, 40)
    ]
    assert max(red_y) - min(red_y) > 200
