# ruff: noqa: E501
"""Build the reviewed-input Case A model contract without running a benchmark.

This command is intentionally benchmark-authoring code. It reads the exact
rerun-13 model and current ProblemState, creates an immutable version-2 model
artifact, and emits a DRAFT policy candidate. It never writes to the database.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine, text

from mathmodel_ai.benchmark.manifests import BenchmarkManifestRegistry
from mathmodel_ai.mathematical.digests import mathematical_model_digest
from mathmodel_ai.schemas.mathematical import MathematicalModel
from mathmodel_ai.verification.metric_recompute import content_digest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "case-003-mcm-2024-a"
SOURCE_RECORD_ID = "cf55e323-9cc5-45b6-a974-f6e1010771fc"
SOURCE_MODEL_DIGEST = "3334b664351b303aa89a0fd5e6efb2eec8862df18b579dab9f07268feb0c63cf"
PROBLEM_SHA256 = "3c69f8f3b56c9aeebc6ff45ab70b4e97f68ca7bd779923edab16ac61e36ce75a"
ASSUMPTION_PARAMETERS = (
    "sex_ratio_shape",
    "cap_min",
    "cap_slope",
    "birth_coef",
    "maturation",
    "mort_larva",
    "mort_adult",
    "fish_growth",
    "fish_capacity",
    "attack_rate",
    "parasite_growth",
    "parasite_capacity",
    "fish_benefit_par",
    "lamprey_cost_par",
)
PARAMETER_ID_BY_SYMBOL = {
    "sex_ratio_shape": "PAR-GAMMA",
    "cap_min": "PAR-CAPMIN",
    "cap_slope": "PAR-CAPSLOPE",
    "birth_coef": "PAR-BIRTH",
    "maturation": "PAR-MAT",
    "mort_larva": "PAR-MUJ",
    "mort_adult": "PAR-MUA",
    "fish_growth": "PAR-RF",
    "fish_capacity": "PAR-KF",
    "attack_rate": "PAR-ATTACK",
    "parasite_growth": "PAR-RP",
    "parasite_capacity": "PAR-KP",
    "fish_benefit_par": "PAR-BPF",
    "lamprey_cost_par": "PAR-PA",
    "adaptation_weight": "PAR-ADAPT",
    "sex_low": "CONST-SEXLOW",
    "sex_high": "CONST-SEXHIGH",
    "J0": "CONST-J0",
    "A0": "CONST-A0",
    "F0": "CONST-F0",
    "P0": "CONST-P0",
}
REFERENCE_STATE = {
    "J": 0.17386363636363633,
    "A": 0.13909090909090907,
    "F": 0.9304545454545454,
    "P": 3.3421212121212127,
}
TIME_STOP = 153.50567286626972


def sym(name: str) -> dict[str, Any]:
    return {"kind": "SYMBOL", "value": None, "symbol": name, "operands": []}


def number(value: float) -> dict[str, Any]:
    return {"kind": "CONSTANT", "value": value, "symbol": None, "operands": []}


def op(kind: str, *operands: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "value": None, "symbol": None, "operands": list(operands)}


def add(*operands: dict[str, Any]) -> dict[str, Any]:
    return op("ADD", *operands)


def sub(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return op("SUBTRACT", left, right)


def mul(*operands: dict[str, Any]) -> dict[str, Any]:
    return op("MULTIPLY", *operands)


def div(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return op("DIVIDE", left, right)


def power(base: dict[str, Any], exponent: dict[str, Any]) -> dict[str, Any]:
    return op("POWER", base, exponent)


def neg(value: dict[str, Any]) -> dict[str, Any]:
    return op("NEGATE", value)


def dimensionless(display: str = "1") -> dict[str, Any]:
    return {"dimensions": {}, "scale": 1.0, "display": display, "unknown_units": []}


def variable(
    variable_id: str,
    symbol: str,
    description: str,
    *,
    domain: str = "CONTINUOUS",
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, Any]:
    return {
        "variable_id": variable_id,
        "symbol": symbol,
        "description": description,
        "index_sets": [],
        "domain": domain,
        "lower_bound": lower,
        "upper_bound": upper,
        "unit": dimensionless("normalized dimensionless scalar"),
        "role": "DERIVED",
        "source_refs": ["ASSUMP-NORM", "ASSUMP-STABILITY-V2"],
    }


def parameter(
    parameter_id: str,
    symbol: str,
    value: float,
    description: str,
    *,
    source_type: str,
    source_ref: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "parameter_id": parameter_id,
        "symbol": symbol,
        "description": description,
        "value": value,
        "data_binding": None,
        "unit": dimensionless("normalized dimensionless scalar"),
        "source_type": source_type,
        "source_ref": source_ref,
        "is_estimated": False,
        "estimation_method": None,
        "confidence": confidence,
    }


def equation(
    equation_id: str,
    lhs: str,
    rhs: dict[str, Any],
    normalized: str,
    meaning: str,
    derivation: str,
    *,
    source_refs: list[str],
    symbol_refs: list[str],
    parameter_refs: list[str] | None = None,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "equation_id": equation_id,
        "latex": normalized,
        "normalized_expression": normalized,
        "lhs": sym(lhs),
        "rhs": rhs,
        "meaning": meaning,
        "source_refs": source_refs,
        "derivation": derivation,
        "symbol_refs": symbol_refs,
        "parameter_refs": (
            parameter_refs
            if parameter_refs is not None
            else [
                PARAMETER_ID_BY_SYMBOL[item]
                for item in symbol_refs
                if item in PARAMETER_ID_BY_SYMBOL
            ]
        ),
        "dependency_refs": dependencies or [],
        "unit_lhs": dimensionless(),
        "unit_rhs": dimensionless(),
        "dimension_status": "PASS",
    }


def constraint(
    constraint_id: str,
    name: str,
    left: str,
    relation: str,
    right: dict[str, Any],
    normalized: str,
    description: str,
    equation_ref: str,
    source_refs: list[str],
) -> dict[str, Any]:
    return {
        "constraint_id": constraint_id,
        "name": name,
        "expression": sym(left),
        "relation": relation,
        "rhs": right,
        "normalized_expression": normalized,
        "description": description,
        "source_refs": source_refs,
        "assumption_refs": ["ASSUMP-INIT-V2"],
        "unit": dimensionless(),
        "index_scope": [],
        "is_hard": True,
        "equation_ref": equation_ref,
    }


def _read_source() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], int]:
    url = os.getenv(
        "MM_DATABASE_URL",
        "postgresql+psycopg://mathmodel:mathmodel@localhost:5432/mathmodel",
    )
    engine = create_engine(url)
    with engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    "SELECT model_json, model_digest FROM mathematical_models WHERE id = :record_id"
                ),
                {"record_id": SOURCE_RECORD_ID},
            )
            .mappings()
            .one()
        )
        if row["model_digest"] != SOURCE_MODEL_DIGEST:
            raise ValueError("rerun-13 source model digest changed")
        source = dict(row["model_json"])
        state = connection.execute(
            text(
                "SELECT state FROM problem_states WHERE problem_id = :problem_id "
                "AND is_current = true"
            ),
            {"problem_id": source["problem_id"]},
        ).scalar_one()
        evidence = list(
            connection.execute(
                text(
                    "SELECT r.id AS result_id, r.solver_run_id, r.execution_record_id, "
                    "r.status AS result_status, s.generated_program_id, "
                    "g.code_hash AS program_code_hash, e.code_hash AS execution_code_hash, "
                    "e.is_mock, e.status AS execution_status "
                    "FROM results r JOIN solver_runs s ON s.id = r.solver_run_id "
                    "LEFT JOIN generated_programs g ON g.id = s.generated_program_id "
                    "JOIN execution_records e ON e.id = r.execution_record_id "
                    "WHERE r.mathematical_model_record_id = :record_id "
                    "ORDER BY r.created_at"
                ),
                {"record_id": SOURCE_RECORD_ID},
            ).mappings()
        )
        literature_count = connection.execute(
            text("SELECT count(*) FROM literature_references WHERE project_id = :project_id"),
            {"project_id": source["project_id"]},
        ).scalar_one()
    serialized_evidence = [
        {key: str(value) if isinstance(value, UUID) else value for key, value in item.items()}
        for item in evidence
    ]
    return source, dict(state), serialized_evidence, int(literature_count)


def _assumption(assumption_id: str, statement: str, support_reason: str) -> dict[str, Any]:
    return {
        "assumption_id": assumption_id,
        "statement": statement,
        "source_refs": ["RESOURCE-A-PROBLEM", SOURCE_MODEL_DIGEST],
        "critical": False,
        "supported": False,
        "support_reason": support_reason,
    }


def _model_v2(source: dict[str, Any]) -> MathematicalModel:
    model = json.loads(json.dumps(source))
    model.update(
        {
            "version": 2,
            "status": "READY",
            "name": "Resource-dependent sea lamprey sex ratio: reviewed coupled ODE contract",
            "description": (
                "Autonomous four-state normalized ecosystem ODE with an explicit adaptive-versus-"
                "fixed sex-ratio mechanism, reproducible initial state, Jacobian spectrum, and "
                "paper-visible assumption boundaries."
            ),
        }
    )
    for item in [
        *model["decision_variables"],
        *model["state_variables"],
        *model["derived_variables"],
        *model["parameters"],
        *model["constants"],
    ]:
        item["unit"] = dimensionless("normalized dimensionless scalar")
    for item in model["state_variables"]:
        item["lower_bound"] = 0.0
    model["decision_variables"][0]["unit"] = dimensionless("normalized food index")
    model["parameters"].append(
        parameter(
            "PAR-ADAPT",
            "adaptation_weight",
            1.0,
            "Mechanism switch: 1 uses the resource-adaptive sex ratio and 0 uses the fixed comparator.",
            source_type="ASSUMPTION",
            source_ref="ASSUMP-COMPARATOR-V2",
            confidence=1.0,
        )
    )
    model["constants"].extend(
        [
            *[
                parameter(
                    f"CONST-{symbol}0",
                    f"{symbol}0",
                    value,
                    f"{symbol} component of the derived positive reference equilibrium at E_ref.",
                    source_type="DERIVATION",
                    source_ref="ASSUMP-INIT-V2",
                    confidence=1.0,
                )
                for symbol, value in REFERENCE_STATE.items()
            ],
        ]
    )
    extra_variables = [
        variable(
            "VAR-FIXED-MALE",
            "fixed_male_ratio",
            "Fixed comparator male fraction, defined as the midpoint of the two approximate anchors.",
            lower=0.0,
            upper=1.0,
        ),
        variable("VAR-JJ", "jac_JJ", "Jacobian partial derivative of dJ_dt with respect to J."),
        variable("VAR-JA", "jac_JA", "Jacobian partial derivative of dJ_dt with respect to A."),
        variable("VAR-AJ", "jac_AJ", "Jacobian partial derivative of dA_dt with respect to J."),
        variable("VAR-AA", "jac_AA", "Jacobian partial derivative of dA_dt with respect to A."),
        variable("VAR-FA", "jac_FA", "Jacobian partial derivative of dF_dt with respect to A."),
        variable("VAR-FF", "jac_FF", "Jacobian partial derivative of dF_dt with respect to F."),
        variable("VAR-PA", "jac_PA", "Jacobian partial derivative of dP_dt with respect to A."),
        variable("VAR-PF", "jac_PF", "Jacobian partial derivative of dP_dt with respect to F."),
        variable("VAR-PP", "jac_PP", "Jacobian partial derivative of dP_dt with respect to P."),
        variable("VAR-TRACE-JA", "trace_JA", "Trace of the larva-adult Jacobian block."),
        variable("VAR-DET-JA", "det_JA", "Determinant of the larva-adult Jacobian block."),
        variable(
            "VAR-DISC-JA",
            "disc_JA",
            "Nonnegative discriminant of the real larva-adult eigenvalue pair.",
            domain="NONNEGATIVE_CONTINUOUS",
            lower=0.0,
        ),
        variable("VAR-LAMBDA-JA-PLUS", "lambda_JA_plus", "Larger larva-adult eigenvalue."),
        variable("VAR-LAMBDA-JA-MINUS", "lambda_JA_minus", "Smaller larva-adult eigenvalue."),
        variable(
            "VAR-LAMBDA-F",
            "lambda_F",
            "Fish-direction eigenvalue of the block-triangular Jacobian.",
        ),
        variable(
            "VAR-LAMBDA-P",
            "lambda_P",
            "Parasite-direction eigenvalue of the block-triangular Jacobian.",
        ),
        variable(
            "VAR-DOMINANT-REAL",
            "dominant_real_part",
            "Maximum real part among the four explicitly derived real Jacobian eigenvalues.",
        ),
        variable(
            "VAR-EQUILIBRIUM-RESIDUAL",
            "equilibrium_residual_sq",
            "Squared Euclidean norm of the four ODE derivatives at the evaluated state.",
            domain="NONNEGATIVE_CONTINUOUS",
            lower=0.0,
        ),
    ]
    model["derived_variables"].extend(extra_variables)

    fixed_ratio = div(add(sym("sex_low"), sym("sex_high")), number(2.0))
    adaptive_ratio = sub(
        sym("sex_low"),
        mul(
            sub(sym("sex_low"), sym("sex_high")),
            power(sym("E"), sym("sex_ratio_shape")),
        ),
    )
    male_ratio = add(
        mul(sym("adaptation_weight"), adaptive_ratio),
        mul(sub(number(1.0), sym("adaptation_weight")), sym("fixed_male_ratio")),
    )
    original = {item["equation_id"]: item for item in model["equations"]}
    original["EQ-SEXRULE"].update(
        {
            "rhs": male_ratio,
            "latex": "male_ratio = alpha adaptive_ratio(E) + (1-alpha) fixed_male_ratio",
            "normalized_expression": (
                "male_ratio = adaptation_weight*(sex_low-(sex_low-sex_high)*"
                "E^sex_ratio_shape)+(1-adaptation_weight)*fixed_male_ratio"
            ),
            "meaning": (
                "One formal mechanism parameter selects the adaptive treatment or fixed comparator "
                "without changing any demographic or interaction coefficient."
            ),
            "derivation": (
                "The adaptive branch preserves the two stated approximate data anchors; the fixed "
                "branch uses their midpoint as a transparent neutral comparator assumption."
            ),
            "source_refs": ["EVID-DATA-1", "EVID-DATA-2", "ASSUMP-COMPARATOR-V2"],
            "symbol_refs": [
                "male_ratio",
                "adaptation_weight",
                "sex_low",
                "sex_high",
                "E",
                "sex_ratio_shape",
                "fixed_male_ratio",
            ],
            "parameter_refs": [
                "PAR-ADAPT",
                "CONST-SEXLOW",
                "CONST-SEXHIGH",
                "PAR-GAMMA",
            ],
            "dependency_refs": ["EQ-FIXED-RATIO"],
        }
    )
    dependencies = {
        "EQ-FEMRATIO": ["EQ-SEXRULE"],
        "EQ-JUVCAP": [],
        "EQ-JDYN": ["EQ-FEMRATIO", "EQ-JUVCAP"],
        "EQ-ADYN": [],
        "EQ-FDYN": [],
        "EQ-PDYN": [],
    }
    for item in original.values():
        item["unit_lhs"] = dimensionless()
        item["unit_rhs"] = dimensionless()
        item["dimension_status"] = "PASS"
        item["dependency_refs"] = dependencies.get(item["equation_id"], item["dependency_refs"])

    equations = [
        equation(
            "EQ-FIXED-RATIO",
            "fixed_male_ratio",
            fixed_ratio,
            "fixed_male_ratio = (sex_low + sex_high) / 2",
            "Fixed-ratio comparator shared by all paired structural scenarios.",
            "Arithmetic midpoint is an explicit comparator design choice, not an observed third datum.",
            source_refs=["EVID-DATA-1", "EVID-DATA-2", "ASSUMP-COMPARATOR-V2"],
            symbol_refs=["fixed_male_ratio", "sex_low", "sex_high"],
            parameter_refs=["CONST-SEXLOW", "CONST-SEXHIGH"],
        ),
        *original.values(),
    ]
    equations.extend(_analysis_equations())
    for symbol in REFERENCE_STATE:
        equations.append(
            equation(
                f"EQ-INIT-{symbol}",
                symbol,
                sym(f"{symbol}0"),
                f"{symbol}(0) = {symbol}0",
                f"Initial value for state {symbol} in state ordering [J,A,F,P].",
                "Derived positive interior equilibrium at E_ref=0.5 under baseline assumptions.",
                source_refs=["ASSUMP-INIT-V2"],
                symbol_refs=[symbol, f"{symbol}0"],
                parameter_refs=[f"CONST-{symbol}0"],
            )
        )
    equations.extend(
        [
            equation(
                "EQ-ALPHA-LOW",
                "adaptation_weight",
                number(0.0),
                "adaptation_weight >= 0",
                "Lower domain boundary of the mechanism switch.",
                "Comparator design domain.",
                source_refs=["ASSUMP-COMPARATOR-V2"],
                symbol_refs=["adaptation_weight"],
            ),
            equation(
                "EQ-ALPHA-HIGH",
                "adaptation_weight",
                number(1.0),
                "adaptation_weight <= 1",
                "Upper domain boundary of the mechanism switch.",
                "Comparator design domain.",
                source_refs=["ASSUMP-COMPARATOR-V2"],
                symbol_refs=["adaptation_weight"],
            ),
        ]
    )
    model["equations"] = equations
    model["constraints"] = [
        constraint(
            "CON-ALPHA-LOW",
            "mechanism switch lower bound",
            "adaptation_weight",
            "GE",
            number(0.0),
            "adaptation_weight >= 0",
            "The mechanism interpolation cannot be below the fixed comparator.",
            "EQ-ALPHA-LOW",
            ["ASSUMP-COMPARATOR-V2"],
        ),
        constraint(
            "CON-ALPHA-HIGH",
            "mechanism switch upper bound",
            "adaptation_weight",
            "LE",
            number(1.0),
            "adaptation_weight <= 1",
            "The mechanism interpolation cannot exceed the adaptive branch.",
            "EQ-ALPHA-HIGH",
            ["ASSUMP-COMPARATOR-V2"],
        ),
    ]
    model["initial_conditions"] = [
        constraint(
            f"CON-INIT-{symbol}",
            f"initial {symbol}",
            symbol,
            "EQ",
            sym(f"{symbol}0"),
            f"{symbol}(0) = {symbol}0",
            f"Normalized reference-state initial condition for {symbol}.",
            f"EQ-INIT-{symbol}",
            ["ASSUMP-INIT-V2"],
        )
        for symbol in REFERENCE_STATE
    ]
    model["boundary_conditions"] = []
    model["assumptions"].extend(
        [
            _assumption(
                "ASSUMP-COMPARATOR-V2",
                "The fixed-ratio comparator uses the midpoint 0.67 of the two approximate male-ratio anchors and differs from the adaptive model only through adaptation_weight=0 rather than 1.",
                "A midpoint comparator is neutral on the normalized two-anchor interval and preserves all non-mechanism inputs for causal comparison.",
            ),
            _assumption(
                "ASSUMP-INIT-V2",
                "The ordered initial state [J,A,F,P] is the derived positive interior equilibrium at E_ref=0.5 for the baseline adaptive model; it is a normalized reference state, not an official field observation.",
                "A common interior equilibrium avoids arbitrary absolute abundance and gives every paired scenario the same reproducible starting point; endpoint conclusions remain scenario- and initial-state-dependent.",
            ),
            _assumption(
                "ASSUMP-TIME-V2",
                "Simulation time is tau=t/T_ref with unknown physical reference duration T_ref; horizon and sampling are numerical analysis choices and cannot be converted to calendar time without external evidence.",
                "The problem supplies no physical time scale, so nondimensionalization is the only traceable executable contract.",
            ),
            _assumption(
                "ASSUMP-SENSITIVITY-V2",
                "Local sensitivity uses a one-sided +1% finite change for each of the 14 provisional ecological parameters, one at a time, from the same reference state.",
                "The perturbation is an analysis design choice large relative to solver tolerance and small relative to baseline scale; it is not a biological uncertainty range.",
            ),
            _assumption(
                "ASSUMP-STABILITY-V2",
                "Stability means local asymptotic stability of an equilibrium reached by the autonomous fixed-E ODE. It is classified from the independently recomputed Jacobian eigenvalues only after the squared derivative residual satisfies the numerical equilibrium criterion.",
                "This is mathematically matched to the autonomous ODE and does not invent a biological percentage threshold.",
            ),
            _assumption(
                "ASSUMP-PERSISTENCE-V2",
                "Mathematical persistence is reported only when the limiting equilibrium component is strictly positive; no practical-extinction cutoff or separate finite-horizon resilience claim is authorized.",
                "Continuous ODE states need not reach exact zero, and the problem supplies no defensible practical extinction or recovery threshold.",
            ),
        ]
    )
    for resolution in model["interpretation_resolutions"]:
        if resolution["ambiguity_id"] == "AMB-1":
            resolution.update(
                {
                    "status": "ASSUMPTION_REQUIRED",
                    "critical": False,
                    "interpretation_id": "INT-AMB1-1",
                    "reason": (
                        "P is explicitly the normalized abundance of a representative non-lamprey "
                        "fish-parasite guild. This is a paper-visible modeling assumption; it is not "
                        "automatically inferred by the verifier."
                    ),
                    "source_refs": ["ASSUMP-PARASITE"],
                }
            )
        if resolution["ambiguity_id"] == "AMB-3":
            resolution.update(
                {
                    "status": "ASSUMPTION_REQUIRED",
                    "critical": False,
                    "interpretation_id": "INT-AMB3-1",
                    "reason": (
                        "The contract adopts local asymptotic equilibrium stability and mathematical "
                        "persistence; it forbids an unsourced practical-extinction or recovery claim."
                    ),
                    "source_refs": ["ASSUMP-STABILITY-V2", "ASSUMP-PERSISTENCE-V2"],
                }
            )
    model["algorithm_requirements"].update(
        {
            "deterministic_required": True,
            "supports_local_solution": True,
            "numerical_tolerance": 1e-7,
            "notes": [
                "Integrate the formal ODE AST with RK45, rtol=1e-8, atol=1e-10.",
                "Require all 401 requested samples on tau in [0,153.50567286626972].",
                "Classify local stability from independently recomputed eigenvalues at a numerical equilibrium.",
            ],
        }
    )
    model["solver_requirements"].update(
        {
            "requires_continuous": True,
            "requires_nonlinear": True,
            "maximum_runtime_seconds": 30.0,
            "required_capabilities": ["CONTINUOUS"],
            "preferred_solver_families": ["SCIPY_MINIMIZE"],
        }
    )
    for item in model["expected_outputs"]:
        item["unit"] = dimensionless("normalized dimensionless scalar")
    model["validation_requirements"] = [
        "Recompute both sex-ratio anchor identities from the formal AST.",
        "Replay adaptive and fixed-ratio treatments through the same isolated RK45 path and common initial state.",
        "Recompute every Jacobian entry, all four real eigenvalues, dominant_real_part, and equilibrium_residual_sq from final raw state values.",
        "Run one-at-a-time +1% local sensitivity for each of the 14 provisional ecological parameters; do not describe these as biological ranges.",
        "Report local stability from dominant_real_part only when equilibrium_residual_sq <= 1e-10; do not require a favorable sign for verification PASS.",
        "Report mathematical persistence from positive limiting components; do not claim practical extinction or finite-horizon resilience without new evidence.",
        "Keep P's representative non-lamprey fish-parasite-guild interpretation explicit in all paper claims.",
    ]
    model["limitations"] = [
        "The 14 ecological coefficients are dimensionless modeling assumptions, not empirical estimates.",
        "The physical duration T_ref is unavailable, so tau cannot be converted to calendar time.",
        "Local asymptotic stability does not establish global stability or ecological resilience after arbitrary disturbances.",
        "The common reference equilibrium improves comparator fairness but endpoint trajectories can remain initial-condition dependent.",
        "P is a representative non-lamprey fish-parasite abundance, not prevalence, burden, or infection pressure.",
    ]
    all_symbols = [
        item["symbol"]
        for item in [
            *model["decision_variables"],
            *model["state_variables"],
            *model["derived_variables"],
            *model["parameters"],
            *model["constants"],
        ]
    ]
    model["units"] = {symbol: dimensionless() for symbol in ["tau", *all_symbols]}
    return MathematicalModel.model_validate(model)


def _analysis_equations() -> list[dict[str, Any]]:
    ja_sqrt = power(sym("disc_JA"), number(0.5))
    max_fp = div(
        add(
            sym("lambda_F"),
            sym("lambda_P"),
            power(power(sub(sym("lambda_F"), sym("lambda_P")), number(2.0)), number(0.5)),
        ),
        number(2.0),
    )
    dominant = div(
        add(
            sym("lambda_JA_plus"),
            max_fp,
            power(power(sub(sym("lambda_JA_plus"), max_fp), number(2.0)), number(0.5)),
        ),
        number(2.0),
    )
    specs = [
        (
            "EQ-JAC-JJ",
            "jac_JJ",
            neg(
                add(
                    div(mul(sym("birth_coef"), sym("female_ratio"), sym("A")), sym("juv_capacity")),
                    sym("mort_larva"),
                )
            ),
            "jac_JJ = -birth_coef*female_ratio*A/juv_capacity-mort_larva",
            ["birth_coef", "female_ratio", "A", "juv_capacity", "mort_larva"],
        ),
        (
            "EQ-JAC-JA",
            "jac_JA",
            mul(
                sym("birth_coef"),
                sym("female_ratio"),
                sub(number(1.0), div(sym("J"), sym("juv_capacity"))),
            ),
            "jac_JA = birth_coef*female_ratio*(1-J/juv_capacity)",
            ["birth_coef", "female_ratio", "J", "juv_capacity"],
        ),
        ("EQ-JAC-AJ", "jac_AJ", sym("maturation"), "jac_AJ = maturation", ["maturation"]),
        ("EQ-JAC-AA", "jac_AA", neg(sym("mort_adult")), "jac_AA = -mort_adult", ["mort_adult"]),
        (
            "EQ-JAC-FA",
            "jac_FA",
            neg(mul(sym("attack_rate"), sym("F"))),
            "jac_FA = -attack_rate*F",
            ["attack_rate", "F"],
        ),
        (
            "EQ-JAC-FF",
            "jac_FF",
            sub(
                mul(
                    sym("fish_growth"),
                    sub(number(1.0), div(mul(number(2.0), sym("F")), sym("fish_capacity"))),
                ),
                mul(sym("attack_rate"), sym("A")),
            ),
            "jac_FF = fish_growth*(1-2*F/fish_capacity)-attack_rate*A",
            ["fish_growth", "F", "fish_capacity", "attack_rate", "A"],
        ),
        (
            "EQ-JAC-PA",
            "jac_PA",
            neg(mul(sym("lamprey_cost_par"), sym("P"))),
            "jac_PA = -lamprey_cost_par*P",
            ["lamprey_cost_par", "P"],
        ),
        (
            "EQ-JAC-PF",
            "jac_PF",
            mul(sym("fish_benefit_par"), sym("P")),
            "jac_PF = fish_benefit_par*P",
            ["fish_benefit_par", "P"],
        ),
        (
            "EQ-JAC-PP",
            "jac_PP",
            add(
                mul(
                    sym("parasite_growth"),
                    sub(number(1.0), div(mul(number(2.0), sym("P")), sym("parasite_capacity"))),
                ),
                mul(sym("fish_benefit_par"), sym("F")),
                neg(mul(sym("lamprey_cost_par"), sym("A"))),
            ),
            "jac_PP = parasite_growth*(1-2*P/parasite_capacity)+fish_benefit_par*F-lamprey_cost_par*A",
            [
                "parasite_growth",
                "P",
                "parasite_capacity",
                "fish_benefit_par",
                "F",
                "lamprey_cost_par",
                "A",
            ],
        ),
        (
            "EQ-TRACE-JA",
            "trace_JA",
            add(sym("jac_JJ"), sym("jac_AA")),
            "trace_JA = jac_JJ + jac_AA",
            ["jac_JJ", "jac_AA"],
        ),
        (
            "EQ-DET-JA",
            "det_JA",
            sub(mul(sym("jac_JJ"), sym("jac_AA")), mul(sym("jac_JA"), sym("jac_AJ"))),
            "det_JA = jac_JJ*jac_AA-jac_JA*jac_AJ",
            ["jac_JJ", "jac_AA", "jac_JA", "jac_AJ"],
        ),
        (
            "EQ-DISC-JA",
            "disc_JA",
            sub(power(sym("trace_JA"), number(2.0)), mul(number(4.0), sym("det_JA"))),
            "disc_JA = trace_JA^2-4*det_JA",
            ["trace_JA", "det_JA"],
        ),
        (
            "EQ-LAMBDA-JA-PLUS",
            "lambda_JA_plus",
            div(add(sym("trace_JA"), ja_sqrt), number(2.0)),
            "lambda_JA_plus = (trace_JA+sqrt(disc_JA))/2",
            ["trace_JA", "disc_JA"],
        ),
        (
            "EQ-LAMBDA-JA-MINUS",
            "lambda_JA_minus",
            div(sub(sym("trace_JA"), ja_sqrt), number(2.0)),
            "lambda_JA_minus = (trace_JA-sqrt(disc_JA))/2",
            ["trace_JA", "disc_JA"],
        ),
        ("EQ-LAMBDA-F", "lambda_F", sym("jac_FF"), "lambda_F = jac_FF", ["jac_FF"]),
        ("EQ-LAMBDA-P", "lambda_P", sym("jac_PP"), "lambda_P = jac_PP", ["jac_PP"]),
        (
            "EQ-DOMINANT-REAL",
            "dominant_real_part",
            dominant,
            "dominant_real_part = max(lambda_JA_plus,lambda_JA_minus,lambda_F,lambda_P)",
            ["lambda_JA_plus", "lambda_JA_minus", "lambda_F", "lambda_P"],
        ),
        (
            "EQ-EQUILIBRIUM-RESIDUAL",
            "equilibrium_residual_sq",
            add(
                power(sym("dJ_dt"), number(2.0)),
                power(sym("dA_dt"), number(2.0)),
                power(sym("dF_dt"), number(2.0)),
                power(sym("dP_dt"), number(2.0)),
            ),
            "equilibrium_residual_sq = dJ_dt^2+dA_dt^2+dF_dt^2+dP_dt^2",
            ["dJ_dt", "dA_dt", "dF_dt", "dP_dt"],
        ),
    ]
    dependencies = {
        "EQ-JAC-JJ": ["EQ-FEMRATIO", "EQ-JUVCAP"],
        "EQ-JAC-JA": ["EQ-FEMRATIO", "EQ-JUVCAP"],
        "EQ-TRACE-JA": ["EQ-JAC-JJ", "EQ-JAC-AA"],
        "EQ-DET-JA": ["EQ-JAC-JJ", "EQ-JAC-AA", "EQ-JAC-JA", "EQ-JAC-AJ"],
        "EQ-DISC-JA": ["EQ-TRACE-JA", "EQ-DET-JA"],
        "EQ-LAMBDA-JA-PLUS": ["EQ-TRACE-JA", "EQ-DISC-JA"],
        "EQ-LAMBDA-JA-MINUS": ["EQ-TRACE-JA", "EQ-DISC-JA"],
        "EQ-LAMBDA-F": ["EQ-JAC-FF"],
        "EQ-LAMBDA-P": ["EQ-JAC-PP"],
        "EQ-DOMINANT-REAL": [
            "EQ-LAMBDA-JA-PLUS",
            "EQ-LAMBDA-JA-MINUS",
            "EQ-LAMBDA-F",
            "EQ-LAMBDA-P",
        ],
        "EQ-EQUILIBRIUM-RESIDUAL": ["EQ-JDYN", "EQ-ADYN", "EQ-FDYN", "EQ-PDYN"],
    }
    return [
        equation(
            equation_id,
            lhs,
            rhs,
            normalized,
            f"Deterministic Jacobian/equilibrium quantity {lhs}.",
            "Symbolic differentiation or algebraic derivation from the formal autonomous ODE.",
            source_refs=["ASSUMP-STABILITY-V2"],
            symbol_refs=[lhs, *refs],
            dependencies=dependencies.get(equation_id, []),
        )
        for equation_id, lhs, rhs, normalized, refs in specs
    ]


def _metric(
    model_digest: str,
    metric_id: str,
    key: str,
    quantity: str,
    calculation: str,
    inputs: list[str],
    *,
    series_key: str | None = None,
    value_symbol: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
    exact: bool = False,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "metric_id": metric_id,
        "key": key,
        "version": "1",
        "required": True,
        "series_key": series_key,
        "value_symbol": value_symbol,
        "reported_key": None,
        "absolute_tolerance": 1e-7,
        "relative_tolerance": 0.0,
        "lower_threshold": lower,
        "upper_threshold": upper,
        "exact": exact,
        "quantity": quantity,
        "calculation": calculation,
        "inputs": inputs,
        "unit": "normalized dimensionless scalar",
        "tolerance_provenance": "MathematicalModel algorithm numerical_tolerance=1e-7.",
        "threshold_provenance": None,
        "model_binding": model_digest,
    }
    if lower is not None or upper is not None:
        data["threshold_provenance"] = (
            "Formal model identity/domain or numerical equilibrium criterion; not a biological threshold."
        )
    return data


def _scenario_metrics(
    model_digest: str, prefix: str, expected_male: float | None
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for state in REFERENCE_STATE:
        metrics.extend(
            [
                _metric(
                    model_digest,
                    f"{prefix}_{state}_minimum",
                    "minimum",
                    f"minimum simulated {state}",
                    f"min({state}[tau_i]) over the reviewed sample grid",
                    [state],
                    series_key=state,
                    lower=-1e-7,
                ),
                _metric(
                    model_digest,
                    f"{prefix}_{state}_final",
                    "final_value",
                    f"final simulated {state}",
                    f"{state}(tau_stop)",
                    [state],
                    series_key=state,
                ),
            ]
        )
    male_lower = expected_male - 1e-7 if expected_male is not None else None
    male_upper = expected_male + 1e-7 if expected_male is not None else None
    metrics.append(
        _metric(
            model_digest,
            f"{prefix}_male_ratio",
            "algebraic_scalar",
            "male lamprey fraction",
            "formal EQ-SEXRULE AST recomputation",
            ["E", "adaptation_weight", "sex_low", "sex_high", "sex_ratio_shape"],
            value_symbol="male_ratio",
            lower=male_lower,
            upper=male_upper,
        )
    )
    metrics.append(
        _metric(
            model_digest,
            f"{prefix}_equilibrium_residual_sq",
            "algebraic_scalar",
            "squared ODE equilibrium residual",
            "dJ_dt^2+dA_dt^2+dF_dt^2+dP_dt^2 at final raw state",
            ["J", "A", "F", "P"],
            value_symbol="equilibrium_residual_sq",
            upper=1e-10,
        )
    )
    for symbol in (
        "lambda_JA_plus",
        "lambda_JA_minus",
        "lambda_F",
        "lambda_P",
        "dominant_real_part",
    ):
        metrics.append(
            _metric(
                model_digest,
                f"{prefix}_{symbol}",
                "algebraic_scalar",
                f"Jacobian spectral quantity {symbol}",
                f"formal AST recomputation of {symbol} at final raw state",
                ["J", "A", "F", "P"],
                value_symbol=symbol,
            )
        )
    return metrics


def _scenario(
    model_digest: str,
    scenario_id: str,
    *,
    decision_values: dict[str, float],
    parameter_values: dict[str, float],
    reason: str,
    comparison: str,
    expected_male: float | None = None,
) -> dict[str, Any]:
    changed = sorted([*decision_values, *parameter_values])
    return {
        "scenario_id": scenario_id,
        "version": "1",
        "required": True,
        "parameter_values": parameter_values,
        "decision_values": decision_values,
        "noise_fraction": 0.0,
        "seed": None,
        "timeout_seconds": 30.0,
        "dynamic": {
            "equation_by_state": {
                "J": "EQ-JDYN",
                "A": "EQ-ADYN",
                "F": "EQ-FDYN",
                "P": "EQ-PDYN",
            },
            "initial_state": REFERENCE_STATE,
            "start": 0.0,
            "stop": TIME_STOP,
            "samples": 401,
            "rtol": 1e-8,
            "atol": 1e-10,
        },
        "metrics": _scenario_metrics(model_digest, scenario_id, expected_male),
        "baseline": "Model v2 baseline parameters and common ordered state [J0,A0,F0,P0].",
        "perturbation": ", ".join(
            f"{key}={value:.12g}" for key, value in {**decision_values, **parameter_values}.items()
        ),
        "reason": reason,
        "input_changes": changed,
        "comparison_quantity": comparison,
        "acceptance_criterion": (
            "Fresh isolated RK45 execution completes every sample; states remain above -1e-7; "
            "final derivative residual squared is <=1e-10; declared algebraic metrics recompute. "
            "The signs of ecological and stability outcomes are evidence, not preconditions."
        ),
        "criterion_provenance": (
            "Formal ODE, domain, and numerical integration contract; no biological outcome threshold."
        ),
    }


def _policy_candidate(
    model_digest: str, contract_digest: str, manifest_digest: str
) -> dict[str, Any]:
    scenarios = [
        _scenario(
            model_digest,
            "adaptive_mid_reference",
            decision_values={"E": 0.5},
            parameter_values={},
            reason="Reference execution for local sensitivity comparisons.",
            comparison="Reference final states, equilibrium residual, and Jacobian spectrum.",
            expected_male=0.67,
        ),
        _scenario(
            model_digest,
            "adaptive_low_resource",
            decision_values={"E": 0.0},
            parameter_values={},
            reason="Official low-food endpoint under the adaptive sex-ratio mechanism.",
            comparison="Adaptive-versus-fixed low-resource trajectories and stability quantities.",
            expected_male=0.78,
        ),
        _scenario(
            model_digest,
            "fixed_low_resource",
            decision_values={"E": 0.0},
            parameter_values={"adaptation_weight": 0.0},
            reason="Paired low-resource structural comparator; only the sex-ratio mechanism changes.",
            comparison="Fixed-versus-adaptive low-resource trajectories and stability quantities.",
            expected_male=0.67,
        ),
        _scenario(
            model_digest,
            "adaptive_high_resource",
            decision_values={"E": 1.0},
            parameter_values={},
            reason="Official high-food endpoint under the adaptive sex-ratio mechanism.",
            comparison="Adaptive-versus-fixed high-resource trajectories and stability quantities.",
            expected_male=0.56,
        ),
        _scenario(
            model_digest,
            "fixed_high_resource",
            decision_values={"E": 1.0},
            parameter_values={"adaptation_weight": 0.0},
            reason="Paired high-resource structural comparator; only the sex-ratio mechanism changes.",
            comparison="Fixed-versus-adaptive high-resource trajectories and stability quantities.",
            expected_male=0.67,
        ),
    ]
    baselines = {
        "sex_ratio_shape": 1.0,
        "cap_min": 0.2,
        "cap_slope": 0.8,
        "birth_coef": 0.8,
        "maturation": 0.2,
        "mort_larva": 0.15,
        "mort_adult": 0.25,
        "fish_growth": 0.6,
        "fish_capacity": 1.0,
        "attack_rate": 0.3,
        "parasite_growth": 0.15,
        "parasite_capacity": 1.0,
        "fish_benefit_par": 0.4,
        "lamprey_cost_par": 0.15,
    }
    for symbol in ASSUMPTION_PARAMETERS:
        scenarios.append(
            _scenario(
                model_digest,
                f"sensitivity_{symbol}_plus_1pct",
                decision_values={"E": 0.5},
                parameter_values={symbol: baselines[symbol] * 1.01},
                reason=(
                    f"One-sided local finite-change sensitivity for assumption parameter {symbol}; "
                    "the +1% range is an analysis design choice, not biological evidence."
                ),
                comparison=f"Change from adaptive_mid_reference in final states and Jacobian spectrum for {symbol}.",
            )
        )
    baseline_metric = _metric(
        model_digest,
        "fixed_ratio_definition",
        "algebraic_scalar",
        "fixed comparator male fraction",
        "(sex_low+sex_high)/2 from formal EQ-FIXED-RATIO",
        ["sex_low", "sex_high"],
        value_symbol="fixed_male_ratio",
        lower=0.67 - 1e-7,
        upper=0.67 + 1e-7,
    )
    return {
        "version": "2",
        "review_status": "DRAFT",
        "production_eligible": False,
        "benchmark_id": "BENCH-MCM2024-A",
        "manifest_digest": manifest_digest,
        "problem_sha256": PROBLEM_SHA256,
        "model_digest": model_digest,
        "model_contract_digest": contract_digest,
        "red_team_report_digest": None,
        "model_jury_report_digest": None,
        "observation_sha256": None,
        "metrics": [baseline_metric],
        "scenarios": scenarios,
        "scientific_scope": (
            "Independently recompute the Case A model's formal sex-ratio mechanism, state-domain "
            "trajectories, adaptive/fixed structural comparisons, numerical equilibrium residuals, "
            "Jacobian spectrum, and one-at-a-time local sensitivity evidence. A PASS verifies faithful "
            "execution, not a favorable ecological conclusion."
        ),
        "unresolved_obligations": [],
    }


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    )


def main() -> None:
    source, state, historical_evidence, literature_count = _read_source()
    manifest_digest = BenchmarkManifestRegistry(ROOT / "benchmarks").digest("BENCH-MCM2024-A")
    problem = ROOT / "var" / "benchmarks" / "cache" / "BENCH-MCM2024-A" / "2024_MCM_Problem_A.pdf"
    import hashlib

    if hashlib.sha256(problem.read_bytes()).hexdigest() != PROBLEM_SHA256:
        raise ValueError("official problem artifact digest mismatch")
    model = _model_v2(source)
    model_digest = mathematical_model_digest(model)
    if model_digest == SOURCE_MODEL_DIGEST:
        raise ValueError("version-2 mathematical meaning did not change")
    parameter_by_symbol = {item.symbol: item for item in model.parameters}
    inventory = [
        {
            "parameter": parameter_by_symbol[symbol].parameter_id,
            "symbol": symbol,
            "meaning": parameter_by_symbol[symbol].description,
            "unit": parameter_by_symbol[symbol].unit.display,
            "baseline_value": parameter_by_symbol[symbol].value,
            "source": parameter_by_symbol[symbol].source_ref,
            "status": "ASSUMPTION",
            "uncertainty": (
                "No retained literature range; local +1% finite-change analysis only, not a "
                "biological uncertainty interval."
            ),
        }
        for symbol in ASSUMPTION_PARAMETERS
    ]
    symbols = []
    for kind, items in (
        ("decision", model.decision_variables),
        ("state", model.state_variables),
        ("derived", model.derived_variables),
        ("parameter", model.parameters),
        ("constant", model.constants),
    ):
        for item in items:
            symbols.append(
                {
                    "symbol": item.symbol,
                    "definition": item.description,
                    "unit": item.unit.display if item.unit else None,
                    "domain": getattr(item, "domain", None),
                    "source_status": (
                        item.source_type.value if hasattr(item, "source_type") else "DERIVATION"
                    ),
                    "kind": kind,
                }
            )
    contract: dict[str, Any] = {
        "contract_version": "2",
        "content_digest": None,
        "source_bindings": {
            "manifest_digest": manifest_digest,
            "problem_resource_id": "RESOURCE-A-PROBLEM",
            "problem_artifact_sha256": PROBLEM_SHA256,
            "source_model_record_id": SOURCE_RECORD_ID,
            "source_model_version": source["version"],
            "source_model_digest": SOURCE_MODEL_DIGEST,
            "source_attempt_id": "1dd1e43e-99fd-4a6a-8332-81dde7b15a69",
            "source_project_id": source["project_id"],
            "source_problem_id": source["problem_id"],
            "problem_state_version": state["version"],
            "problem_state_digest": content_digest(state),
            "historical_result_solver_execution_artifacts": historical_evidence,
            "retained_literature_reference_count": literature_count,
        },
        "model_digest": model_digest,
        "model": model.model_dump(mode="json"),
        "fact_data_assumption_derivation": {
            "FACT": [
                "Sea-lamprey sex depends on larval growth conditions and food availability.",
                "The task asks about lampreys, ecosystem stability, and other organisms including parasites.",
            ],
            "DATA": [
                "Approximate low-food male fraction 0.78.",
                "Approximate high-food male fraction 0.56.",
            ],
            "ASSUMPTION": [
                "Normalized food index, state scaling, provisional coefficients, midpoint fixed comparator, common reference equilibrium, nondimensional time, parasite-guild interpretation, and local sensitivity design.",
            ],
            "DERIVATION": [
                "Reference state, comparator equation, ODE residual, Jacobian entries, eigenvalues, dominant real part, and all replay metrics follow from the formal AST.",
            ],
        },
        "state_contract": {
            "state_vector": ["J", "A", "F", "P"],
            "state_ordering": "[larvae, adult lamprey, fish hosts, representative non-lamprey fish-parasite abundance]",
            "unit": "normalized dimensionless abundance",
            "valid_domain": "R_{≥0}^4",
            "initial_state": REFERENCE_STATE,
            "initial_state_source": "DERIVATION under ASSUMP-INIT-V2 at E_ref=0.5",
            "normalization_rule": "Each abundance is scaled to the model's provisional baseline capacity; no absolute population claim is authorized.",
            "initial_condition_dependence": "Endpoint and transient magnitudes depend on this common initial proportion and require scenario-visible reporting.",
        },
        "time_contract": {
            "time_variable": "tau",
            "unit": "dimensionless",
            "conversion": "tau=t/T_ref; T_ref has no retained physical calibration",
            "source_status": "ASSUMPTION",
            "start": 0.0,
            "stop": TIME_STOP,
            "horizon_derivation": "-ln(1e-10)/0.15 using the smallest positive baseline rate",
            "samples": 401,
            "termination": "all requested samples on the closed interval must be returned",
        },
        "integration_contract": {
            "solver_family": "SciPy solve_ivp RK45",
            "step_policy": "adaptive internal steps with fixed 401-point output grid",
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-8,
            "maximum_integration_time": TIME_STOP,
            "wall_timeout_seconds": 30.0,
            "failure_conditions": [
                "solver reports failure",
                "not all requested samples are returned",
                "non-finite output",
                "state below -1e-7 numerical domain tolerance",
            ],
        },
        "fixed_ratio_comparator": {
            "adaptive_model": "adaptation_weight=1",
            "fixed_model": "adaptation_weight=0 and fixed_male_ratio=(0.78+0.56)/2=0.67",
            "same_contract": [
                "state variables",
                "initial state",
                "horizon",
                "sampling",
                "all ecological parameters",
                "output metrics",
            ],
            "only_changed_mechanism": "EQ-SEXRULE branch selected by adaptation_weight",
            "source_status": "ASSUMPTION plus DERIVATION",
        },
        "stability_contract": {
            "system": "autonomous ODE at fixed E and adaptation_weight",
            "equilibrium_criterion": "equilibrium_residual_sq <= 1e-10 (numerical, not biological)",
            "jacobian_order": ["J", "A", "F", "P"],
            "spectrum": ["lambda_JA_plus", "lambda_JA_minus", "lambda_F", "lambda_P"],
            "dominant_real_part": "max of the four formal real eigenvalues via max(a,b)=(a+b+sqrt((a-b)^2))/2",
            "local_asymptotic_stability": "dominant_real_part < 0 after equilibrium criterion passes",
            "boundary": "dominant_real_part=0 is non-asymptotic/indeterminate, not stable",
            "source_status": "ASSUMPTION definition plus DERIVATION from the ODE",
        },
        "parameter_inventory": inventory,
        "sensitivity_contract": {
            "kind": "one-at-a-time local finite-change sensitivity",
            "parameters": list(ASSUMPTION_PARAMETERS),
            "range": "+1% from each baseline, one parameter per fresh replay",
            "source_status": "ASSUMPTION analysis design choice",
            "not_authorized_as": "biological confidence interval or real-world uncertainty bound",
        },
        "robustness_contract": {
            "kind": "structural scenario comparison",
            "treatments": [
                "adaptive versus fixed ratio at E=0",
                "adaptive versus fixed ratio at E=1",
            ],
            "controlled_inputs": "all non-mechanism parameters, initial state, and time contract identical within each pair",
            "source_status": "ASSUMPTION design plus manifest requirement",
        },
        "resilience_persistence_contract": {
            "resilience": "No separate finite-horizon resilience claim is authorized; local asymptotic return is represented only by the Jacobian stability definition.",
            "persistence": "A limiting equilibrium component is mathematically persistent only when strictly positive.",
            "extinction": "No practical-extinction cutoff is used because no domain source supplies one.",
            "source_status": "ASSUMPTION/DERIVATION",
        },
        "parasite_contract": {
            "P": "normalized abundance of a representative non-lamprey fish-parasite guild",
            "not_P": ["prevalence", "burden", "infection pressure", "lamprey population"],
            "mechanism": "fish hosts subsidize P; adult lamprey reduces P indirectly through shared-host depletion",
            "source_status": "ASSUMPTION_REQUIRED and paper-visible",
        },
        "symbol_table": symbols,
        "derived_metric_contract": [
            "state minimum and final value from raw trajectory series",
            "formal sex-ratio scalar",
            "squared equilibrium residual",
            "four Jacobian eigenvalues",
            "dominant real part",
        ],
        "unresolved_obligations": [],
    }
    contract["content_digest"] = content_digest(
        {key: value for key, value in contract.items() if key != "content_digest"}
    )
    _write(CASE / "mathematical-model-v2.json", contract)
    draft = {
        "draft_version": "2",
        "content_digest": None,
        "review_status": "DRAFT",
        "production_eligible": False,
        "model_contract_digest": contract["content_digest"],
        "candidate_policy": _policy_candidate(
            model_digest, contract["content_digest"], manifest_digest
        ),
        "unresolved_obligations": [],
        "promotion_blockers": [
            "INDEPENDENT_MODEL_CONTRACT_RED_TEAM_PENDING",
            "INDEPENDENT_MODEL_JURY_PENDING",
        ],
    }
    draft["content_digest"] = content_digest(
        {key: value for key, value in draft.items() if key != "content_digest"}
    )
    _write(CASE / "independent-verification.draft.json", draft)
    print(
        json.dumps(
            {
                "model_version": model.version,
                "model_digest": model_digest,
                "model_contract_digest": contract["content_digest"],
                "scenario_count": len(draft["candidate_policy"]["scenarios"]),
                "fresh_case_a_run": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
