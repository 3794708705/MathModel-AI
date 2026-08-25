from mathmodel_ai.schemas.model_selection import (
    CandidateJuryAssessment,
    CapabilityAssessment,
    ComplexityLevel,
    JuryAssessment,
    ModelCandidate,
    ModelChainStage,
    ModelExploration,
    ModelFamily,
    ModelIOContract,
)
from mathmodel_ai.schemas.problem_analysis import (
    Ambiguity,
    CoreProblem,
    DataAvailability,
    DataRequirement,
    EvidenceItem,
    EvidenceSource,
    EvidenceStatus,
    EvidenceType,
    Interpretation,
    ObjectiveItem,
    ProblemAnalysis,
    ProblemTaskType,
    SubProblem,
    SubProblemDependency,
)


def analysis_fixture(*, low_confidence_ambiguity: bool = False) -> ProblemAnalysis:
    ambiguity = Ambiguity(
        ambiguity_id="AMB-demand",
        description="Whether demand means observed or forecast demand",
        interpretations=[
            Interpretation(
                interpretation_id="observed",
                meaning="Use observed demand",
                support="Historical values are provided",
            ),
            Interpretation(
                interpretation_id="forecast",
                meaning="Use forecast demand",
                support="The planning period is future-facing",
            ),
        ],
        preferred_interpretation_id="forecast",
        reason="The optimization is for the next period",
        confidence=0.4 if low_confidence_ambiguity else 0.85,
    )
    return ProblemAnalysis(
        core_problem=CoreProblem(
            statement="Forecast demand, optimize allocation, and evaluate the plan.",
            success_criteria=["forecast error is reported", "resource limits hold"],
            evidence_refs=["EVID-fact-1"],
        ),
        objectives=[
            ObjectiveItem(
                objective_id="OBJ-main",
                description="Produce a validated allocation plan",
                required_output="forecast, allocation, and evaluation",
                evidence_refs=["EVID-fact-1"],
            )
        ],
        subproblems=[
            SubProblem(
                subproblem_id="Q1",
                order=1,
                original_text="Predict next-period demand.",
                normalized_goal="Estimate next-period demand with uncertainty.",
                output_required=["demand forecast"],
                task_types=[ProblemTaskType.PREDICTION, ProblemTaskType.TIME_SERIES],
                output_dependencies=["Q2"],
                data_requirements=[
                    DataRequirement(
                        name="demand history",
                        description="past demand by period",
                        availability=DataAvailability.PROVIDED,
                    )
                ],
                ambiguity_refs=["AMB-demand"],
            ),
            SubProblem(
                subproblem_id="Q2",
                order=2,
                original_text="Optimize allocations.",
                normalized_goal="Minimize cost subject to demand and capacity.",
                output_required=["allocation decision"],
                task_types=[ProblemTaskType.OPTIMIZATION],
                input_dependencies=["Q1"],
                output_dependencies=["Q3"],
            ),
            SubProblem(
                subproblem_id="Q3",
                order=3,
                original_text="Evaluate the resulting plan.",
                normalized_goal="Evaluate cost and service trade-offs.",
                output_required=["evaluation metrics"],
                task_types=[ProblemTaskType.EVALUATION],
                input_dependencies=["Q2"],
            ),
        ],
        facts=[
            EvidenceItem(
                evidence_id="EVID-fact-1",
                type=EvidenceType.FACT,
                content="The task asks for a future allocation plan.",
                source=EvidenceSource.PROBLEM_TEXT,
                source_location="paragraph 1",
                confidence=1,
                status=EvidenceStatus.ACCEPTED,
            )
        ],
        assumptions_required=[
            EvidenceItem(
                evidence_id="EVID-assumption-1",
                type=EvidenceType.ASSUMPTION,
                content="Demand-generation behavior remains stable over the forecast horizon.",
                source=EvidenceSource.AGENT_DERIVATION,
                confidence=0.6,
                status=EvidenceStatus.PROPOSED,
            )
        ],
        ambiguities=[ambiguity],
        task_types=[
            ProblemTaskType.PREDICTION,
            ProblemTaskType.OPTIMIZATION,
            ProblemTaskType.EVALUATION,
        ],
        dependencies_between_subproblems=[
            SubProblemDependency(
                upstream_id="Q1",
                downstream_id="Q2",
                transferred_output="demand forecast",
            ),
            SubProblemDependency(
                upstream_id="Q2",
                downstream_id="Q3",
                transferred_output="allocation decision",
            ),
        ],
        confidence=0.8,
    )


def candidate_fixture(
    candidate_id: str,
    *,
    name: str,
    family: ModelFamily,
    targets: list[str] | None = None,
    data_availability: DataAvailability = DataAvailability.PROVIDED,
) -> ModelCandidate:
    io_input = ModelIOContract(
        name="input data",
        description="problem observations and parameters",
        source_or_destination="provided data",
    )
    io_output = ModelIOContract(
        name="model output",
        description="candidate-specific estimates or decisions",
        source_or_destination="next model stage",
    )
    assessment = CapabilityAssessment(score=8, rationale="Suitable for the stated task.")
    return ModelCandidate(
        candidate_id=candidate_id,
        name=name,
        normalized_name=name.casefold(),
        family=family,
        target_subproblems=targets or ["Q1", "Q2", "Q3"],
        description=f"Use {name} for the linked competition tasks.",
        fit_reason="The mathematical structure matches the requested outputs.",
        mathematical_core=f"{name} mathematical formulation",
        required_inputs=[io_input],
        expected_outputs=[io_output],
        assumptions_required=["Input relationships remain usable during the planning horizon."],
        data_requirements=[
            DataRequirement(
                name="candidate input",
                description="data required by this candidate",
                availability=data_availability,
            )
        ],
        advantages=["clear mathematical contract"],
        disadvantages=["requires parameter checking"],
        explainability=assessment,
        mathematical_rigor=assessment,
        computational_complexity=ComplexityLevel.MEDIUM,
        implementation_complexity=ComplexityLevel.MEDIUM,
        validation_potential=assessment,
        robustness_potential=assessment,
        innovation_potential=assessment,
        competition_feasibility=assessment,
        risks=["parameter misspecification"],
        model_chain_position=[
            ModelChainStage(
                order=1,
                name=name,
                family=family,
                mathematical_method=f"{name} formulation",
                inputs=[io_input],
                outputs=[io_output],
            )
        ],
        confidence=0.8,
    )


def candidate_set() -> list[ModelCandidate]:
    return [
        candidate_fixture(
            "CAND-chain",
            name="Forecast MILP Evaluation Chain",
            family=ModelFamily.MODEL_CHAIN,
        ),
        candidate_fixture(
            "CAND-regression",
            name="Regularized Regression and Linear Allocation",
            family=ModelFamily.REGRESSION,
        ),
        candidate_fixture(
            "CAND-transformer",
            name="Large Transformer Decision System",
            family=ModelFamily.NEURAL_NETWORK,
        ),
    ]


def exploration_fixture() -> ModelExploration:
    return ModelExploration(
        candidates=candidate_set(),
        exploration_summary=(
            "Three distinct model families cover prediction, optimization, and evaluation."
        ),
    )


def jury_fixture() -> JuryAssessment:
    common = {
        "problem_fit": 8,
        "data_fit": 8,
        "mathematical_validity": 8,
        "explainability": 8,
        "validation_potential": 8,
        "innovation_potential": 7,
        "competition_feasibility": 8,
        "computational_cost": 8,
        "rationale": "The candidate has a credible mathematical and competition workflow.",
    }
    return JuryAssessment(
        assessments=[
            CandidateJuryAssessment(candidate_id="CAND-chain", **common),
            CandidateJuryAssessment(
                candidate_id="CAND-regression",
                **{**common, "problem_fit": 7, "innovation_potential": 5},
            ),
            CandidateJuryAssessment(
                candidate_id="CAND-transformer",
                **{
                    **common,
                    "problem_fit": 4,
                    "data_fit": 1,
                    "explainability": 3,
                    "validation_potential": 3,
                    "competition_feasibility": 2,
                    "computational_cost": 1,
                    "rationale": "Twenty observations cannot support a large Transformer reliably.",
                },
            ),
        ],
        overall_rationale="Simple traceable candidates dominate the data-hungry alternative.",
        confidence=0.85,
    )
