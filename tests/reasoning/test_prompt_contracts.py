from mathmodel_ai.reasoning.prompts import PromptRegistry


def test_model_explorer_distinguishes_missing_from_mitigated_data_gaps() -> None:
    prompt = PromptRegistry().get("model_explorer")

    assert prompt.version == "2.5.0"
    assert '"external_required"' in prompt.system
    assert '"missing" means no credible' in prompt.system
    assert "bounded assumption" in prompt.system
    assert "AUTOMATED_RETRY_FEEDBACK" in prompt.system
    assert "Escape quotation marks, newlines, and backslashes" in prompt.system
    assert 'no unresolved mandatory "missing" input' in prompt.system
    assert "Do not merely change an" in prompt.system
    assert "EACH address ALL supplied" in prompt.system
    assert "expected_outputs are arrays of objects" in prompt.system
    assert "name, description, and source_or_destination" in prompt.system
    assert "Do not\nreplace any schema object with a shorthand string" in prompt.system


def test_model_jury_preserves_hard_failures_without_disqualifying_uncertainty() -> None:
    prompt = PromptRegistry().get("model_jury")

    assert prompt.version == "2.2.0"
    assert "include the required overall confidence" in prompt.system
    assert "use it only when the candidate cannot credibly be executed" in prompt.system
    assert "never conceal a real hard failure" in prompt.system
    assert "VIOLATES_PROBLEM_REQUIREMENT" in prompt.system


def test_math_modeler_requires_traceable_executable_parameter_values() -> None:
    prompt = PromptRegistry().get("math_modeler")

    assert prompt.version == "4.6.1"
    assert "every parameter and constant requires a finite" in prompt.system
    assert "source_type=ASSUMPTION" in prompt.system
    assert "source_type=ESTIMATED" in prompt.system
    assert "do not emit a null value" in prompt.system
    assert "will correctly\nfail the deterministic model gate" in prompt.system
    assert "Every SYMBOL node used anywhere is declared exactly once" in prompt.system
    assert "AUTOMATED_RETRY_FEEDBACK" in prompt.system
    assert "preferred_solver_families may contain only" in prompt.system
    assert "exact key parameter_id with a CONST- identifier" in prompt.system
    assert "Equation dependency_refs lists only EQ- IDs" in prompt.system
    assert "A citation that merely names a CSV does" in prompt.system
    assert "constant-objective pseudo-optimization" in prompt.system
    assert "Listing CSV columns as data_bindings" in prompt.system
    assert "count_nonmissing" in prompt.system
