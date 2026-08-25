import re
from dataclasses import dataclass

from mathmodel_ai.schemas.model_selection import ModelCandidate

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_IGNORED_TOKENS = frozenset({"model", "method", "approach", "algorithm"})


def normalize_model_name(value: str) -> str:
    folded = value.casefold()
    if "milp" in folded or "mixed integer linear" in folded or "整数线性" in folded:
        return "mixed integer linear programming"
    tokens = [token for token in _TOKEN_PATTERN.findall(folded) if token not in _IGNORED_TOKENS]
    return " ".join(tokens) or folded.strip()


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(normalize_model_name(left).split())
    right_tokens = set(normalize_model_name(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def are_semantic_duplicates(left: ModelCandidate, right: ModelCandidate) -> bool:
    left_name = normalize_model_name(left.normalized_name or left.name)
    right_name = normalize_model_name(right.normalized_name or right.name)
    if left_name == right_name:
        return True
    if left.family is not right.family:
        return False
    if _token_similarity(left_name, right_name) >= 0.75:
        return True
    return normalize_model_name(left.mathematical_core) == normalize_model_name(
        right.mathematical_core
    )


@dataclass(frozen=True)
class DeduplicationResult:
    candidates: list[ModelCandidate]
    removed_ids: list[str]


def deduplicate_candidates(candidates: list[ModelCandidate]) -> DeduplicationResult:
    kept: list[ModelCandidate] = []
    removed: list[str] = []
    for candidate in candidates:
        normalized = normalize_model_name(candidate.normalized_name or candidate.name)
        normalized_candidate = candidate.model_copy(update={"normalized_name": normalized})
        if any(are_semantic_duplicates(normalized_candidate, existing) for existing in kept):
            removed.append(candidate.candidate_id)
        else:
            kept.append(normalized_candidate)
    return DeduplicationResult(candidates=kept, removed_ids=removed)
