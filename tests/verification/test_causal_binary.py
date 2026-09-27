import hashlib

import pytest

from mathmodel_ai.verification.causal_binary import CausalBinarySpec, iter_causal_binary_points


def _spec(source: bytes) -> CausalBinarySpec:
    return CausalBinarySpec(
        source_sha256=hashlib.sha256(source).hexdigest(),
        group_column="match",
        condition_column="server",
        outcome_column="winner",
        positive_value="1",
        negative_value="2",
        history_window=2,
    )


def test_features_use_only_prior_points_and_reset_by_group() -> None:
    source = b"match,server,winner,after_point\nA,1,1,99\nA,2,2,99\nA,1,2,99\nB,1,2,99\nA,1,1,99\n"
    iterator = iter_causal_binary_points(source, _spec(source))
    first, first_label = next(iterator)
    assert first_label == 1.0
    assert first.prior_group_count == first.prior_condition_count == first.recent_count == 0
    second, second_label = next(iterator)
    assert second_label == 0.0
    assert second.prior_group_count == 1 and second.prior_group_positive == 1
    assert second.prior_condition_count == 0
    third, _ = next(iterator)
    assert third.prior_condition_count == 1
    assert third.prior_condition_positive == 1
    assert third.recent_count == 2 and third.recent_positive == 1
    fourth, _ = next(iterator)
    assert fourth.group == "B" and fourth.prior_group_count == 0
    fifth, _ = next(iterator)
    assert fifth.group == "A" and fifth.prior_group_count == 3
    assert fifth.recent_count == 2 and fifth.recent_positive == 0
    with pytest.raises(StopIteration):
        next(iterator)


def test_future_labels_and_post_point_columns_cannot_change_prefix_features() -> None:
    original = b"match,server,winner,after_point\nA,1,1,1\nA,2,2,2\nA,1,1,3\n"
    altered = b"match,server,winner,after_point\nA,1,1,999\nA,2,1,999\nA,1,2,999\n"
    first = list(iter_causal_binary_points(original, _spec(original)))
    second = list(iter_causal_binary_points(altered, _spec(altered)))
    assert [feature for feature, _ in first][:2] == [feature for feature, _ in second][:2]
    assert first[2][0].prior_group_positive == 1
    assert second[2][0].prior_group_positive == 2
    assert first[-1][1] != second[-1][1]


def test_causal_csv_rejects_changed_source_and_invalid_labels_or_columns() -> None:
    source = b"match,server,winner\nA,1,1\n"
    with pytest.raises(ValueError, match="CAUSAL_CSV_SIZE_OR_HASH_MISMATCH"):
        list(iter_causal_binary_points(source.replace(b"A,1,1", b"A,1,2"), _spec(source)))
    invalid = b"match,server,winner\nA,1,3\n"
    with pytest.raises(ValueError, match="CAUSAL_CSV_VALUE_INVALID"):
        list(iter_causal_binary_points(invalid, _spec(invalid)))
    duplicated = b"match,server,winner,winner\nA,1,1,1\n"
    with pytest.raises(ValueError, match="CAUSAL_CSV_COLUMNS_INVALID"):
        list(iter_causal_binary_points(duplicated, _spec(duplicated)))
    empty = b"match,server,winner\n"
    with pytest.raises(ValueError, match="CAUSAL_CSV_NO_ROWS"):
        list(iter_causal_binary_points(empty, _spec(empty)))
