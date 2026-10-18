import pytest

from assistant.retrieval.hybrid import rrf_fuse


def test_rrf_scores_match_formula() -> None:
    fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_item_in_both_lists_beats_item_first_in_one() -> None:
    # "b" is 2nd in both lists, "a" and "c" are 1st in one list only.
    ranked = [item for item, _ in rrf_fuse([["a", "b"], ["c", "b"]])]
    assert ranked[0] == "b"


def test_ties_broken_by_first_appearance() -> None:
    ranked = [item for item, _ in rrf_fuse([["x", "y"], ["y", "x"]])]
    assert ranked == ["x", "y"]  # equal scores; "x" was seen first


def test_small_k_favours_top_rank() -> None:
    # k=0: "a" (1st once) scores 1/1 = 1.0 and "b" (2nd twice) scores 1/2 + 1/2 = 1.0;
    # the tie goes to "a" (seen first). k=60: "b" scores 2/62 > 1/61, so agreement wins.
    lists = [["a", "b"], ["c", "b"]]
    assert [i for i, _ in rrf_fuse(lists, k=0)][0] == "a"
    assert [i for i, _ in rrf_fuse(lists, k=60)][0] == "b"


def test_empty_and_single_list() -> None:
    assert rrf_fuse([]) == []
    assert [i for i, _ in rrf_fuse([["a", "b", "c"]])] == ["a", "b", "c"]


def test_negative_k_rejected() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=-1)
