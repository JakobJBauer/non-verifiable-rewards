"""chrF++ arithmetic, batch semantics, and reproducible settings."""

import pytest

from non_verifiable_rewards.rewards import ChrFPlusPlusReward


def test_chrf_plus_plus_includes_both_word_orders():
    # Character 1/2/3-gram overlaps: 2/3, 1/2, 0.
    # Word 1/2-gram overlaps: 1/2, 0. Higher orders are unavailable.
    # P = R = (2/3 + 1/2 + 0 + 1/2 + 0) / 5 = 1/3.
    reward = ChrFPlusPlusReward()
    assert reward.score(["ab c"], references=["ab d"]) == pytest.approx([100 / 3])
    assert "nc:6|nw:2" in reward.signature
    assert "case:mixed|eff:yes" in reward.signature
    assert "space:no|version:" in reward.signature


def test_chrf_beta_two_and_candidate_reference_orientation():
    # For the shorter candidate: P=1, R=7/12 over the four available orders.
    # F2 = 5PR / (4P + R); reversing candidate/reference changes the score.
    reward = ChrFPlusPlusReward()
    assert reward.score(["a b", "a b c"], references=["a b c", "a b"]) == pytest.approx(
        [100 * 5 * (7 / 12) / (4 + 7 / 12), 87.5]
    )


def test_chrf_short_unicode_empty_case_and_whitespace():
    reward = ChrFPlusPlusReward()
    assert reward.score(
        ["x", "é🙂漢字", "", " \t", "ABC", "a  b\nc"],
        references=["x", "é🙂漢字", "a", "a", "abc", "a b c"],
    ) == [100, 100, 0, 0, 0, 100]


def test_chrf_returns_independent_aligned_scores_without_corpus_pooling():
    reward = ChrFPlusPlusReward()
    candidates = ["x", "a b", "x"]
    references = ["y", "a b c", "x"]
    batch = reward.score(candidates, references=references, prompts=["ignored"] * 3)
    single = [
        reward.score([candidate], references=[reference])[0]
        for candidate, reference in zip(candidates, references, strict=True)
    ]
    assert batch == single
    assert batch[0] == 0 and batch[2] == 100


def test_chrf_validates_before_initializing_sacrebleu():
    reward = ChrFPlusPlusReward()
    assert reward.score([]) == []
    with pytest.raises(ValueError, match="requires references"):
        reward.score(["candidate"])
    with pytest.raises(ValueError, match="same length"):
        reward.score(["candidate"], references=[])
    with pytest.raises(ValueError, match="must not be empty"):
        reward.score(["candidate"], references=[" "])
    with pytest.raises(ValueError, match="prompts and candidates"):
        reward.score(["candidate"], references=["reference"], prompts=[])
    with pytest.raises(TypeError, match="single string"):
        reward.score("candidate", references=["reference"])
    with pytest.raises(TypeError, match="must be a string"):
        reward.score([None], references=["reference"])
    assert reward._scorer is None and reward.signature is None
