import math

import pytest

from non_verifiable_rewards.rewards import (
    ReferenceLikelihoodReward,
    SelfCertaintyReward,
    VRCLIReward,
)
from non_verifiable_rewards.rewards.reference_likelihood import reference_context
from non_verifiable_rewards.rewards.vr_cli import perplexity_improvement


class RecordingScorer:
    def __init__(self):
        self.calls = []

    def log_likelihood(self, contexts, continuations, *, reduction):
        self.calls.append((contexts, continuations, reduction))
        return [
            -1.0 if "Candidate response:" in context else -2.0 for context in contexts
        ]

    def self_certainty(self, contexts, continuations):
        self.calls.append((contexts, continuations))
        return [0.4] * len(contexts)


def test_reference_likelihood_scores_reference_conditioned_on_candidate():
    scorer = RecordingScorer()
    reward = ReferenceLikelihoodReward(scorer)
    assert reward.score(["candidate"], references=["gold"], prompts=["input"]) == [-1.0]
    contexts, targets, reduction = scorer.calls[0]
    assert contexts == [reference_context("input", "candidate")]
    assert "gold" not in contexts[0]
    assert targets == ["gold"]
    assert reduction == "sum"


def test_self_certainty_scores_candidate_without_reference():
    scorer = RecordingScorer()
    reward = SelfCertaintyReward(
        scorer, context_builder=lambda prompt: "CHAT:" + prompt
    )
    assert reward.score(["story"], prompts=["prompt"]) == [0.4]
    assert scorer.calls == [(["CHAT:prompt"], ["story"])]


def test_vrcli_reuses_baseline_within_call_and_keys_by_reference():
    scorer = RecordingScorer()
    reward = VRCLIReward(scorer)
    kwargs = {
        "candidates": ["a", "b", "c"],
        "references": ["gold", "gold", "other"],
        "prompts": ["input"] * 3,
    }
    assert reward.score(**kwargs) == [1.0, 1.0, 1.0]
    assert scorer.calls[0][1:] == (["gold", "gold", "other"], "mean")
    assert scorer.calls[1] == (
        [reference_context("input", None)] * 2,
        ["gold", "other"],
        "mean",
    )
    reward.score(**kwargs)
    assert len(scorer.calls) == 4  # No stale cross-call cache if model weights change.


@pytest.mark.parametrize(
    "improvement,expected",
    [
        (-20, 0),
        (0, 0),
        (0.04, 0),
        (0.1, 0.5),
        (0.99, 0.5),
        (1.1, 0.9),
        (1.99, 0.9),
        (2.1, 1),
        (50, 1),
    ],
)
def test_vrcli_published_percent_schedule(improvement, expected):
    baseline = -3.0
    conditioned = baseline - math.log1p(-improvement / 100)
    assert perplexity_improvement(conditioned, baseline) == pytest.approx(improvement)
    assert VRCLIReward(RecordingScorer()).score_from_log_likelihoods(
        [conditioned], [baseline]
    ) == [expected]


def test_vrcli_threshold_equality_and_order():
    improvement = perplexity_improvement(-1.9, -2.0)
    reward = VRCLIReward(RecordingScorer(), thresholds=[improvement], levels=[0, 1])
    assert reward.score_from_log_likelihoods([-1.9], [-2.0]) == [1.0]
    assert reward.score_from_log_likelihoods([-1.9001], [-2.0]) == [0.0]


def test_vrcli_stability_for_long_references_and_extreme_regressions():
    assert perplexity_improvement(-1000.0, -1000.1) == pytest.approx(
        100 * (1 - math.exp(-0.1))
    )
    reward = VRCLIReward(RecordingScorer())
    assert reward.score_from_log_likelihoods([-10000], [-1]) == [0]
    with pytest.raises(ValueError, match="finite"):
        reward.score_from_log_likelihoods([math.nan], [-1])
    with pytest.raises(ValueError, match="same length"):
        reward.score_from_log_likelihoods([-1], [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thresholds": [1, 0]},
        {"thresholds": [1, 1], "levels": [0, 0.5, 1]},
        {"thresholds": [math.nan], "levels": [0, 1]},
        {"thresholds": [1], "levels": [1, 0]},
    ],
)
def test_vrcli_rejects_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        VRCLIReward(RecordingScorer(), **kwargs)


@pytest.mark.parametrize(
    "cls", [ReferenceLikelihoodReward, VRCLIReward, SelfCertaintyReward]
)
def test_lm_reward_required_inputs_and_empty_batches(cls):
    reward = cls(RecordingScorer())
    assert reward.score([]) == []
    with pytest.raises(ValueError, match="requires prompts"):
        reward.score(["candidate"], references=["reference"])
