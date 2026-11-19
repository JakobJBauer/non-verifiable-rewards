import math

import pytest

from non_verifiable_rewards.rewards import (
    ReferenceLikelihoodReward,
    SelfCertaintyReward,
    VRCLIReward,
)
from non_verifiable_rewards.rewards.reference_likelihood import reference_context
from non_verifiable_rewards.rewards.vr_cli import perplexity_improvement, vr_cli_context


class RecordingScorer:
    def __init__(self):
        self.calls = []

    def log_likelihood(self, contexts, continuations, *, reduction):
        self.calls.append((contexts, continuations, reduction))
        return [
            -1.0
            if "Candidate response:" in context or "Here is an example" in context
            else -2.0
            for context in contexts
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
    assert reward.score(**kwargs) == pytest.approx([100 * (1 - math.exp(-1))] * 3)
    assert scorer.calls[0][1:] == (["gold", "gold", "other"], "mean")
    assert scorer.calls[1] == (
        [vr_cli_context("input", None, task="summarization")] * 2,
        ["gold", "other"],
        "mean",
    )
    reward.score(**kwargs)
    assert len(scorer.calls) == 4  # No stale cross-call cache if model weights change.


@pytest.mark.parametrize(
    "task,example,instruction",
    [
        ("summarization", "summary of the same source text", "summary of the source text"),
        (
            "story_generation",
            "story for the same writing prompt",
            "standalone story for the writing prompt",
        ),
    ],
)
def test_vrcli_task_prompt_keeps_reference_as_target(task, example, instruction):
    scorer = RecordingScorer()
    reward = VRCLIReward(scorer, task=task)
    reward.score(["candidate"], references=["held-out reference"], prompts=["source"])
    conditioned, targets, reduction = scorer.calls[0]
    baseline, baseline_targets, baseline_reduction = scorer.calls[1]
    block = f"Here is an example {example}:\ncandidate\n\n"
    assert block in conditioned[0]
    assert conditioned[0].replace(block, "") == baseline[0]
    assert conditioned[0].endswith(f"Now write a {instruction}:\n")
    assert "held-out reference" not in conditioned[0]
    assert "held-out reference" not in baseline[0]
    assert "source" in conditioned[0]
    assert targets == baseline_targets == ["held-out reference"]
    assert reduction == baseline_reduction == "mean"


def test_vrcli_rejects_unknown_task_and_accepts_custom_context():
    with pytest.raises(ValueError, match="task must"):
        VRCLIReward(RecordingScorer(), task="unknown")
    scorer = RecordingScorer()
    reward = VRCLIReward(scorer, context_builder=reference_context)
    reward.score(["candidate"], references=["gold"], prompts=["input"])
    assert scorer.calls[0][0] == [reference_context("input", "candidate")]


@pytest.mark.parametrize(
    "improvement", [-200, -20, 0, 0.04, 0.05, 0.1, 0.99, 1, 1.1, 1.99, 2, 2.1, 50]
)
def test_vrcli_returns_percent_improvement_without_thresholds(improvement):
    baseline = -3.0
    conditioned = baseline - math.log1p(-improvement / 100)
    assert perplexity_improvement(conditioned, baseline) == pytest.approx(improvement)
    assert VRCLIReward(RecordingScorer()).score_from_log_likelihoods(
        [conditioned], [baseline]
    ) == pytest.approx([improvement])


@pytest.mark.parametrize(
    "improvements", [(-20, -10), (0.01, 0.04), (0.1, 0.9), (1.1, 1.9), (3, 50)]
)
def test_vrcli_distinguishes_candidates_within_each_former_bucket(improvements):
    baseline = -3.0
    conditioned = [baseline - math.log1p(-value / 100) for value in improvements]
    scores = VRCLIReward(RecordingScorer()).score_from_log_likelihoods(
        conditioned, [baseline] * 2
    )
    assert scores == pytest.approx(improvements)
    assert scores[0] < scores[1]


def test_vrcli_preserves_true_ties_and_small_improvements():
    reward = VRCLIReward(RecordingScorer())
    scores = reward.score_from_log_likelihoods([-2, -2], [-3, -3])
    assert scores[0] == scores[1]
    improvement = reward.score_from_log_likelihoods([-2 + 1e-12], [-2])[0]
    assert 0 < improvement < 1e-9


def test_vrcli_stability_for_long_references_and_extreme_regressions():
    assert perplexity_improvement(-1000.0, -1000.1) == pytest.approx(
        100 * (1 - math.exp(-0.1))
    )
    reward = VRCLIReward(RecordingScorer())
    # Overflow must fail explicitly, rather than become an artificial tied floor.
    for conditioned in (-10000, -710):
        with pytest.raises(ValueError, match="finite"):
            reward.score_from_log_likelihoods([conditioned], [-1])
    with pytest.raises(ValueError, match="same length"):
        reward.score_from_log_likelihoods([-1], [])


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_vrcli_rejects_nonfinite_log_probabilities(invalid):
    reward = VRCLIReward(RecordingScorer())
    with pytest.raises(ValueError, match="finite"):
        reward.score_from_log_likelihoods([invalid], [-1])
    with pytest.raises(ValueError, match="finite"):
        reward.score_from_log_likelihoods([-1], [invalid])


@pytest.mark.parametrize(
    "cls", [ReferenceLikelihoodReward, VRCLIReward, SelfCertaintyReward]
)
def test_lm_reward_required_inputs_and_empty_batches(cls):
    reward = cls(RecordingScorer())
    assert reward.score([]) == []
    with pytest.raises(ValueError, match="requires prompts"):
        reward.score(["candidate"], references=["reference"])
