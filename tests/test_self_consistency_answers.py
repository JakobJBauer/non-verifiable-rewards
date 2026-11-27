"""Joint SuperGPQA answer extraction and grouped-reward regressions."""

import json
from types import SimpleNamespace

import pytest

from non_verifiable_rewards.rewards import SelfConsistencyReward
from non_verifiable_rewards.training.config import validate_config
from non_verifiable_rewards.training.rewards import RewardFactory, WeightedReward


@pytest.mark.parametrize(
    "options,completions,votes",
    [
        (
            ["one", "two", "three", "four"],
            ["Answer: one", "one", "Option A", "Answer: B: one", "two", "B", "partial", ""],
            ["A", "A", "A", "A", "B", "B", None, None],
        ),
        (
            ["A", "C", "K", "B"],
            ["Answer: C", "Option B", "C", "<think>private</think>C", "Option C", "K", "Option D", ""],
            ["B", "B", "B", "B", "C", "C", "D", None],
        ),
    ],
)
def test_consensus_matches_answer_text_and_label_semantics(options, completions, votes):
    rows = [
        dict(dataset="supergpqa", prompt_id="q", prompt="Question", options=options)
    ] * 8
    scores, details = SelfConsistencyReward(group_size=8).score_groups(rows, completions)
    assert details[0]["votes"] == votes
    assert scores == [1, 1, 1, 1, 0, 0, 0, 0]


@pytest.mark.parametrize("reference_weight", [1, -1])
def test_consistency_mixture_preserves_answer_views_and_invalid_penalty(
    tmp_path, reference_weight
):
    factory = RewardFactory(validate_config({}))
    callback = WeightedReward(
        SimpleNamespace(score=factory.score),
        {"rewards": {"self_consistency": 1, "rouge_l": reference_weight}},
        None,
        tmp_path / "trace.jsonl",
        self_consistency=SelfConsistencyReward(group_size=8),
    )
    scores = callback(
        prompts=["Q"] * 8,
        completions=[
            "Answer: one", "Option A", "Answer: B: one", "Option B",
            "", "Option J", "<think>unfinished", "partial",
        ],
        source_prompt=["Q"] * 8,
        dataset=["supergpqa"] * 8,
        reference=["two"] * 8,
        prompt_id=["q"] * 8,
        options=[["one", "two"]] * 8,
        answer_letter=["B"] * 8,
    )
    assert scores == [1, 1, 1, reference_weight, -1000, -1000, -1000, 0]
    trace = json.loads(callback.trace_path.read_text())
    assert trace["self_consistency"][0]["winner"] == "A"
    assert trace["components"]["self_consistency"] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert trace["reward_candidates"]["rouge_l"] == [
        "one", "one", "one", "two", None, None, None, "partial"
    ]
