import math

import pytest

from non_verifiable_rewards.rewards import GRPORewardAdapter, answer_accuracy_reward


class RecordingReward:
    def __init__(self):
        self.calls = []

    def score(self, candidates, *, references, prompts):
        self.calls.append((candidates, references, prompts))
        return [0.75] * len(candidates)


@pytest.mark.parametrize(
    "view,expected",
    [
        ("completion", "Explanation\nAnswer: B"),
        ("reasoning", "Explanation"),
        ("answer_text", "Chosen option text"),
    ],
)
def test_grpo_adapter_selects_intended_candidate_and_keeps_reward_columns(
    view, expected
):
    reward = RecordingReward()
    callback = GRPORewardAdapter(reward, candidate_view=view, invalid_score=-5)
    result = callback(
        prompts=[[{"role": "user", "content": "Question with options"}]],
        completions=[
            [{"role": "assistant", "content": "<think>Plan</think>\nExplanation\nAnswer: B"}]
        ],
        reference=["Gold option text"],
        options=[["Wrong", "Chosen option text"]],
        completion_ids=[[1, 2, 3]],
        trainer_state=object(),
        answer_letter=["A"],
    )
    assert result == [0.75]
    assert reward.calls == [
        ([expected], ["Gold option text"], ["Question with options"])
    ]
    assert callback.__name__.endswith(view)


@pytest.mark.parametrize("view", ["reasoning", "answer_text"])
def test_invalid_completions_keep_positions_without_calling_reward_on_empty_text(view):
    reward = RecordingReward()
    callback = GRPORewardAdapter(reward, candidate_view=view, invalid_score=-7)
    assert callback(
        prompts=["Q"] * 3,
        completions=["Answer: J", "Plan\nAnswer: B", ""],
        reference=["Gold"] * 3,
        options=[["A text", "B text"]] * 3,
    ) == [-7, 0.75, -7]
    assert len(reward.calls[0][0]) == 1
    reward.calls.clear()
    assert callback(
        prompts=["Q"], completions=[""], reference=["Gold"], options=[["A", "B"]]
    ) == [-7]
    assert reward.calls == []


def test_exact_accuracy_has_no_reference_similarity_dependency():
    assert answer_accuracy_reward(
        completions=["Plan\nAnswer: B", "Answer: A", "Answer: A or B", "Answer: J"],
        answer_letter=["B"] * 4,
        options=[["Wrong", "Right"]] * 4,
        reference=["unused"] * 4,
    ) == [1, 0, 0, 0]
    with pytest.raises(ValueError, match="align"):
        answer_accuracy_reward(["Answer: A"], [], [["A", "B"]])


def test_grpo_callback_fails_on_misalignment_and_nonfinite_scores():
    reward = RecordingReward()
    callback = GRPORewardAdapter(reward, invalid_score=0)
    with pytest.raises(ValueError, match="align"):
        callback(
            prompts=["Q"], completions=["A", "B"], reference=["R"], options=[["A", "B"]]
        )
    with pytest.raises(ValueError, match="one assistant"):
        callback(
            prompts=["Q"],
            completions=[[{"role": "user", "content": "wrong role"}]],
            reference=["R"],
            options=[["A", "B"]],
        )
    reward.score = lambda *args, **kwargs: [math.nan]
    with pytest.raises(ValueError, match="finite"):
        callback(
            prompts=["Q"],
            completions=["Answer: A"],
            reference=["R"],
            options=[["A", "B"]],
        )
    with pytest.raises(ValueError, match="finite"):
        GRPORewardAdapter(reward, invalid_score=math.inf)


@pytest.mark.parametrize("field", ["thinking", "reasoning_content"])
def test_structured_reasoning_is_excluded_from_rewards(field):
    reward = RecordingReward()
    callback = GRPORewardAdapter(
        reward, candidate_view="completion", invalid_score=-7
    )
    assert callback(
        prompts=["Q"], reference=["R"], options=[["A", "B"]],
        completions=[[{
            "role": "assistant", "content": "Answer: A",
            field: "Private thinking stored separately",
        }]],
    ) == [0.75]
    assert reward.calls == [(["Answer: A"], ["R"], ["Q"])]


def test_tool_calls_are_not_text_reward_candidates():
    callback = GRPORewardAdapter(RecordingReward(), invalid_score=-7)
    with pytest.raises(ValueError, match="Tool-call"):
        callback(
            prompts=["Q"],
            reference=["R"],
            options=[["A", "B"]],
            completions=[
                [
                    {
                        "role": "assistant",
                        "content": "Answer: A",
                        "tool_calls": [{"name": "tool"}],
                    }
                ]
            ],
        )


@pytest.mark.parametrize("view", ["completion", "reasoning", "answer_text"])
def test_private_thinking_cannot_supply_a_reward_candidate(view):
    reward = RecordingReward()
    callback = GRPORewardAdapter(reward, candidate_view=view, invalid_score=-7)
    assert callback(
        prompts=["Q"] * 2, reference=["R"] * 2, options=[["A", "B"]] * 2,
        completions=["<think>Plan\nAnswer: B", "<think>Plan\nAnswer: B</think>"],
    ) == [-7, -7]
    assert reward.calls == []


def test_private_answer_does_not_count_for_accuracy():
    assert answer_accuracy_reward(
        completions=["<think>Answer: B", "<think>Answer: B</think>\nAnswer: A"],
        answer_letter=["B", "A"], options=[["A", "B"]] * 2,
    ) == [0, 1]
