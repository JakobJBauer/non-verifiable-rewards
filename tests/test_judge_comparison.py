import json
import math

import pytest

from non_verifiable_rewards.experiments.judge_comparison import (
    distribution,
    evaluate_batch,
    load_journal,
)
from non_verifiable_rewards.rewards.llm_judge import LLMJudgeReward
from test_judge_thinking import TraceScorer


def test_comparison_rating_matches_existing_reward_reduction():
    scorer = TraceScorer()
    expected = LLMJudgeReward(
        scorer, enable_thinking=True, max_thinking_tokens=8
    ).score(
        ["candidate"],
        prompts=["prompt"],
        references=["reference"],
    )[0]
    actual = distribution([math.log(p) for p in [0.05, 0.05, 0.1, 0.1, 0.2]])
    assert actual["score"] == pytest.approx(expected)
    assert actual["rating_token_mass"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        distribution([0] * 5)


def test_one_unfinished_rationale_does_not_discard_successful_batch_members():
    scorer = TraceScorer()
    scorer.traces = [[7, 9], [7] * 8]
    rows = [
        dict(
            key=str(i), dataset="reddit_tldr", prompt="P", candidate="C", reference="R"
        )
        for i in range(2)
    ]
    results = evaluate_batch(scorer, rows, thinking=True, budget=8)
    assert results[0]["status"] == "ok"
    assert results[0]["score"] == pytest.approx(0.675)
    assert results[1]["status"] == "thinking_limit"
    assert results[1]["score"] is None
    assert scorer.scored == [[6, 7, 9, 8]]
    assert results[1]["thinking_token_ids"] == [7] * 8


def test_context_budget_failure_keeps_input_and_has_no_score():
    scorer = TraceScorer()
    scorer.max_length = 10
    rows = [
        dict(key="k", dataset="reddit_tldr", prompt="P", candidate="C", reference="R")
    ]
    result = evaluate_batch(scorer, rows, thinking=True, budget=8)[0]
    assert result["status"] == "context_limit" and result["score"] is None
    assert scorer.scored == []


def test_journal_repairs_only_torn_tail_and_rejects_duplicates(tmp_path):
    path = tmp_path / "scores.jsonl"
    good = json.dumps({"key": "one", "score": 0.5}) + "\n"
    path.write_text(good + '{"key": "unfinished"')
    assert list(load_journal(path)) == ["one"]
    assert path.read_text() == good
    path.write_text(good * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_journal(path)
