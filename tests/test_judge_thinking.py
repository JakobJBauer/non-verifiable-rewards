"""Thinking ends before rating, and both backends score the exact trace IDs."""

import math
import sys
from types import SimpleNamespace

import pytest

from non_verifiable_rewards.rewards import CausalLMScorer, LLMJudgeReward
from non_verifiable_rewards.rewards.llm_judge import validate_judge_config
from non_verifiable_rewards.training.config import load_config, validate_config


@pytest.mark.parametrize(
    "path",
    [
        "configs/grpo_qwen3_4b.toml",
        "configs/eidf_grpo_qwen3_4b.toml",
        "configs/eidf_grpo_vllm_qwen3_4b.toml",
    ],
)
def test_standard_configs_default_to_thinking_policy_and_direct_judge(path):
    config = load_config(path)
    assert config["policy"]["enable_thinking"] is True
    assert config["judge"] == {"enable_thinking": False, "max_thinking_tokens": 1024}


@pytest.mark.parametrize("policy", [True, False])
@pytest.mark.parametrize("judge", [True, False])
def test_policy_and_judge_switches_are_independent(policy, judge):
    config = validate_config(
        {
            "policy": {"enable_thinking": policy},
            "judge": {"enable_thinking": judge},
        }
    )
    assert config["policy"]["enable_thinking"] is policy
    assert config["judge"]["enable_thinking"] is judge


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"enable_thinking": "false"},
        {"max_thinking_tokens": True},
        {"max_thinking_tokens": 0},
        {"max_thinking_tokens": 1.5},
        {"unknown": True},
    ],
)
def test_invalid_judge_settings_fail_before_loading_models(raw):
    with pytest.raises((TypeError, ValueError)):
        validate_judge_config(raw)
    with pytest.raises((TypeError, ValueError)):
        validate_config({"judge": raw})


class TraceTokenizer:
    bos_token_id = None
    eos_token_id = pad_token_id = 0
    all_special_ids = (0,)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is True
        return "assistant\n"

    def encode(self, text, **kwargs):
        return {
            "</think>": [9],
            "\n\n": [8],
            **{str(i): [i + 1] for i in range(5)},
        }.get(text, [6])

    def decode(self, ids, **kwargs):
        # Deliberately not reversible: a generated trace must never be re-encoded.
        return "</think>" if ids == [9] else "a decoded rationale </think>"


class TraceScorer(CausalLMScorer):
    def __init__(self):
        self.tokenizer = TraceTokenizer()
        self.max_length = 100
        self.traces = [[7, 9], [7, 7, 9]]
        self.scored = []

    def generate_ids(self, prompts, **kwargs):
        self.prompts, self.options = prompts, kwargs
        return self.traces[: len(prompts)]

    def label_logprobs_from_ids(self, prompts, labels):
        self.scored = prompts
        return [[math.log(p) for p in [0.05, 0.05, 0.1, 0.1, 0.2]]] * len(prompts)


def test_training_factory_forwards_judge_settings_independently_of_policy():
    from non_verifiable_rewards.training.rewards import RewardFactory

    factory = RewardFactory(
        validate_config(
            {
                "policy": {"enable_thinking": False},
                "judge": {"enable_thinking": True, "max_thinking_tokens": 8},
            }
        )
    )
    factory.scorer = TraceScorer()
    reward = factory.get("llm_judge", "reddit_tldr")
    assert reward.enable_thinking and reward.max_thinking_tokens == 8
    assert reward.score(["A"], references=["R"], prompts=["P"]) == pytest.approx(
        [0.675]
    )


def test_offline_experiment_config_accepts_the_same_judge_table(tmp_path):
    from pathlib import Path

    from non_verifiable_rewards.experiments.io import load_config as load_experiment

    path = tmp_path / "experiment.toml"
    base = Path("configs/local_cpu.toml").read_text()
    path.write_text(
        base + "\n[judge]\nenable_thinking = true\nmax_thinking_tokens = 72\n"
    )
    assert load_experiment(path)["judge"] == {
        "enable_thinking": True,
        "max_thinking_tokens": 72,
    }
    path.write_text(base + "\n[judge]\nenable_thinking = 'false'\n")
    with pytest.raises(ValueError, match="boolean"):
        load_experiment(path)


def test_thinking_judge_scores_after_exact_closed_trace_and_records_it():
    scorer = TraceScorer()
    results = LLMJudgeReward(
        scorer, enable_thinking=True, max_thinking_tokens=8
    ).score_with_details(
        ["A", "B"],
        references=["R"] * 2,
        prompts=["P"] * 2,
    )
    assert scorer.options == {"max_new_tokens": 8, "stop_token_id": 9}
    assert scorer.scored == [[6, 7, 9, 8], [6, 7, 7, 9, 8]]
    assert [r["score"] for r in results] == pytest.approx([0.675] * 2)
    assert [r["thinking_token_ids"] for r in results] == scorer.traces
    assert [r["conditioning_tokens"] for r in results] == [4, 5]
    assert all(r["enable_thinking"] and r["max_thinking_tokens"] == 8 for r in results)
    assert (
        results[0]["judge_context_token_ids_sha256"]
        != results[1]["judge_context_token_ids_sha256"]
    )


@pytest.mark.parametrize("trace", [[], [7, 0], [7] * 8, [7] * 8 + [9]])
def test_unfinished_or_overbudget_thinking_never_becomes_a_rating(trace):
    scorer = TraceScorer()
    scorer.traces = [trace]
    with pytest.raises(ValueError, match="No rating was scored"):
        LLMJudgeReward(scorer, enable_thinking=True, max_thinking_tokens=8).score(
            ["A"],
            references=["R"],
            prompts=["P"],
        )
    assert scorer.scored == []


def test_judge_reserves_space_for_trace_separator_and_rating():
    scorer = TraceScorer()
    scorer.max_length = 10  # prompt=1, trace budget=8, separator=1, rating=1
    with pytest.raises(ValueError, match="budget exceeds"):
        LLMJudgeReward(scorer, enable_thinking=True, max_thinking_tokens=8).score(
            ["A"],
            references=["R"],
            prompts=["P"],
        )
    assert scorer.scored == []
    assert not hasattr(scorer, "prompts")


def test_vllm_stops_at_delimiter_then_requests_unconstrained_label_logprobs(
    monkeypatch,
):
    from non_verifiable_rewards.inference.scorer import VLLMScorer

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace)
    )
    scorer = object.__new__(VLLMScorer)
    scorer.tokenizer = TraceTokenizer()
    scorer.max_length, scorer.batch_size = 100, 2
    scorer._generation_config = SimpleNamespace(eos_token_id=[0, 10])
    scorer._sleeping, scorer._sleep = True, True
    calls = []

    def generate(prompts, params, **kwargs):
        calls.append((prompts, params))
        if params.max_tokens > 1:
            assert params.temperature == 0 and params.stop_token_ids == [0, 10, 9]
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7, 9])])]
        assert params.logprob_token_ids == [1, 2, 3, 4, 5]
        assert not hasattr(params, "allowed_token_ids")
        assert params.temperature == 1
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        logprobs=[
                            {
                                i + 1: SimpleNamespace(logprob=math.log(p))
                                for i, p in enumerate([0.05, 0.05, 0.1, 0.1, 0.2])
                            }
                        ]
                    )
                ]
            )
        ]

    scorer.llm = SimpleNamespace(
        generate=generate, wake_up=lambda: None, sleep=lambda **kw: None
    )
    score = LLMJudgeReward(scorer, enable_thinking=True, max_thinking_tokens=8).score(
        ["A"],
        references=["R"],
        prompts=["P"],
    )
    assert score == pytest.approx([0.675])
    assert calls[0][0] == [{"prompt_token_ids": [6]}]
    assert calls[1][0] == [{"prompt_token_ids": [6, 7, 9, 8]}]
    assert scorer._sleeping


def test_pytorch_generation_stops_before_rating_and_scores_raw_token_context():
    torch = pytest.importorskip("torch")
    from transformers import LogitsProcessorList, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(42)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=11,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=0,
            pad_token_id=0,
        )
    ).eval()
    original = model.generate

    def force_trace(ids, scores):
        # Force a known trajectory; the real generation loop must stop at 9,
        # leaving the proposed rating (5) ungenerated.
        token = [7, 9, 5][min(ids.shape[1] - 1, 2)]
        scores[:] = -torch.inf
        scores[:, token] = 0
        return scores

    model.generate = lambda **kwargs: original(
        **kwargs,
        logits_processor=LogitsProcessorList([force_trace]),
    )
    scorer = CausalLMScorer(model, TraceTokenizer(), max_length=100)
    result = LLMJudgeReward(
        scorer, enable_thinking=True, max_thinking_tokens=8
    ).score_with_details(
        ["A"],
        references=["R"],
        prompts=["P"],
    )[0]
    assert result["thinking_token_ids"] == [7, 9]
    with torch.no_grad():
        logp = (
            model(input_ids=torch.tensor([[6, 7, 9, 8]]))
            .logits[0, -1]
            .float()
            .log_softmax(-1)
        )
    labels = logp[1:6].softmax(-1)
    assert list(result["rating_probabilities"].values()) == pytest.approx(
        labels.tolist()
    )
    assert result["rating_token_mass"] == pytest.approx(logp[1:6].exp().sum().item())
    assert all(p.grad is None for p in model.parameters())
