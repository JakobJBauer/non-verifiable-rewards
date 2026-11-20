"""Training boundary tests plus real, download-free GRPO/checkpoint integration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from non_verifiable_rewards.training.config import REWARDS, load_config, validate_config
from non_verifiable_rewards.training.data import (
    policy_prompt,
    prepare,
    training_dataset,
)
from non_verifiable_rewards.training.rewards import (
    LM_REWARDS,
    RewardClient,
    RewardFactory,
    WeightedReward,
)
from non_verifiable_rewards.training.state import (
    ensure_sweep,
    verify_data,
    verify_model,
)


def test_default_sweep_has_each_reward_once_and_explicit_scope():
    config = load_config("configs/grpo_qwen3_4b.toml")
    assert [next(iter(r["rewards"])) for r in config["runs"]] == list(REWARDS)
    assert all(len(r["rewards"]) == 1 for r in config["runs"])
    assert config["policy"]["id"] == "Qwen/Qwen3-4B"
    assert config["benchmark"]["pairs_per_dataset"] == 2000
    assert config["benchmark"]["smart_bleurt_prompts"] == 100
    eidf = load_config("configs/eidf_grpo_qwen3_4b.toml")
    assert eidf["runs"] == config["runs"]
    assert eidf["output_dir"].startswith("/workspace/writeable/")


@pytest.mark.parametrize(
    "update",
    [
        {"optimization": {"num_generations": 1}},
        {"optimization": {"num_generations": 3}},
        {"optimization": {"learning_rate": float("nan")}},
        {"runs": [{"name": "../escape", "rewards": {"length": 1}}]},
        {"runs": [{"name": "mix", "rewards": {"missing": 1}}]},
        {"runs": [{"name": "mix", "rewards": {"length": 0}}]},
        {"runs": [{"name": "mix", "rewards": {"answer_accuracy": 1}}]},
        {
            "runs": [
                {
                    "name": "mix",
                    "rewards": {"length": 1},
                    "candidate_views": {"length": "reasoning"},
                }
            ]
        },
        {"benchmark": {"generation_prompts_per_dataset": 2, "smart_bleurt_prompts": 3}},
        {"optimization": {"max_step": 2}},
        {"benchmark": {"enabled": "false"}},
        {"benchmark": {"rewards": ["answer_accuracy"]}},
        {"data": {"datasets": "reddit_tldr"}},
    ],
)
def test_invalid_sweeps_fail_before_loading_models(update):
    with pytest.raises(ValueError):
        validate_config(update)


def test_task_prompts_request_the_right_output_without_duplicate_translation_instruction():
    assert "summary" in policy_prompt("reddit_tldr", "Post")
    assert "story" in policy_prompt("litbench", "Writing prompt")
    source = "Translate the following German text into English:\n\nHallo Welt"
    assert policy_prompt("wmt14_de_en", source) == source
    assert policy_prompt("supergpqa", "Question and options") == "Question and options"


def test_mixture_preserves_batch_alignment_raw_reasoning_and_signed_weights(tmp_path):
    calls = []

    class Client:
        def score(self, name, rows, texts, view):
            calls.append((name, rows, texts, view))
            return {"chrf_pp": [10, 80], "length": [4, 2]}[name]

    tokenizer = SimpleNamespace(
        batch_decode=lambda ids, **kwargs: [
            "Rationale A\nAnswer: B",
            "Rationale B\nAnswer: A",
        ]
    )
    run = {
        "rewards": {"chrf_pp": 0.01, "length": -0.1},
        "candidate_views": {"chrf_pp": "answer_text"},
    }
    callback = WeightedReward(Client(), run, tokenizer, tmp_path / "trace.jsonl")
    kwargs = dict(
        prompts=[[], []],
        completions=["discarded", "discarded"],
        source_prompt=["Q1", "Q2"],
        dataset=["supergpqa"] * 2,
        reference=["Gold1", "Gold2"],
        prompt_id=["p1", "p2"],
        options=[["A", "B"]] * 2,
        answer_letter=["B", "A"],
        completion_ids=[[1], [2]],
    )
    assert callback(**kwargs) == pytest.approx([-0.3, 0.6])
    assert calls[0][1][1]["reference"] == "Gold2"
    assert calls[0][2][0].startswith("Rationale A")
    assert calls[0][3] == "answer_text"
    assert json.loads((tmp_path / "trace.jsonl").read_text())["components"][
        "length"
    ] == [4.0, 2.0]
    kwargs["reference"] = ["one"]
    with pytest.raises(ValueError, match="aligned"):
        callback(**kwargs)


def test_factory_routes_mixed_tasks_and_keeps_invalid_positions(tmp_path, monkeypatch):
    config = validate_config({"output_dir": str(tmp_path)})
    factory = RewardFactory(config)
    seen = []

    class Reward:
        def score(self, texts, *, prompts, references):
            seen.append((texts, prompts, references))
            return [len(text) for text in texts]

    monkeypatch.setattr(factory, "get", lambda name, dataset: Reward())
    rows = [
        dict(dataset=d, prompt=p, reference="R")
        for d, p in [
            ("litbench", "story"),
            ("reddit_tldr", "post"),
            ("litbench", "story2"),
        ]
    ]
    assert factory.score("length", rows, ["abc", "", "hello"]) == [3, -1000, 5]
    assert seen == [(["abc", "hello"], ["story", "story2"], ["R", "R"])]
    row = dict(
        dataset="supergpqa",
        prompt="Q",
        reference="right",
        options=["wrong", "right"],
        answer_letter="B",
    )
    assert factory.score("answer_accuracy", [row] * 2, ["Answer: B", "Answer: A"]) == [
        1,
        0,
    ]
    assert factory.score("length", [row], ["Work\nAnswer: B"], "answer_text") == [5]


def test_invalid_completion_penalty_cannot_flip_sign_in_a_mixture(tmp_path):
    client = SimpleNamespace(score=lambda *args: [-1000.0])
    callback = WeightedReward(
        client, {"rewards": {"length": -1}}, None, tmp_path / "trace.jsonl"
    )
    assert callback(
        prompts=["p"],
        completions=[""],
        source_prompt=["p"],
        dataset=["reddit_tldr"],
        reference=["r"],
        prompt_id=["p"],
        options=[[]],
        answer_letter=[""],
    ) == [-1000]


def test_reward_worker_reports_errors_without_fabricating_scores(tmp_path):
    config = validate_config({"output_dir": str(tmp_path)})
    ensure_sweep(config)
    with RewardClient(config, tmp_path / "logs") as client:
        with pytest.raises(RuntimeError, match="Unknown reward"):
            client.score(
                "unknown",
                [dict(dataset="reddit_tldr", prompt="P", reference="R")],
                ["candidate"],
            )


@pytest.fixture
def tiny_sweep(tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3Config,
        Qwen3ForCausalLM,
        set_seed,
    )

    set_seed(42)
    base = tmp_path / "tiny-qwen"
    words = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "user",
        "assistant",
        "good",
        "bad",
        "story",
        "summary",
        "gold",
        "train",
        "test",
        "one",
        "two",
        "three",
        "four",
        "source",
        "post",
        "reference",
        "the",
        "a",
        "words",
        "answer",
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    backend = Tokenizer(
        WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tokenizer.save_pretrained(base)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(words),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(base)
    train = [
        dict(
            dataset="reddit_tldr",
            prompt_id=f"train-{i}",
            prompt=f"train source {word}",
            reference="gold summary",
            reference_id=f"ref-train-{i}",
            metadata={"split": "train"},
        )
        for i, word in enumerate(["one", "two", "three", "four"])
    ]
    test = [
        dict(
            dataset="reddit_tldr",
            pair_id=f"test-pair-{i}",
            prompt_id=f"test-{i // 2}",
            prompt=f"test source {['one', 'two'][i // 2]}",
            reference="gold summary",
            reference_id=f"ref-test-{i // 2}",
            preferred="good summary words",
            rejected="bad story",
            metadata={"split": "test"},
        )
        for i in range(4)
    ]
    for name, rows in (("train", train), ("test", test)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    config = validate_config(
        {
            "output_dir": str(tmp_path / "sweep"),
            "cache_dir": str(tmp_path / "cache"),
            "policy": {
                "id": str(base),
                "device": "cpu",
                "dtype": "float32",
                "lora": True,
                "lora_r": 2,
                "lora_alpha": 4,
            },
            "data": {
                "datasets": ["reddit_tldr"],
                "train_examples_per_dataset": 4,
                "max_prompt_tokens": 64,
                "local": {
                    "reddit_tldr": {
                        name: str(tmp_path / f"{name}.jsonl")
                        for name in ("train", "test")
                    }
                },
            },
            "optimization": {
                "max_steps": 2,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "num_generations": 4,
                "max_completion_length": 8,
                "save_steps": 1,
                "logging_steps": 1,
                "gradient_checkpointing": False,
                "learning_rate": 0.001,
            },
            "reward_models": {
                "device": "cpu",
                "dtype": "float32",
                "batch_size": 2,
                "max_context_tokens": 4096,
                "lm_id": str(base),
            },
            "benchmark": {
                "pairs_per_dataset": 4,
                "generation_prompts_per_dataset": 2,
                "smart_bleurt_prompts": 1,
                "rewards": ["length", "rouge_l"],
                "max_new_tokens": 8,
                "bootstrap_samples": 100,
            },
            "runs": [{"name": "mixed", "rewards": {"length": 0.1, "rouge_l": 1.0}}],
        }
    )
    return config, tokenizer


def test_frozen_data_never_sends_reference_or_preferences_to_policy(tiny_sweep):
    config, tokenizer = tiny_sweep
    root = ensure_sweep(config)
    manifest = prepare(config, tokenizer)
    assert manifest["counts"]["reddit_tldr"]["train"] == 4
    dataset = training_dataset(config, config["runs"][0], tokenizer)
    assert all("gold" not in row["prompt"][0]["content"] for row in dataset)
    assert (
        "preferred" not in dataset.column_names
        and "rejected" not in dataset.column_names
    )
    assert prepare(config, tokenizer) == manifest
    with (root / "data/reddit_tldr-train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Frozen data changed"):
        verify_data(root)


def test_train_test_overlap_is_rejected(tiny_sweep):
    config, tokenizer = tiny_sweep
    path = Path(config["data"]["local"]["reddit_tldr"]["test"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["prompt"] = "train source one"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="leakage"):
        prepare(config, tokenizer)


def test_fixed_qwen_rewards_share_frozen_weights_and_task_rubrics(
    tiny_sweep, monkeypatch
):
    config, _ = tiny_sweep
    ensure_sweep(config)
    factory = RewardFactory(config)
    for name in LM_REWARDS:
        reward = factory.get(name, "reddit_tldr")
        assert reward.scorer is factory.scorer
        # The full judge rubric is intentionally larger than the tiny model's
        # context; inspect its task selection here, score the other four.
        if name != "llm_judge":
            assert (
                len(
                    factory.score(
                        name,
                        [
                            dict(
                                dataset="reddit_tldr",
                                prompt="post",
                                reference="gold summary",
                            )
                        ],
                        ["good summary"],
                    )
                )
                == 1
            )
    assert factory.get("llm_judge", "wmt14_de_en").task == "translation_de_en"
    assert factory.get("llm_judge", "litbench").task == "story_generation"
    assert all(
        not parameter.requires_grad for parameter in factory.scorer.model.parameters()
    )
    assert not factory.scorer.model.training
    config["policy"]["enable_thinking"] = True
    calls = []

    def template(messages, **kwargs):
        calls.append((messages, kwargs))
        return "serialized prompt"

    monkeypatch.setattr(factory.scorer.tokenizer, "apply_chat_template", template)
    assert (
        factory.get("self_certainty", "litbench").context_builder("story")
        == "serialized prompt"
    )
    assert calls[0][1]["enable_thinking"] is True
    assert "standalone story" in calls[0][0][0]["content"]


def test_real_grpo_resume_export_reload_and_benchmark(tiny_sweep, monkeypatch):
    from trl import GRPOTrainer
    from transformers import TrainerCallback, AutoModelForCausalLM
    from non_verifiable_rewards.training.train import train_run
    from non_verifiable_rewards.training.benchmark import generate, score_generations
    from non_verifiable_rewards.training.report import report

    config, tokenizer = tiny_sweep
    root = ensure_sweep(config)
    prepare(config, tokenizer)
    run = config["runs"][0]
    original_train = GRPOTrainer.train

    class StopAfterOne(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True
            return control

    def interrupted(trainer, *args, **kwargs):
        trainer.add_callback(StopAfterOne())
        return original_train(trainer, *args, **kwargs)

    monkeypatch.setattr(GRPOTrainer, "train", interrupted)
    with pytest.raises(RuntimeError, match="requested number"):
        train_run(config, run)
    assert (root / "runs/mixed/checkpoints/checkpoint-1/trainer_state.json").exists()
    monkeypatch.setattr(GRPOTrainer, "train", original_train)
    done = train_run(config, run)
    assert done["steps"] == 2 and done["reward_models_frozen"]
    state = json.loads((root / "runs/mixed/optimization_complete.json").read_text())
    assert state["resumed_from"].endswith("checkpoint-1")
    exported = AutoModelForCausalLM.from_pretrained(
        root / "runs/mixed/model", local_files_only=True
    )
    assert exported.config.model_type == "qwen3"
    assert not (root / "runs/mixed/model/adapter_config.json").exists()
    import torch

    base = AutoModelForCausalLM.from_pretrained(
        config["policy"]["id"], local_files_only=True
    )
    assert any(
        not torch.equal(value, base.state_dict()[key])
        for key, value in exported.state_dict().items()
    )
    assert train_run(config, run) == verify_model(root / "runs/mixed")
    for name in ("base", "mixed"):
        result = generate(config, name)
        assert result["generations"] == 2 and result["preference_pairs"] == 4
        assert generate(config, name) == result
        scores = score_generations(config, name)
        assert scores["counts"] == {"length": 2, "rouge_l": 2}
        assert score_generations(config, name) == scores
    scope = report(config, ["mixed"])
    assert scope["models"] == ["base", "mixed"]
    assert (root / "report.html").exists()
    import pandas as pd

    table = pd.read_parquet(root / "tables/policy_preference_summary.parquet")
    assert len(table) == 4 and table.examples.eq(4).all()
    assert table.loc[table.model.eq("base"), "delta_vs_base"].eq(0).all()
    # Recover the narrow interruption window after export but before its marker.
    (root / "runs/mixed/trained.json").unlink()
    assert train_run(config, run) == done


def test_full_finetuning_recovers_a_finished_checkpoint_without_extra_updates(
    tiny_sweep, monkeypatch
):
    from trl import GRPOTrainer
    from non_verifiable_rewards.training.train import train_run

    config, tokenizer = tiny_sweep
    config["policy"]["lora"] = False
    config["optimization"]["max_steps"] = 1
    config["optimization"]["warmup_ratio"] = 0
    prepare(config, tokenizer)
    original_save = GRPOTrainer.save_model

    def interrupted_save(trainer, output_dir=None, **kwargs):
        if output_dir and str(output_dir).endswith("/policy_state"):
            raise RuntimeError("Interrupted after final optimizer checkpoint")
        return original_save(trainer, output_dir, **kwargs)

    monkeypatch.setattr(GRPOTrainer, "save_model", interrupted_save)
    with pytest.raises(RuntimeError, match="final optimizer checkpoint"):
        train_run(config, config["runs"][0])
    trace = Path(config["output_dir"]) / "runs/mixed/reward_trace.jsonl"
    original_trace = trace.read_bytes()
    monkeypatch.setattr(GRPOTrainer, "save_model", original_save)
    result = train_run(config, config["runs"][0])
    assert result["steps"] == 1
    assert "model.safetensors" in result["model_files"]
    assert trace.read_bytes() == original_trace


def test_cli_runs_all_stages_and_skips_completed_training(tiny_sweep, tmp_path):
    import os
    import subprocess
    import sys

    config, _ = tiny_sweep
    config["optimization"].update(max_steps=1, warmup_ratio=0)

    def toml_value(value):
        if isinstance(value, dict):
            return (
                "{"
                + ", ".join(
                    json.dumps(k) + " = " + toml_value(v) for k, v in value.items()
                )
                + "}"
            )
        if isinstance(value, list):
            return "[" + ", ".join(map(toml_value, value)) + "]"
        return json.dumps(value)

    path = tmp_path / "sweep.toml"
    path.write_text("\n".join(k + " = " + toml_value(v) for k, v in config.items()))
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    command = [
        sys.executable,
        "-m",
        "non_verifiable_rewards.training",
        "run",
        "--config",
        str(path),
        "--run",
        "mixed",
    ]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    root = Path(config["output_dir"])
    trace = (root / "runs/mixed/reward_trace.jsonl").read_bytes()
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert trace == (root / "runs/mixed/reward_trace.jsonl").read_bytes()
    assert json.loads((root / "report_scope.json").read_text())["models"] == [
        "base",
        "mixed",
    ]
