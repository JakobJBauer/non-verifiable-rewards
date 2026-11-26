"""Switching invariants independent of CUDA; real GPU audits live in scripts/."""

import copy
from types import SimpleNamespace

import pytest

from non_verifiable_rewards.inference import grpo_inference_options, use_vllm
from non_verifiable_rewards.training.config import validate_config


def test_backend_selection_is_explicit_and_never_silently_falls_back():
    config = validate_config({})
    assert use_vllm(config)
    assert not use_vllm(config, device="cpu")
    options = grpo_inference_options(config)
    assert options["vllm_enable_sleep_mode"]
    assert options["vllm_max_model_length"] == 5120
    config["inference"]["backend"] = "transformers"
    assert not grpo_inference_options(config)["use_vllm"]


@pytest.mark.parametrize(
    "inference",
    [
        {"backend": "missing"},
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": float("nan")},
        {"reward_gpu_memory_utilization": 1},
        {"batch_size": True},
        {"batch_size": 0},
    ],
)
def test_invalid_inference_config(inference):
    with pytest.raises(ValueError):
        validate_config({"inference": inference})


def test_lora_sync_preserves_base_exactly_and_follows_optimizer_updates():
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from non_verifiable_rewards.inference.training import policy_weights

    torch.manual_seed(8)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            tie_word_embeddings=False,
        )
    ).to(torch.bfloat16)
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.5
    )
    frozen = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }
    previous = None
    for _ in range(3):
        model(
            input_ids=torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]])
        ).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        actual = {name: value.clone() for name, value in policy_weights(model)}
        reference = copy.deepcopy(model).merge_and_unload().state_dict()
        assert all(
            torch.equal(value, reference[name]) for name, value in actual.items()
        )
        if previous is not None:
            assert not torch.equal(
                actual["model.layers.0.self_attn.q_proj.weight"], previous
            )
        previous = actual["model.layers.0.self_attn.q_proj.weight"]
        for _ in range(20):
            for _, value in policy_weights(model):
                pass
        assert all(
            torch.equal(p, frozen[name])
            for name, p in model.named_parameters()
            if name in frozen
        )


def test_switch_syncs_after_update_and_sleep_and_releases_on_failure(
    tmp_path, monkeypatch
):
    import torch
    from non_verifiable_rewards.inference.training import TrainingSwitch

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    copied = []
    model = torch.nn.Linear(1, 1, bias=False)

    class Backend:
        _llm_weights_sleeping = True
        fail = False

        def sync_weights(self):
            self._llm_weights_sleeping = False
            copied.append(next(self._iter_named_params())[1].clone())

        def generate(self):
            if self.fail:
                raise RuntimeError("generation failed")
            return copied[-1]

    backend = Backend()
    backend.model = model
    backend._iter_named_params = model.named_parameters
    backend.llm = SimpleNamespace(sleep=lambda **kwargs: None)
    trainer = SimpleNamespace(
        vllm_generation=backend, state=SimpleNamespace(global_step=0)
    )
    switch = TrainingSwitch(trainer, tmp_path / "trace.jsonl")
    first = backend.generate()
    assert backend._llm_weights_sleeping
    with torch.no_grad():
        model.weight.add_(1)
    trainer.state.global_step = 1
    second = backend.generate()
    assert torch.equal(second, first + 1)
    assert switch.synced_step == 1
    backend.generate()  # Level-2 sleep discards weights, even in the same step.
    assert switch.sync_count == 3
    backend.fail = True
    with pytest.raises(RuntimeError, match="generation failed"):
        backend.generate()
    assert backend._llm_weights_sleeping


def test_nested_frozen_reward_calls_share_one_wake_and_sleep_even_on_error():
    from non_verifiable_rewards.inference.scorer import VLLMScorer

    calls = []
    scorer = object.__new__(VLLMScorer)
    scorer._sleeping = True
    scorer._sleep = True
    scorer.llm = SimpleNamespace(
        wake_up=lambda: calls.append("wake"),
        sleep=lambda **kwargs: calls.append("sleep"),
    )
    with pytest.raises(RuntimeError, match="failed"):
        with scorer.awake():
            with scorer.awake():
                pass
            assert not scorer._sleeping
            raise RuntimeError("failed")
    assert calls == ["wake", "sleep"]
    assert scorer._sleeping


def test_trl_constructor_receives_policy_dtype_and_is_restored(monkeypatch):
    from trl.generation import vllm_generation
    from non_verifiable_rewards.inference.training import engine_dtype

    import os

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    constructor = lambda **kwargs: kwargs
    monkeypatch.setattr(vllm_generation, "LLM", constructor, raising=False)
    with pytest.raises(RuntimeError):
        with engine_dtype("float32"):
            assert os.environ["VLLM_BATCH_INVARIANT"] == "0"
            assert vllm_generation.LLM(model="local")["dtype"] == "float32"
            raise RuntimeError("constructor failed")
    assert vllm_generation.LLM is constructor
    assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
