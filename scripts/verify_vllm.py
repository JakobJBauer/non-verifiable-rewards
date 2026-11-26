#!/usr/bin/env python3
"""GPU acceptance audit. Writes measurements and fails on violated invariants.

Run each mode in a fresh process; see docs/vllm_inference.md.
"""

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path


def memory():
    import psutil
    import torch

    torch.cuda.synchronize()
    process = psutil.Process()
    return dict(
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        device_used=torch.cuda.device_memory_used(),
        rss=sum(
            p.memory_info().rss
            for p in [process, *process.children(recursive=True)]
            if p.is_running()
        ),
    )


def parity(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from non_verifiable_rewards.inference.policy import TransformersPolicy
    from non_verifiable_rewards.inference.scorer import VLLMScorer
    from non_verifiable_rewards.rewards.llm_judge import LLMJudgeReward

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    model.requires_grad_(False)
    hf = TransformersPolicy(model, tokenizer, batch_size=1, max_length=4096)
    texts = [
        "Translate into English: Guten Morgen.",
        "What is 2 + 2? Reply with only the number.",
        "Complete the sequence: 1, 2, 3,",
        "What is the capital of France? Answer in one word.",
    ]
    contexts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in texts
    ]
    prompts = [tokenizer.encode(text, add_special_tokens=False) for text in contexts]
    targets = ["Good morning.", "4", "4, 5, 6.", "Paris"]
    start = time.perf_counter()
    hf_ids = hf.generate_ids(prompts, max_new_tokens=32)
    hf_seconds = time.perf_counter() - start
    throughput_prompts = [prompts[2]] * 8
    start = time.perf_counter()
    hf_batch = hf.generate_ids(throughput_prompts, max_new_tokens=64)
    hf_batch_seconds = time.perf_counter() - start
    hf_logp = hf.log_likelihood(contexts, targets, reduction="mean")
    hf_labels = hf.label_logprobs(contexts, ["0", "1", "2", "3", "4"])
    hf_tie = hf.label_logprobs([contexts[2]], ["Let", "We"])
    judge_kwargs = dict(
        candidates=["The cat sat on the mat.", "A dog flew to the moon."],
        prompts=["A cat sat quietly on the mat."] * 2,
        references=["The cat sat on the mat."] * 2,
    )
    hf_judge = LLMJudgeReward(hf).score(**judge_kwargs)
    hf_certainty = hf.self_certainty(contexts, targets)
    model.cpu()
    torch.cuda.empty_cache()
    start = time.perf_counter()
    with VLLMScorer(
        args.model,
        max_length=4096,
        batch_size=16,
        worker_extension_cls="non_verifiable_rewards.inference.verification.WeightVerification",
    ) as engine:
        initialization_seconds = time.perf_counter() - start
        with engine.awake():
            initial_weight_hash = engine.llm.collective_rpc("weight_digest")[0]
        start = time.perf_counter()
        vl_ids = engine.generate_ids(prompts, max_new_tokens=32)
        vl_seconds = time.perf_counter() - start
        with engine.awake():
            start = time.perf_counter()
            vl_batch = engine.generate_ids(throughput_prompts, max_new_tokens=64)
            vl_batch_seconds = time.perf_counter() - start
        vl_logp = engine.log_likelihood(contexts, targets, reduction="mean")
        vl_sum = engine.log_likelihood(contexts, targets)
        vl_labels = engine.label_logprobs(contexts, ["0", "1", "2", "3", "4"])
        vl_tie = engine.label_logprobs([contexts[2]], ["Let", "We"])
        vl_judge = LLMJudgeReward(engine).score(**judge_kwargs)
        vl_certainty = engine.self_certainty(contexts, targets)
        cycles = []
        for i in range(args.cycles):
            start = time.perf_counter()
            with engine.awake():
                awake_at = time.perf_counter()
                ids = engine.generate_ids(prompts, max_new_tokens=32)
                generated_at = time.perf_counter()
            slept_at = time.perf_counter()
            if ids != vl_ids:
                with engine.awake():
                    final_weight_hash = engine.llm.collective_rpc("weight_digest")[0]
                save(
                    args,
                    {
                        "hf_ids": hf_ids,
                        "vllm_ids": vl_ids,
                        "cycle_ids": ids,
                        "cycle": i,
                        "initial_weight_hash": initial_weight_hash,
                        "final_weight_hash": final_weight_hash,
                        "hf_logp": hf_logp,
                        "hf_tie": hf_tie,
                        "vllm_tie": vl_tie,
                        "vllm_logp": vl_logp,
                        "hf_judge": hf_judge,
                        "vllm_judge": vl_judge,
                        "texts": [tokenizer.decode(row) for row in ids],
                        "initial_texts": [tokenizer.decode(row) for row in vl_ids],
                    },
                )
                raise AssertionError("Frozen outputs changed across sleep/wake cycles")
            cycles.append(
                dict(
                    cycle=i,
                    seconds=slept_at - start,
                    wake_seconds=awake_at - start,
                    generation_seconds=generated_at - awake_at,
                    sleep_seconds=slept_at - generated_at,
                    **memory(),
                )
            )
        with engine.awake():
            final_weight_hash = engine.llm.collective_rpc("weight_digest")[0]
        assert initial_weight_hash == final_weight_hash, "Frozen weights changed"
        result = dict(
            initial_weight_hash=initial_weight_hash,
            final_weight_hash=final_weight_hash,
            hf_ids=hf_ids,
            vllm_ids=vl_ids,
            greedy_equal=hf_ids == vl_ids,
            hf_logp=hf_logp,
            vllm_logp=vl_logp,
            hf_labels=hf_labels,
            hf_tie=hf_tie,
            vllm_tie=vl_tie,
            vllm_labels=vl_labels,
            hf_judge=hf_judge,
            vllm_judge=vl_judge,
            hf_certainty=hf_certainty,
            vllm_certainty=vl_certainty,
            hf_batch_seconds=hf_batch_seconds,
            vllm_batch_seconds=vl_batch_seconds,
            hf_batch_tokens=sum(map(len, hf_batch)),
            vllm_batch_tokens=sum(map(len, vl_batch)),
            hf_seconds=hf_seconds,
            vllm_seconds=vl_seconds,
            initialization_seconds=initialization_seconds,
            cycles=cycles,
        )
        save(args, result)
        # Exact generation is required for the factual probes. The intentionally
        # ambiguous sequence probe has a tied BF16 argmax in vLLM; different
        # kernels/tie-breaking can change its entire trajectory. Preserve the
        # mismatch in greedy_equal and require an actual tie in either engine.
        assert all(hf_ids[i] == vl_ids[i] for i in (0, 1, 3))
        assert all(ids[: len(vl_ids[2])] == vl_ids[2] for ids in vl_batch), (
            "Batch size changed invariant vLLM outputs"
        )
        if hf_ids[2] != vl_ids[2]:
            gaps = [abs(row[0] - row[1]) for row in (hf_tie[0], vl_tie[0])]
            assert min(gaps) < 1e-6, (hf_tie, vl_tie)
            assert max(abs(a - b) for a, b in zip(hf_tie[0], vl_tie[0])) < 0.25

        assert max(abs(a - b) for a, b in zip(hf_logp, vl_logp)) < 0.08
        # BF16 rounds large negative logits coarsely. Validate probabilities
        # directly as well as bounding the log-probability error.
        label_pairs = [
            (a, b) for ar, br in zip(hf_labels, vl_labels) for a, b in zip(ar, br)
        ]
        assert max(abs(math.exp(a) - math.exp(b)) for a, b in label_pairs) < 0.005
        assert max(abs(a - b) for a, b in label_pairs) < 0.5
        assert max(abs(a - b) for a, b in zip(hf_judge, vl_judge)) < 0.01
        assert max(abs(a - b) for a, b in zip(hf_certainty, vl_certainty)) < 1e-5
        assert all(
            abs(total - mean * len(tokenizer.encode(target, add_special_tokens=False)))
            < 1e-4
            for total, mean, target in zip(vl_sum, vl_logp, targets)
        )
        stable = cycles[3:]
        for key, allowance in (("device_used", 64 * 1024**2), ("rss", 128 * 1024**2)):
            assert (
                max(x[key] for x in stable) - min(x[key] for x in stable) < allowance
            ), (key, stable)
    return result


def check_live_weights(trainer):
    """Compare every current HF tensor to the actual packed vLLM model."""
    import hashlib
    import torch
    from non_verifiable_rewards.inference.training import policy_weights

    hf = trainer.vllm_generation.model
    vl = trainer.vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model
    config = hf.config
    qsize = config.num_attention_heads * config.head_dim
    kvsize = config.num_key_value_heads * config.head_dim
    count = 0
    checksum = hashlib.sha256()
    for name, expected in policy_weights(hf):
        target, offset = name, 0
        for old, new, start in (
            ("q_proj", "qkv_proj", 0),
            ("k_proj", "qkv_proj", qsize),
            ("v_proj", "qkv_proj", qsize + kvsize),
            ("gate_proj", "gate_up_proj", 0),
            ("up_proj", "gate_up_proj", config.intermediate_size),
        ):
            if f".{old}." in name:
                target = name.replace(f".{old}.", f".{new}.")
                offset = start
                break
        actual = vl.get_parameter(target)
        if actual.shape != expected.shape or offset:
            actual = actual[offset : offset + expected.shape[0]]
        assert torch.equal(actual, expected), f"Stale or incorrect vLLM weights: {name}"
        count += 1
        # Hash all adapter tensors separately: every update must be observable,
        # even if a very small change rounds away in the merged BF16 weights.
    for name, value in hf.named_parameters():
        if value.requires_grad:
            checksum.update(value.detach().float().cpu().numpy().tobytes())
    return dict(checked_tensors=count, trainable_sha256=checksum.hexdigest())


def grpo(args):
    import torch
    import trl
    from transformers import TrainerCallback
    from non_verifiable_rewards.training.config import validate_config
    from non_verifiable_rewards.training.data import prepare
    from non_verifiable_rewards.training.train import train_run, load_tokenizer
    from non_verifiable_rewards.inference.training import TrainingSwitch

    torch.set_num_threads(4)
    if args.mode == "grpo-full":
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

        tiny_path = args.output / "tiny-base"
        tiny_path.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tokenizer.save_pretrained(tiny_path)
        torch.manual_seed(1729)
        tiny = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                max_position_embeddings=4096,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                tie_word_embeddings=True,
            )
        )
        tiny.save_pretrained(tiny_path)
        args.model = str(tiny_path)
    args.output.mkdir(parents=True, exist_ok=True)
    train_path, test_path = args.output / "train.jsonl", args.output / "test.jsonl"
    rows = [
        dict(
            dataset="reddit_tldr",
            prompt_id=f"train-{i}",
            prompt=f"Write a short, original story inspired by this idea: {text}",
            reference=text,
            reference_id=f"ref-{i}",
            metadata={"split": "train"},
        )
        for i, text in enumerate(
            [
                "Cats sleep a lot.",
                "Dogs enjoy walks.",
                "The train arrived late.",
                "The shop opens at nine.",
                "It rained all day.",
                "The river flooded.",
                "The team won the match.",
                "The cake tasted sweet.",
            ]
        )
    ]
    tests = [
        dict(
            dataset="reddit_tldr",
            pair_id=f"test-{i}",
            prompt_id=f"test-{i}",
            prompt=f"A held-out story about a bird number {i}.",
            reference="A bird sang.",
            reference_id=f"test-ref-{i}",
            preferred="A bird sang.",
            rejected="A fish swam.",
            metadata={"split": "test"},
        )
        for i in range(2)
    ]
    train_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    test_path.write_text("".join(json.dumps(row) + "\n" for row in tests))
    config = validate_config(
        {
            "output_dir": str(args.output / "sweep"),
            "cache_dir": str(args.output / "cache"),
            "policy": {"id": args.model, "lora": True, "lora_r": 4, "lora_alpha": 8},
            "data": {
                "datasets": ["reddit_tldr"],
                "train_examples_per_dataset": 8,
                "max_prompt_tokens": 256,
                "local": {
                    "reddit_tldr": {"train": str(train_path), "test": str(test_path)}
                },
            },
            "optimization": {
                "max_steps": 24,
                "learning_rate": 0.001,
                "temperature": 1.0,
                "top_p": 0.99,
                "warmup_ratio": 0,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "num_generations": 4,
                "max_completion_length": 32,
                "save_steps": 3,
                "logging_steps": 1,
            },
            "reward_models": {"lm_id": args.model, "max_context_tokens": 4096},
            "inference": {
                "backend": "vllm",
                "gpu_memory_utilization": 0.35,
                "reward_gpu_memory_utilization": 0.35,
            },
            "benchmark": {
                "pairs_per_dataset": 2,
                "generation_prompts_per_dataset": 2,
                "smart_bleurt_prompts": 1,
                "max_new_tokens": 32,
                "rewards": ["length", "llm_judge"],
            },
            "runs": [{"name": "audit", "rewards": {"length": 0.1, "llm_judge": 1.0}}],
        }
    )
    if args.mode == "grpo-full":
        config["policy"]["lora"] = False
        config["inference"]["gpu_memory_utilization"] = 0.08
        config["optimization"]["max_steps"] = 6
        config["runs"][0]["rewards"] = {"length": 1.0}
    tokenizer = load_tokenizer(args.model)
    prepare(config, tokenizer)
    records = []
    original_sync = TrainingSwitch.sync_weights

    def audited_sync(switch):
        original_sync(switch)
        record = dict(
            step=switch.trainer.state.global_step,
            **check_live_weights(switch.trainer),
            memory=memory(),
        )
        records.append(record)
        save(args, {"weight_checks": records})

    TrainingSwitch.sync_weights = audited_sync
    original_train = trl.GRPOTrainer.train

    class Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 3:
                control.should_training_stop = True

    def train(trainer, *a, **kw):
        if args.mode == "grpo-start":
            trainer.add_callback(Stop())
        return original_train(trainer, *a, **kw)

    trl.GRPOTrainer.train = train
    try:
        trained = train_run(config, config["runs"][0])
    except RuntimeError as exc:
        if args.mode != "grpo-start" or "requested number" not in str(exc):
            raise
        trained = {"interrupted_at_step": 3}
    finally:
        TrainingSwitch.sync_weights = original_sync
        trl.GRPOTrainer.train = original_train
    steps = [row["step"] for row in records]
    expected = (
        list(range(6))
        if args.mode == "grpo-full"
        else list(range(3))
        if args.mode == "grpo-start"
        else list(range(3, 24))
    )
    assert steps == expected, (steps, expected)
    assert len({row["trainable_sha256"] for row in records}) > 1, (
        "The audit did not exercise a changed policy"
    )
    if args.mode == "grpo-resume":
        initial = json.loads((args.output / "grpo-start.json").read_text())[
            "weight_checks"
        ]
        assert records[0]["trainable_sha256"] != initial[0]["trainable_sha256"]
        assert trained["steps"] == 24
        steady = records[3:]
        for key, allowance in (("device_used", 128 * 1024**2), ("rss", 256 * 1024**2)):
            samples = [row["memory"][key] for row in steady]
            assert max(samples) - min(samples) < allowance, (key, samples)
    result = {"weight_checks": records, "trained": trained, "memory": memory()}
    save(args, result)
    return result


def save(args, result):
    result["packages"] = {
        p: importlib.metadata.version(p)
        for p in ("torch", "transformers", "vllm", "trl", "peft")
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n")


def benchmark(args):
    from non_verifiable_rewards.training.benchmark import generate

    config = json.loads((args.output / "sweep" / "sweep.json").read_text())["config"]
    start = time.perf_counter()
    result = generate(config, "audit")
    seconds = time.perf_counter() - start
    assert generate(config, "audit") == result, "Completed benchmark did not resume"
    return {"benchmark": result, "seconds": seconds, "memory": memory()}


def context_limit(args):
    import torch
    from non_verifiable_rewards.inference.scorer import VLLMScorer

    # Reserve the equivalent of a resident 4B BF16 training policy, then check
    # that the production reward budget admits its configured full context.
    resident_policy = torch.empty(4 * 1024**3, dtype=torch.bfloat16, device="cuda")
    with VLLMScorer(args.model, max_length=32768) as engine:
        context = " x" * 32767
        target = " y"
        ids, _ = engine._encode(context, target, 0)
        assert len(ids) == 32768
        start = time.perf_counter()
        value = engine.log_likelihood([context], [target])[0]
        result = {
            "tokens": len(ids),
            "log_probability": value,
            "seconds": time.perf_counter() - start,
            "resident_policy_bytes": resident_policy.numel()
            * resident_policy.element_size(),
            "memory": memory(),
        }
        assert math.isfinite(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "parity",
            "grpo-start",
            "grpo-resume",
            "grpo-full",
            "benchmark",
            "context",
        ],
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=20)
    args = parser.parse_args()
    if args.mode == "parity":
        result = parity(args)
    elif args.mode == "benchmark":
        result = benchmark(args)
    elif args.mode == "context":
        result = context_limit(args)
    else:
        result = grpo(args)
    result["passed"] = True
    save(args, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("cycles", "hf_ids", "vllm_ids")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
