#!/usr/bin/env python3
"""GPU acceptance checks for Qwen thinking; run each mode in a fresh process.

python scripts/verify_thinking.py judge --model /path/to/Qwen3-4B --output /path/to/audit
python scripts/verify_thinking.py grpo --backend vllm --model ... --output ...
python scripts/verify_thinking.py grpo --backend transformers --model ... --output ...
"""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path


def judge_check(args):
    import torch

    from non_verifiable_rewards.inference.scorer import VLLMScorer
    from non_verifiable_rewards.rewards import CausalLMScorer, LLMJudgeReward
    from non_verifiable_rewards.rewards.llm_judge import RATING_LABELS, judge_context
    from non_verifiable_rewards.training.data import render_prompt

    torch.set_num_threads(4)
    records, fixed_contexts, fixed_logps = {}, [], []
    for backend in ("transformers", "vllm"):
        if backend == "transformers":
            scorer = CausalLMScorer.from_pretrained(
                args.model,
                device="cuda",
                max_length=4096,
                model_kwargs={
                    "dtype": torch.bfloat16,
                    "local_files_only": True,
                    "attn_implementation": "sdpa",
                },
                tokenizer_kwargs={"local_files_only": True},
            )
        else:
            scorer = VLLMScorer(args.model, max_length=4096, batch_size=2)
        tokenizer = scorer.tokenizer
        records[backend] = {}
        for thinking in (False, True):
            prompt = render_prompt(
                tokenizer,
                {
                    "dataset": "wmt14_de_en",
                    "prompt": "Guten Morgen.",
                },
                thinking,
            )
            ids = scorer.generate_ids(
                [tokenizer.encode(prompt, add_special_tokens=False)],
                max_new_tokens=1024,
            )[0]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            assert ("<think>" in text) == thinking, text
            judge = LLMJudgeReward(
                scorer, enable_thinking=thinking, max_thinking_tokens=1024
            )
            details = judge.score_with_details(
                ["The cat sat on the mat."],
                prompts=["A cat sat quietly on the mat."],
                references=["The cat sat on the mat."],
            )[0]
            assert details["enable_thinking"] == thinking
            assert 0 <= details["score"] <= 1
            records[backend][str(thinking)] = {
                "policy_text": text,
                "policy_ids": ids,
                "judge": details,
                "policy_thinking_closed": "</think>" in text,
            }
            if thinking:
                context = judge_context(
                    tokenizer,
                    "A cat sat quietly on the mat.",
                    "The cat sat on the mat.",
                    "The cat sat on the mat.",
                    task="summarization",
                    enable_thinking=True,
                )
                prompt_ids = scorer._encode(context, "0", 0)[0][:-1]
                exact = (
                    prompt_ids
                    + details["thinking_token_ids"]
                    + tokenizer.encode("\n\n", add_special_tokens=False)
                )
                logps = scorer.label_logprobs_from_ids([exact], RATING_LABELS)[0]
                if backend == "transformers":
                    fixed_contexts.append(exact)
                    fixed_logps.append(logps)
                else:
                    # Compare both kernels on IDENTICAL reasoning tokens. Their
                    # independently generated rationales need not be identical.
                    other = scorer.label_logprobs_from_ids(
                        fixed_contexts, RATING_LABELS
                    )
                    difference = max(
                        abs(a - b)
                        for x, y in zip(fixed_logps, other)
                        for a, b in zip(x, y)
                    )
                    assert difference < 0.5, difference
                    records["same_trace_max_logprob_difference"] = difference
        if backend == "vllm":
            scorer.close()
        else:
            scorer.model.cpu()
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    return records


def grpo_check(args):
    from contextlib import nullcontext

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from non_verifiable_rewards.inference import configure_vllm, grpo_inference_options
    from non_verifiable_rewards.inference.training import TrainingSwitch, engine_dtype
    from non_verifiable_rewards.training.config import validate_config
    from non_verifiable_rewards.training.train import load_policy, load_tokenizer

    torch.set_num_threads(4)
    config = validate_config(
        {
            "policy": {"id": args.model, "enable_thinking": True},
            "inference": {"backend": args.backend},
            "data": {"max_prompt_tokens": 256},
            "optimization": {"max_completion_length": 256},
        }
    )
    configure_vllm()
    tokenizer = load_tokenizer(args.model)
    dataset = Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": p}]}
            for p in (
                "What is 17 times 23? Give the numerical answer.",
                "Translate into English: Guten Morgen.",
            )
        ]
    )
    sampled, rewards, losses, syncs = [], [], [], []

    def reward(completion_ids, **kwargs):
        rewards.extend(completion_ids)
        # A synthetic, nonconstant signal exercises an update without loading
        # a second model. Judge correctness is covered separately above.
        return [sum(ids) % 997 / 997 for ids in completion_ids]

    options = GRPOConfig(
        output_dir=str(args.output / args.backend),
        max_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=256,
        learning_rate=0.001,
        beta=0.04,
        warmup_steps=0,
        bf16=True,
        loss_type="grpo",
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        chat_template_kwargs={"enable_thinking": True},
        mask_truncated_completions=False,
        **grpo_inference_options(config),
    )
    with engine_dtype("bfloat16") if options.use_vllm else nullcontext():
        trainer = GRPOTrainer(
            model=load_policy(config, args.model),
            processing_class=tokenizer,
            train_dataset=dataset,
            reward_funcs=[reward],
            args=options,
            peft_config=LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    switch = (
        TrainingSwitch(trainer, args.output / "switch.jsonl")
        if options.use_vllm
        else None
    )
    if switch:
        from verify_vllm import check_live_weights

        original_sync = switch._sync

        def sync():
            original_sync()
            syncs.append(check_live_weights(trainer))

        switch._sync = sync
    original_generate = trainer._generate_single_turn
    original_loss = trainer._compute_loss

    def generate(prompts, *a, **kw):
        result = original_generate(prompts, *a, **kw)
        sampled.extend(zip(prompts, result[0]))
        return result

    def loss(model, inputs):
        for i in range(len(inputs["completion_ids"])):
            prompt = inputs["prompt_ids"][i][inputs["prompt_mask"][i].bool()].tolist()
            trace = inputs["completion_ids"][i][
                inputs["completion_mask"][i].bool()
            ].tolist()
            assert (prompt, trace) in sampled
            assert trace in rewards
            assert inputs["ref_per_token_logps"].shape == inputs["completion_ids"].shape
            losses.append(
                {
                    "prompt_ids": prompt,
                    "completion_ids": trace,
                    "text": tokenizer.decode(trace, skip_special_tokens=True),
                }
            )
        return original_loss(model, inputs)

    trainer._generate_single_turn, trainer._compute_loss = generate, loss
    before = {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    }
    try:
        trainer.train()
        assert trainer.state.global_step == 2
        changed = [
            n
            for n, p in trainer.model.named_parameters()
            if n in before and not torch.equal(p, before[n])
        ]
        assert changed, "No policy update was exercised"
        assert len(losses) == 4
        assert all("<think>" in row["text"] for row in losses)
        if switch:
            assert (
                len(syncs) == 2
                and syncs[0]["trainable_sha256"] != syncs[1]["trainable_sha256"]
            )
        return {
            "loss_sequences": losses,
            "changed_parameters": changed,
            "weight_checks": syncs,
        }
    finally:
        if switch:
            switch.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["judge", "grpo"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="vllm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = judge_check(args) if args.mode == "judge" else grpo_check(args)
    result["packages"] = {
        p: importlib.metadata.version(p)
        for p in ("torch", "transformers", "trl", "vllm")
    }
    result["passed"] = True
    path = args.output / f"{args.mode}-{args.backend}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {path}")


if __name__ == "__main__":
    main()
