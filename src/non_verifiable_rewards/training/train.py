"""One independent TRL GRPO run, resumable optimizer state, standalone export."""

import json
import math
from contextlib import nullcontext
from pathlib import Path

from ..experiments.io import write_json
from ..inference import configure_vllm, grpo_inference_options, use_vllm
from .assets import policy_path
from .data import training_dataset
from .rewards import RewardClient, WeightedReward
from .state import (
    assert_single_process,
    checkpoint_manifest,
    ensure_sweep,
    verify_data,
    verify_model,
)


def load_tokenizer(path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, padding_side="left"
    )
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError(
            "The policy tokenizer must define an EOS token and a chat template"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy(config, path):
    import torch
    from transformers import AutoModelForCausalLM

    policy = config["policy"]
    if policy["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Policy training/generation requested CUDA; use the GPU node"
        )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=getattr(torch, policy["dtype"]),
        local_files_only=True,
        attn_implementation="sdpa",
    )
    limit = getattr(model.config, "max_position_embeddings", None)
    required = config["data"]["max_prompt_tokens"] + max(
        config["optimization"]["max_completion_length"],
        config["benchmark"]["max_new_tokens"],
    )
    if limit is not None and required > limit:
        raise ValueError(
            f"Prompt + completion budget {required} exceeds policy context {limit}"
        )
    return model


def train_run(config, run):
    if use_vllm(config):
        configure_vllm()
    import torch
    from transformers import set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    assert_single_process()
    if run not in config["runs"]:
        raise ValueError("Run must match its frozen sweep definition")
    root = ensure_sweep(config)
    verify_data(root)
    target = root / "runs" / run["name"]
    target.mkdir(parents=True, exist_ok=True)
    if (target / "trained.json").exists():
        return verify_model(target)
    # Recover an interruption between the atomic model rename and its marker.
    if (target / "model").exists() and (target / "export_manifest.json").exists():
        metadata = json.loads((target / "export_manifest.json").read_text())
        if metadata["model_files"] != checkpoint_manifest(target / "model"):
            raise ValueError("Interrupted final export is incomplete or changed")
        write_json(target / "trained.json", metadata)
        return metadata
    write_json(target / "run.json", run)
    set_seed(config["seed"])
    torch.set_num_threads(config["reward_models"]["threads"])
    base = policy_path(config)
    tokenizer = load_tokenizer(base)
    policy, opt = config["policy"], config["optimization"]
    checkpoints = target / "checkpoints"
    resume = get_last_checkpoint(str(checkpoints)) if checkpoints.exists() else None
    saved_step = (
        json.loads((Path(resume) / "trainer_state.json").read_text())["global_step"]
        if resume
        else 0
    )
    if saved_step > opt["max_steps"]:
        raise ValueError("Checkpoint exceeds the configured training budget")
    optimized = target / "optimization_complete.json"
    if optimized.exists():
        metadata = json.loads(optimized.read_text())
        if metadata["files"] != checkpoint_manifest(target / "policy_state"):
            raise ValueError("Saved optimized policy changed before export")
        if policy["lora"]:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                load_policy(config, base), target / "policy_state"
            )
        else:
            model = load_policy(config, target / "policy_state")
    elif saved_step == opt["max_steps"]:
        # Trainer.train() on an already finished checkpoint can take an extra
        # step. Recover the saved weights directly after a post-training crash.
        if policy["lora"]:
            from peft import PeftModel

            model = PeftModel.from_pretrained(load_policy(config, base), resume)
        else:
            model = load_policy(config, resume)
        model.save_pretrained(target / "policy_state", safe_serialization=True)
        tokenizer.save_pretrained(target / "policy_state")
        write_json(
            optimized,
            {
                "steps": saved_step,
                "files": checkpoint_manifest(target / "policy_state"),
                "resumed_from": resume,
                "metrics": {"recovered_completed_checkpoint": True},
            },
        )
    else:
        model = load_policy(config, base)
        peft_config = None
        if policy["lora"]:
            from peft import LoraConfig

            peft_config = LoraConfig(
                r=policy["lora_r"],
                lora_alpha=policy["lora_alpha"],
                lora_dropout=policy["lora_dropout"],
                target_modules=policy["lora_target_modules"],
                task_type="CAUSAL_LM",
                bias="none",
            )
        trainer_options = {
            key: value for key, value in opt.items() if key != "warmup_ratio"
        }
        args = GRPOConfig(
            output_dir=str(checkpoints),
            seed=config["seed"],
            data_seed=config["seed"],
            **trainer_options,
            warmup_steps=math.ceil(opt["max_steps"] * opt["warmup_ratio"]),
            loss_type="grpo",
            scale_rewards="group",
            num_iterations=1,
            optim="adamw_torch",
            **grpo_inference_options(config),
            use_cpu=policy["device"] == "cpu",
            bf16=policy["dtype"] == "bfloat16",
            fp16=policy["dtype"] == "float16",
            gradient_checkpointing_kwargs={"use_reentrant": False},
            chat_template_kwargs={"enable_thinking": policy["enable_thinking"]},
            mask_truncated_completions=False,
            remove_unused_columns=False,
            report_to="none",
            push_to_hub=False,
            eval_strategy="no",
            save_strategy="steps",
            dataloader_num_workers=0,
        )
        dataset = training_dataset(config, run, tokenizer)
        with RewardClient(config, target / "logs") as client:
            callback = WeightedReward(
                client,
                run,
                tokenizer,
                target / "reward_trace.jsonl",
                invalid_score=config["reward_models"]["invalid_score"],
            )
            if args.use_vllm:
                from ..inference.training import engine_dtype

            with engine_dtype(policy["dtype"]) if args.use_vllm else nullcontext():
                trainer = GRPOTrainer(
                    model=model,
                    args=args,
                    train_dataset=dataset,
                    processing_class=tokenizer,
                    reward_funcs=[callback],
                    peft_config=peft_config,
                )
            switch = None
            if args.use_vllm:
                from ..inference.training import TrainingSwitch

                switch = TrainingSwitch(trainer, target / "inference_trace.jsonl")
            if "self_consistency" in run["rewards"]:
                from ..rewards.self_consistency import SelfConsistencyReward
                from .self_consistency import PolicySelector

                settings = config["self_consistency"]
                callback.self_consistency = SelfConsistencyReward(
                    group_size=opt["num_generations"],
                    selector=PolicySelector(
                        trainer,
                        selection_temperature=settings["selection_temperature"],
                        max_context_tokens=settings["max_context_tokens"],
                    ),
                    seed=config["seed"],
                    **settings,
                )
            try:
                result = trainer.train(resume_from_checkpoint=resume)
            finally:
                # Release the callback -> selector -> trainer reference cycle.
                callback.self_consistency = None
                if switch is not None:
                    switch.close()
            if trainer.state.global_step != opt["max_steps"]:
                raise RuntimeError(
                    "Training did not reach the requested number of optimizer steps"
                )
            trainer.save_model(str(target / "policy_state"))
            tokenizer.save_pretrained(target / "policy_state")
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)
            model = trainer.accelerator.unwrap_model(trainer.model)
            write_json(
                optimized,
                {
                    "steps": trainer.state.global_step,
                    "files": checkpoint_manifest(target / "policy_state"),
                    "resumed_from": resume,
                    "metrics": result.metrics,
                },
            )
    # Export merged weights as a complete reloadable model, not only a LoRA
    # adapter pointing at a base checkpoint in an ephemeral VM cache.
    if policy["lora"]:
        model = model.merge_and_unload()
    model.config.use_cache = True
    export = target / "model-export"
    export.mkdir(exist_ok=True)
    model.save_pretrained(export, safe_serialization=True)
    tokenizer.save_pretrained(export)
    # An interrupted export is rewritten from the persisted optimized policy.
    final = target / "model"
    if final.exists():
        raise ValueError(
            "An unmarked final model directory already exists; inspect it before retrying"
        )
    metadata = {
        "name": run["name"],
        "rewards": run["rewards"],
        "datasets": run["datasets"],
        "steps": opt["max_steps"],
        "model_files": checkpoint_manifest(export),
        "base_asset": json.loads((root / "models.json").read_text())["policy"],
        "reward_models_frozen": "self_consistency" not in run["rewards"],
        "online_rewards": [name for name in run["rewards"] if name == "self_consistency"],
        "export": "Standalone merged weights and tokenizer",
    }
    write_json(target / "export_manifest.json", metadata)
    export.rename(final)
    write_json(target / "trained.json", metadata)
    return metadata
