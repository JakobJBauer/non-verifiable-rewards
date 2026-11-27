"""Strict, reviewable sweep configuration; importing this module needs no GPU."""

import copy
import math
import re
import tomllib
from pathlib import Path

from ..experiments.io import ALL_REWARDS as REWARDS
from ..rewards.llm_judge import JUDGE_DEFAULTS, validate_judge_config

DATASETS = ("reddit_tldr", "litbench", "wmt14_de_en", "supergpqa")
DEFAULTS = {
    "output_dir": "outputs/grpo_qwen3_4b",
    "cache_dir": "artifacts/cache",
    "seed": 1729,
    "wandb": {
        "enabled": False,
        "project": "non-verifiable rewards",
        "entity": "",
        "mode": "online",
    },
    "inference": {
        "backend": "auto",
        "gpu_memory_utilization": 0.35,
        "reward_gpu_memory_utilization": 0.35,
        "batch_size": 16,
    },
    "policy": {
        "id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "device": "cuda",
        "dtype": "bfloat16",
        "enable_thinking": True,
        "lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "data": {
        "datasets": list(DATASETS[:3]),
        "train_examples_per_dataset": 4096,
        "max_prompt_tokens": 4096,
        "stream_shuffle_buffer": 10000,
        "local": {},
    },
    "optimization": {
        "max_steps": 500,
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_generations": 8,
        "max_completion_length": 2048,
        "beta": 0.04,
        "temperature": 0.8,
        "top_p": 0.95,
        "save_steps": 100,
        "save_total_limit": 2,
        "logging_steps": 5,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.03,
    },
    "judge": JUDGE_DEFAULTS.copy(),
    "reward_models": {
        "python": "",
        "bleurt_python": "",
        "device": "cuda",
        "dtype": "bfloat16",
        "lm_id": "Qwen/Qwen3-4B",
        "lm_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "bertscore_id": "FacebookAI/roberta-large",
        "bertscore_revision": "722cf37b1afa9454edce342e7895e588b6ff1d59",
        "bertscore_layers": 17,
        "bleurt_checkpoint": "",
        "bleurt_url": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip",
        "bleurt_sha256": "bd4f376076f493d76ea486ce4d7c081fb0b512cc757794db0ed25c2272fca9bd",
        "batch_size": 32,
        "lm_batch_size": 1,
        "max_context_tokens": 32768,
        "allow_truncation": True,
        "bleurt_tf32": False,
        "threads": 2,
        "invalid_score": -1000.0,
    },
    "benchmark": {
        "enabled": True,
        "include_base_model": True,
        "pairs_per_dataset": 2000,
        "generation_prompts_per_dataset": 200,
        "smart_bleurt_prompts": 100,
        "rewards": list(REWARDS),
        "max_new_tokens": 1024,
        "bootstrap_samples": 10000,
        "tie_tolerance": 1e-7,
    },
}


def _merge(default, supplied, context):
    if not isinstance(supplied, dict):
        raise ValueError(f"{context} must be a TOML table")
    unknown = set(supplied) - set(default)
    if unknown:
        raise ValueError(f"Unknown {context} settings: {sorted(unknown)}")
    return {**copy.deepcopy(default), **supplied}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite(value, name, *, minimum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("Sweep configuration must be a TOML table")
    raw = copy.deepcopy(raw)
    runs = raw.pop("runs", None)
    config = _merge(DEFAULTS, raw, "sweep")
    for section in (
        "policy",
        "data",
        "optimization",
        "reward_models",
        "benchmark",
        "inference",
        "wandb",
    ):
        config[section] = _merge(DEFAULTS[section], raw.get(section, {}), section)
    config["judge"] = validate_judge_config(raw.get("judge", {}))
    tracking = config["wandb"]
    if not isinstance(tracking["enabled"], bool):
        raise ValueError("wandb.enabled must be boolean")
    project = tracking["project"]
    if (
        not isinstance(project, str)
        or not project.strip()
        or project != project.strip()
        or len(project) > 128
        or any(char in project for char in "/\\#?%:")
    ):
        raise ValueError("Invalid wandb.project")
    if (
        not isinstance(tracking["entity"], str)
        or tracking["entity"] != tracking["entity"].strip()
    ):
        raise ValueError("wandb.entity must be an account/team name or an empty string")
    if tracking["mode"] not in ("online", "offline"):
        raise ValueError("wandb.mode must be online or offline")
    inference = config["inference"]
    if inference["backend"] not in ("auto", "transformers", "vllm"):
        raise ValueError("inference.backend must be auto, transformers or vllm")
    _positive(inference["batch_size"], "inference.batch_size")
    for key in ("gpu_memory_utilization", "reward_gpu_memory_utilization"):
        _finite(inference[key], key, minimum=1e-12)
        if inference[key] >= 1:
            raise ValueError(f"{key} must be less than 1")
    if inference["backend"] == "vllm" and config["policy"]["device"] != "cuda":
        raise ValueError("vllm requires policy.device=cuda")
    if (
        isinstance(config["seed"], bool)
        or not isinstance(config["seed"], int)
        or config["seed"] < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    for key in ("output_dir", "cache_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty path")
        config[key] = str(Path(config[key]).expanduser().resolve())
    for section in ("policy", "reward_models"):
        if config[section]["device"] not in ("cpu", "cuda"):
            raise ValueError(f"{section}.device must be cpu or cuda")
        if config[section]["dtype"] not in ("float32", "bfloat16", "float16"):
            raise ValueError(f"Unsupported {section}.dtype")
    for key in ("enable_thinking", "lora"):
        if not isinstance(config["policy"][key], bool):
            raise ValueError(f"policy.{key} must be boolean")
    for key in ("lora_r", "lora_alpha"):
        _positive(config["policy"][key], key)
    if not 0 <= config["policy"]["lora_dropout"] < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    modules = config["policy"]["lora_target_modules"]
    if (
        not isinstance(modules, list)
        or not modules
        or any(not isinstance(x, str) or not x for x in modules)
    ):
        raise ValueError("lora_target_modules must list module names")
    data, opt, bench, rewards = (
        config[k] for k in ("data", "optimization", "benchmark", "reward_models")
    )
    for key in (
        "train_examples_per_dataset",
        "max_prompt_tokens",
        "stream_shuffle_buffer",
    ):
        _positive(data[key], key)
    names = data["datasets"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) for n in names)
        or len(names) != len(set(names))
        or not set(names) <= set(DATASETS)
    ):
        raise ValueError("data.datasets must contain distinct supported datasets")
    if not isinstance(data["local"], dict):
        raise ValueError("data.local must be a table of dataset overrides")
    for dataset, paths in data["local"].items():
        if (
            dataset not in names
            or not isinstance(paths, dict)
            or not set(paths) <= {"train", "test"}
            or any(not isinstance(p, str) or not p for p in paths.values())
        ):
            raise ValueError(
                "Local data overrides must map selected datasets to train/test JSONL paths"
            )
    for section, keys in (
        (opt, ("gradient_checkpointing",)),
        (bench, ("enabled", "include_base_model")),
        (rewards, ("allow_truncation", "bleurt_tf32")),
    ):
        for key in keys:
            if not isinstance(section[key], bool):
                raise ValueError(f"{key} must be boolean")
    for key in (
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_generations",
        "max_completion_length",
        "save_steps",
        "save_total_limit",
        "logging_steps",
    ):
        _positive(opt[key], key)
    if opt["num_generations"] < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    batch = opt["per_device_train_batch_size"] * opt["gradient_accumulation_steps"]
    if batch % opt["num_generations"]:
        raise ValueError(
            "Effective generation batch must be divisible by num_generations"
        )
    for key in ("learning_rate", "temperature", "max_grad_norm"):
        _finite(opt[key], key, minimum=1e-12)
    _finite(opt["beta"], "beta", minimum=0)
    if not 0 < opt["top_p"] <= 1 or not 0 <= opt["warmup_ratio"] < 1:
        raise ValueError("Invalid top_p or warmup_ratio")
    for key in (
        "pairs_per_dataset",
        "generation_prompts_per_dataset",
        "smart_bleurt_prompts",
        "bootstrap_samples",
        "max_new_tokens",
    ):
        _positive(bench[key], key)
    if (
        not isinstance(bench["rewards"], list)
        or not bench["rewards"]
        or any(not isinstance(name, str) for name in bench["rewards"])
        or len(set(bench["rewards"])) != len(bench["rewards"])
        or not set(bench["rewards"]) <= set(REWARDS) | {"answer_accuracy"}
    ):
        raise ValueError("benchmark.rewards must contain distinct supported rewards")
    if "answer_accuracy" in bench["rewards"] and "supergpqa" not in names:
        raise ValueError("Answer-accuracy benchmarking requires SuperGPQA")
    if bench["smart_bleurt_prompts"] > bench["generation_prompts_per_dataset"]:
        raise ValueError("smart_bleurt_prompts exceeds the generation sample size")
    _finite(bench["tie_tolerance"], "tie_tolerance", minimum=0)
    _finite(rewards["invalid_score"], "invalid_score")
    for key in (
        "batch_size",
        "lm_batch_size",
        "max_context_tokens",
        "threads",
        "bertscore_layers",
    ):
        _positive(rewards[key], key)
    if runs is None:
        runs = [{"name": reward, "rewards": {reward: 1.0}} for reward in REWARDS]
    if not isinstance(runs, list) or not runs:
        raise ValueError("Provide at least one [[runs]] entry")
    seen = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Each run must be a TOML table")
        if set(run) - {"name", "rewards", "datasets", "candidate_views"}:
            raise ValueError("Unknown run setting")
        name = run.get("name", "")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name)
            or name == "base"
            or name in seen
        ):
            raise ValueError(
                "Run names must be unique safe directory names; 'base' is reserved"
            )
        seen.add(name)
        weights = run.get("rewards", {})
        if (
            not isinstance(weights, dict)
            or not weights
            or not set(weights) <= set(REWARDS) | {"answer_accuracy"}
        ):
            raise ValueError(f"Unknown or empty reward mixture in {name}")
        for reward, weight in weights.items():
            _finite(weight, f"{name}.{reward}.weight")
            if weight == 0:
                raise ValueError(
                    "Remove zero-weight rewards instead of evaluating them"
                )
        run.setdefault("datasets", names.copy())
        if (
            not isinstance(run["datasets"], list)
            or not run["datasets"]
            or any(not isinstance(name, str) for name in run["datasets"])
            or len(set(run["datasets"])) != len(run["datasets"])
            or not set(run["datasets"]) <= set(names)
        ):
            raise ValueError("Run datasets must be distinct members of data.datasets")
        if "answer_accuracy" in weights and run["datasets"] != ["supergpqa"]:
            raise ValueError("answer_accuracy training requires a SuperGPQA-only run")
        run.setdefault("candidate_views", {})
        if not isinstance(run["candidate_views"], dict):
            raise ValueError("candidate_views must be a table")
        for reward, view in run["candidate_views"].items():
            if reward not in weights or view not in (
                "completion",
                "reasoning",
                "answer_text",
            ):
                raise ValueError("Invalid candidate view")
            if view != "completion" and run["datasets"] != ["supergpqa"]:
                raise ValueError("Answer/rationale views require SuperGPQA-only runs")
    config["runs"] = runs
    return config


def load_config(path):
    with Path(path).open("rb") as stream:
        return validate_config(tomllib.load(stream))
