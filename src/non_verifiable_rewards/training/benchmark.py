"""Generated-output evaluation and policy agreement with held-out preferences."""

import gc
import json
from pathlib import Path
import time

from .._hashing import file_sha256
from ..experiments.io import Journal, read_jsonl, write_json
from ..rewards import CausalLMScorer
from .assets import policy_path
from .data import render_prompt
from .rewards import RewardClient
from .state import ensure_sweep, verify_data, verify_model
from .train import load_policy, load_tokenizer


def evaluation_dir(config, name):
    return Path(config["output_dir"]) / "benchmarks" / name


def sample_key(row):
    return row["dataset"] + ":" + row["prompt_id"]


def benchmark_identity(config, name):
    root = ensure_sweep(config)
    manifest = verify_data(root)
    if name == "base":
        model_path = policy_path(config)
        model_id = json.loads((root / "models.json").read_text())["policy"]
    else:
        model_id = verify_model(root / "runs" / name)
        model_path = str(root / "runs" / name / "model")
    target = evaluation_dir(config, name)
    target.mkdir(parents=True, exist_ok=True)
    identity = {
        "model": model_id,
        "data": manifest["files"],
        "benchmark": config["benchmark"],
        "policy": config["policy"],
    }
    path = target / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(
            "Benchmark identity changed; do not mix policy checkpoints or inputs"
        )
    write_json(path, identity)
    return target, model_path


def generate(config, name):
    import torch
    from transformers import set_seed

    target, path = benchmark_identity(config, name)
    if (target / "policy_complete.json").exists():
        return verify_policy_outputs(target)
    tokenizer = load_tokenizer(path)
    torch.set_num_threads(config["reward_models"]["threads"])
    model = load_policy(config, path).to(config["policy"]["device"])
    model.requires_grad_(False)
    model.eval()
    scorer = CausalLMScorer(
        model, tokenizer, batch_size=config["reward_models"]["lm_batch_size"]
    )
    root = Path(config["output_dir"])
    bench = config["benchmark"]
    set_seed(config["seed"])
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [tokenizer.eos_token_id])
    expected_generations, expected_pairs = set(), set()
    with (
        Journal(target / "generations.jsonl") as outputs,
        Journal(target / "preferences.jsonl") as preferences,
    ):
        for dataset in config["data"]["datasets"]:
            rows = read_jsonl(root / "data" / f"{dataset}-generation.jsonl")
            for row in rows:
                key = sample_key(row)
                expected_generations.add(key)
                if key in outputs.completed:
                    continue
                text = render_prompt(
                    tokenizer, row, config["policy"]["enable_thinking"]
                )
                inputs = tokenizer(
                    text, add_special_tokens=False, return_tensors="pt"
                ).to(model.device)
                length = inputs.input_ids.shape[1]
                if length > config["data"]["max_prompt_tokens"]:
                    raise ValueError(
                        f"Held-out prompt {key} exceeds max_prompt_tokens; no silent truncation"
                    )
                start = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(
                        **inputs,
                        max_new_tokens=bench["max_new_tokens"],
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )[0, length:].tolist()
                completion = tokenizer.decode(ids, skip_special_tokens=True)
                outputs.append(
                    {
                        "pair_id": key,
                        "dataset": dataset,
                        "prompt_id": row["prompt_id"],
                        "completion": completion,
                        "token_ids": ids,
                        "tokens": len(ids),
                        "words": len(completion.split()),
                        "empty": not completion.strip(),
                        "reached_token_limit": len(ids) == bench["max_new_tokens"]
                        and (not ids or ids[-1] not in eos_ids),
                        "seconds": time.perf_counter() - start,
                    }
                )
            for row in read_jsonl(root / "data" / f"{dataset}-pairs.jsonl"):
                key = dataset + ":" + row["pair_id"]
                expected_pairs.add(key)
                if key in preferences.completed:
                    continue
                context = render_prompt(
                    tokenizer, row, config["policy"]["enable_thinking"]
                )
                candidates = [row["preferred"], row["rejected"]]
                means = scorer.log_likelihood(
                    [context] * 2, candidates, reduction="mean"
                )
                counts = [
                    len(tokenizer.encode(text, add_special_tokens=False))
                    for text in candidates
                ]
                preferences.append(
                    {
                        "pair_id": key,
                        "dataset": dataset,
                        "prompt_id": row["prompt_id"],
                        "mean_margin": means[0] - means[1],
                        "sum_margin": means[0] * counts[0] - means[1] * counts[1],
                        "preferred_logp_mean": means[0],
                        "rejected_logp_mean": means[1],
                        "preferred_tokens": counts[0],
                        "rejected_tokens": counts[1],
                    }
                )
        if (
            set(outputs.completed) != expected_generations
            or set(preferences.completed) != expected_pairs
        ):
            raise ValueError("Unexpected, duplicate or missing policy benchmark rows")
    write_json(
        target / "policy_complete.json",
        {
            "files": {
                name: file_sha256(target / name)
                for name in ("generations.jsonl", "preferences.jsonl")
            },
            "generations": len(expected_generations),
            "preference_pairs": len(expected_pairs),
            "generation": "Greedy, one raw decoded completion per frozen prompt; no reference in policy input",
            "preference_protocol": "Teacher-forced candidate likelihood, mean and sum, no EOS scored; ties count half",
        },
    )
    del scorer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return verify_policy_outputs(target)


def verify_policy_outputs(target):
    target = Path(target)
    complete = json.loads((target / "policy_complete.json").read_text())
    for name, checksum in complete["files"].items():
        if file_sha256(target / name) != checksum:
            raise ValueError(f"Policy benchmark output changed: {name}")
    return complete


def score_generations(config, name):
    target, _ = benchmark_identity(config, name)
    verify_policy_outputs(target)
    if (target / "complete.json").exists():
        complete = json.loads((target / "complete.json").read_text())
        for file, checksum in complete["files"].items():
            if file_sha256(target / file) != checksum:
                raise ValueError("Generated-output reward scores changed")
        return complete
    root = Path(config["output_dir"])
    outputs = {row["pair_id"]: row for row in read_jsonl(target / "generations.jsonl")}
    rewards = list(config["benchmark"]["rewards"])
    if "supergpqa" in config["data"]["datasets"] and "answer_accuracy" not in rewards:
        rewards.append("answer_accuracy")
    counts, hashes = {}, {}
    # Model generation runs in another CLI process, so these fixed scorers do
    # not share memory with the evaluated policy. Families have separate envs.
    with RewardClient(config, target / "logs") as client:
        for reward in rewards:
            path = target / "scores" / f"{reward}.jsonl"
            expected = set()
            with Journal(path) as journal:
                for dataset in config["data"]["datasets"]:
                    if reward == "answer_accuracy" and dataset != "supergpqa":
                        continue
                    rows = read_jsonl(root / "data" / f"{dataset}-generation.jsonl")
                    if reward == "smart1_bleurt":
                        rows = rows[: config["benchmark"]["smart_bleurt_prompts"]]
                    expected.update(sample_key(row) for row in rows)
                    pending = [
                        row for row in rows if sample_key(row) not in journal.completed
                    ]
                    batch_size = config["reward_models"]["batch_size"]
                    for start in range(0, len(pending), batch_size):
                        batch = pending[start : start + batch_size]
                        completions = [
                            outputs[sample_key(row)]["completion"] for row in batch
                        ]
                        scores = client.score(reward, batch, completions)
                        for row, score in zip(batch, scores, strict=True):
                            journal.append(
                                {
                                    "pair_id": sample_key(row),
                                    "dataset": dataset,
                                    "prompt_id": row["prompt_id"],
                                    "reward": reward,
                                    "score": score,
                                }
                            )
                if set(journal.completed) != expected:
                    raise ValueError(f"Incorrect generated-score coverage for {reward}")
                counts[reward] = len(expected)
            hashes[str(path.relative_to(target))] = file_sha256(path)
    complete = {
        "files": hashes,
        "counts": counts,
        "reward_policy": "Same frozen reward checkpoints for base and every trained policy",
        "smart_bleurt_scope": "First configured number of frozen generation prompts per dataset",
    }
    write_json(target / "complete.json", complete)
    return complete
