"""Generated-output evaluation and policy agreement with held-out preferences."""

import json
from pathlib import Path
import time

from .._hashing import file_sha256
from ..completions import answer_text, decode_completions
from ..experiments.io import Journal, read_jsonl, write_json
from ..inference.policy import policy_backend
from .assets import policy_path
from .data import render_prompt
from .rewards import RewardClient, reward_candidate
from .state import ensure_sweep, verify_data, verify_model


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
        "inference": config["inference"],
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
    torch.set_num_threads(config["reward_models"]["threads"])
    set_seed(config["seed"])
    with policy_backend(config, path) as scorer:
        return _generate(config, target, scorer)


def _generate(config, target, scorer):
    tokenizer = scorer.tokenizer
    root = Path(config["output_dir"])
    bench = config["benchmark"]
    eos = scorer.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [tokenizer.eos_token_id])
    expected_generations, expected_pairs = set(), set()
    with (
        Journal(target / "generations.jsonl") as outputs,
        Journal(target / "preferences.jsonl") as preferences,
    ):
        for dataset in config["data"]["datasets"]:
            rows = read_jsonl(root / "data" / f"{dataset}-generation.jsonl")
            expected_generations.update(sample_key(row) for row in rows)
            pending = [row for row in rows if sample_key(row) not in outputs.completed]
            batch_size = config["inference"]["batch_size"]
            for start_index in range(0, len(pending), batch_size):
                batch = pending[start_index : start_index + batch_size]
                prompts = [
                    tokenizer.encode(
                        render_prompt(
                            tokenizer, row, config["policy"]["enable_thinking"]
                        ),
                        add_special_tokens=False,
                    )
                    for row in batch
                ]
                if any(
                    len(ids) > config["data"]["max_prompt_tokens"] for ids in prompts
                ):
                    raise ValueError(
                        "Held-out prompt exceeds max_prompt_tokens; no silent truncation"
                    )
                start = time.perf_counter()
                completions = scorer.generate_ids(
                    prompts, max_new_tokens=bench["max_new_tokens"]
                )
                seconds = (time.perf_counter() - start) / len(batch)
                for row, ids in zip(batch, completions, strict=True):
                    key = sample_key(row)
                    completion = decode_completions(tokenizer, [ids])[0]
                    outputs.append(
                        {
                            "pair_id": key,
                            "dataset": dataset,
                            "prompt_id": row["prompt_id"],
                            "completion": completion,
                            "visible_completion": answer_text(completion),
                            "reward_input": reward_candidate(row, completion),
                            "token_ids": ids,
                            "tokens": len(ids),
                            "words": len(completion.split()),
                            "empty": not completion.strip(),
                            "reached_token_limit": len(ids) == bench["max_new_tokens"]
                            and (not ids or ids[-1] not in eos_ids),
                            "seconds": seconds,
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
        "candidate_policy": "SuperGPQA generated answer text with bare-label fallback; visible completions for other tasks; private thinking excluded",
        "answer_accuracy_protocol": "supergpqa_unique_exact_option_text_v1_case_whitespace_normalized",
        "smart_bleurt_scope": "First configured number of frozen generation prompts per dataset",
    }
    write_json(target / "complete.json", complete)
    return complete
