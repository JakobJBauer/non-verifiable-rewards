"""Matched, resumable thinking/direct judge comparison on frozen baseline pairs."""

import argparse
import fcntl
import importlib.metadata
import json
import math
import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from .._hashing import digest, file_sha256
from ..rewards.base import checked_scores
from ..rewards.llm_judge import RATING_LABELS, judge_context
from ..rewards.prompts import task_for_dataset
from .io import read_jsonl, write_json, write_jsonl


def read_config(path):
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if config["backend"] != "vllm" or config["dtype"] != "bfloat16":
        raise ValueError("This comparison uses the approved vLLM bfloat16 protocol")
    for key in (
        "batch_size",
        "max_thinking_tokens",
        "max_context_tokens",
        "bootstrap_samples",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    return config


def prepare(config, source=None, output=None, tokenizer=None):
    """Freeze every existing pair, validate judge prompts, deduplicate contexts."""
    source, output = (
        Path(source or config["source_run"]),
        Path(output or config["run_dir"]),
    )
    source_config = json.loads((source / "config.json").read_text())
    models = json.loads((source / "models.json").read_text())
    if (
        source_config["dtype"] != config["dtype"]
        or source_config["max_context_tokens"] != config["max_context_tokens"]
    ):
        raise ValueError(
            "Precision/context settings differ from the historical baseline"
        )
    preprocessing = json.loads((source / "preprocessing.json").read_text())
    sources = {
        name: file_sha256(source / name)
        for name in (
            "config.json",
            "models.json",
            "data_manifest.json",
            "preprocessing.json",
        )
    }
    candidates, pairs = {}, []
    for name in source_config["datasets"]:
        data_path = source / "scored_text" / f"{name}.jsonl"
        sources[str(data_path.relative_to(source))] = file_sha256(data_path)
        if file_sha256(data_path) != preprocessing["files"][name]:
            raise ValueError("Frozen scored inputs changed")
        old = {}
        for path in sorted((source / "scores/lm").glob(f"{name}-shard-*.jsonl")):
            sources[str(path.relative_to(source))] = file_sha256(path)
            for record in read_jsonl(path):
                for row in record["rows"]:
                    if row["reward"] != "llm_judge":
                        continue
                    key = (row["pair_id"], row["side"])
                    if (
                        key in old
                        or row["status"] != "ok"
                        or row["statistics"]["enable_thinking"]
                    ):
                        raise ValueError(
                            "Historical judge scores must be unique, complete and non-thinking"
                        )
                    old[key] = row
        data = read_jsonl(data_path)
        if len(data) != source_config["pairs_per_dataset"] or len(old) != 2 * len(data):
            raise ValueError(
                "Historical baseline coverage differs from its configured scope"
            )
        for pair in data:
            matched = {k: pair[k] for k in ("dataset", "pair_id", "prompt_id")}
            for side in ("preferred", "rejected"):
                row = old.pop((pair["pair_id"], side))
                if (
                    row["candidate_id"] != pair[side + "_id"]
                    or row["reference_id"] != pair["reference_id"]
                ):
                    raise ValueError("Candidate/reference identity mismatch")
                candidate = dict(
                    dataset=name,
                    prompt=pair["prompt"],
                    candidate=pair[side],
                    reference=pair["reference"],
                )
                key = digest(candidate)
                candidate["key"] = key
                if tokenizer is not None:
                    context = judge_context(
                        tokenizer,
                        candidate["prompt"],
                        candidate["candidate"],
                        candidate["reference"],
                        task=task_for_dataset(name),
                    )
                    if digest(context) != row["statistics"]["judge_context_sha256"]:
                        raise ValueError(
                            "Current non-thinking prompt differs from the saved baseline"
                        )
                candidate["historical_context_sha256"] = row["statistics"][
                    "judge_context_sha256"
                ]
                if key in candidates and candidates[key] != candidate:
                    raise ValueError("Conflicting duplicate candidate context")
                candidates[key] = candidate
                matched[side + "_key"] = key
                matched[side + "_historical_score"] = row["score"]
                matched[side + "_words"] = len(pair[side].split())
            pairs.append(matched)
        if old:
            raise ValueError("Unexpected historical judge scores")
    # Interleave task queues so early progress covers all three datasets.
    queues = defaultdict(list)
    for item in candidates.values():
        queues[item["dataset"]].append(item)
    ordered = []
    for i in range(max(map(len, queues.values()))):
        for name in source_config["datasets"]:
            if i < len(queues[name]):
                ordered.append(queues[name][i])
    manifest = dict(
        config=config,
        model=models["lm"],
        source_files=sources,
        pairs=len(pairs),
        candidate_occurrences=2 * len(pairs),
        unique_candidates=len(ordered),
        pairs_by_dataset=dict(Counter(p["dataset"] for p in pairs)),
        unique_candidates_by_dataset=dict(Counter(p["dataset"] for p in ordered)),
        candidates_sha256=digest(ordered),
        pairs_sha256=digest(pairs),
        historical_prompt_hashes_checked=tokenizer is not None,
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Comparison inputs/settings changed; use a fresh run directory"
            )
    else:
        write_jsonl(output / "candidates.jsonl", ordered)
        write_jsonl(output / "pairs.jsonl", pairs)
        write_json(manifest_path, manifest)
    return manifest


def distribution(logprobs):
    values = checked_scores(logprobs, 5)
    maximum = max(values)
    weights = [math.exp(v - maximum) for v in values]
    total = math.fsum(weights)
    probabilities = [v / total for v in weights]
    mass = math.exp(maximum) * total
    if mass > 1 + 1e-5:
        raise ValueError("Label scores must be full-vocabulary log probabilities")
    return dict(
        score=sum(i * p for i, p in enumerate(probabilities)) / 4,
        rating_probabilities=probabilities,
        rating_token_mass=mass,
    )


def evaluate_batch(scorer, rows, *, thinking, budget):
    """Keep successful cases when another trace hits its limit; never fake a rating."""
    tokenizer = scorer.tokenizer
    contexts = [
        judge_context(
            tokenizer,
            r["prompt"],
            r["candidate"],
            r["reference"],
            task=task_for_dataset(r["dataset"]),
            enable_thinking=thinking,
        )
        for r in rows
    ]
    if not thinking:
        for row, context in zip(rows, contexts, strict=True):
            if digest(context) != row["historical_context_sha256"]:
                raise ValueError("Non-thinking prompt failed historical hash check")
    encoded = [scorer._encode(c, "0", i)[0][:-1] for i, c in enumerate(contexts)]
    stop = tokenizer.encode("</think>", add_special_tokens=False)
    if thinking and (
        len(stop) != 1
        or tokenizer.decode(stop, skip_special_tokens=False) != "</think>"
    ):
        raise ValueError("Missing Qwen reasoning delimiter")
    suffix = tokenizer.encode("\n\n", add_special_tokens=False) if thinking else []
    results = [
        dict(
            key=r["key"],
            dataset=r["dataset"],
            thinking=thinking,
            status="pending",
            score=None,
            prompt_tokens=len(ids),
            prompt_sha256=digest(c),
        )
        for r, ids, c in zip(rows, encoded, contexts, strict=True)
    ]
    valid = []
    for i, ids in enumerate(encoded):
        if len(ids) + (budget if thinking else 0) + len(suffix) + 1 > scorer.max_length:
            results[i].update(
                status="context_limit",
                error="Insufficient room for the full reasoning budget; input preserved",
            )
        else:
            valid.append(i)
    if not valid:
        return results
    start = time.perf_counter()
    traces = (
        scorer.generate_ids(
            [encoded[i] for i in valid], max_new_tokens=budget, stop_token_id=stop[0]
        )
        if thinking
        else [[] for _ in valid]
    )
    generation_seconds = time.perf_counter() - start if thinking else 0.0
    scored, scoring_ids = [], []
    for i, trace in zip(valid, traces, strict=True):
        results[i].update(
            thinking_tokens=len(trace),
            thinking_token_ids=trace,
            generation_batch_seconds=generation_seconds,
            generation_batch_size=len(valid),
        )
        if thinking and (not trace or trace[-1] != stop[0]):
            results[i].update(
                status="thinking_limit" if len(trace) == budget else "early_eos",
                error="Reasoning did not close; no rating was computed",
            )
            continue
        scoring_ids.append(encoded[i] + trace + suffix)
        scored.append(i)
    start = time.perf_counter()
    values = (
        scorer.label_logprobs_from_ids(scoring_ids, RATING_LABELS) if scored else []
    )
    seconds = time.perf_counter() - start
    for i, ids, logp in zip(scored, scoring_ids, values, strict=True):
        results[i].update(
            distribution(logp),
            status="ok",
            scoring_context_sha256=digest(ids),
            scoring_batch_seconds=seconds,
            scoring_batch_size=len(scored),
        )
    return results


def load_journal(path):
    """Accept only complete lines; reject duplicates rather than averaging them."""
    result = {}
    if not path.exists():
        return result
    with path.open("rb+") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                break
            row = json.loads(line)
            if row["key"] in result:
                raise ValueError("Duplicate candidate in comparison journal")
            result[row["key"]] = row
    return result


def score(config, *, limit=None):
    import torch
    from ..inference.scorer import VLLMScorer

    root = Path(config["run_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    if config != manifest["config"]:
        raise ValueError("Frozen comparison configuration changed")
    rows = read_jsonl(root / "candidates.jsonl")
    if digest(rows) != manifest["candidates_sha256"]:
        raise ValueError("Frozen candidate file changed")
    package = Path(__file__).parents[1]
    identity = dict(
        manifest_sha256=file_sha256(root / "manifest.json"),
        code={
            str(p.relative_to(package)): file_sha256(p)
            for p in [
                Path(__file__),
                package / "rewards/llm_judge.py",
                package / "rewards/causal_lm.py",
                package / "inference/scorer.py",
            ]
        },
        packages={
            p: importlib.metadata.version(p) for p in ("torch", "transformers", "vllm")
        },
    )
    path = root / "scoring_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Scoring implementation/environment changed")
    write_json(path, identity)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    with (root / ".score.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with VLLMScorer(
            manifest["model"]["path"],
            dtype=config["dtype"],
            max_length=config["max_context_tokens"],
            batch_size=config["batch_size"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            sleep=False,
        ) as scorer:
            for thinking in (False, True):
                name = "thinking" if thinking else "control"
                path = root / f"{name}.jsonl"
                done = load_journal(path)
                if set(done) - {r["key"] for r in rows}:
                    raise ValueError("Unexpected scored candidate")
                pending = [r for r in rows if r["key"] not in done]
                if limit is not None:
                    pending = pending[:limit]
                started = time.perf_counter()
                processed = 0
                with path.open("a") as stream:
                    for start in range(0, len(pending), config["batch_size"]):
                        batch = pending[start : start + config["batch_size"]]
                        results = evaluate_batch(
                            scorer,
                            batch,
                            thinking=thinking,
                            budget=config["max_thinking_tokens"],
                        )
                        for row in results:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            done[row["key"]] = row
                        stream.flush()
                        processed += len(results)
                        elapsed = time.perf_counter() - started
                        status = dict(
                            condition=name,
                            completed=len(done),
                            total=len(rows),
                            statuses=dict(Counter(r["status"] for r in done.values())),
                            invocation_seconds=elapsed,
                            invocation_candidates=processed,
                            eta_seconds=(len(rows) - len(done)) * elapsed / processed,
                        )
                        write_json(root / "progress.json", status)
                        print(json.dumps(status), flush=True)
            write_json(
                root / "run_status.json",
                dict(
                    complete=all(
                        len(load_journal(root / f"{n}.jsonl")) == len(rows)
                        for n in ("control", "thinking")
                    )
                ),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "score", "report"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--tokenizer")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.mode == "prepare":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer
            or json.loads(
                (Path(args.source or config["source_run"]) / "models.json").read_text()
            )["lm"]["path"],
            local_files_only=True,
        )
        print(
            json.dumps(prepare(config, args.source, args.output, tokenizer), indent=2)
        )
    elif args.mode == "score":
        score(config, limit=args.limit)
    else:
        from .judge_report import report

        report(
            Path(args.source or config["run_dir"]),
            Path(args.output or "reports/judge_thinking_comparison"),
        )


if __name__ == "__main__":
    main()
