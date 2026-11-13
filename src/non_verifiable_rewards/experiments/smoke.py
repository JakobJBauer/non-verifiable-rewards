"""One-pair LM execution check, kept separate from preference accuracy."""

import json
import math
import time
from pathlib import Path

from .io import (
    BACKEND_REWARDS,
    DATASETS,
    Journal,
    digest,
    ensure_run,
    file_sha256,
    read_jsonl,
    utc_now,
    write_json,
)

PILOT_REWARDS = ("rouge_l", "bertscore", "bleurt", "length")


def set_lm_scope(config, *, skip=False):
    run = ensure_run(config)
    path = run / "evaluation_scope.json"
    scope = json.loads(path.read_text()) if path.exists() else {}
    scope.update(
        benchmark_rewards=list(PILOT_REWARDS),
        lm_mode="skip" if skip else "smoke",
        partial_lm_used_in_accuracy_figures=False,
    )
    scope.setdefault("created_at", utc_now())
    scope.setdefault(
        "reason",
        "LM evaluation limited locally; full comparison deferred to the GPU run.",
    )
    write_json(path, scope)


def lm_mode(config):
    path = Path(config["run_dir"]) / "evaluation_scope.json"
    if not path.exists():
        return "full"
    scope = json.loads(path.read_text())
    if any(reward in scope["benchmark_rewards"] for reward in BACKEND_REWARDS["lm"]):
        return "full"
    return scope.get("lm_mode", "smoke")


def successful_pair(record):
    rows = record.get("rows", [])
    expected = {
        (reward, side)
        for reward in BACKEND_REWARDS["lm"]
        for side in ("preferred", "rejected")
    }
    return (
        len(rows) == len(expected)
        and {(row["reward"], row["side"]) for row in rows} == expected
        and all(
            row["status"] == "ok"
            and isinstance(row.get("score"), (float, int))
            and math.isfinite(row["score"])
            for row in rows
        )
    )


def smoke_check(config):
    """Reuse a successful saved pair before considering any new inference."""
    run = ensure_run(config)
    path = run / "qwen_smoke_check.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if not successful_pair(saved["validation"]):
            raise ValueError("Saved LM smoke check is incomplete or unsuccessful")
        return saved

    from .assets import model_paths
    from .runner import Backend, scoring_code_digest

    models = model_paths(config)
    validation, selected = None, None
    for name in DATASETS:
        for journal in sorted((run / "scores" / "lm").glob(f"{name}-shard-*.jsonl")):
            for record in read_jsonl(journal):
                if successful_pair(record):
                    validation, selected = record, name
                    break
            if validation is not None:
                break
        if validation is not None:
            break

    identity = {
        "config_sha256": digest(config),
        "scoring_code_sha256": scoring_code_digest(),
        "preprocessing_sha256": file_sha256(run / "preprocessing.json"),
    }
    if validation is None:
        selected = DATASETS[0]
        pair = read_jsonl(run / "scored_text" / f"{selected}.jsonl")[0]
        original = next(
            item
            for item in read_jsonl(run / "data" / f"{selected}.jsonl")
            if item["pair_id"] == pair["pair_id"]
        )
        started = time.perf_counter()
        engine = Backend("lm", config, models)
        model_load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        # A separate journal cannot overlap the eventual full benchmark shards.
        with Journal(run / "smoke" / "baselines.jsonl") as baselines:
            rows, timings = engine.evaluate(pair, original, baselines)
        validation = {
            "pair_id": pair["pair_id"],
            "dataset": selected,
            "rows": rows,
            "timings": timings,
            "elapsed_seconds": time.perf_counter() - started,
            "model_load_seconds": model_load_seconds,
            "completed_at": utc_now(),
        }
        if not successful_pair(validation):
            raise ValueError(
                "LM smoke check did not produce six finite candidate scores"
            )

    original = next(
        item
        for item in read_jsonl(run / "data" / f"{selected}.jsonl")
        if item["pair_id"] == validation["pair_id"]
    )
    scored = next(
        item
        for item in read_jsonl(run / "scored_text" / f"{selected}.jsonl")
        if item["pair_id"] == validation["pair_id"]
    )
    result = {
        "validation": validation,
        "input": original,
        "scored_input": scored,
        "model": models["lm"],
        "identity": identity,
        "cpu_dynamic_int8": config.get("cpu_dynamic_int8", False),
        "note": "One successful pair checks execution only; it does not estimate preference accuracy.",
    }
    write_json(path, result)
    return result
