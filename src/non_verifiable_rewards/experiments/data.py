"""Freeze samples from the shared data loaders and prepare scoring inputs."""

import json
import logging
import random
from pathlib import Path

from ..data import load_litbench, load_sff, load_wmt14
from .io import configured_datasets, ensure_run, file_sha256, write_json, write_jsonl

logger = logging.getLogger(__name__)


def sample_pairs(pairs, count, seed):
    if len(pairs) < count:
        raise ValueError(f"Requested {count} pairs, but only {len(pairs)} eligible")
    # Prefixes of this fixed shuffle are nested across pilot/full sample sizes.
    ordered = sorted(pairs, key=lambda pair: pair["pair_id"])
    random.Random(seed).shuffle(ordered)
    return ordered[:count]


def prepare(
    config,
    *,
    sff_arrow=None,
    litbench_arrow=None,
    tldr_references=None,
    wmt14_submissions=None,
    wmt14_judgments=None,
):
    import pandas as pd

    run = ensure_run(config)
    names = configured_datasets(config)
    if all((run / "data" / f"{name}.jsonl").exists() for name in names):
        manifest = json.loads((run / "data_manifest.json").read_text())
        for name in names:
            if manifest[name]["sha256"] != file_sha256(run / "data" / f"{name}.jsonl"):
                raise ValueError("Prepared benchmark data changed after it was frozen")
        logger.info("Prepared datasets already exist and their hashes match")
        return
    cache = str(Path(config["cache_dir"]) / "datasets")
    loaders = {
        "reddit_tldr": lambda: load_sff(
            split="test",
            cache_dir=cache,
            sff_arrow=sff_arrow,
            tldr_references=tldr_references,
            return_metadata=True,
        ),
        "litbench": lambda: load_litbench(
            split="test",
            cache_dir=cache,
            litbench_arrow=litbench_arrow,
            return_metadata=True,
        ),
        "wmt14_de_en": lambda: load_wmt14(
            split="test",
            cache_dir=cache,
            submissions_archive=wmt14_submissions,
            judgments_archive=wmt14_judgments,
            return_metadata=True,
        ),
    }
    manifest = {
        "datasets": names,
        "sources": {},
        "sampling": "Uniform labeled pairs after filtering, fixed shuffled order",
        "seed": config["seed"],
    }
    for name in names:
        eligible, metadata_by_split = loaders[name]()
        metadata = next(iter(metadata_by_split.values()))
        audit = metadata["audit"]
        manifest["sources"].update(metadata["sources"])
        selected = sample_pairs(eligible, config["pairs_per_dataset"], config["seed"])
        target = run / "data" / f"{name}.jsonl"
        write_jsonl(target, selected)
        pd.DataFrame(selected).to_parquet(target.with_suffix(".parquet"), index=False)
        manifest[name] = {
            "split": "test",
            "split_policy": metadata["split_policy"],
            "audit": audit,
            "pairs": len(selected),
            "prompts": len({p["prompt_id"] for p in selected}),
            "sha256": file_sha256(target),
        }
        logger.info(
            "Prepared %s: %d pairs from %d eligible", name, len(selected), len(eligible)
        )
    write_json(run / "data_manifest.json", manifest)


def preprocess(config, tokenizer, models):
    """Save the exact text every backend sees, plus original/scored token counts."""
    from .io import read_jsonl

    run = Path(config["run_dir"])
    names = configured_datasets(config)
    manifest_path = run / "preprocessing.json"
    expected = {
        "lm_revision": models["lm"]["revision"],
        "limits": {
            key: config[key]
            for key in (
                "max_prompt_tokens",
                "max_candidate_tokens",
                "max_reference_tokens",
            )
        },
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if any(existing[key] != value for key, value in expected.items()):
            raise ValueError("Preprocessing configuration changed")
        for name in names:
            if (
                file_sha256(run / "scored_text" / f"{name}.jsonl")
                != existing["files"][name]
            ):
                raise ValueError("Scored text changed after preparation")
        return
    hashes = {}
    for name in names:
        records = []
        for pair in read_jsonl(run / "data" / f"{name}.jsonl"):
            record = dict(pair)
            lengths = {}
            for field in ("prompt", "preferred", "rejected", "reference"):
                text = pair[field]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                limit_name = (
                    "candidate" if field in ("preferred", "rejected") else field
                )
                limit = config[f"max_{limit_name}_tokens"]
                truncated = bool(limit and len(tokens) > limit)
                scored = (
                    tokenizer.decode(tokens[:limit], skip_special_tokens=True)
                    if truncated
                    else text
                )
                if not scored.strip():
                    raise ValueError(f"Empty scored {field} after preprocessing")
                record[field] = scored
                lengths[field] = {
                    "original_tokens": len(tokens),
                    "scored_tokens": len(
                        tokenizer.encode(scored, add_special_tokens=False)
                    ),
                    "truncated": truncated,
                    "original_words": len(text.split()),
                    "scored_words": len(scored.split()),
                }
            record["lengths"] = lengths
            record["reference_prefix_overlaps_candidate"] = record["reference"] in (
                record["preferred"],
                record["rejected"],
            )
            records.append(record)
        target = run / "scored_text" / f"{name}.jsonl"
        write_jsonl(target, records)
        hashes[name] = file_sha256(target)
    write_json(
        manifest_path,
        {
            **expected,
            "files": hashes,
            "policy": "Keep first N Qwen tokens; zero means full text. Original texts are retained in data/.",
        },
    )
