"""Build auditable preference pairs with references independent of either side."""

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..data.loaders import _sff_prompt
from .assets import TLDR_URL, download_file
from .io import DATASETS, digest, ensure_run, file_sha256, write_json, write_jsonl

logger = logging.getLogger(__name__)


def make_pair(
    dataset, prompt_id, prompt, preferred, rejected, reference, reference_id, metadata
):
    preferred, rejected, reference = (
        preferred.strip(),
        rejected.strip(),
        reference.strip(),
    )
    if not all((prompt.strip(), preferred, rejected, reference)):
        raise ValueError("Empty benchmark field")
    if preferred == rejected or reference in (preferred, rejected):
        raise ValueError("Candidates and the held-out reference must be distinct")
    return {
        "dataset": dataset,
        "prompt_id": prompt_id,
        "pair_id": digest([dataset, prompt_id, preferred, rejected])[:24],
        "prompt": prompt.strip(),
        "preferred": preferred,
        "rejected": rejected,
        "reference": reference,
        "reference_id": reference_id,
        "preferred_id": digest([prompt_id, preferred])[:24],
        "rejected_id": digest([prompt_id, rejected])[:24],
        "metadata": metadata,
    }


def reddit_pairs(rows, references):
    audit = Counter()
    pairs = {}
    for index, row in enumerate(rows):
        audit["source_rows"] += 1
        info = row["info"]
        if info.get("site") in ("cnn", "dailymail") or not info.get("post"):
            audit["excluded_non_reddit"] += 1
            continue
        gold = references.get(info["id"])
        if gold is None:
            audit["excluded_missing_reference"] += 1
            continue
        if gold["post"].strip() != info["post"].strip():
            audit["excluded_source_post_mismatch"] += 1
            continue
        summaries, choice = row["summaries"], row["choice"]
        if len(summaries) != 2 or choice not in (0, 1):
            audit["excluded_invalid_choice"] += 1
            continue
        try:
            pair = make_pair(
                "reddit_tldr",
                info["id"],
                _sff_prompt(info),
                summaries[choice]["text"],
                summaries[1 - choice]["text"],
                gold["summary"],
                info["id"] + ":human_tldr",
                {
                    "source_row": index,
                    "split": row.get("split"),
                    "batch": row.get("batch"),
                    "confidence": row.get("extra", {}).get("confidence"),
                    "preferred_policy": summaries[choice].get("policy"),
                    "rejected_policy": summaries[1 - choice].get("policy"),
                    "reference_rule": "Original human TL;DR joined by post ID; source post must match",
                },
            )
        except ValueError:
            audit["excluded_empty_or_overlapping_texts"] += 1
            continue
        if pair["pair_id"] in pairs:
            audit["excluded_duplicate_labeled_pair"] += 1
            continue
        pairs[pair["pair_id"]] = pair
    audit["eligible_pairs"] = len(pairs)
    audit["eligible_prompts"] = len({p["prompt_id"] for p in pairs.values()})
    return list(pairs.values()), dict(audit)


def litbench_pairs(rows):
    audit = Counter()
    stories = defaultdict(dict)
    for row in rows:
        prompt = row["prompt"].strip()
        for side in ("chosen", "rejected"):
            text = row[side + "_story"].strip()
            votes = int(row[side + "_upvotes"])
            values = stories[prompt].setdefault(text, set())
            values.add(votes)
    audit["source_prompts"] = len(stories)
    audit["prompts_with_at_least_three_stories"] = sum(
        len(group) >= 3 for group in stories.values()
    )
    audit["stories_with_multiple_observed_vote_counts"] = sum(
        len(votes) > 1 for group in stories.values() for votes in group.values()
    )
    pairs = {}
    for index, row in enumerate(rows):
        audit["source_rows"] += 1
        prompt = row["prompt"].strip()
        group = stories[prompt]
        if len(group) < 3:
            audit["excluded_fewer_than_three_stories"] += 1
            continue
        audit["rows_with_at_least_three_stories"] += 1
        if row["chosen_upvotes"] <= row["rejected_upvotes"]:
            audit["excluded_non_strict_upvote_preference"] += 1
            continue
        preferred, rejected = row["chosen_story"].strip(), row["rejected_story"].strip()
        candidates = [
            text for text in group if text and text not in (preferred, rejected)
        ]
        if not candidates:
            audit["excluded_missing_held_out_story"] += 1
            continue
        # Counts can vary across scrapes. Freeze the maximum observed count for
        # each exact story; break vote ties by a stable content hash.
        reference = min(candidates, key=lambda text: (-max(group[text]), digest(text)))
        prompt_id = digest(["litbench", prompt])[:24]
        try:
            pair = make_pair(
                "litbench",
                prompt_id,
                prompt,
                preferred,
                rejected,
                reference,
                digest([prompt_id, reference])[:24],
                {
                    "source_row": index,
                    "preferred_upvotes": int(row["chosen_upvotes"]),
                    "rejected_upvotes": int(row["rejected_upvotes"]),
                    "reference_upvotes": max(group[reference]),
                    "stories_for_prompt": len(group),
                    "reference_observed_upvotes": sorted(group[reference]),
                    "reference_rule": "Highest maximum-observed upvotes outside the pair; hash breaks ties",
                },
            )
        except ValueError:
            audit["excluded_empty_or_overlapping_texts"] += 1
            continue
        if pair["pair_id"] in pairs:
            audit["excluded_duplicate_labeled_pair"] += 1
            continue
        pairs[pair["pair_id"]] = pair
    audit["eligible_pairs"] = len(pairs)
    audit["eligible_prompts"] = len({p["prompt_id"] for p in pairs.values()})
    return list(pairs.values()), dict(audit)


def sample_pairs(pairs, count, seed):
    if len(pairs) < count:
        raise ValueError(f"Requested {count} pairs, but only {len(pairs)} eligible")
    # Prefixes of this fixed shuffle are nested across pilot/full sample sizes.
    ordered = sorted(pairs, key=lambda pair: pair["pair_id"])
    random.Random(seed).shuffle(ordered)
    return ordered[:count]


def source_info(path, description):
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "description": description,
    }


def prepare(config, *, sff_arrow=None, litbench_arrow=None, tldr_references=None):
    import pandas as pd
    from datasets import Dataset, load_dataset

    run = ensure_run(config)
    if all((run / "data" / f"{name}.jsonl").exists() for name in DATASETS):
        manifest = json.loads((run / "data_manifest.json").read_text())
        for name in DATASETS:
            if manifest[name]["sha256"] != file_sha256(run / "data" / f"{name}.jsonl"):
                raise ValueError("Prepared benchmark data changed after it was frozen")
        logger.info("Prepared datasets already exist and their hashes match")
        return
    gold_path = (
        Path(tldr_references)
        if tldr_references
        else download_file(
            TLDR_URL, Path(config["cache_dir"]) / "data" / "tldr_valid.jsonl"
        )
    )
    references = {}
    with gold_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in references and references[row["id"]] != row:
                raise ValueError(
                    "Conflicting human references for the same Reddit post"
                )
            references[row["id"]] = row
    sources = {"reddit_references": source_info(gold_path, TLDR_URL)}
    if sff_arrow:
        sff = Dataset.from_file(str(sff_arrow))
        sources["reddit_comparisons"] = source_info(
            sff_arrow, "openai/summarize_from_feedback comparisons validation"
        )
    else:
        sff = load_dataset(
            "parquet",
            data_files={
                "validation": "hf://datasets/openai/summarize_from_feedback@refs/convert/parquet/comparisons/validation/0000.parquet"
            },
            split="validation",
            cache_dir=str(Path(config["cache_dir"]) / "datasets"),
        )
        sources["reddit_comparisons"] = {
            "dataset": "openai/summarize_from_feedback",
            "split": "validation",
            "fingerprint": sff._fingerprint,
        }
    if litbench_arrow:
        litbench = Dataset.from_file(str(litbench_arrow))
        sources["litbench"] = source_info(
            litbench_arrow, "SAA-Lab/LitBench-Train train"
        )
    else:
        litbench = load_dataset(
            "SAA-Lab/LitBench-Train",
            split="train",
            cache_dir=str(Path(config["cache_dir"]) / "datasets"),
        )
        sources["litbench"] = {
            "dataset": "SAA-Lab/LitBench-Train",
            "split": "train",
            "fingerprint": litbench._fingerprint,
        }
    manifest = {
        "sources": sources,
        "sampling": "Uniform labeled pairs after filtering, fixed shuffled order",
        "seed": config["seed"],
    }
    for name, builder in (
        ("reddit_tldr", lambda: reddit_pairs(sff, references)),
        ("litbench", lambda: litbench_pairs(litbench)),
    ):
        eligible, audit = builder()
        selected = sample_pairs(eligible, config["pairs_per_dataset"], config["seed"])
        target = run / "data" / f"{name}.jsonl"
        write_jsonl(target, selected)
        pd.DataFrame(selected).to_parquet(target.with_suffix(".parquet"), index=False)
        manifest[name] = {
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
        for name in DATASETS:
            if (
                file_sha256(run / "scored_text" / f"{name}.jsonl")
                != existing["files"][name]
            ):
                raise ValueError("Scored text changed after preparation")
        return
    hashes = {}
    for name in DATASETS:
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
