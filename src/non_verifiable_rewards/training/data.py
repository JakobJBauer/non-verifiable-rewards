"""Freeze training references separately from held-out benchmark labels."""

import random
from pathlib import Path

from .._hashing import digest, file_sha256
from ..data import (
    load_sff,
    load_litbench,
    load_wmt14,
    load_supergpqa,
    to_prompt_dataset,
)
from ..experiments.data import sample_pairs
from ..experiments.io import read_jsonl, write_json, write_jsonl
from ..rewards.prompts import translation_prompt
from .state import ensure_sweep, verify_data


def policy_prompt(dataset, prompt):
    if dataset == "reddit_tldr":
        return (
            "Write an accurate, concise summary of the following source post. Return only the summary.\n\n"
            + prompt
        )
    if dataset == "litbench":
        return (
            "Write a standalone story responding to this writing prompt. Return only the story.\n\n"
            + prompt
        )
    if dataset == "wmt14_de_en":
        prefix = "Translate the following German text into English:\n\n"
        return translation_prompt(prompt.removeprefix(prefix))
    if dataset == "supergpqa":
        return prompt  # Loader already includes options and the final-answer format.
    raise ValueError(f"Unknown task: {dataset}")


def render_prompt(tokenizer, row, enable_thinking=False):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": policy_prompt(row["dataset"], row["prompt"])}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def load_split(config, dataset, split):
    local = config["data"]["local"].get(dataset, {}).get(split)
    if local:
        rows = read_jsonl(local)
        for row in rows:
            if (
                row.get("dataset") != dataset
                or row.get("metadata", {}).get("split") != split
            ):
                raise ValueError(
                    "Local rows must record their dataset and metadata.split"
                )
        return rows, {
            "local_jsonl": str(Path(local).resolve()),
            "sha256": file_sha256(local),
        }
    loader = {
        "reddit_tldr": load_sff,
        "litbench": load_litbench,
        "wmt14_de_en": load_wmt14,
        "supergpqa": load_supergpqa,
    }[dataset]
    kwargs = {
        "split": split,
        "cache_dir": str(Path(config["cache_dir"]) / "datasets"),
        "return_metadata": True,
    }
    if dataset == "wmt14_de_en" and split == "train":
        kwargs["streaming"] = True
    rows, metadata = loader(**kwargs)
    return rows, metadata


def sample_train(rows, count, seed, tokenizer, config):
    # WMT's streaming shuffle is a bounded-buffer sample, not an exact uniform
    # sample of 4.5M translations. The selected examples are frozen for all runs.
    if hasattr(rows, "shuffle"):
        from datasets import IterableDataset

        options = (
            {"buffer_size": config["data"]["stream_shuffle_buffer"]}
            if isinstance(rows, IterableDataset)
            else {}
        )
        rows = rows.shuffle(seed=seed, **options)
    else:
        rows = sorted(
            rows, key=lambda r: digest([seed, r["prompt_id"], r["reference"]])
        )
    selected, seen, seen_ids, excluded = (
        [],
        set(),
        set(),
        {"duplicate_prompt": 0, "prompt_too_long": 0, "reference_budget": 0},
    )
    for row in rows:
        if any(key in row for key in ("preferred", "rejected", "chosen")):
            raise ValueError(
                "Training requires the prompt/reference train split, not preference pairs"
            )
        if not row["prompt"].strip() or not row["reference"].strip():
            raise ValueError("Training prompt and reference must be nonempty")
        key = " ".join(row["prompt"].split())
        if key in seen or row["prompt_id"] in seen_ids:
            excluded["duplicate_prompt"] += 1
            continue
        tokens = len(
            tokenizer.encode(
                render_prompt(tokenizer, row, config["policy"]["enable_thinking"]),
                add_special_tokens=False,
            )
        )
        if tokens > config["data"]["max_prompt_tokens"]:
            excluded["prompt_too_long"] += 1
            continue
        # Reserve reference/rubric space for fixed Qwen likelihood/judge rewards.
        budget = (
            tokens
            + len(tokenizer.encode(row["reference"], add_special_tokens=False))
            + config["optimization"]["max_completion_length"]
            + 2048
        )
        if budget > config["reward_models"]["max_context_tokens"]:
            excluded["reference_budget"] += 1
            continue
        seen.add(key)
        seen_ids.add(row["prompt_id"])
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"Requested {count} eligible training prompts; found {len(selected)}"
        )
    return selected, excluded


def prepare(config, tokenizer):
    from datasets import Dataset

    root = ensure_sweep(config)
    if (root / "data_manifest.json").exists():
        return verify_data(root)
    files, sources, counts = {}, {}, {}
    for name in config["data"]["datasets"]:
        train, train_meta = load_split(config, name, "train")
        test, test_meta = load_split(config, name, "test")
        test = list(test)
        train, excluded = sample_train(
            train,
            config["data"]["train_examples_per_dataset"],
            config["seed"],
            tokenizer,
            config,
        )
        train_texts = {" ".join(r["prompt"].split()) for r in train}
        train_ids = {r["prompt_id"] for r in train}
        if any(
            " ".join(r["prompt"].split()) in train_texts or r["prompt_id"] in train_ids
            for r in test
        ):
            raise ValueError(f"Train/test prompt leakage in {name}")
        if name == "supergpqa":
            unique = {}
            for row in sorted(test, key=lambda r: r["example_id"]):
                unique.setdefault(row["prompt_id"], row)
            ordered = list(unique.values())
            random.Random(config["seed"]).shuffle(ordered)
            pairs = []
            generation = ordered[
                : config["benchmark"]["generation_prompts_per_dataset"]
            ]
        else:
            pairs = sample_pairs(
                test, config["benchmark"]["pairs_per_dataset"], config["seed"]
            )
            # Unique prompts avoid silently giving frequently compared prompts
            # extra weight in generated-output evaluation. References are fixed.
            ordered = list(to_prompt_dataset(Dataset.from_list(test)))
            random.Random(config["seed"]).shuffle(ordered)
            generation = ordered[
                : config["benchmark"]["generation_prompts_per_dataset"]
            ]
        if len(generation) != config["benchmark"]["generation_prompts_per_dataset"]:
            raise ValueError(f"Not enough held-out generation prompts for {name}")
        for split, records in (
            ("train", train),
            ("pairs", pairs),
            ("generation", generation),
        ):
            target = root / "data" / f"{name}-{split}.jsonl"
            write_jsonl(target, records)
            files[str(target.relative_to(root))] = file_sha256(target)
        sources[name] = {"train": train_meta, "test": test_meta}
        counts[name] = {
            "train": len(train),
            "test_pairs": len(pairs),
            "generation_prompts": len(generation),
            "excluded_train": excluded,
        }
    manifest = {
        "files": files,
        "sources": sources,
        "counts": counts,
        "seed": config["seed"],
        "training_sampling": "Seeded shuffle; WMT uses a bounded streaming shuffle; deduplicated prompts; equal counts per selected dataset",
        "reference_policy": "Reference is reward-side data only, never supplied to the policy",
    }
    write_json(root / "data_manifest.json", manifest)
    return manifest


def training_dataset(config, run, tokenizer):
    from datasets import Dataset

    root = Path(config["output_dir"])
    verify_data(root)
    rows = []
    for name in run["datasets"]:
        for row in read_jsonl(root / "data" / f"{name}-train.jsonl"):
            rows.append(
                {
                    "prompt": [
                        {"role": "user", "content": policy_prompt(name, row["prompt"])}
                    ],
                    "source_prompt": row["prompt"],
                    "dataset": name,
                    "reference": row["reference"],
                    "prompt_id": row["prompt_id"],
                    "options": row.get("options", []),
                    "answer_letter": row.get("answer_letter", ""),
                }
            )
    return Dataset.from_list(rows).shuffle(seed=config["seed"])
