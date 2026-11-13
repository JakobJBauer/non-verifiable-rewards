#!/usr/bin/env python3
"""Audit frozen references and likelihood arithmetic without model inference.

Uses the original raw files named in data_manifest.json and a local copy of
the recorded Qwen tokenizer. Does not import the project's reference builders
or reward formulas, and writes only to a separate audit directory.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def audit(run, tokenizer_dir, output):
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError("Write audit outputs outside the frozen run")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "data_manifest.json").read_text())
    sources = manifest["sources"]
    source_hashes = {}
    for name, source in sources.items():
        source_hashes[name] = sha256(source["path"])
        assert source_hashes[name] == source["sha256"], name
    selected = {
        name: read_rows(run / "data" / f"{name}.jsonl")
        for name in ("reddit_tldr", "litbench")
    }
    for name, pairs in selected.items():
        assert sha256(run / "data" / f"{name}.jsonl") == manifest[name]["sha256"]
        assert len(pairs) == 2000
        for pair in pairs:
            assert len({pair["preferred"], pair["rejected"], pair["reference"]}) == 3

    gold = {row["id"]: row for row in read_rows(sources["reddit_references"]["path"])}
    raw_reddit = Dataset.from_file(sources["reddit_comparisons"]["path"])
    for pair in selected["reddit_tldr"]:
        row = raw_reddit[pair["metadata"]["source_row"]]
        info = row["info"]
        assert pair["prompt_id"] == info["id"]
        assert info.get("site") not in ("cnn", "dailymail") and info["post"]
        reference = gold[info["id"]]
        assert reference["post"].strip() == info["post"].strip()
        assert pair["reference"] == reference["summary"].strip()
        assert pair["reference_id"] == info["id"] + ":human_tldr"
        assert pair["preferred"] == row["summaries"][row["choice"]]["text"].strip()
        assert pair["rejected"] == row["summaries"][1 - row["choice"]]["text"].strip()
        fields = {
            key: (info.get(key) or "").strip() for key in ("subreddit", "title", "post")
        }
        expected_prompt = (
            f"SUBREDDIT: r/{fields['subreddit']}\n\nTITLE: {fields['title']}"
            f"\n\nPOST: {fields['post']}\n\nTL;DR:"
        )
        assert pair["prompt"] == expected_prompt

    raw_lit = Dataset.from_file(sources["litbench"]["path"])
    prompts = {pair["prompt"] for pair in selected["litbench"]}
    pool = defaultdict(dict)
    for row in raw_lit:
        prompt = row["prompt"].strip()
        if prompt not in prompts:
            continue
        for side in ("chosen", "rejected"):
            text, votes = row[side + "_story"].strip(), int(row[side + "_upvotes"])
            pool[prompt][text] = max(votes, pool[prompt].get(text, votes))
    below_both = above_both = 0
    for pair in selected["litbench"]:
        row = raw_lit[pair["metadata"]["source_row"]]
        assert pair["prompt"] == row["prompt"].strip()
        assert pair["preferred"] == row["chosen_story"].strip()
        assert pair["rejected"] == row["rejected_story"].strip()
        assert row["chosen_upvotes"] > row["rejected_upvotes"]
        group = pool[pair["prompt"]]
        assert len(group) >= 3
        eligible = [
            text
            for text in group
            if text and text not in (pair["preferred"], pair["rejected"])
        ]
        best_votes = max(group[text] for text in eligible)
        tied_best = [text for text in eligible if group[text] == best_votes]
        expected = min(tied_best, key=content_hash)
        assert pair["reference"] == expected
        assert pair["metadata"]["reference_upvotes"] == group[expected]
        below_both += group[expected] < min(
            row["chosen_upvotes"], row["rejected_upvotes"]
        )
        above_both += group[expected] > max(
            row["chosen_upvotes"], row["rejected_upvotes"]
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    baselines = {
        row["pair_id"]: row
        for path in (run / "scores" / "lm").glob("*.baselines.jsonl")
        for row in read_rows(path)
    }
    results = {}
    for name, pairs in selected.items():
        prepared = {
            row["pair_id"]: row
            for row in read_rows(run / "scored_text" / f"{name}.jsonl")
        }
        journals = {
            row["pair_id"]: row
            for path in (run / "scores" / "lm").glob(f"{name}-shard-*.jsonl")
            for row in read_rows(path)
        }
        gains, levels = [], []
        raw_credit, vr_credit, tie_levels = [], [], defaultdict(int)
        reference_in_prompt = []
        arithmetic_error = 0.0
        max_sequence_tokens = 0
        for pair in pairs:
            reference_in_prompt.append(pair["reference"] in pair["prompt"])
            for field in ("prompt", "preferred", "rejected", "reference"):
                assert prepared[pair["pair_id"]][field] == pair[field]
            rows = {
                (row["reward"], row["side"]): row
                for row in journals[pair["pair_id"]]["rows"]
            }
            reference_ids = tokenizer.encode(
                pair["reference"], add_special_tokens=False
            )
            base_context = tokenizer.apply_chat_template(
                [{"role": "user", "content": pair["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            baseline = baselines[content_hash([base_context, pair["reference"]])]
            assert baseline["reference_id"] == pair["reference_id"]
            likelihoods, thresholded = [], []
            for side in ("preferred", "rejected"):
                likelihood = rows[("reference_likelihood", side)]
                vr = rows[("vr_cli", side)]
                stats = likelihood["statistics"]
                assert likelihood["status"] == vr["status"] == "ok"
                assert stats == vr["statistics"]
                assert stats["reference_tokens"] == len(reference_ids)
                c, b = stats["conditioned_logp_mean"], stats["baseline_logp_mean"]
                assert b == baseline["logp_mean"]
                assert likelihood["score"] == c
                assert math.isclose(
                    stats["conditioned_logp_sum"], c * len(reference_ids), abs_tol=1e-9
                )
                assert math.isclose(
                    stats["baseline_logp_sum"], b * len(reference_ids), abs_tol=1e-9
                )
                # Direct paper expression, independent of the project's expm1 implementation.
                original_ppl, conditioned_ppl = math.exp(-b), math.exp(-c)
                gain = 100.0 * (original_ppl - conditioned_ppl) / original_ppl
                expected_reward = (
                    0.0
                    if gain < 0.05
                    else 0.5
                    if gain < 1
                    else 0.9
                    if gain < 2
                    else 1.0
                )
                arithmetic_error = max(
                    arithmetic_error,
                    abs(gain - stats["perplexity_improvement_percent"]),
                )
                assert math.isclose(
                    gain,
                    stats["perplexity_improvement_percent"],
                    rel_tol=1e-12,
                    abs_tol=1e-10,
                )
                assert vr["score"] == expected_reward
                context = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": pair["prompt"]
                            + "\n\nUse this candidate response as context:\n"
                            + pair[side],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                context_ids = tokenizer.encode(context, add_special_tokens=False)
                assert len(context_ids) == stats["conditioning_tokens"]
                target_ids = tokenizer.encode(
                    pair["reference"], add_special_tokens=False
                )
                assert target_ids == reference_ids
                sequence_length = len(context_ids) + len(target_ids)
                if (
                    tokenizer.bos_token_id is not None
                    and context_ids[0] != tokenizer.bos_token_id
                ):
                    sequence_length += 1
                max_sequence_tokens = max(max_sequence_tokens, sequence_length)
                assert sequence_length <= 32768
                gains.append(gain)
                levels.append(expected_reward)
                likelihoods.append(c)
                thresholded.append(expected_reward)
            raw_margin, vr_margin = (
                likelihoods[0] - likelihoods[1],
                thresholded[0] - thresholded[1],
            )
            assert raw_margin * vr_margin >= 0
            raw_credit.append(
                1.0 if raw_margin > 1e-7 else 0.0 if raw_margin < -1e-7 else 0.5
            )
            vr_credit.append(
                1.0 if vr_margin > 1e-7 else 0.0 if vr_margin < -1e-7 else 0.5
            )
            if vr_margin == 0:
                tie_levels[str(thresholded[0])] += 1
        results[name] = {
            "pairs_checked": len(pairs),
            "candidate_likelihoods_and_thresholds_checked": len(levels),
            "raw_likelihood_accuracy": float(np.mean(raw_credit)),
            "vr_cli_accuracy": float(np.mean(vr_credit)),
            "vr_cli_tie_rate": float(np.mean(np.array(vr_credit) == 0.5)),
            "vr_cli_ties_by_level": dict(tie_levels),
            "perplexity_improvement_quantiles_percent": dict(
                zip(
                    ("min", "p25", "median", "p75", "max"),
                    np.quantile(gains, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                    strict=True,
                )
            ),
            "max_absolute_improvement_recomputation_error": arithmetic_error,
            "max_conditioned_sequence_tokens": max_sequence_tokens,
            "reference_already_present_in_prompt_pairs": sum(reference_in_prompt),
            "raw_accuracy_without_reference_in_prompt": float(
                np.mean(np.array(raw_credit)[~np.array(reference_in_prompt)])
            ),
            "vr_cli_accuracy_without_reference_in_prompt": float(
                np.mean(np.array(vr_credit)[~np.array(reference_in_prompt)])
            ),
        }
    report = {
        "status": "PASS",
        "source_files_sha256_verified": source_hashes,
        "references": {
            "reddit_original_tldr_matches": 2000,
            "reddit_source_post_and_preference_matches": 2000,
            "litbench_highest_outside_pair_matches": 2000,
            "litbench_source_prompt_and_preference_matches": 2000,
            "references_equal_to_a_candidate": 0,
            "litbench_reference_votes_below_both_candidates": below_both,
            "litbench_reference_votes_above_both_candidates": above_both,
            "vote_note": "Maximum reference votes compared with candidate source-row votes; not a calibrated quality comparison.",
        },
        "unique_cached_baselines": len(baselines),
        "datasets": results,
        "scope": "Source provenance, tokenizer/target/context counts, cache identities, and saved scalar arithmetic; does not recompute Qwen3-4B forward passes.",
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_dir, args.tokenizer_dir, args.output_dir)
