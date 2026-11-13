"""Pairwise evaluation, prompt-cluster bootstrap, and analysis-ready exports."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import BACKENDS, DATASETS, REWARDS, read_jsonl, write_jsonl


def preference_credit(margins, tolerance=1e-7):
    values = np.asarray(margins, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Pairwise margins must be finite")
    return np.where(values > tolerance, 1.0, np.where(values < -tolerance, 0.0, 0.5))


def cluster_bootstrap(credits, prompt_ids, *, samples, seed):
    """Resample prompts, retaining all their pairs; estimate pair-weighted accuracy.

    The same draws are used for every reward, preserving paired comparisons.
    Chunking bounds memory for the 2,000-pair / 10,000-bootstrap cluster run.
    """
    credits = np.asarray(credits, dtype=float)
    if credits.ndim != 2 or len(credits) == 0 or len(credits) != len(prompt_ids):
        raise ValueError("Need a nonempty matrix and one prompt ID per row")
    groups, inverse = np.unique(np.asarray(prompt_ids), return_inverse=True)
    totals = np.zeros((len(groups), credits.shape[1]))
    np.add.at(totals, inverse, credits)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    result = np.empty((samples, credits.shape[1]))
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        indices = rng.integers(0, len(groups), size=(size, len(groups)))
        result[start : start + size] = (
            totals[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
        )
    return result


def report_rewards(config):
    path = Path(config["run_dir"]) / "evaluation_scope.json"
    rewards = (
        tuple(json.loads(path.read_text())["benchmark_rewards"])
        if path.exists()
        else REWARDS
    )
    if (
        len(rewards) != len(set(rewards))
        or "length" not in rewards
        or not set(rewards).issubset(REWARDS)
    ):
        raise ValueError(
            "The report scope must contain distinct known rewards and the length baseline"
        )
    return rewards


def load_results(run, required_rewards=REWARDS):
    run = Path(run)
    originals = {name: read_jsonl(run / "data" / f"{name}.jsonl") for name in DATASETS}
    rows, timing_rows = [], []
    for backend in BACKENDS:
        for name in DATASETS:
            for path in sorted(
                (run / "scores" / backend).glob(f"{name}-shard-*.jsonl")
            ):
                for record in read_jsonl(path):
                    rows.extend(record["rows"])
                    timing_rows.append(
                        {
                            "dataset": name,
                            "backend": backend,
                            "pair_id": record["pair_id"],
                            "elapsed_seconds": record["elapsed_seconds"],
                            **{
                                key: value
                                for key, value in record["timings"].items()
                                if key != "traceback"
                            },
                        }
                    )
    if not rows:
        raise ValueError("There are no scores to report")
    scores = pd.json_normalize(rows)
    keys = ["dataset", "pair_id", "side", "reward"]
    if scores.duplicated(keys).any():
        raise ValueError(
            "Duplicate scores: overlapping shards or repeated pairs must not be averaged"
        )
    expected = {
        (name, pair["pair_id"], side, reward)
        for name, pairs in originals.items()
        for pair in pairs
        for side in ("preferred", "rejected")
        for reward in REWARDS
    }
    observed = set(scores[keys].itertuples(index=False, name=None))
    required = {key for key in expected if key[-1] in required_rewards}
    if required - observed or observed - expected:
        raise ValueError(
            f"Run is incomplete or inconsistent: {len(required - observed)} missing and {len(observed - expected)} unexpected candidate/reward records"
        )
    return scores, pd.DataFrame(timing_rows), originals


def export_table(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False, float_format="%.12g")
    frame.to_parquet(path.with_suffix(".parquet"), index=False)


def analyze(config):
    run = Path(config["run_dir"])
    rewards = report_rewards(config)
    scores, timings, originals = load_results(run, rewards)
    scores["included_in_primary_benchmark"] = scores.reward.isin(rewards)
    export_table(scores, run / "tables" / "candidate_scores")
    export_table(timings, run / "tables" / "pair_timings")
    export_table(scores[scores.status != "ok"], run / "tables" / "failures")
    summary_rows, coverage_rows, margin_frames, bootstrap_frames, length_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    correlations, agreements, examples = {}, {}, []
    for name in DATASETS:
        subset = scores[scores.dataset == name]
        pairs = originals[name]
        pair_lookup = {pair["pair_id"]: pair for pair in pairs}
        ok = subset[(subset.status == "ok") & np.isfinite(pd.to_numeric(subset.score))]
        pivot = ok.pivot(index="pair_id", columns=["reward", "side"], values="score")
        pivot = pivot.reindex(
            columns=pd.MultiIndex.from_product([rewards, ("preferred", "rejected")])
        )
        margins = pd.DataFrame(
            {
                reward: pivot[(reward, "preferred")] - pivot[(reward, "rejected")]
                for reward in rewards
            }
        )
        margins = margins.dropna()
        if margins.empty:
            raise ValueError(f"No pairs have complete scores for all rewards on {name}")
        for reward in REWARDS:
            reward_rows = subset[subset.reward == reward]
            valid = reward_rows[reward_rows.status == "ok"]
            complete_pairs = int((valid.groupby("pair_id").size() == 2).sum())
            coverage_rows.append(
                {
                    "dataset": name,
                    "reward": reward,
                    "expected_pairs": len(pairs),
                    "complete_pairs": complete_pairs,
                    "common_pairs": len(margins) if reward in rewards else 0,
                    "included_in_primary_benchmark": reward in rewards,
                    "missing_candidates": 2 * len(pairs) - len(reward_rows),
                    "failed_candidates": int((reward_rows.status != "ok").sum()),
                    "preprocessing_truncated_candidates": int(
                        valid.preprocessing_truncated.eq(True).sum()
                    ),
                    "native_truncated_candidates": int(
                        valid.native_truncated.eq(True).sum()
                    ),
                    "reference_prefix_overlap_candidates": int(
                        valid.get(
                            "reference_prefix_overlaps_candidate", pd.Series(dtype=bool)
                        )
                        .eq(True)
                        .sum()
                    ),
                }
            )
        prompt_ids = [pair_lookup[pair_id]["prompt_id"] for pair_id in margins.index]
        credits = preference_credit(margins.to_numpy(), config["tie_tolerance"])
        bootstrap = cluster_bootstrap(
            credits,
            prompt_ids,
            samples=config["bootstrap_samples"],
            seed=config["seed"],
        )
        baseline_index = rewards.index("length")
        differences = bootstrap - bootstrap[:, baseline_index, None]
        ci = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        delta_ci = np.quantile(differences, [0.025, 0.975], axis=0)
        for index, reward in enumerate(rewards):
            summary_rows.append(
                {
                    "dataset": name,
                    "reward": reward,
                    "pairs": len(margins),
                    "prompts": len(set(prompt_ids)),
                    "accuracy": float(credits[:, index].mean()),
                    "ci_low": float(ci[0, index]),
                    "ci_high": float(ci[1, index]),
                    "tie_rate": float((credits[:, index] == 0.5).mean()),
                    "delta_vs_length": float(
                        (credits[:, index] - credits[:, baseline_index]).mean()
                    ),
                    "delta_ci_low": float(delta_ci[0, index]),
                    "delta_ci_high": float(delta_ci[1, index]),
                }
            )
        boot_frame = pd.DataFrame(bootstrap, columns=rewards)
        boot_frame.insert(0, "dataset", name)
        boot_frame.insert(1, "replicate", np.arange(len(bootstrap)))
        bootstrap_frames.append(boot_frame)
        correlations[name] = margins.corr(method="spearman")
        # Three-way decision agreement includes ties as a distinct decision.
        agreements[name] = pd.DataFrame(
            (credits[:, :, None] == credits[:, None, :]).mean(axis=0),
            index=rewards,
            columns=rewards,
        )
        max_words = np.array(
            [
                max(
                    len(pair_lookup[key]["preferred"].split()),
                    len(pair_lookup[key]["rejected"].split()),
                )
                for key in margins.index
            ]
        )
        length_difference = np.array(
            [
                len(pair_lookup[key]["preferred"].split())
                - len(pair_lookup[key]["rejected"].split())
                for key in margins.index
            ]
        )
        for condition, mask in (
            ("Preferred shorter", length_difference < 0),
            ("Equal length", length_difference == 0),
            ("Preferred longer", length_difference > 0),
        ):
            if not mask.any():
                continue
            for index, reward in enumerate(rewards):
                length_rows.append(
                    {
                        "dataset": name,
                        "condition": condition,
                        "reward": reward,
                        "pairs": int(mask.sum()),
                        "accuracy": float(credits[mask, index].mean()),
                    }
                )
        margin_frame = margins.copy().reset_index()
        margin_frame.insert(0, "dataset", name)
        margin_frame["prompt_id"] = prompt_ids
        margin_frame["max_original_candidate_words"] = max_words
        margin_frame["preferred_minus_rejected_words"] = length_difference
        for index, reward in enumerate(rewards):
            margin_frame[reward + "_credit"] = credits[:, index]
        margin_frames.append(margin_frame)
        metric_credits = credits[
            :, [i for i, reward in enumerate(rewards) if reward != "length"]
        ]
        disagreement = np.minimum(
            (metric_credits == 1).sum(axis=1), (metric_credits == 0).sum(axis=1)
        )
        ranked = sorted(
            range(len(margins)), key=lambda i: (-disagreement[i], margins.index[i])
        )
        for index in ranked[:6]:
            if disagreement[index] == 0:
                continue
            pair = pair_lookup[margins.index[index]]
            examples.append(
                {
                    **pair,
                    "margins": margins.iloc[index].to_dict(),
                    "credits": dict(zip(rewards, credits[index].tolist(), strict=True)),
                    "selection": "Most evenly split correct/incorrect neural+ROUGE decisions; pair ID breaks ties",
                }
            )
    summary = pd.DataFrame(summary_rows)
    coverage = pd.DataFrame(coverage_rows)
    margins = pd.concat(margin_frames, ignore_index=True)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    lengths = pd.DataFrame(length_rows)
    for label, frame in (
        ("summary", summary),
        ("coverage", coverage),
        ("pair_margins", margins),
        ("bootstrap_accuracy", bootstrap),
        ("length_diagnostics", lengths),
    ):
        export_table(frame, run / "tables" / label)
    for name in DATASETS:
        for label, frame in (
            ("spearman", correlations[name]),
            ("decision_agreement", agreements[name]),
        ):
            frame.rename_axis("reward").reset_index().to_csv(
                run / "tables" / f"{name}_{label}.csv", index=False
            )
    write_jsonl(run / "tables" / "disagreement_examples.jsonl", examples)
    runtime = timings.groupby(["dataset", "backend"], as_index=False).agg(
        pairs=("pair_id", "size"), scoring_seconds=("elapsed_seconds", "sum")
    )
    runtime["seconds_per_candidate"] = runtime.scoring_seconds / (2 * runtime.pairs)
    export_table(runtime, run / "tables" / "runtime")
    workers = []
    for path in sorted((run / "scores").glob("*/*.invocation-*.json")):
        worker = json.loads(path.read_text())
        workers.append(
            {
                "backend": worker["identity"]["backend"],
                "shard_index": worker["identity"]["shard_index"],
                **{
                    key: worker.get(key)
                    for key in (
                        "started_at",
                        "finished_at",
                        "wall_seconds_this_invocation",
                        "model_load_seconds",
                        "pairs_processed_this_invocation",
                        "errors",
                        "peak_process_rss_gib",
                        "peak_gpu_allocated_gib",
                        "gpu_name",
                    )
                },
            }
        )
    worker_frame = pd.DataFrame(workers)
    if not worker_frame.empty:
        export_table(worker_frame, run / "tables" / "workers")
    return {
        "rewards": rewards,
        "scores": scores,
        "summary": summary,
        "coverage": coverage,
        "margins": margins,
        "lengths": lengths,
        "correlations": correlations,
        "agreements": agreements,
        "runtime": runtime,
        "workers": worker_frame,
        "examples": examples,
    }
