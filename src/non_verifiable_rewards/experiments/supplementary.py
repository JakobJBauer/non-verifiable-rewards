"""Additional diagnostics from completed scores; no model loading or inference."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import export_table, preference_credit
from .io import DATASETS


def supplementary(run_dir, pilot_dir):
    run, pilot = Path(run_dir), Path(pilot_dir)
    config = json.loads((run / "config.json").read_text())
    margins = pd.read_parquet(run / "tables" / "pair_margins.parquet")
    scores = pd.read_parquet(run / "tables" / "candidate_scores.parquet")
    bootstrap = pd.read_parquet(run / "tables" / "bootstrap_accuracy.parquet")
    summary = pd.read_parquet(run / "tables" / "summary.parquet")
    pilot_margins = pd.read_parquet(pilot / "tables" / "pair_margins.parquet")
    pilot_summary = pd.read_parquet(pilot / "tables" / "summary.parquet")
    rewards = list(summary.reward.unique())
    comparisons, overlaps, truncation, weighting, thresholds = [], [], [], [], []

    for dataset in DATASETS:
        frame = margins[margins.dataset == dataset].set_index("pair_id")
        boot = bootstrap[bootstrap.dataset == dataset]
        for first, second in combinations(rewards, 2):
            delta = boot[first] - boot[second]
            ci = np.quantile(delta, [0.025, 0.975])
            comparisons.append(
                {
                    "dataset": dataset,
                    "reward_a": first,
                    "reward_b": second,
                    "pairs": len(frame),
                    "accuracy_a_minus_b": float(
                        (frame[first + "_credit"] - frame[second + "_credit"]).mean()
                    ),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                }
            )
        for reward in rewards:
            weighting.append(
                {
                    "dataset": dataset,
                    "reward": reward,
                    "pairs": len(frame),
                    "prompts": frame.prompt_id.nunique(),
                    "pair_weighted_accuracy": frame[reward + "_credit"].mean(),
                    "prompt_weighted_accuracy": frame.groupby("prompt_id")[
                        reward + "_credit"
                    ]
                    .mean()
                    .mean(),
                }
            )

        old = pilot_margins[pilot_margins.dataset == dataset].set_index("pair_id")
        shared = old.index.intersection(frame.index)
        if len(shared) != len(old):
            raise ValueError("The new run must include every reported pilot pair")
        for reward in pilot_summary.reward.unique():
            old_credit = old.loc[shared, reward + "_credit"]
            new_credit = frame.loc[shared, reward + "_credit"]
            if reward in ("rouge_l", "length") and not np.allclose(
                old.loc[shared, reward], frame.loc[shared, reward], rtol=0, atol=1e-12
            ):
                raise ValueError(
                    f"Full-text {reward} changed on shared {dataset} pairs"
                )
            overlaps.append(
                {
                    "dataset": dataset,
                    "reward": reward,
                    "shared_pairs": len(shared),
                    "pilot_accuracy": old_credit.mean(),
                    "gpu_shared_accuracy": new_credit.mean(),
                    "gpu_all_accuracy": frame[reward + "_credit"].mean(),
                    "decision_agreement": (old_credit == new_credit).mean(),
                    "note": "Encoder checkpoint and text policy both changed; differences cannot be attributed to either alone.",
                }
            )

        for reward in ("bertscore", "bleurt"):
            subset = scores[
                (scores.dataset == dataset)
                & (scores.reward == reward)
                & (scores.status == "ok")
            ]
            shortened = (
                subset.groupby("pair_id").native_truncated.any().reindex(frame.index)
            )
            if shortened.isna().any():
                raise ValueError("Missing encoder truncation flags for reported pairs")
            for flag in (False, True):
                selected = frame[shortened == flag]
                truncation.append(
                    {
                        "dataset": dataset,
                        "reward": reward,
                        "native_truncated": flag,
                        "pairs": len(selected),
                        "accuracy": selected[reward + "_credit"].mean()
                        if len(selected)
                        else None,
                        "median_max_candidate_words": selected.max_original_candidate_words.median()
                        if len(selected)
                        else None,
                        "note": "Descriptive groups contain different examples; not a causal truncation experiment.",
                    }
                )

        if "vr_cli" in rewards:
            raw = preference_credit(frame.reference_likelihood, config["tie_tolerance"])
            bucketed = preference_credit(frame.vr_cli, config["tie_tolerance"])
            inversions = int(
                ((raw == 1) & (bucketed == 0) | (raw == 0) & (bucketed == 1)).sum()
            )
            thresholds.append(
                {
                    "dataset": dataset,
                    "pairs": len(frame),
                    "raw_likelihood_accuracy": raw.mean(),
                    "thresholded_accuracy": bucketed.mean(),
                    "thresholded_tie_rate": (bucketed == 0.5).mean(),
                    "raw_correct_collapsed_to_tie": int(
                        ((raw == 1) & (bucketed == 0.5)).sum()
                    ),
                    "raw_incorrect_collapsed_to_tie": int(
                        ((raw == 0) & (bucketed == 0.5)).sum()
                    ),
                    "ranking_inversions": inversions,
                }
            )
            if inversions:
                raise ValueError(
                    "Monotone VR-CLI thresholds reversed a raw likelihood ranking"
                )

    for name, rows in (
        ("all_paired_comparisons", comparisons),
        ("pilot_overlap", overlaps),
        ("encoder_truncation_diagnostics", truncation),
        ("prompt_weighting_sensitivity", weighting),
        ("vr_cli_information_loss", thresholds),
    ):
        if rows:
            export_table(pd.DataFrame(rows), run / "tables" / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pilot-dir", default="outputs/local_cpu_int8")
    args = parser.parse_args()
    supplementary(args.run_dir, args.pilot_dir)
