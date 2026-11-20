"""Derive NOVER proxy scores from a frozen run without new model inference."""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

from ..rewards.nover import NOVERReward, nover_length_normalizer
from .analysis import load_results, report_rewards
from .io import (
    DATASETS,
    REWARDS,
    ensure_run,
    file_sha256,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
)


def _snapshot(run):
    return {
        str(path.relative_to(run)): file_sha256(path)
        for path in sorted(run.rglob("*"))
        if path.is_file()
    }


def _derive_row(row):
    derived = {**row, "reward": "nover", "statistics": {}}
    if row["status"] != "ok":
        # Preserve failures rather than silently dropping them from coverage.
        derived["score"] = None
        return derived
    stats = row["statistics"]
    mean, count = stats["conditioned_logp_mean"], stats["candidate_tokens"]
    if not math.isclose(row["score"], mean, rel_tol=0, abs_tol=1e-12):
        raise ValueError("NOVER derivation requires mean reference-likelihood scores")
    derived["score"] = NOVERReward.score_from_log_likelihoods([mean], [count])[0]
    derived["statistics"] = {
        **{
            key: stats[key]
            for key in (
                "conditioned_logp_mean", "candidate_tokens", "reference_tokens",
                "conditioned_logp_sum", "conditioning_tokens", "reference_prompt_template",
            )
            if key in stats
        },
        "nover_variant": NOVERReward.variant,
        "length_normalizer": nover_length_normalizer(count),
        "length_log_base": 10,
        "source_reward": "reference_likelihood",
    }
    return derived


def derive_nover(run_dir, output_dir):
    """Create a separate, report-only run; source scores and hashes are retained.

    Input must have a complete reference-likelihood journal for both datasets.
    The saved token counts must describe the candidate used for that inference.
    Original scoring times are inherited, not presented as fresh measurements.
    """
    source, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the source run")
    if output.exists():
        raise ValueError("Output already exists; choose a new output directory")
    hashes = _snapshot(source)
    config = json.loads((source / "config.json").read_text())
    scope = report_rewards({**config, "run_dir": str(source)})
    if "reference_likelihood" not in scope:
        raise ValueError("Derivation requires reference likelihood in the benchmark scope")
    scores, _, _ = load_results(source, scope)
    if "nover" in set(scores.reward):
        raise ValueError("The source already contains NOVER scores")
    # Validate every derived value before creating output artifacts.
    journals = {}
    derived_count = 0
    for dataset in DATASETS:
        for path in sorted((source / "scores" / "lm").glob(f"{dataset}-shard-*.jsonl")):
            records = read_jsonl(path)
            for record in records:
                additions = [
                    _derive_row(row)
                    for row in record["rows"]
                    if row["reward"] == "reference_likelihood"
                ]
                if len(additions) != 2:
                    raise ValueError(
                        "Each LM pair must contain both reference-likelihood scores"
                    )
                record["rows"].extend(additions)
                derived_count += len(additions)
            journals[path.relative_to(source)] = records
    if not derived_count:
        raise ValueError("No reference-likelihood records to derive")
    config.update(
        name=config.get("name", "Benchmark") + " + NOVER proxy (derived)",
        run_dir=str(output),
    )
    ensure_run(config)
    # Retain exact inputs, original worker metadata/timings and model revisions.
    # Do not copy old reports/tables or imply that inference ran with new code.
    for directory in ("data", "scored_text", "scores"):
        if (source / directory).exists():
            shutil.copytree(source / directory, output / directory)
    for filename in ("models.json", "data_manifest.json", "preprocessing.json"):
        if (source / filename).exists():
            shutil.copy2(source / filename, output / filename)
    for filename in ("config.json", "environment.json"):
        if (source / filename).exists():
            shutil.copy2(source / filename, output / ("source_" + filename))
    for path, records in journals.items():
        write_jsonl(output / path, records)
    write_json(
        output / "evaluation_scope.json",
        {
            "benchmark_rewards": [r for r in REWARDS if r in scope or r == "nover"],
            "lm_mode": "full",
            "reason": "Original benchmark plus NOVER derived from saved likelihoods and token counts.",
        },
    )
    if _snapshot(source) != hashes:
        raise ValueError("Source run changed during derivation")
    write_json(
        output / "derivation.json",
        {
            "created_at": utc_now(),
            "source_run": str(source),
            "source_files_sha256": hashes,
            "source_run_unchanged": True,
            "derived_candidate_records": derived_count,
            "nover_variant": NOVERReward.variant,
            "formula": "conditioned_logp_mean / max(1, 1 + log10(candidate_tokens))",
            "new_model_inference": False,
            "timings": "Inherited source inference timings; excludes derivation and reporting.",
            "original_rewards": "Unchanged, including the source VR-CLI variant.",
            "usage": "Report-only; prepare, preprocess and score require a new run_dir.",
        },
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = derive_nover(args.run_dir, args.output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(config["run_dir"]) / "plot_cache"))
    from .report import generate_report

    generate_report(config)
    print(Path(config["run_dir"]) / "report.html")


if __name__ == "__main__":
    main()
