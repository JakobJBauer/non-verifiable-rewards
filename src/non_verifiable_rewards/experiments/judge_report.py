"""Paired inference and a publication-style PDF for the frozen judge experiment."""

import importlib.metadata
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .._hashing import digest, file_sha256
from .analysis import cluster_bootstrap, preference_credit
from .io import read_jsonl, write_json

DATASETS = ("reddit_tldr", "litbench", "wmt14_de_en")
NAMES = {
    "reddit_tldr": "TL;DR",
    "litbench": "LitBench",
    "wmt14_de_en": "WMT14 de → en",
    "macro": "Macro average",
}
CONDITIONS = ("historical", "control", "thinking")
LABELS = {
    "historical": "Historical PyTorch · off",
    "control": "Matched vLLM · off",
    "thinking": "Matched vLLM · on",
}
COLORS = {"historical": "#97A4B4", "control": "#2165A6", "thinking": "#D46A36"}


def freeze_observed(root, *, reason):
    """Freeze a deliberately stopped run; never silently accept missing scores."""
    root = Path(root)
    path = root / "analysis_scope.json"
    if path.exists():
        load_completed(root)
        return json.loads(path.read_text())
    candidates = read_jsonl(root / "candidates.jsonl")
    pairs = read_jsonl(root / "pairs.jsonl")
    with (root / "thinking.jsonl").open() as stream:
        keys = [json.loads(line)["key"] for line in stream]
    observed = set(keys)
    included = [
        {"dataset": p["dataset"], "pair_id": p["pair_id"]}
        for p in pairs
        if p["preferred_key"] in observed and p["rejected_key"] in observed
    ]
    scope = {
        "kind": "user_stopped_prefix_v1",
        "reason": reason,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": file_sha256(root / "manifest.json"),
        "journal_sha256": {
            name: file_sha256(root / f"{name}.jsonl")
            for name in ("control", "thinking")
        },
        "observed_thinking_candidates": len(keys),
        "planned_unique_candidates": len(candidates),
        "included_pairs_sha256": digest(included),
        "included_pairs_by_dataset": dict(Counter(p["dataset"] for p in included)),
        "selection": "Both candidate contexts were recorded before the user stopped generation; successful closure is not required for inclusion.",
    }
    load_completed(root, scope=scope)
    write_json(path, scope)
    return scope


def load_completed(root, *, scope=None):
    """Audit exact coverage and join scores without copying rationale text to Git."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("historical_prompt_hashes_checked"):
        raise ValueError("Historical prompt hashes were not verified")
    scope_path = root / "analysis_scope.json"
    if scope is None and scope_path.exists():
        scope = json.loads(scope_path.read_text())
    if scope is not None:
        if (
            scope.get("kind") != "user_stopped_prefix_v1"
            or not isinstance(scope.get("reason"), str)
            or not scope["reason"].strip()
            or scope["manifest_sha256"] != file_sha256(root / "manifest.json")
        ):
            raise ValueError("Invalid explicitly frozen analysis scope")
        for name in ("control", "thinking"):
            if scope["journal_sha256"][name] != file_sha256(root / f"{name}.jsonl"):
                raise ValueError("Scores changed after freezing the analysis scope")
    if (
        not json.loads((root / "run_status.json").read_text())["complete"]
        and scope is None
    ):
        raise ValueError("Scoring run is incomplete")
    identity = json.loads((root / "scoring_identity.json").read_text())
    if identity["manifest_sha256"] != file_sha256(root / "manifest.json"):
        raise ValueError("Scoring manifest identity changed")
    candidates, pairs = (
        read_jsonl(root / name) for name in ("candidates.jsonl", "pairs.jsonl")
    )
    for name, rows in (("candidates", candidates), ("pairs", pairs)):
        if digest(rows) != manifest[name + "_sha256"]:
            raise ValueError(f"Frozen {name} changed")
    lookup = {r["key"]: r for r in candidates}
    if (
        len(lookup) != len(candidates)
        or len(candidates) != manifest["unique_candidates"]
    ):
        raise ValueError("Candidate count/uniqueness mismatch")
    if len(pairs) != manifest["pairs"] or len(
        {(p["dataset"], p["pair_id"]) for p in pairs}
    ) != len(pairs):
        raise ValueError("Pair count/uniqueness mismatch")
    if dict(Counter(p["dataset"] for p in pairs)) != manifest["pairs_by_dataset"]:
        raise ValueError("Dataset pair counts changed")
    journals = {}
    for condition in ("control", "thinking"):
        records = {}
        with (root / f"{condition}.jsonl").open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    raise ValueError("Unfinished journal record")
                row = json.loads(line)
                key = row["key"]
                if (
                    key in records
                    or key not in lookup
                    or row["dataset"] != lookup[key]["dataset"]
                ):
                    raise ValueError("Unknown, duplicate or mismatched candidate")
                if row["thinking"] != (condition == "thinking"):
                    raise ValueError("Wrong condition in score journal")
                if (
                    condition == "control"
                    and row["prompt_sha256"] != lookup[key]["historical_context_sha256"]
                ):
                    raise ValueError("Historical control prompt mismatch")
                if row["status"] == "ok":
                    if (
                        not isinstance(row["score"], (int, float))
                        or not math.isfinite(row["score"])
                        or not 0 <= row["score"] <= 1
                    ):
                        raise ValueError("Invalid successful score")
                    probs = np.asarray(row["rating_probabilities"], dtype=float)
                    if (
                        probs.shape != (5,)
                        or not np.isfinite(probs).all()
                        or (probs < 0).any()
                        or not np.isclose(probs.sum(), 1)
                    ):
                        raise ValueError("Invalid rating distribution")
                    if not np.isclose(
                        probs @ np.arange(5) / 4, row["score"], atol=1e-12, rtol=0
                    ):
                        raise ValueError("Score differs from recorded probabilities")
                    mass = row["rating_token_mass"]
                    if not math.isfinite(mass) or not 0 <= mass <= 1 + 1e-5:
                        raise ValueError("Invalid full-vocabulary rating mass")
                elif (
                    row["status"]
                    not in ("thinking_limit", "context_limit", "early_eos")
                    or row["score"] is not None
                ):
                    raise ValueError("Invalid failed judgment")
                if (
                    "thinking_token_ids" in row
                    and len(row["thinking_token_ids"]) != row["thinking_tokens"]
                ):
                    raise ValueError("Reasoning token count mismatch")
                records[key] = {
                    k: v for k, v in row.items() if k != "thinking_token_ids"
                }
        expected_keys = list(lookup)
        if condition == "thinking" and scope is not None:
            count = scope["observed_thinking_candidates"]
            if not 0 < count <= len(expected_keys):
                raise ValueError("Invalid stopped-run candidate count")
            expected_keys = expected_keys[:count]
            if list(records) != expected_keys:
                raise ValueError(
                    "Observed thinking contexts are not the frozen input prefix"
                )
        if set(records) != set(expected_keys):
            raise ValueError("Scoring run has incomplete candidate coverage")
        journals[condition] = records
    frame = pd.DataFrame(pairs)
    used = set()
    for side in ("preferred", "rejected"):
        for pair in pairs:
            key = pair[side + "_key"]
            if key not in lookup or lookup[key]["dataset"] != pair["dataset"]:
                raise ValueError("Pair candidate identity mismatch")
            used.add(key)
        for condition in ("control", "thinking"):
            frame[f"{condition}_{side}"] = [
                journals[condition].get(k, {}).get("score")
                for k in frame[side + "_key"]
            ]
            frame[f"{condition}_{side}_status"] = [
                journals[condition].get(k, {}).get("status", "not_attempted")
                for k in frame[side + "_key"]
            ]
        frame[f"historical_{side}"] = frame.pop(side + "_historical_score")
    if used != set(lookup):
        raise ValueError("Unreferenced candidate in frozen inputs")
    frame["in_analysis_scope"] = frame.thinking_preferred_status.ne(
        "not_attempted"
    ) & frame.thinking_rejected_status.ne("not_attempted")
    if scope is not None:
        included = frame.loc[frame.in_analysis_scope, ["dataset", "pair_id"]].to_dict(
            "records"
        )
        if (
            digest(included) != scope["included_pairs_sha256"]
            or dict(Counter(p["dataset"] for p in included))
            != scope["included_pairs_by_dataset"]
        ):
            raise ValueError("Frozen pair selection changed")
    for condition in CONDITIONS:
        cols = [condition + "_" + side for side in ("preferred", "rejected")]
        frame[cols] = frame[cols].astype(float)
        frame[condition + "_margin"] = frame[cols[0]] - frame[cols[1]]
        valid = frame[cols].notna().all(axis=1)
        frame[condition + "_credit"] = np.nan
        frame.loc[valid, condition + "_credit"] = preference_credit(
            frame.loc[valid, condition + "_margin"], manifest["config"]["tie_tolerance"]
        )
    provenance = {
        "manifest": manifest,
        "scoring_identity": identity,
        "analysis_scope": scope or {"kind": "complete_run"},
        "raw_files_sha256": {
            name: file_sha256(root / name)
            for name in (
                "manifest.json",
                "candidates.jsonl",
                "pairs.jsonl",
                "control.jsonl",
                "thinking.jsonl",
                "run_status.json",
                "scoring_identity.json",
            )
        },
    }
    if scope_path.exists():
        provenance["raw_files_sha256"]["analysis_scope.json"] = file_sha256(scope_path)
    return frame, journals, provenance


def estimate(value, draws):
    return {
        "estimate": float(value),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def paired_statistics(frame, *, samples, seed):
    """Shared prompt-cluster draws retain pairing and repeated-prompt dependence."""
    columns = [c + "_credit" for c in CONDITIONS]
    paired = frame.dropna(subset=columns)
    if paired.empty:
        raise ValueError("No complete matched pairs")
    values = paired[columns].to_numpy()
    draws = cluster_bootstrap(values, paired.prompt_id, samples=samples, seed=seed)
    delta = values[:, 2] - values[:, 1]
    result = {
        "total_pairs": len(frame),
        "matched_pairs": len(paired),
        "matched_prompts": paired.prompt_id.nunique(),
        "conditions": {
            c: estimate(values[:, i].mean(), draws[:, i])
            for i, c in enumerate(CONDITIONS)
        },
        "thinking_minus_control": estimate(delta.mean(), draws[:, 2] - draws[:, 1]),
        "improved": int((delta > 0).sum()),
        "regressed": int((delta < 0).sum()),
        "unchanged": int((delta == 0).sum()),
        "ties": {c: int((paired[c + "_credit"] == 0.5).sum()) for c in CONDITIONS},
    }
    # Unknown pairs receive the mathematically possible credit range [0, 1].
    # Bounds are descriptive identification bounds, not confidence intervals.
    result["scope_delta_bounds"] = [
        float(
            (frame.thinking_credit.fillna(0) - frame.control_credit.fillna(1)).mean()
        ),
        float(
            (frame.thinking_credit.fillna(1) - frame.control_credit.fillna(0)).mean()
        ),
    ]
    result["scope_sample"] = {}
    for i, condition in enumerate(CONDITIONS):
        credit = frame[condition + "_credit"]
        valid = frame.loc[credit.notna()]
        result["scope_sample"][condition] = {
            "valid_pairs": len(valid),
            "failed_pairs": int(credit.isna().sum()),
            "valid_accuracy": float(credit.mean()) if len(valid) else None,
            "accuracy_bounds": [
                float(credit.fillna(0).mean()),
                float(credit.fillna(1).mean()),
            ],
        }
    backend = frame.dropna(subset=["historical_credit", "control_credit"])
    backend_delta = backend.control_credit - backend.historical_credit
    backend_draws = cluster_bootstrap(
        backend_delta.to_numpy()[:, None],
        backend.prompt_id,
        samples=samples,
        seed=seed + 100,
    )
    result["backend"] = {
        **estimate(backend_delta.mean(), backend_draws[:, 0]),
        "pairs": len(backend),
        "decision_changes": int((backend_delta != 0).sum()),
        "candidate_score_mae": float(
            np.abs(
                backend[["control_preferred", "control_rejected"]].to_numpy()
                - backend[["historical_preferred", "historical_rejected"]].to_numpy()
            ).mean()
        ),
    }
    transition = pd.crosstab(paired.control_credit, paired.thinking_credit).reindex(
        index=[0.0, 0.5, 1.0], columns=[0.0, 0.5, 1.0], fill_value=0
    )
    result["transitions"] = transition.to_numpy().astype(int).tolist()
    return result, draws


def analyze(frame, journals, config):
    source_frame = frame
    frame = frame.loc[frame.in_analysis_scope]
    result = {
        "datasets": {},
        "bootstrap_samples": config["bootstrap_samples"],
        "seed": config["seed"],
        "tie_tolerance": config["tie_tolerance"],
        "planned_pairs": len(source_frame),
        "attempted_pairs": len(frame),
        "unattempted_pairs": len(source_frame) - len(frame),
        "planned_control_contexts": len(journals["control"]),
        "attempted_thinking_contexts": len(journals["thinking"]),
    }
    bootstrap = []
    for i, dataset in enumerate(DATASETS):
        subset = frame[frame.dataset == dataset]
        statistics, draws = paired_statistics(
            subset, samples=config["bootstrap_samples"], seed=config["seed"] + i * 1000
        )
        result["datasets"][dataset] = statistics
        statistics["planned_pairs"] = int((source_frame.dataset == dataset).sum())
        statistics["unattempted_pairs"] = statistics["planned_pairs"] - len(subset)
        bootstrap.append(draws)
    macro_draws = np.stack(bootstrap).mean(axis=0)
    stats = list(result["datasets"].values())
    result["macro"] = {
        "conditions": {
            c: estimate(
                np.mean([s["conditions"][c]["estimate"] for s in stats]),
                macro_draws[:, i],
            )
            for i, c in enumerate(CONDITIONS)
        },
        "thinking_minus_control": estimate(
            np.mean([s["thinking_minus_control"]["estimate"] for s in stats]),
            macro_draws[:, 2] - macro_draws[:, 1],
        ),
        "scope_delta_bounds": np.mean(
            [s["scope_delta_bounds"] for s in stats], axis=0
        ).tolist(),
        "matched_pairs": sum(s["matched_pairs"] for s in stats),
        "total_pairs": len(frame),
    }
    diagnostics = {}
    for condition, journal in journals.items():
        diagnostics[condition] = {}
        for dataset in (*DATASETS, "all"):
            rows = [
                r
                for r in journal.values()
                if r["key"] in journals["thinking"]
                and (dataset == "all" or r["dataset"] == dataset)
            ]
            valid = [r for r in rows if r["status"] == "ok"]
            tokens = [r["thinking_tokens"] for r in rows if "thinking_tokens" in r]
            masses = [r["rating_token_mass"] for r in valid]
            seconds = sum(
                r.get("generation_batch_seconds", 0) / r.get("generation_batch_size", 1)
                + r.get("scoring_batch_seconds", 0) / r.get("scoring_batch_size", 1)
                for r in rows
            )
            diagnostics[condition][dataset] = {
                "unique_candidates": len(rows),
                "statuses": dict(Counter(r["status"] for r in rows)),
                "inference_seconds": seconds,
                "thinking_tokens_total": sum(tokens),
                "thinking_tokens_median": float(np.median(tokens)) if tokens else None,
                "thinking_tokens_p90": float(np.quantile(tokens, 0.9))
                if tokens
                else None,
                "rating_mass_median": float(np.median(masses)) if masses else None,
                "rating_mass_p10": float(np.quantile(masses, 0.1)) if masses else None,
            }
    result["diagnostics"] = diagnostics
    return result


def fmt_interval(stat, *, signed=False):
    point, (low, high) = stat["estimate"], stat["ci95"]
    sign = "+" if signed else ""
    return f"{point * 100:{sign}.2f} [{low * 100:{sign}.2f}, {high * 100:{sign}.2f}]"


def interpretation(summary):
    stat = summary["macro"]["thinking_minus_control"]
    low, high = stat["ci95"]
    if low > 0:
        return "Thinking improves agreement on matched pairs"
    if high < 0:
        return "Thinking reduces agreement on matched pairs"
    return "No clear average agreement gain from thinking"


def report(root, output):
    frame, journals, provenance = load_completed(root)
    provenance["analysis_identity"] = {
        "judge_report_sha256": file_sha256(Path(__file__)),
        "analysis_sha256": file_sha256(Path(__file__).with_name("analysis.py")),
        "packages": {
            p: importlib.metadata.version(p) for p in ("numpy", "pandas", "matplotlib")
        },
    }
    for name in ("hardware.csv", "scoring_source.tar.gz"):
        path = Path(root) / name
        if path.exists():
            provenance["raw_files_sha256"][name] = file_sha256(path)
    if (Path(root) / "hardware.csv").exists():
        provenance["hardware"] = (Path(root) / "hardware.csv").read_text().strip()
    config = provenance["manifest"]["config"]
    summary = analyze(frame, journals, config)
    summary["stopped_early"] = provenance["analysis_scope"]["kind"] != "complete_run"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", provenance)
    frame.to_csv(output / "pair_scores.csv", index=False, float_format="%.12g")
    compact = pd.DataFrame(
        [
            {
                **r,
                "condition": c,
                "in_matched_context_scope": r["key"] in journals["thinking"],
            }
            for c, journal in journals.items()
            for r in journal.values()
        ]
    )
    # CSV keeps numerical audit data; raw text and full token traces stay in the run.
    probabilities = compact.pop("rating_probabilities")
    for rating in range(5):
        compact[f"rating_probability_{rating}"] = [
            p[rating] if isinstance(p, list) else np.nan for p in probabilities
        ]
    compact.to_csv(
        output / "candidate_diagnostics.csv", index=False, float_format="%.12g"
    )
    render_pdf(
        summary,
        frame,
        journals,
        provenance,
        output,
        previews=Path(root) / "report_previews",
    )
    write_readme(summary, output)
    return summary


def write_readme(summary, output):
    stopping_note = (
        f"Generation was stopped at the user's request after {summary['attempted_thinking_contexts']:,} distinct thinking contexts; no further completions were generated for this report. The available contexts are a prefix of the fixed interleaved dataset queues. A pair enters the analysis if both its contexts were recorded, irrespective of whether reasoning closed. This is an available-data subset, not a new random sample."
        if summary["stopped_early"]
        else "All planned contexts were attempted; the paired estimates use the common valid judgments."
    )
    rows = [
        "| Dataset | Matched pairs | Thinking off | Thinking on | On − off (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, stat in [*summary["datasets"].items(), ("macro", summary["macro"])]:
        rows.append(
            f"| {NAMES[dataset]} | {stat['matched_pairs']:,} | {fmt_interval(stat['conditions']['control'])} | {fmt_interval(stat['conditions']['thinking'])} | {fmt_interval(stat['thinking_minus_control'], signed=True)} |"
        )
    (output / "README.md").write_text(
        "# Qwen3-4B: judge thinking comparison\n\n"
        "[PDF report](judge_thinking_comparison.pdf) · [Summary JSON](summary.json) · [Provenance](provenance.json)\n\n"
        f"**{interpretation(summary)}.** Intervals below are 95% prompt-cluster bootstrap intervals (10,000 draws). Accuracy is percent; delta is percentage points. Primary estimates use the same valid pairs in both fresh vLLM conditions.\n\n"
        f"Analysis covers **{summary['attempted_pairs']:,} attempted pairs** from the frozen {summary['planned_pairs']:,}-pair source; {summary['macro']['matched_pairs']:,} have valid scores in both conditions. {summary['unattempted_pairs']:,} source pairs lack one or both thinking judgments and are excluded from inference, not counted as failures. {stopping_note}\n\n"
        + "\n".join(rows)
        + "\n\nThe source is the frozen 20 November benchmark: 2,000 preference pairs each for TL;DR, LitBench and WMT14 de→en. All 12,000 historical non-thinking prompt hashes match. Identical contexts are evaluated once (10,344 planned unique contexts) and expanded back to their original pairs. The fresh non-thinking control completed all 10,344 contexts; the main comparisons use only the matching available sample. Historical PyTorch scores are a separate execution-stack diagnostic; the thinking effect uses fresh vLLM controls.\n\n"
        "Both fresh conditions use Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, bfloat16, batch size 128, a 32,768-token context and the same task rubric, candidates and references. Thinking adds an explicit reasoning instruction and prefilled `<think>`; greedy generation stops at `</think>` with a 4,096-token budget. The exact generated IDs plus two newlines condition the next-token probabilities of ratings 0–4. The score is their renormalized expected rating divided by four. This tests the implemented thinking protocol, including its prompt changes.\n\n"
        "Unclosed reasoning has no score. The PDF and JSON report failures, paired coverage and conservative worst/best bounds within the attempted-pair sample. Those bounds do not cover unattempted source pairs and are not confidence intervals. Ties within 1e-7 receive half credit. Bootstrap draws resample prompt clusters and preserve condition pairing. The macro average weights the three datasets equally and combines independent bootstrap streams. Intervals are exploratory and unadjusted for multiple comparisons; they do not account for the stopped-prefix selection.\n\n"
        "Runtime compares the same observed candidate contexts in both conditions. It sums generation and label-scoring calls for saved batches, including the pilot, excluding engine startup, excess control contexts, the interrupted unsaved batch and report generation. Mixed-task batch time is allocated equally to its members: only the total is a direct measured duration. The new runs share one A100-SXM4-40GB; historical scores used an H200 and a different software stack. Runtime ratios do not compare those GPUs.\n\n"
        "This is offline judge agreement, not a GRPO training experiment. LitBench preferences are upvote-derived proxies. One deterministic rationale is used per candidate; no claim is made about repeated-run uncertainty, probability calibration, rationale correctness or downstream training gains. The main policy's thinking setting is independent.\n\n"
        "## Reproduce\n\n"
        "Regenerate this report from the archived run, with the benchmark dependencies installed:\n\n"
        "```bash\npython -m non_verifiable_rewards.experiments.judge_comparison report --config configs/eidf_judge_thinking_comparison.toml --source outputs/judge_thinking_comparison_20261126 --output reports/judge_thinking_comparison\n```\n\n"
        "The original inference sequence is recorded below for audit; it is not needed to regenerate this report. For any new inference, copy the config, change `run_dir`, and substitute the new config path below. Preserve the archived run unchanged. The stopped-sample scope and journal hashes are retained in the report provenance.\n\n"
        "```bash\npython -m non_verifiable_rewards.experiments.judge_comparison prepare --config configs/eidf_judge_thinking_comparison.toml\n# The measured run used a pilot, then resumed in a fresh process.\npython -m non_verifiable_rewards.experiments.judge_comparison score --config configs/eidf_judge_thinking_comparison.toml --limit 128\npython -m non_verifiable_rewards.experiments.judge_comparison score --config configs/eidf_judge_thinking_comparison.toml\npython -m non_verifiable_rewards.experiments.judge_comparison report --config configs/eidf_judge_thinking_comparison.toml\n```\n\n"
        "GPU scoring runs on `eidf-dev` in detached tmux. Raw frozen inputs, journals and reasoning IDs are retained at `/workspace/writeable/non-verifiable-rewards/outputs/judge_thinking_comparison_20261126`; local mirrors live under `outputs/judge_thinking_comparison_20261126`. Pass `--source` and `--output` to regenerate the report from a mirror. Preparation accepts the same overrides and `--tokenizer`. Scoring refuses changed configuration or scoring-source identity and resumes flushed candidates. Failed judgments are retained rather than silently retried or replaced. An explicitly stopped run is frozen with `python -m non_verifiable_rewards.experiments.judge_report freeze-observed --source RUN --reason REASON`. Its `analysis_scope.json` fixes journal hashes, prefix length and included pair identities; changed journals are rejected.\n\n"
        "`pair_scores.csv` retains every original pair with an `in_analysis_scope` flag, condition scores, statuses, margins and preference credit. Unattempted thinking scores are blank and marked `not_attempted`. `candidate_diagnostics.csv` contains one row per recorded unique candidate per fresh condition, all five rating probabilities and an `in_matched_context_scope` flag, without source text or rationale IDs. `provenance.json` records the stopped scope, frozen-input hashes, raw-result hashes, model revision and scoring-source/package identity.\n"
    )


def render_pdf(summary, frame, journals, provenance, output, *, previews=None):
    """Vector charts, embedded fonts and directly rendered page previews."""
    import textwrap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#29394D",
            "text.color": "#162A43",
            "axes.edgecolor": "#CCD4DE",
            "xtick.color": "#526176",
            "ytick.color": "#526176",
            "pdf.fonttype": 42,
            "savefig.facecolor": "#FFFFFF",
        }
    )
    output = Path(output)
    frame = frame.loc[frame.in_analysis_scope]
    observed = set(journals["thinking"])
    journals = {
        c: {key: row for key, row in rows.items() if key in observed}
        for c, rows in journals.items()
    }
    ink, muted, orange, blue = (
        "#162A43",
        "#526176",
        COLORS["thinking"],
        COLORS["control"],
    )
    manifest = provenance["manifest"]
    config = manifest["config"]
    stats = summary["datasets"]
    diagnostics = summary["diagnostics"]
    matched = frame.dropna(subset=[c + "_credit" for c in CONDITIONS])
    scope_note = (
        f"Generation stopped at the user's request; {summary['unattempted_pairs']:,} source pairs lack complete attempts and are outside this analysis."
        if summary["stopped_early"]
        else "Every source pair has both candidate contexts attempted."
    )
    selection_note = (
        f"The user stopped generation after {summary['attempted_thinking_contexts']:,} contexts in the fixed interleaved input order. This available-data prefix is not a new random sample."
        if summary["stopped_early"]
        else "All planned candidate contexts were attempted."
    )
    page_number = 0

    def paragraph(fig, text, y, *, size=10, width=92, color=muted):
        wrapped = "\n".join(
            textwrap.fill(line, width=width) for line in text.split("\n")
        )
        fig.text(
            0.085, y, wrapped, va="top", fontsize=size, color=color, linespacing=1.55
        )

    def page(title, subtitle):
        nonlocal page_number
        page_number += 1
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        fig.add_artist(
            Rectangle((0, 0.956), 1, 0.044, transform=fig.transFigure, color=ink)
        )
        fig.text(
            0.085,
            0.974,
            "NON-VERIFIABLE REWARDS",
            fontsize=8,
            color="white",
            weight="bold",
            va="center",
        )
        fig.text(
            0.915,
            0.974,
            "RESEARCH NOTE  /  NOVEMBER 2026",
            fontsize=7,
            color="#D4DEEA",
            ha="right",
            va="center",
        )
        fig.text(0.085, 0.912, title, fontsize=22, weight="bold", va="top")
        paragraph(fig, subtitle, 0.868, size=9, width=103)
        fig.add_artist(
            plt.Line2D(
                [0.085, 0.915],
                [0.065, 0.065],
                transform=fig.transFigure,
                color="#DCE2E9",
                lw=0.7,
            )
        )
        fig.text(
            0.085,
            0.043,
            "Qwen3-4B  /  matched vLLM judge comparison",
            fontsize=7,
            color=muted,
        )
        fig.text(
            0.915,
            0.043,
            f"{page_number:02d}  /  06",
            fontsize=7,
            color=muted,
            ha="right",
        )
        return fig

    def table(fig, bounds, headers, rows, widths=None, size=9):
        ax = fig.add_axes(bounds)
        ax.axis("off")
        tab = ax.table(
            cellText=rows,
            colLabels=headers,
            colWidths=widths,
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0, 1, 1],
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(size)
        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor("white")
            cell.PAD = 0.08
            cell.set_facecolor(
                "#E9EFF5" if r == 0 else ("#F7F9FB" if r % 2 else "white")
            )
            if r == 0:
                cell.get_text().set_weight("bold")
        return tab

    def save(pdf, fig):
        pdf.savefig(fig)
        if previews is not None:
            Path(previews).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(previews) / f"page-{page_number:02d}.png", dpi=130)
        plt.close(fig)

    def forest(ax, entries, field, color):
        for i, (label, stat) in enumerate(entries):
            point = stat[field]["estimate"] * 100
            low, high = np.asarray(stat[field]["ci95"]) * 100
            ax.plot([low, high], [i, i], color=color, lw=2.3, solid_capstyle="round")
            ax.scatter(
                point,
                i,
                s=48 if label == "Macro average" else 33,
                color=color,
                zorder=3,
            )
            ax.text(
                high,
                i,
                f"  {point:+.2f}",
                va="center",
                fontsize=9,
                color=ink,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
            )
        ax.axvline(0, color="#7B899B", lw=0.8, ls="--")
        ax.set_yticks(range(len(entries)), [e[0] for e in entries])
        ax.invert_yaxis()
        ax.margins(x=0.25, y=0.2)
        ax.set_xlabel("Change in agreement (percentage points)", labelpad=10)
        ax.grid(axis="x", alpha=0.13)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=12)

    path = output / "judge_thinking_comparison.pdf"
    with PdfPages(
        path,
        metadata={
            "Title": "Qwen3-4B: LLM judge thinking enabled vs disabled",
            "Author": "Non-verifiable rewards project",
            "Subject": "Available-pair subset of a frozen preference benchmark with prompt-cluster uncertainty",
        },
    ) as pdf:
        fig = page(
            "Does a thinking judge judge better?",
            f"{summary['attempted_pairs']:,} attempted pairs from {summary['planned_pairs']:,} frozen source pairs · fresh vLLM control",
        )
        paragraph(fig, interpretation(summary), 0.803, size=16, width=54, color=ink)
        macro = summary["macro"]
        delta = macro["thinking_minus_control"]
        ratio = (
            diagnostics["thinking"]["all"]["inference_seconds"]
            / diagnostics["control"]["all"]["inference_seconds"]
        )
        paragraph(
            fig,
            f"The matched macro change is {delta['estimate'] * 100:+.2f} percentage points (95% interval {delta['ci95'][0] * 100:+.2f} to {delta['ci95'][1] * 100:+.2f}). Thinking costs {ratio:.1f}× as much recorded inference time on matching contexts. {macro['matched_pairs']:,} of {macro['total_pairs']:,} attempted pairs have valid judgments in both conditions.",
            0.734,
            size=10,
            width=94,
        )
        ax = fig.add_axes([0.255, 0.430, 0.56, 0.225])
        forest(
            ax,
            [(NAMES[d], stats[d]) for d in DATASETS] + [(NAMES["macro"], macro)],
            "thinking_minus_control",
            orange,
        )
        rows = []
        for dataset, stat in [*stats.items(), ("macro", macro)]:
            rows.append(
                [
                    NAMES[dataset],
                    f"{stat['matched_pairs']:,}",
                    f"{stat['conditions']['control']['estimate'] * 100:.2f}%",
                    f"{stat['conditions']['thinking']['estimate'] * 100:.2f}%",
                    f"{stat['thinking_minus_control']['estimate'] * 100:+.2f}",
                ]
            )
        table(
            fig,
            [0.085, 0.210, 0.83, 0.145],
            ["Dataset", "Pairs", "Off", "On", "Δ (pp)"],
            rows,
            widths=[0.30, 0.16, 0.18, 0.18, 0.18],
        )
        paragraph(
            fig,
            f"Accuracy is agreement with the dataset preference, with half credit for ties. Bars are 95% paired prompt-cluster bootstrap intervals (10,000 draws); datasets receive equal macro weight. Estimates use common valid pairs. {scope_note}",
            0.176,
            size=9,
            width=103,
        )
        save(pdf, fig)

        fig = page(
            "Separate thinking from the backend",
            "The historical PyTorch run is a diagnostic; the primary thinking effect compares two fresh vLLM runs.",
        )
        rows = []
        for dataset in DATASETS:
            stat = stats[dataset]
            rows.append(
                [
                    NAMES[dataset],
                    f"{stat['scope_sample']['historical']['valid_accuracy'] * 100:.2f}%",
                    f"{stat['scope_sample']['control']['valid_accuracy'] * 100:.2f}%",
                    fmt_interval(stat["backend"], signed=True),
                    str(stat["backend"]["decision_changes"]),
                ]
            )
        table(
            fig,
            [0.085, 0.680, 0.83, 0.115],
            ["Dataset", "Torch off", "vLLM off", "Δ pp [95% CI]", "Changes"],
            rows,
            widths=[0.22, 0.15, 0.15, 0.35, 0.13],
            size=8,
        )
        paragraph(
            fig,
            "Both non-thinking conditions use the original frozen task prompts, verified by all 12,000 serialized-context hashes. Differences can reflect execution backend, package versions, hardware or batching; they are not evidence about thinking.",
            0.646,
            size=9,
            width=103,
        )
        for i, dataset in enumerate(DATASETS):
            ax = fig.add_axes([0.10 + i * 0.285, 0.332, 0.23, 0.22])
            sub = frame[frame.dataset == dataset]
            x = sub[["historical_preferred", "historical_rejected"]].to_numpy().ravel()
            y = sub[["control_preferred", "control_rejected"]].to_numpy().ravel()
            valid = np.isfinite(x) & np.isfinite(y)
            ax.hexbin(
                x[valid],
                y[valid],
                gridsize=34,
                mincnt=1,
                cmap="Blues",
                bins="log",
                extent=(0, 1, 0, 1),
            )
            ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#8797A8")
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Historical score",
                title=NAMES[dataset],
                aspect="equal",
            )
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            if i == 0:
                ax.set_ylabel("Fresh control score")
        paragraph(
            fig,
            "Score agreement within the attempted-pair sample",
            0.278,
            size=11,
            color=ink,
        )
        paragraph(
            fig,
            "Each panel compares the same candidate under historical PyTorch and fresh vLLM non-thinking scoring. Darker bins contain more occurrences. A small numerical score shift can change a near-tie decision; the table counts every changed preference credit, including transitions to or from ties.",
            0.246,
            size=9,
            width=103,
        )
        versions = provenance["scoring_identity"]["packages"]
        paragraph(
            fig,
            f"New run: NVIDIA A100-SXM4-40GB; torch {versions['torch']}; vLLM {versions['vllm']}; transformers {versions['transformers']}. Historical run: H200; torch 2.9.1+cu128; transformers 5.22.2; batch size 1. Fresh conditions: batch size 128.",
            0.160,
            size=9,
            width=103,
        )
        save(pdf, fig)

        fig = page(
            "Which preferences change?",
            "Matched pairs only. A correct decision ranks the dataset-preferred candidate above the rejected candidate.",
        )
        for i, dataset in enumerate(DATASETS):
            ax = fig.add_axes([0.115 + i * 0.285, 0.600, 0.21, 0.19])
            matrix = np.asarray(stats[dataset]["transitions"])
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
            for r in range(3):
                for c in range(3):
                    ax.text(
                        c,
                        r,
                        f"{matrix[r, c]:,}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if matrix[r, c] > matrix.max() * 0.55 else ink,
                    )
            ax.set_xticks([0, 1, 2], ["Wrong", "Tie", "Right"], fontsize=8)
            ax.set_yticks(
                [0, 1, 2],
                ["Wrong", "Tie", "Right"] if i == 0 else ["", "", ""],
                fontsize=8,
            )
            ax.set_title(NAMES[dataset], pad=10)
            ax.set_xlabel("Thinking on", fontsize=8)
            if i == 0:
                ax.set_ylabel("Thinking off", fontsize=8)
        rows = [
            [
                NAMES[d],
                f"{stats[d]['improved']:,}",
                f"{stats[d]['regressed']:,}",
                f"{stats[d]['unchanged']:,}",
                f"{stats[d]['ties']['control']:,} → {stats[d]['ties']['thinking']:,}",
            ]
            for d in DATASETS
        ]
        table(
            fig,
            [0.085, 0.424, 0.83, 0.115],
            ["Dataset", "Improved", "Regressed", "Unchanged", "Ties: off → on"],
            rows,
            widths=[0.26, 0.17, 0.17, 0.18, 0.22],
            size=8.5,
        )
        for i, dataset in enumerate(DATASETS):
            ax = fig.add_axes([0.11 + i * 0.285, 0.188, 0.23, 0.165])
            sub = matched[matched.dataset == dataset]
            for condition in ("control", "thinking"):
                values = np.sort(sub[condition + "_margin"].to_numpy())
                ax.plot(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    color=COLORS[condition],
                    label="On" if condition == "thinking" else "Off",
                    lw=1.6,
                )
            ax.axvline(0, color="#9AA6B5", lw=0.7, ls="--")
            ax.set(xlim=(-1, 1), ylim=(0, 1), xlabel="Preferred − rejected score")
            ax.set_xticks([-1, 0, 1])
            if i == 0:
                ax.set_ylabel("Cumulative fraction")
                ax.legend(frameon=False, fontsize=8)
        paragraph(
            fig,
            "Lower panels show the distribution of pairwise score margins. Positive margins agree with the preference. A higher numerical margin is not a calibrated measure of preference strength; the upper panels measure changes in the actual decision.",
            0.125,
            size=8.5,
            width=110,
        )
        save(pdf, fig)

        fig = page(
            "The cost of thinking",
            f"Recorded inference time on the same {summary['attempted_thinking_contexts']:,} contexts in both conditions; explicit reasoning failures.",
        )
        control_time = diagnostics["control"]["all"]["inference_seconds"]
        thinking_time = diagnostics["thinking"]["all"]["inference_seconds"]
        for x, value, label in (
            (0.085, f"{control_time / 60:.1f} min", "THINKING OFF"),
            (0.37, f"{thinking_time / 60:.1f} min", "THINKING ON"),
            (0.665, f"{ratio:.1f}×", "INFERENCE TIME RATIO"),
        ):
            fig.text(
                x,
                0.785,
                value,
                fontsize=24,
                color=orange if x > 0.3 else blue,
                weight="bold",
            )
            fig.text(x, 0.755, label, fontsize=8, color=muted)
        rows = []
        for dataset in DATASETS:
            diag = diagnostics["thinking"][dataset]
            failed = diag["unique_candidates"] - diag["statuses"].get("ok", 0)
            rows.append(
                [
                    NAMES[dataset],
                    f"{diag['unique_candidates']:,}",
                    f"{diag['thinking_tokens_median']:,.0f}",
                    f"{diag['thinking_tokens_p90']:,.0f}",
                    str(failed),
                    f"{stats[dataset]['matched_pairs']:,} / {stats[dataset]['total_pairs']:,}",
                ]
            )
        table(
            fig,
            [0.085, 0.570, 0.83, 0.125],
            ["Dataset", "Contexts", "Median", "P90", "Failed", "Valid pairs"],
            rows,
            widths=[0.24, 0.16, 0.13, 0.13, 0.12, 0.22],
            size=8.5,
        )
        ax = fig.add_axes([0.14, 0.302, 0.73, 0.205])
        task_colors = [blue, orange, "#4C8873"]
        for dataset, color in zip(DATASETS, task_colors, strict=True):
            tokens = np.sort(
                [
                    r["thinking_tokens"]
                    for r in journals["thinking"].values()
                    if r["dataset"] == dataset and "thinking_tokens" in r
                ]
            )
            ax.plot(
                tokens,
                np.arange(1, len(tokens) + 1) / len(tokens) * 100,
                label=NAMES[dataset],
                color=color,
                lw=1.7,
            )
        ax.axvline(config["max_thinking_tokens"], color="#A9B3BF", ls="--", lw=0.8)
        ax.set(
            xlabel="Generated reasoning tokens (closing token included)",
            ylabel="Cumulative share (%)",
            ylim=(0, 101),
            xlim=(0, config["max_thinking_tokens"] * 1.025),
        )
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        statuses = diagnostics["thinking"]["all"]["statuses"]
        bounds = macro["scope_delta_bounds"]
        paragraph(
            fig,
            f"Failure counts: {statuses.get('thinking_limit', 0):,} token-budget limits; {statuses.get('early_eos', 0):,} early EOS; {statuses.get('context_limit', 0):,} insufficient-context cases. Including all attempted pairs, the macro effect is bounded by {bounds[0] * 100:+.2f} to {bounds[1] * 100:+.2f} pp when failed judgments receive conservative worst/best credit. These are bounds, not confidence intervals, and do not cover the unattempted remainder.",
            0.246,
            size=9,
            width=103,
        )
        paragraph(
            fig,
            "Time covers saved batches on identical contexts, including the pilot. It excludes engine startup, additional control contexts and the interrupted unsaved batch. Batch time is counted once. Failed traces remain in the token statistics. Both conditions ran on the same A100; no H200 runtime comparison is made.",
            0.143,
            size=8.5,
            width=110,
        )
        save(pdf, fig)

        fig = page(
            "Inspect the scoring distribution",
            "Rating concentration and score shifts; a confident quality rating is not a calibrated preference probability.",
        )
        for i, dataset in enumerate(DATASETS):
            ax = fig.add_axes([0.11 + i * 0.285, 0.574, 0.23, 0.205])
            for condition in ("control", "thinking"):
                concentration = np.sort(
                    [
                        max(r["rating_probabilities"])
                        for r in journals[condition].values()
                        if r["dataset"] == dataset and r["status"] == "ok"
                    ]
                )
                ax.plot(
                    concentration,
                    np.arange(1, len(concentration) + 1) / len(concentration),
                    color=COLORS[condition],
                    label="On" if condition == "thinking" else "Off",
                    lw=1.7,
                )
            ax.set(
                xlim=(0, 1.01),
                ylim=(0, 1),
                xlabel="Largest rating probability",
                title=NAMES[dataset],
            )
            ax.set_xticks([0, 0.5, 1])
            if i == 0:
                ax.set_ylabel("Cumulative fraction")
                ax.legend(frameon=False, fontsize=8)
        paragraph(
            fig,
            "Higher maximum probability means a more concentrated rating distribution, not greater correctness. The total probability mass on digits 0–4 is effectively one in both modes (see the numerical diagnostics). No mass threshold or fallback was applied. Thinking produces more concentrated ratings and more pairwise ties.",
            0.512,
            size=9,
            width=103,
        )
        for i, dataset in enumerate(DATASETS):
            ax = fig.add_axes([0.11 + i * 0.285, 0.238, 0.23, 0.195])
            for condition in ("control", "thinking"):
                scores = [
                    r["score"]
                    for r in journals[condition].values()
                    if r["dataset"] == dataset and r["status"] == "ok"
                ]
                ax.hist(
                    scores,
                    bins=np.linspace(0, 1, 21),
                    weights=np.ones(len(scores)) / len(scores) * 100,
                    histtype="step",
                    color=COLORS[condition],
                    lw=1.5,
                )
            ax.set(xlim=(0, 1), xlabel="Expected rating / 4")
            ax.set_xticks([0, 0.5, 1])
            if i == 0:
                ax.set_ylabel("Valid contexts (%)")
        paragraph(
            fig,
            "Lower panels show score distributions over valid unique contexts, separately for each condition. Repeated occurrences are not counted twice here. These diagnostics use each condition’s successful judgments; the preference-effect estimates use matched pairs. Reasoning can shift the rating scale without improving its ordering of candidates.",
            0.173,
            size=9,
            width=103,
        )
        save(pdf, fig)

        fig = page(
            "Protocol, provenance and limits",
            "A controlled offline comparison of the implemented thinking protocol, with auditable raw results.",
        )
        sections = [
            (
                "01  Frozen comparison",
                f"Source: 2,000 pairs per dataset from SFF valid2, the prompt-disjoint LitBench holdout and human-ranked WMT14 translations. Texts are reused exactly. {selection_note} Analysis uses {summary['attempted_pairs']:,} pairs with both contexts attempted; all 6,000 remain in the CSV.",
            ),
            (
                "02  What changes",
                "The non-thinking prompt is verified against the historical serialized context. Thinking enables the chat-template flag, adds the reasoning instruction and prefills <think>. One greedy rationale is generated, stopping at </think> within 4,096 tokens. Its exact IDs plus two newlines condition probabilities for digits 0–4. Both scores use the same expected-rating reduction. This includes the prompt change, not only a hidden inference switch.",
            ),
            (
                "03  Inference and uncertainty",
                "Qwen/Qwen3-4B, bfloat16, 32,768-token context, batches of 128, prefix caching disabled. Ties within 1e-7 earn half credit. Prompt-cluster bootstrap: 10,000 draws; seed 1729 with independent dataset offsets; paired condition draws. Macro estimates give each dataset equal weight. Percentile intervals are exploratory and have no multiplicity correction. Failed judgments remain explicit; complete-case estimates may have selection bias.",
            ),
            (
                "04  Scope of the conclusion",
                "These results measure offline agreement, including upvote-derived LitBench labels. They do not validate rationale correctness, calibrate scores, or establish a GRPO benefit. Judge and policy thinking controls are independent. Intervals condition on the available sample and omit stopping-selection uncertainty. One deterministic rationale is used per context; alternate-rationale or execution-repeat uncertainty is not estimated.",
            ),
            (
                "05  Reproduce and audit",
                "The companion README gives prepare / score / report commands. pair_scores.csv retains all original comparisons; candidate_diagnostics.csv retains per-context statistics. provenance.json records source/result hashes and exact scoring-code/package identity. Raw inputs and reasoning IDs remain in the persistent EIDF run directory, outside Git.",
            ),
        ]
        positions = [0.803, 0.668, 0.512, 0.356, 0.221]
        for (title, body), y in zip(sections, positions, strict=True):
            fig.text(0.085, y, title, fontsize=11, weight="bold", color=ink, va="top")
            paragraph(fig, body, y - 0.031, size=8.6, width=105)
        fig.text(
            0.085,
            0.093,
            "MODEL REVISION   " + manifest["model"]["revision"],
            fontsize=6.7,
            color=muted,
        )
        save(pdf, fig)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Freeze an explicitly stopped judge comparison for reporting"
    )
    parser.add_argument("mode", choices=["freeze-observed"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_observed(args.source, reason=args.reason), indent=2))
