"""Paired policy comparisons on identical frozen test inputs."""

import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .._hashing import file_sha256
from ..experiments.analysis import cluster_bootstrap, export_table, preference_credit
from ..experiments.io import read_jsonl, write_json
from .benchmark import evaluation_dir, verify_policy_outputs
from .state import verify_data


def summarize(values, prompts, models, config):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Benchmark values must be finite")
    boot = cluster_bootstrap(
        values,
        prompts,
        samples=config["benchmark"]["bootstrap_samples"],
        seed=config["seed"],
    )
    baseline = models.index("base") if "base" in models else None
    results = []
    for i, name in enumerate(models):
        low, high = np.quantile(boot[:, i], [0.025, 0.975])
        result = {
            "model": name,
            "examples": len(values),
            "prompts": len(set(prompts)),
            "mean": float(values[:, i].mean()),
            "ci_low": float(low),
            "ci_high": float(high),
        }
        if baseline is not None:
            delta = boot[:, i] - boot[:, baseline]
            low, high = np.quantile(delta, [0.025, 0.975])
            result.update(
                delta_vs_base=float((values[:, i] - values[:, baseline]).mean()),
                delta_ci_low=float(low),
                delta_ci_high=float(high),
            )
        results.append(result)
    return results


def report(config, names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(config["output_dir"])
    verify_data(root)
    names = list(names)
    if config["benchmark"]["include_base_model"] and "base" not in names:
        names.insert(0, "base")
    if len(set(names)) != len(names) or not names:
        raise ValueError("Report model names must be distinct and nonempty")
    policy_frames, reward_frames, generated = [], [], []
    for name in names:
        target = evaluation_dir(config, name)
        verify_policy_outputs(target)
        complete = json.loads((target / "complete.json").read_text())
        for file, checksum in complete["files"].items():
            if file_sha256(target / file) != checksum:
                raise ValueError(f"Benchmark scores changed for {name}")
        rows = read_jsonl(target / "preferences.jsonl")
        if rows:
            policy_frames.append(pd.DataFrame(rows).assign(model=name))
        generated.extend(
            {**row, "model": name} for row in read_jsonl(target / "generations.jsonl")
        )
        for reward in complete["counts"]:
            reward_frames.append(
                pd.DataFrame(read_jsonl(target / "scores" / f"{reward}.jsonl")).assign(
                    model=name
                )
            )
    rewards = pd.concat(reward_frames, ignore_index=True)
    policy = (
        pd.concat(policy_frames, ignore_index=True) if policy_frames else pd.DataFrame()
    )
    if rewards.duplicated(["model", "dataset", "reward", "prompt_id"]).any():
        raise ValueError("Duplicate generated-output scores")
    reward_summary, policy_summary = [], []
    for (dataset, reward), frame in rewards.groupby(["dataset", "reward"]):
        table = frame.pivot(index="prompt_id", columns="model", values="score").reindex(
            columns=names
        )
        if table.isna().any().any():
            raise ValueError("Models were not evaluated on the same generation prompts")
        expected = config["benchmark"][
            "smart_bleurt_prompts"
            if reward == "smart1_bleurt"
            else "generation_prompts_per_dataset"
        ]
        if len(table) != expected:
            raise ValueError("Incorrect generation benchmark coverage")
        reward_summary.extend(
            {"dataset": dataset, "reward": reward, **row}
            for row in summarize(table.to_numpy(), table.index, names, config)
        )
    if len(policy):
        if policy.duplicated(["model", "pair_id"]).any():
            raise ValueError("Duplicate preference evaluations")
        for dataset, frame in policy.groupby("dataset"):
            for measure in ("mean_margin", "sum_margin"):
                table = frame.pivot(
                    index="pair_id", columns="model", values=measure
                ).reindex(columns=names)
                prompts = (
                    frame.drop_duplicates("pair_id")
                    .set_index("pair_id")
                    .loc[table.index, "prompt_id"]
                )
                if (
                    table.isna().any().any()
                    or len(table) != config["benchmark"]["pairs_per_dataset"]
                ):
                    raise ValueError(
                        "Policies were not evaluated on all the same preference pairs"
                    )
                credits = preference_credit(
                    table.to_numpy(), config["benchmark"]["tie_tolerance"]
                )
                rows = summarize(credits, prompts, names, config)
                policy_summary.extend(
                    {"dataset": dataset, "measure": measure, **row} for row in rows
                )
    reward_summary = pd.DataFrame(reward_summary)
    policy_summary = pd.DataFrame(policy_summary)
    generated = pd.DataFrame(generated)
    diagnostics = generated.groupby(["model", "dataset"], as_index=False).agg(
        prompts=("prompt_id", "size"),
        mean_words=("words", "mean"),
        mean_tokens=("tokens", "mean"),
        empty_rate=("empty", "mean"),
        token_limit_rate=("reached_token_limit", "mean"),
    )
    for label, frame in (
        ("generated_reward_scores", rewards),
        ("generated_reward_summary", reward_summary),
        ("policy_preferences", policy),
        ("policy_preference_summary", policy_summary),
        ("generation_diagnostics", diagnostics),
    ):
        if len(frame):
            export_table(frame, root / "tables" / label)
    images = []
    if len(policy_summary):
        datasets = list(policy_summary.dataset.unique())
        fig, axes = plt.subplots(
            1,
            len(datasets),
            figsize=(6 * len(datasets), max(4, len(names) * 0.38)),
            squeeze=False,
        )
        for ax, dataset in zip(axes[0], datasets, strict=True):
            for offset, measure, color in (
                (-0.13, "mean_margin", "#2266aa"),
                (0.13, "sum_margin", "#bd6725"),
            ):
                frame = (
                    policy_summary[
                        (policy_summary.dataset == dataset)
                        & (policy_summary.measure == measure)
                    ]
                    .set_index("model")
                    .loc[names]
                )
                ax.errorbar(
                    frame["mean"],
                    np.arange(len(names)) + offset,
                    xerr=[frame["mean"] - frame.ci_low, frame.ci_high - frame["mean"]],
                    fmt="o",
                    color=color,
                    label=measure.removesuffix("_margin"),
                )
            ax.set_yticks(range(len(names)), names)
            ax.invert_yaxis()
            ax.axvline(0.5, color="gray", linestyle=":")
            ax.set(title=dataset, xlabel="Preference agreement (ties count half)")
            ax.legend(title="Candidate log likelihood")
        fig.suptitle(
            "Saved policies on the same held-out preference pairs · 95% prompt-bootstrap intervals"
        )
        fig.tight_layout()
        (root / "figures").mkdir(exist_ok=True)
        fig.savefig(root / "figures/policy_agreement.png", dpi=160, bbox_inches="tight")
        fig.savefig(root / "figures/policy_agreement.pdf", bbox_inches="tight")
        images.append(
            base64.b64encode(
                (root / "figures/policy_agreement.png").read_bytes()
            ).decode()
        )
        plt.close(fig)
    notes = (
        "Each trained policy starts from the same base checkpoint. Reward models are frozen and shared in definition across runs. "
        "Generated outputs have no human preference labels: their reward means are model/metric assessments, not human accuracy. "
        "Policy preference accuracy instead ranks the existing held-out candidates using the saved policy's own likelihood; "
        "both mean-token and summed log likelihood are shown because length affects their rankings. "
        "LitBench labels are upvote-derived proxies and its references are third stories. "
        "Every model uses identical frozen evaluation inputs. Bootstrap intervals resample prompts with paired draws across models, "
        "and are exploratory without a multiple-comparison correction. Reward scales differ; do not average columns or compare their raw magnitudes. "
        "SMART-BLEURT uses only its configured common subset; every other reward uses all generation prompts. "
        "BERTScore and BLEURT may truncate at their native encoder limits when enabled in the config. "
        "SuperGPQA, when selected, has exact answer accuracy and no invented preference pairs."
    )
    scope = {
        "models": names,
        "datasets": config["data"]["datasets"],
        "preference_pairs_per_dataset": config["benchmark"]["pairs_per_dataset"],
        "generation_prompts_per_dataset": config["benchmark"][
            "generation_prompts_per_dataset"
        ],
        "smart_bleurt_prompts_per_dataset": config["benchmark"]["smart_bleurt_prompts"],
        "notes": notes,
    }
    write_json(root / "report_scope.json", scope)
    title = "GRPO reward sweep: saved-policy comparison"
    doc = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
    doc += f"<title>{title}</title><style>body{{font:16px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:20px;color:#182333}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}details{{margin:20px 0}}img{{max-width:100%}}.scroll{{overflow-x:auto}}</style><h1>{title}</h1>"
    doc += f"<p>{html.escape(notes)}</p><pre>{html.escape(json.dumps({k: v for k, v in scope.items() if k != 'notes'}, indent=2))}</pre>"
    for encoded in images:
        doc += f'<img alt="Policy preference agreement" src="data:image/png;base64,{encoded}">'

    def table(frame):
        return (
            '<div class="scroll">'
            + frame.to_html(index=False, float_format=lambda x: f"{x:.4f}", escape=True)
            + "</div>"
        )

    definitions = [
        {
            "model": run["name"],
            "training_datasets": ", ".join(run["datasets"]),
            "reward_weights": json.dumps(run["rewards"], sort_keys=True),
            "optimizer_steps": config["optimization"]["max_steps"],
        }
        for run in config["runs"]
        if run["name"] in names
    ]
    if definitions:
        doc += "<h2>Training runs</h2>" + table(pd.DataFrame(definitions))
    if len(policy_summary):
        doc += "<h2>Policy likelihood agreement</h2>" + table(policy_summary)
    doc += "<h2>Generated-output rewards</h2><p>Means, 95% intervals and paired differences from the base model, in each reward's native units.</p>"
    for reward, frame in reward_summary.groupby("reward", sort=False):
        doc += (
            f"<details><summary>{html.escape(reward)}</summary>{table(frame)}</details>"
        )
    doc += "<h2>Length, empty responses and generation limits</h2>" + table(diagnostics)
    doc += "<h2>Saved artifacts</h2><p>Full completions, token IDs and score journals: benchmarks/&lt;model&gt;/. Standalone model weights and tokenizer: runs/&lt;run&gt;/model/. Resumable optimizer states: runs/&lt;run&gt;/checkpoints/. Raw reward components: runs/&lt;run&gt;/reward_trace.jsonl. CSV/Parquet tables: tables/.</p></html>"
    (root / "report.html").write_text(doc)
    (root / "report.md").write_text(
        f"# {title}\n\n{notes}\n\n[HTML report](report.html)\n\nModels: {', '.join(names)}.\n\nSee `report_scope.json` for exact sample counts and `tables/` for paired confidence intervals.\n"
    )
    # Model/checkpoint checksums are in trained.json; avoid rehashing optimizer
    # checkpoints and all 13 full model copies every time tables are rendered.
    files = [
        p
        for folder in ("benchmarks", "tables", "figures")
        for p in (root / folder).rglob("*")
        if p.is_file()
    ]
    files += [
        root / name
        for name in (
            "report.html",
            "report.md",
            "report_scope.json",
            "sweep.json",
            "data_manifest.json",
            "models.json",
        )
        if (root / name).exists()
    ]
    write_json(
        root / "report_artifacts.json",
        {
            str(p.relative_to(root)): {
                "sha256": file_sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted(files)
        },
    )
    return scope
