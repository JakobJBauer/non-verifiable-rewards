"""Generate publication-friendly figures, tables, and a self-contained HTML report."""

import base64
import html
import json
import logging
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from .analysis import analyze
from .io import DATASETS, REWARDS, code_files, file_sha256, utc_now, write_json

logger = logging.getLogger(__name__)
LABELS = {
    "rouge_l": "ROUGE-L",
    "bertscore": "BERTScore",
    "bleurt": "BLEURT",
    "self_certainty": "Self-certainty",
    "reference_likelihood": "Reference likelihood",
    "vr_cli": "Thresholded VR-CLI",
    "length": "Length baseline",
}
DATASET_LABELS = {"reddit_tldr": "Reddit TL;DR", "litbench": "LitBench"}
COLORS = dict(
    zip(
        REWARDS,
        ("#4477AA", "#228833", "#AA3377", "#AA8800", "#EE7733", "#0099BB", "#777777"),
        strict=True,
    )
)


def caption(config):
    models = config["models"]
    text = f"{config['name']} · {models['lm']['id'].split('/')[-1]} · {models['bertscore']['id'].split('/')[-1]} · {models['bleurt']['name']}"
    if "reference_likelihood" not in config.get("report_rewards", REWARDS):
        text += " (Qwen: smoke check only)"
    if config["pilot"]:
        text += "\nNeural scores use capped text; ROUGE-L and the length baseline use full outputs. Pipeline validation only."
    else:
        text += "\nEncoder metrics may truncate; see coverage. Confidence intervals resample prompts."
    return text


def save_figure(fig, name, config, *, bottom=0.09):
    directory = Path(config["run_dir"]) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.text(
        0.5,
        0.012,
        caption(config),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, bottom, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(
            directory / f"{name}.{suffix}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def accuracy_figures(result, config):
    rewards = result["rewards"]
    for delta in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
        fig.suptitle(
            "Improvement over the length baseline"
            if delta
            else "Agreement with human preferences",
            fontsize=17,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.90,
            "Points: estimates · bars: 95% prompt-bootstrap intervals"
            + (
                " · dotted: no improvement"
                if delta
                else " · dotted: chance · dashed: length baseline"
            ),
            ha="center",
            fontsize=10,
            color="#555555",
        )
        for ax, name in zip(axes, DATASETS, strict=True):
            rows = (
                result["summary"]
                .query("dataset == @name")
                .set_index("reward")
                .loc[list(rewards)]
            )
            field = "delta_vs_length" if delta else "accuracy"
            low = "delta_ci_low" if delta else "ci_low"
            high = "delta_ci_high" if delta else "ci_high"
            scale = 100 if delta else 1
            for index, reward in enumerate(rewards):
                row = rows.loc[reward]
                value = row[field] * scale
                ax.errorbar(
                    value,
                    index,
                    xerr=[
                        [max(0, (row[field] - row[low]) * scale)],
                        [max(0, (row[high] - row[field]) * scale)],
                    ],
                    fmt="o",
                    color=COLORS[reward],
                    capsize=4,
                    markersize=7,
                    linewidth=1.8,
                )
            ax.axvline(
                0 if delta else 0.5, color="#333333", linestyle=":", linewidth=1.3
            )
            if not delta:
                ax.axvline(
                    rows.loc["length", "accuracy"],
                    color=COLORS["length"],
                    linestyle="--",
                    alpha=0.6,
                )
                ax.xaxis.set_major_formatter(PercentFormatter(1))
                ax.set_xlim(
                    max(0, min(0.4, rows[low].min() - 0.04)),
                    min(1, max(0.7, rows[high].max() + 0.04)),
                )
            ax.set_yticks(range(len(rewards)), [LABELS[reward] for reward in rewards])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.18)
            ax.set_xlabel(
                "Accuracy difference (percentage points)"
                if delta
                else "Pairwise accuracy · ties count ½"
            )
            ax.set_title(
                f"{DATASET_LABELS[name]} · {int(rows.iloc[0]['pairs'])} pairs / {int(rows.iloc[0]['prompts'])} prompts"
            )
        save_figure(fig, "accuracy_vs_length" if delta else "pairwise_accuracy", config)


def heatmaps(result, config):
    rewards = result["rewards"]
    for key, filename, title, low, high in (
        (
            "correlations",
            "reward_correlations",
            "Spearman correlation of pairwise score margins",
            -1,
            1,
        ),
        (
            "agreements",
            "decision_agreement",
            "Agreement between reward decisions (including ties)",
            0,
            1,
        ),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(title, fontsize=17, fontweight="bold")
        for ax, name in zip(axes, DATASETS, strict=True):
            matrix = result[key][name].loc[list(rewards), list(rewards)].to_numpy()
            cmap = plt.get_cmap("RdBu_r" if low < 0 else "Blues").copy()
            cmap.set_bad("#e6e6e6")
            picture = ax.imshow(
                np.ma.masked_invalid(matrix), vmin=low, vmax=high, cmap=cmap
            )
            for row in range(len(rewards)):
                for col in range(len(rewards)):
                    value = matrix[row, col]
                    label = f"{value:.2f}" if np.isfinite(value) else "n/a"
                    white = np.isfinite(value) and (
                        abs(value) > 0.65 if low < 0 else value > 0.65
                    )
                    ax.text(
                        col,
                        row,
                        label,
                        ha="center",
                        va="center",
                        color="white" if white else "#222222",
                        fontsize=10,
                    )
            names = [
                LABELS[reward]
                .replace("Reference likelihood", "Ref. likelihood")
                .replace("Thresholded ", "")
                for reward in rewards
            ]
            ax.set_xticks(
                range(len(rewards)), names, rotation=45, ha="right", fontsize=10
            )
            ax.set_yticks(range(len(rewards)), names, fontsize=10)
            ax.set_title(DATASET_LABELS[name])
            fig.colorbar(picture, ax=ax, fraction=0.046, pad=0.03)
        save_figure(fig, filename, config)


def length_figure(result, config):
    rewards = result["rewards"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle(
        "Does each reward follow the longer output?", fontsize=17, fontweight="bold"
    )
    for ax, name in zip(axes, DATASETS, strict=True):
        frame = result["lengths"].query("dataset == @name")
        conditions = [
            item
            for item in ("Preferred shorter", "Equal length", "Preferred longer")
            if item in set(frame.condition)
        ]
        for index, reward in enumerate(rewards):
            rows = frame[frame.reward == reward].set_index("condition").loc[conditions]
            ax.bar(
                np.arange(len(conditions))
                + (index - (len(rewards) - 1) / 2) * (0.75 / len(rewards)),
                rows.accuracy,
                width=0.7 / len(rewards),
                color=COLORS[reward],
                label=LABELS[reward],
            )
        counts = frame[frame.reward == "length"].set_index("condition")
        ax.set_xticks(
            range(len(conditions)),
            [f"{item}\nn={int(counts.loc[item, 'pairs'])}" for item in conditions],
        )
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.axhline(0.5, color="#555555", linestyle=":")
        ax.set_ylabel("Pairwise accuracy")
        ax.set_title(DATASET_LABELS[name])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    # Reserve more room for the shared legend than the standard figure footer.
    fig.subplots_adjust(bottom=0.29)
    save_figure(fig, "length_diagnostics", config, bottom=0.24)


def vrcli_figure(result, config):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "VR-CLI: continuous improvement and threshold saturation",
        fontsize=17,
        fontweight="bold",
    )
    for col, name in enumerate(DATASETS):
        rows = result["scores"].query(
            "dataset == @name and reward == 'vr_cli' and status == 'ok'"
        )
        improvements = (
            rows["statistics.perplexity_improvement_percent"].dropna().to_numpy()
        )
        axes[0, col].hist(
            np.clip(improvements, -100, 100),
            bins=np.linspace(-100, 100, 41),
            color="#0099BB",
            alpha=0.85,
        )
        axes[0, col].axvline(0, color="#333333", linestyle=":")
        axes[0, col].set_title(DATASET_LABELS[name])
        axes[0, col].set_xlabel("Reference perplexity improvement (%)")
        axes[0, col].set_ylabel("Candidate outputs")
        clipped = int((improvements < -100).sum())
        axes[0, col].text(
            0.02,
            0.96,
            f"{clipped} values below −100% shown at −100%",
            transform=axes[0, col].transAxes,
            va="top",
            fontsize=9,
        )
        levels = [0, 0.5, 0.9, 1]
        fractions = [(rows.score == level).mean() for level in levels]
        axes[1, col].bar(range(4), fractions, color="#0099BB")
        axes[1, col].set_xticks(
            range(4),
            ["0\nI < 0.05%", "0.5\n0.05% ≤ I < 1%", "0.9\n1% ≤ I < 2%", "1\nI ≥ 2%"],
            fontsize=9,
        )
        axes[1, col].set_ylim(0, 1.1)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, col].set_ylabel("Share of candidate outputs")
        axes[1, col].set_xlabel("Published thresholded reward")
        tie = (
            result["summary"]
            .query("dataset == @name and reward == 'vr_cli'")
            .iloc[0]
            .tie_rate
        )
        axes[1, col].text(
            0.98,
            0.96,
            f"Pairwise ties: {tie:.1%}",
            transform=axes[1, col].transAxes,
            ha="right",
            va="top",
        )
    save_figure(fig, "vr_cli_diagnostics", config)


def smoke_figure(config):
    path = Path(config["run_dir"]) / "qwen_smoke_check.json"
    if not path.exists():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No completed Qwen smoke check\nLM inference was skipped or did not complete.\n"
            "The primary benchmark contains ROUGE-L, BERTScore, BLEURT and length.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        save_figure(fig, "qwen_smoke_check", config)
        return
    smoke = json.loads(path.read_text())
    rows = smoke["validation"]["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.suptitle(
        f"Qwen smoke check · one {DATASET_LABELS[smoke['validation']['dataset']]} pair · not an accuracy estimate",
        fontsize=15,
        fontweight="bold",
    )
    for ax, reward in zip(
        axes, ("reference_likelihood", "self_certainty", "vr_cli"), strict=True
    ):
        values = [
            next(
                row["score"]
                for row in rows
                if row["reward"] == reward and row["side"] == side
            )
            for side in ("preferred", "rejected")
        ]
        ax.scatter([0, 1], values, color=["#228833", "#CC6677"], s=75, zorder=3)
        for index, value in enumerate(values):
            ax.annotate(
                f"{value:.4f}",
                (index, value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
            )
        ax.set_xticks([0, 1], ["Human-preferred", "Rejected"], fontsize=10)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(LABELS[reward])
        ax.set_ylabel("Reward value (higher is better)")
        ax.grid(axis="y", alpha=0.2)
        ax.margins(y=0.22)
        if reward == "reference_likelihood":
            ax.set_ylim(min(values) - 0.12, max(values) + 0.12)
            ax.text(
                0.02,
                0.97,
                "Mean token log probability\nAxis zoomed to show difference",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        else:
            ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1.3)
    save_figure(fig, "qwen_smoke_check", config)


def coverage_figure(result, config):
    rewards = result["rewards"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2))
    fig.suptitle("Which scores used shortened inputs?", fontsize=17, fontweight="bold")
    for ax, field, title in zip(
        axes,
        ("preprocessing_truncated_candidates", "native_truncated_candidates"),
        (
            "Explicit Qwen-token caps before neural scoring",
            "Additional BERTScore / BLEURT context-limit truncation",
        ),
        strict=True,
    ):
        frame = result["coverage"].copy()
        frame["rate"] = frame[field] / (2 * frame.expected_pairs)
        matrix = (
            frame.pivot(index="dataset", columns="reward", values="rate")
            .loc[list(DATASETS), list(rewards)]
            .to_numpy()
        )
        ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        for row in range(len(DATASETS)):
            for col in range(len(rewards)):
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.1%}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > 0.65 else "black",
                )
        ax.set_yticks(range(2), [DATASET_LABELS[name] for name in DATASETS])
        ax.set_xticks(
            range(len(rewards)),
            [
                LABELS[reward]
                .replace("Reference likelihood", "Ref. likelihood")
                .replace("Thresholded ", "")
                for reward in rewards
            ],
            fontsize=10,
        )
        ax.set_title(title, fontsize=12)
    save_figure(fig, "input_coverage", config)


def runtime_figure(result, config):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Scoring cost on this machine", fontsize=17, fontweight="bold")
    order = ["lexical", "bertscore", "bleurt", "lm"]
    labels = ["ROUGE-L + length", "BERTScore", "BLEURT", "Qwen: all 3 LM rewards"]
    if "reference_likelihood" not in result["rewards"]:
        order, labels = order[:-1], labels[:-1]
    for index, name in enumerate(DATASETS):
        rows = (
            result["runtime"].query("dataset == @name").set_index("backend").loc[order]
        )
        positions = np.arange(len(order)) + (index - 0.5) * 0.33
        ax.scatter(
            rows.seconds_per_candidate,
            positions,
            s=70,
            label=DATASET_LABELS[name],
            color=("#4477AA", "#EE7733")[index],
        )
        for position, value in zip(positions, rows.seconds_per_candidate, strict=True):
            ax.annotate(
                f"{value:.3f}s",
                (value, position),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
    ax.set_yticks(range(len(order)), labels)
    ax.set_xscale("log")
    ax.set_xlabel(
        "Seconds per candidate output, excluding model loading and warmup (log scale)"
    )
    ax.margins(x=0.18, y=0.15)
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "runtime", config)


def display_summary(summary):
    rows = []
    for row in summary.itertuples():
        rows.append(
            {
                "Dataset": DATASET_LABELS[row.dataset],
                "Reward": LABELS[row.reward],
                "Accuracy (95% CI)": f"{row.accuracy:.1%} [{row.ci_low:.1%}, {row.ci_high:.1%}]",
                "Δ vs length (95% CI), pp": f"{100 * row.delta_vs_length:+.1f} [{100 * row.delta_ci_low:+.1f}, {100 * row.delta_ci_high:+.1f}]",
                "Ties": f"{row.tie_rate:.1%}",
                "Pairs": row.pairs,
            }
        )
    import pandas as pd

    return pd.DataFrame(rows)


def markdown_table(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_reports(result, config):
    run = Path(config["run_dir"])
    models = json.loads((run / "models.json").read_text())
    display = display_summary(result["summary"])
    model_description = "; ".join(
        f"{key}: {item['id']}" for key, item in models.items()
    )
    if "reference_likelihood" not in result["rewards"]:
        model_description += (
            " (Qwen smoke check only; excluded from benchmark accuracy)"
        )
    limits = f"Prompt ≤ {config['max_prompt_tokens']} Qwen tokens; candidate ≤ {config['max_candidate_tokens']}; reference ≤ {config['max_reference_tokens']} (zero means unlimited)."
    notes = [
        "This is an offline preference comparison. There is no training or output generation.",
        "Each reward scores both members of the same pair. A higher human-preferred score earns 1, a lower score earns 0, and a tie earns 0.5.",
        "Accuracy uses pairs completed by every reward included in the primary benchmark. The 95% percentile bootstrap intervals resample prompts, preserving dependent pairs and paired differences versus length. Intervals are exploratory and not adjusted for multiple comparisons.",
        "Reddit references are original human TL;DRs joined by post ID with matching post text. Pairs containing the reference itself are excluded. LitBench references are the highest-upvote distinct story outside the pair; maximum observed upvotes resolve varying scraped counts.",
        "The length baseline counts full-output words. ROUGE-L uses full texts. Neural scores use the saved scored_text/ inputs and may undergo additional encoder truncation. Human preferences refer to the full original outputs.",
        limits,
        "Reference likelihood uses mean reference-token log probability. VR-CLI reuses it and the no-candidate baseline, applying the paper's percent-perplexity thresholds. Policy and frozen scorers coincide here because they share the same checkpoint.",
        "Correlations use preferred-minus-rejected score margins; decision agreement separately measures matching win/tie/loss decisions. n/a means correlation is undefined for a constant reward. High correlation alone does not establish redundancy or RL robustness.",
    ]
    if config["pilot"]:
        notes.insert(
            0,
            f"CPU PILOT: small checkpoints, {config['pairs_per_dataset']} pairs per dataset, and capped neural inputs. Validate the pipeline and figures; do not treat these as Qwen3-4B or full-story benchmark results.",
        )
    if config.get("cpu_dynamic_int8"):
        notes.append(
            "The local Qwen linear layers use PyTorch dynamic int8 quantization; score reductions remain float32. Quantization is disabled in the GPU configuration."
        )
    findings = []
    for name in DATASETS:
        frame = result["summary"].query("dataset == @name")
        winner = frame.loc[frame.accuracy.idxmax()]
        above_length = frame[(frame.reward != "length") & (frame.delta_ci_low > 0)]
        findings.append(
            f"{DATASET_LABELS[name]}: highest point estimate is {LABELS[winner.reward]} ({winner.accuracy:.1%}). "
            + (
                "Paired intervals above the length baseline: "
                + ", ".join(LABELS[reward] for reward in above_length.reward)
                + "."
                if len(above_length)
                else "No reward's paired 95% interval lies entirely above the length baseline."
            )
        )
        if "vr_cli" in result["rewards"]:
            vr = frame[frame.reward == "vr_cli"].iloc[0]
            findings.append(
                f"{DATASET_LABELS[name]}: VR-CLI ties on {vr.tie_rate:.1%} of pairs; inspect the continuous improvement distribution before interpreting its thresholded accuracy."
            )
    if "reference_likelihood" not in result["rewards"]:
        notes.insert(
            1,
            "Qwen is excluded from the primary benchmark because local LM evaluation was limited. "
            + (
                "One successful pair is displayed as an execution smoke check. "
                if (run / "qwen_smoke_check.json").exists()
                else "No successful LM smoke check was recorded. "
            )
            + "Any partial LM outputs are retained but excluded from accuracy, correlations, and cost comparisons. See evaluation_scope.json for the saved scope.",
        )
    figures = [
        ("pairwise_accuracy", "Preference agreement"),
        ("accuracy_vs_length", "Paired differences from length"),
        ("reward_correlations", "Reward-margin correlations"),
        ("decision_agreement", "Decision agreement"),
        ("length_diagnostics", "Length effects"),
        ("vr_cli_diagnostics", "VR-CLI thresholds")
        if "vr_cli" in result["rewards"]
        else ("qwen_smoke_check", "Qwen: one-pair execution check"),
        ("input_coverage", "Input shortening"),
        ("runtime", "Runtime"),
    ]
    markdown = f"# {config['name']}\n\n{model_description}\n\n" + "\n".join(
        "- " + note for note in notes
    )
    markdown += (
        "\n\n## Results\n\n"
        + markdown_table(display)
        + "\n\n"
        + "\n".join("- " + finding for finding in findings)
    )
    for stem, title in figures:
        markdown += f"\n\n## {title}\n\n![{title}](figures/{stem}.png)\n\n[PDF](figures/{stem}.pdf)\n"
    markdown += "\n## Saved outputs\n\nFull original pairs: `data/`. Exact neural inputs: `scored_text/`. Resumable pair journals: `scores/`. Flat scores, margins, confidence intervals, bootstrap replicates, coverage and timing tables: `tables/` (CSV and Parquet). Config, resolved model revisions, source hashes, environment and logs are retained.\n"
    (run / "report.md").write_text(markdown)
    figure_html = []
    for stem, title in figures:
        encoded = base64.b64encode(
            (run / "figures" / f"{stem}.png").read_bytes()
        ).decode()
        figure_html.append(
            f'<section><h2>{html.escape(title)}</h2><img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}"></section>'
        )
    example_html = []
    for example in result["examples"]:
        content = f"<p>Pair {html.escape(example['pair_id'])}; {html.escape(example['selection'])}</p>"
        content += (
            "<table><tr><th>Reward</th><th>Preferred − rejected</th><th>Credit</th></tr>"
            + "".join(
                f"<tr><td>{LABELS[reward]}</td><td>{example['margins'][reward]:.5g}</td><td>{example['credits'][reward]:g}</td></tr>"
                for reward in result["rewards"]
            )
            + "</table>"
        )
        for field in ("prompt", "preferred", "rejected", "reference"):
            content += f"<h3>{field.title()} (full original text)</h3><pre>{html.escape(example[field])}</pre>"
        example_html.append(
            f"<details><summary>{DATASET_LABELS[example['dataset']]} · {example['pair_id']}</summary>{content}</details>"
        )
    document = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
    document += f"<title>{html.escape(config['name'])}</title><style>body{{font:16px/1.55 system-ui,sans-serif;background:#f4f6f8;color:#182333;margin:0}}main{{max-width:1200px;margin:auto;padding:36px}}h1{{font-size:32px}}h2{{margin-top:30px}}section,.notes,details{{background:white;padding:22px;border:1px solid #dce2e8;border-radius:8px;margin:20px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}th{{background:#eaf0f5}}pre{{white-space:pre-wrap;font:14px/1.5 system-ui}}summary{{cursor:pointer;font-weight:600}}.scroll{{overflow-x:auto}}li{{margin:8px 0}}</style><main><h1>{html.escape(config['name'])}</h1><p>{html.escape(model_description)}</p>"
    document += (
        "<div class='notes'><h2>Protocol and interpretation</h2><ul>"
        + "".join("<li>" + html.escape(note) + "</li>" for note in notes)
        + "</ul></div>"
    )
    document += (
        "<section><h2>Results</h2><div class='scroll'>"
        + display.to_html(index=False, border=0, escape=True)
        + "</div><ul>"
        + "".join("<li>" + html.escape(finding) + "</li>" for finding in findings)
        + "</ul></section>"
    )
    document += (
        "".join(figure_html)
        + "<h2>Disagreement examples</h2><p>Deterministically selected cases with split reward decisions; full texts are shown below.</p>"
        + "".join(example_html)
        + "</main></html>"
    )
    (run / "report.html").write_text(document)
    # A standalone LaTeX table needs only booktabs in the consuming document.
    (run / "tables" / "summary.tex").write_text(
        display.to_latex(index=False, escape=True)
    )


def generate_report(config):
    config = dict(config)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    result = analyze(config)
    config["report_rewards"] = result["rewards"]
    accuracy_figures(result, config)
    heatmaps(result, config)
    length_figure(result, config)
    if "vr_cli" in result["rewards"]:
        vrcli_figure(result, config)
    else:
        smoke_figure(config)
    coverage_figure(result, config)
    runtime_figure(result, config)
    write_reports(result, config)
    run = Path(config["run_dir"])
    root, files = code_files()
    with tarfile.open(run / "source_at_report.tar.gz", "w:gz") as archive:
        for file in files:
            archive.add(file, arcname=file.relative_to(root))
    write_json(
        run / "artifacts.json",
        {
            "generated_at": utc_now(),
            "files": {
                str(path.relative_to(run)): {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in sorted(run.rglob("*"))
                if path.is_file() and path.name != "artifacts.json"
            },
        },
    )
    logger.info("Wrote %s/report.html and eight PNG/PDF figures", run)
