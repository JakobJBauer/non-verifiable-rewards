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

from ..rewards.prompts import REFERENCE_PROMPT_TEMPLATE
from .analysis import analyze
from .io import BACKEND_REWARDS, REWARDS, code_files, file_sha256, utc_now, write_json

logger = logging.getLogger(__name__)
LABELS = {
    "rouge_l": "ROUGE-L",
    "chrf_pp": "chrF++",
    "meteor": "METEOR (NLTK)",
    "bertscore": "BERTScore",
    "bleurt": "BLEURT",
    "smart1_bertscore": "SMART-1 BERTScore",
    "smart1_bleurt": "SMART-1 BLEURT",
    "self_certainty": "Self-certainty",
    "reference_likelihood": "Reference likelihood",
    "vr_cli": "VR-CLI",
    "nover": "NOVER proxy",
    "llm_judge": "LLM judge",
    "length": "Length baseline",
}
DATASET_LABELS = {"reddit_tldr": "Reddit TL;DR", "litbench": "LitBench", "wmt14_de_en": "WMT14 German → English"}
COLORS = dict(
    zip(
        REWARDS,
        (
            "#4477AA", "#66CCEE", "#DDCC77", "#228833", "#AA3377", "#AA8800",
            "#EE7733", "#0099BB", "#8855AA", "#332288", "#777777",
        ),
        strict=True,
    )
)
COLORS.update(smart1_bertscore="#44AA99", smart1_bleurt="#CC6677")


def caption(config):
    models = config["models"]
    text = f"{config['name']} · {models['lm']['id'].split('/')[-1]} · {models['bertscore']['id'].split('/')[-1]} · {models['bleurt']['name']}"
    if "reference_likelihood" not in config.get("report_rewards", REWARDS):
        text += " (Qwen: smoke check only)"
    if config["pilot"]:
        text += "\nBase neural scores use capped text; SMART, METEOR and lexical rewards use full outputs. Pipeline validation only."
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
        fig, axes = plt.subplots(1, len(result["datasets"]), figsize=(6.5 * len(result["datasets"]), 5.7), squeeze=False)
        axes = axes[0]
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
        for ax, name in zip(axes, result["datasets"], strict=True):
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
        # Thirteen annotated columns become illegible when three panels are
        # shrunk side by side to the HTML report's page width.
        stacked = len(rewards) > 10 and len(result["datasets"]) > 1
        rows, columns = (len(result["datasets"]), 1) if stacked else (1, len(result["datasets"]))
        fig, axes = plt.subplots(
            rows, columns,
            figsize=(10, 8.8 * rows) if stacked else (7.5 * columns, 7),
            squeeze=False,
        )
        axes = axes.ravel()
        fig.suptitle(title, fontsize=17, fontweight="bold")
        for ax, name in zip(axes, result["datasets"], strict=True):
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
        save_figure(fig, filename, config, bottom=0.035 if stacked else 0.09)


def length_figure(result, config):
    rewards = result["rewards"]
    fig, axes = plt.subplots(1, len(result["datasets"]), figsize=(7 * len(result["datasets"]), 5.8), squeeze=False)
    axes = axes[0]
    fig.suptitle(
        "Does each reward follow the longer output?", fontsize=17, fontweight="bold"
    )
    for ax, name in zip(axes, result["datasets"], strict=True):
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
    fig, axes = plt.subplots(2, len(result["datasets"]), figsize=(6 * len(result["datasets"]), 8), squeeze=False)
    variant_label = (
        "task prompts"
        if result["vr_cli_variant"] == "continuous_task_specific"
        else result["vr_cli_variant"]
    )
    fig.suptitle(
        f"VR-CLI ({variant_label}): improvement and preference margins",
        fontsize=17,
        fontweight="bold",
    )
    for col, name in enumerate(result["datasets"]):
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
        margins = result["margins"].query("dataset == @name")
        axes[1, col].scatter(
            margins.reference_likelihood,
            margins.vr_cli,
            color="#0099BB",
            alpha=0.5,
            s=12,
        )
        axes[1, col].axhline(0, color="#333333", linestyle=":")
        axes[1, col].axvline(0, color="#333333", linestyle=":")
        axes[1, col].set_ylabel("VR-CLI margin (preferred − rejected)")
        axes[1, col].set_xlabel("Reference log-likelihood margin (preferred − rejected)")
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
            "The primary benchmark contains the completed non-LM rewards.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        save_figure(fig, "qwen_smoke_check", config)
        return
    smoke = json.loads(path.read_text())
    rows = smoke["validation"]["rows"]
    rewards = [
        reward
        for reward in ("reference_likelihood", "self_certainty", "vr_cli", "nover", "llm_judge")
        if any(row["reward"] == reward for row in rows)
    ]
    fig, axes = plt.subplots(1, len(rewards), figsize=(4.3 * len(rewards), 4.8))
    fig.suptitle(
        f"Qwen smoke check · one {DATASET_LABELS[smoke['validation']['dataset']]} pair · not an accuracy estimate",
        fontsize=15,
        fontweight="bold",
    )
    for ax, reward in zip(axes, rewards, strict=True):
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
        if reward in ("reference_likelihood", "nover"):
            ax.set_ylim(min(values) - 0.12, max(values) + 0.12)
            ax.text(
                0.02,
                0.97,
                "Mean token log probability\nAxis zoomed to show difference"
                if reward == "reference_likelihood"
                else "Negative log normalized perplexity\nAxis zoomed to show difference",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        else:
            ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1.3)
    save_figure(fig, "qwen_smoke_check", config)


def coverage_figure(result, config):
    rewards = result["rewards"]
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(rewards)), 7.5))
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
            .loc[list(result["datasets"]), list(rewards)]
            .to_numpy()
        )
        ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        for row in range(len(result["datasets"])):
            for col in range(len(rewards)):
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.1%}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > 0.65 else "black",
                )
        ax.set_yticks(range(len(result["datasets"])), [DATASET_LABELS[name] for name in result["datasets"]])
        ax.set_xticks(
            range(len(rewards)),
            [
                LABELS[reward]
                .replace("Reference likelihood", "Ref. likelihood")
                .replace("Thresholded ", "")
                for reward in rewards
            ],
            fontsize=10,
            rotation=35,
            ha="right",
        )
        ax.set_title(title, fontsize=12)
    save_figure(fig, "input_coverage", config)


def runtime_figure(result, config):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    derived = (Path(config["run_dir"]) / "derivation.json").exists()
    fig.suptitle(
        "Source-run inference cost" if derived else "Scoring cost on this machine",
        fontsize=17,
        fontweight="bold",
    )
    order = ["lexical", "bertscore", "bleurt", "lm"]
    lm_label = (
        "Qwen: LM rewards (NOVER reuses likelihoods)"
        if "nover" in result["rewards"] else "Qwen: all 3 LM rewards"
    )
    lexical_label = (
        "ROUGE-L + chrF++ + length"
        if "chrf_pp" in result["rewards"] else "ROUGE-L + length"
    )
    labels = [lexical_label, "BERTScore", "BLEURT", lm_label]
    if "reference_likelihood" not in result["rewards"]:
        order, labels = order[:-1], labels[:-1]
    if "meteor" in result["rewards"]:
        order.insert(1, "meteor")
        labels.insert(1, LABELS["meteor"])
    for reward in ("smart1_bertscore", "smart1_bleurt"):
        if reward in result["rewards"]:
            order.append(reward)
            labels.append(LABELS[reward])
    selected = [
        (backend, label) for backend, label in zip(order, labels, strict=True)
        if set(BACKEND_REWARDS[backend]) & set(result["rewards"])
    ]
    order, labels = zip(*selected, strict=True)
    for index, name in enumerate(result["datasets"]):
        rows = (
            result["runtime"].query("dataset == @name").set_index("backend").loc[list(order)]
        )
        positions = np.arange(len(order)) + (index - (len(result["datasets"]) - 1) / 2) * (0.75 / len(result["datasets"]))
        ax.scatter(
            rows.seconds_per_candidate,
            positions,
            s=70,
            label=DATASET_LABELS[name],
            color=("#4477AA", "#EE7733", "#228833")[index],
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
        "The length baseline, ROUGE-L, chrF++, METEOR and SMART use full original texts. Other neural scores use the saved scored_text/ inputs and may undergo additional encoder truncation. SMART segments long texts to cover their full content. Human preferences refer to the full original outputs.",
        limits,
        "Reference likelihood uses mean reference-token log probability. VR-CLI measures percent reference-perplexity improvement relative to a no-candidate baseline. LM scores use a fixed checkpoint; this run does not train a policy.",
        "Correlations use preferred-minus-rejected score margins; decision agreement separately measures matching win/tie/loss decisions. n/a means correlation is undefined for a constant reward. High correlation alone does not establish redundancy or RL robustness.",
    ]
    manifest_path = run / "data_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(manifest[name].get("split") == "test" for name in result["datasets"]):
            notes.append(
                "Pairs are sampled from the task loaders' test partitions using the saved seed. "
                "Reddit uses SFF valid2 posts; LitBench uses our reproducible prompt-disjoint "
                "20% holdout of LitBench-Train, not its official text-free test release; "
                "WMT14 uses the human-ranked newstest2014 translations. The historical "
                "November 13 report used different Reddit/LitBench sampling pools and older "
                "likelihood prompts, so changes in its aggregate accuracies are not controlled "
                "estimates of the effect of any single implementation change."
            )
    if "wmt14_de_en" in result["datasets"]:
        notes.append(
            "WMT14 German-to-English pairs use official newstest2014 English references "
            "and strict human ranks of system translations. Ties/unranked judgments and "
            "overlapping texts are excluded. Duplicate labeled text pairs are merged; "
            "opposing judgments remain explicit. The parallel training corpus is separate "
            "and is not used for this preference evaluation."
        )
    if "vr_cli" in result["rewards"]:
        vr_notes = {
            "continuous_task_specific": "This run uses continuous VR-CLI with task-specific summary/story/translation instructions. The candidate is an example in the prompt; the held-out reference is the scored continuation. The no-candidate baseline uses the same task instruction. VR-CLI requires separate likelihood evaluations and can rank candidates differently from the generic reference-likelihood benchmark. This is an unthresholded adaptation of the published reward; better preference accuracy is not established by the prompt change alone.",
            "continuous": "This run uses continuous VR-CLI with the original generic conditioning: percent perplexity improvement is the reward, without thresholds or clipping. For a shared baseline it has the same mathematical pairwise ordering as the reference-likelihood benchmark; floating-point precision and tie tolerance can still produce ties.",
            "thresholded": "This historical run uses thresholded VR-CLI: the published cutoffs at 0.05%, 1%, and 2% map improvement to rewards 0, 0.5, 0.9, and 1. Saved scores have not been converted to the continuous variant.",
        }
        if result["reference_prompt_template"] == REFERENCE_PROMPT_TEMPLATE:
            vr_notes["continuous_task_specific"] = (
                "Reference likelihood and continuous VR-CLI share the same task-specific "
                "summary/story/translation prompt, target tokens, and conditioned likelihood evaluation. "
                "The candidate is an example in the prompt; the held-out reference is scored. "
                "VR-CLI additionally uses the same task prompt without the candidate as its baseline. "
                "Their mathematical pairwise ordering is identical; numerical precision and tie "
                "tolerance can still affect ties. Saved results from earlier prompt variants are unchanged."
            )
        notes.append(vr_notes[result["vr_cli_variant"]])
    if "nover" in result["rewards"]:
        notes.append(
            "NOVER proxy reuses the reference-likelihood prompt and mean log probability, divided by "
            "1 + log10(candidate token count), following the authors' length normalization. "
            "This negative-log-perplexity score rewards length at fixed likelihood. "
            "Completed candidates substitute for reasoning; group ranks, format gating, efficiency "
            "rewards, and policy/proxy synchronization are absent. This is an offline proxy, not full NOVER RL."
        )
    if "chrf_pp" in result["rewards"]:
        notes.append(
            "chrF++ scores full original candidates and references using SacreBLEU: "
            "character n-grams through order 6, word n-grams through order 2, beta=2, "
            "case-sensitive, no character whitespace, and effective-order averaging. "
            "Each candidate is scored separately on the native 0–100 scale. "
            "The SacreBLEU settings/version signature is saved with every score."
        )
    if "meteor" in result["rewards"]:
        notes.append(
            "METEOR uses NLTK's English implementation on complete texts with "
            "lowercasing, Porter stemming, WordNet synonyms, and alpha=0.9, beta=3, gamma=0.5. "
            "The tokenizer is nltk.word_tokenize(preserve_line=True); scores are per text pair on a 0–1 scale. "
            "This is NLTK METEOR, not the Java release with paraphrase tables. "
            "WordNet checksums are recorded in models.json, and NLTK/WordNet versions in each score."
        )
    if "llm_judge" in result["rewards"]:
        notes.append(
            "The LLM judge shares the fixed Qwen checkpoint and independently rates each "
            "candidate with a task-specific reference-aware quality rubric. It never sees "
            "preference labels or the competing candidate. Its score is the expected 0–4 "
            "rating, divided by four, using next-token probabilities renormalized over the "
            "five digits with thinking disabled. The rubric, rating probabilities and total "
            "valid-rating probability mass are recorded per score. It is a model judgment, "
            "not a calibrated probability of human preference."
        )
    if any(reward.startswith("smart1_") for reward in result["rewards"]):
        notes.append(
            "The SMART-1 variants match candidate and reference sentence segments using "
            "BERTScore or BLEURT, take directional best-match averages, and combine them "
            "with the published harmonic mean. They cover complete texts without encoder "
            "truncation. BLEURT can yield nonpositive components; diagnostic flags are "
            "retained and its raw harmonic mean is not silently clipped."
        )
    if (run / "execution" / "parallel_workers.sh").exists():
        notes.append(
            "The Qwen worker and separate PyTorch/TensorFlow encoder groups shared one "
            "H200 GPU concurrently. Runtime measurements include resource contention; "
            "they are execution costs for this schedule, not isolated backend-throughput "
            "comparisons or direct speed comparisons with the earlier A100 run."
        )
    if config.get("bleurt_tf32") is False:
        notes.append(
            "TensorFlow TF32 is disabled for both BLEURT variants. SMART-BLEURT "
            "uses the official length-batched scorer, removing padding only. "
            "SMART-BERTScore retains unique sentence embeddings on the GPU and "
            "uses upstream greedy cosine matching. All sentence comparisons are "
            "retained; validation results are saved under execution/."
        )
    if (run / "derivation.json").exists():
        notes.append(
            "This report derives NOVER from saved reference-likelihood statistics without new model inference. "
            "Other scores and inference timings are inherited from the source run; VR-CLI retains its original variant. "
            "See derivation.json for source checksums and provenance."
        )
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
    for name in result["datasets"]:
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
                f"{DATASET_LABELS[name]}: {result['vr_cli_variant'].replace('_', ' ').capitalize()} VR-CLI ties on {vr.tie_rate:.1%} of pairs."
            )
    if "reference_likelihood" not in result["rewards"]:
        scope_path = run / "evaluation_scope.json"
        scope_reason = (
            json.loads(scope_path.read_text()).get("reason")
            if scope_path.exists() else None
        )
        notes.insert(
            1,
            scope_reason or (
            "Qwen is excluded from the primary benchmark because local LM evaluation was limited. "
            + (
                "One successful pair is displayed as an execution smoke check. "
                if (run / "qwen_smoke_check.json").exists()
                else "No successful LM smoke check was recorded. "
            )
            + "Any partial LM outputs are retained but excluded from accuracy, correlations, and cost comparisons. See evaluation_scope.json for the saved scope."),
        )
    if (run / "subset_selection.json").exists():
        notes.insert(
            0,
            "Runtime-limited comparison: all thirteen rewards use the same fixed "
            "100 pairs per dataset, taken from the original shuffled test samples. "
            "Full 2,000-pair BERTScore and inexpensive-baseline results are retained "
            "separately. The smaller comparison has wider uncertainty; do not treat "
            "small differences in point estimates as established rankings."
        )
    figures = [
        ("pairwise_accuracy", "Preference agreement"),
        ("accuracy_vs_length", "Paired differences from length"),
        ("reward_correlations", "Reward-margin correlations"),
        ("decision_agreement", "Decision agreement"),
        ("length_diagnostics", "Length effects"),
        ("vr_cli_diagnostics", "VR-CLI diagnostics")
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
    scope_path = Path(config["run_dir"]) / "evaluation_scope.json"
    if scope_path.exists():
        config["name"] = json.loads(scope_path.read_text()).get("title", config["name"])
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
