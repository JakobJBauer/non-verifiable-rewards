import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from non_verifiable_rewards.data.pairs import (
    litbench_pairs,
    make_pair,
    reddit_pairs,
)
from non_verifiable_rewards.experiments.analysis import (
    analyze,
    cluster_bootstrap,
    load_results,
    preference_credit,
)
from non_verifiable_rewards.experiments.data import sample_pairs
from non_verifiable_rewards.experiments.io import (
    BACKEND_REWARDS,
    LEGACY_REWARDS,
    REWARDS,
    Journal,
    read_jsonl,
    write_json,
    write_jsonl,
)
from non_verifiable_rewards.experiments.io import (
    LEGACY_DATASETS as DATASETS,
)
from non_verifiable_rewards.experiments.runner import Backend
from non_verifiable_rewards.experiments.smoke import (
    PILOT_REWARDS,
    lm_mode,
    set_lm_scope,
    smoke_check,
)
from non_verifiable_rewards.rewards import VRCLIReward
from non_verifiable_rewards.rewards.prompts import (
    REFERENCE_PROMPT_TEMPLATE,
    chat_context,
    make_reference_context,
    task_for_dataset,
)


def test_reference_selection_never_uses_either_litbench_candidate():
    rows = [
        {
            "prompt": "Prompt",
            "chosen_story": "A",
            "rejected_story": "B",
            "chosen_upvotes": 10,
            "rejected_upvotes": 5,
        },
        {
            "prompt": "Prompt",
            "chosen_story": "C",
            "rejected_story": "A",
            "chosen_upvotes": 30,
            "rejected_upvotes": 10,
        },
        {
            "prompt": "Prompt",
            "chosen_story": "D",
            "rejected_story": "B",
            "chosen_upvotes": 20,
            "rejected_upvotes": 6,
        },
    ]
    pairs, audit = litbench_pairs(rows)
    assert pairs[0]["reference"] == "C"
    assert pairs[1]["reference"] == "D"
    assert audit["stories_with_multiple_observed_vote_counts"] == 1
    assert all(
        pair["reference"] not in (pair["preferred"], pair["rejected"]) for pair in pairs
    )


def test_reddit_filters_gold_candidate_and_mismatched_source_posts():
    gold = {"p": {"post": "Post", "summary": "Human reference"}}
    base = {
        "info": {"id": "p", "post": "Post", "title": "Title", "subreddit": "test"},
        "summaries": [{"text": "Summary A"}, {"text": "Summary B"}],
        "choice": 1,
    }
    overlapping = {
        **base,
        "summaries": [{"text": "Human reference"}, {"text": "Summary B"}],
    }
    mismatch = {**base, "info": {**base["info"], "post": "Different post"}}
    pairs, audit = reddit_pairs([base, base, overlapping, mismatch], gold)
    assert len(pairs) == 1
    assert pairs[0]["preferred"] == "Summary B"
    assert audit["excluded_duplicate_labeled_pair"] == 1
    assert audit["excluded_empty_or_overlapping_texts"] == 1
    assert audit["excluded_source_post_mismatch"] == 1


def test_seeded_sampling_is_nested_and_input_order_independent():
    pairs = [{"pair_id": str(index)} for index in range(20)]
    small = sample_pairs(pairs, 3, 17)
    assert small == sample_pairs(list(reversed(pairs)), 10, 17)[:3]
    with pytest.raises(ValueError, match="only"):
        sample_pairs(pairs, 21, 17)


def test_journal_recovers_only_an_incomplete_tail_and_prevents_duplicates(tmp_path):
    path = tmp_path / "scores.jsonl"
    with Journal(path) as journal:
        journal.append({"pair_id": "a", "rows": []})
    with path.open("ab") as stream:
        stream.write(b'{"pair_id": "interrupted')
    with Journal(path) as journal:
        assert set(journal.completed) == {"a"}
        with pytest.raises(ValueError, match="twice"):
            journal.append({"pair_id": "a", "rows": []})
        journal.append({"pair_id": "b", "rows": []})
    assert [json.loads(line)["pair_id"] for line in path.read_text().splitlines()] == [
        "a",
        "b",
    ]


def test_journal_does_not_hide_corruption_in_complete_records(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text('{"broken":\n')
    with pytest.raises(json.JSONDecodeError), Journal(path):
        pass


def test_preference_credit_and_cluster_resampling():
    assert preference_credit([1, 0, -1, 1e-10]).tolist() == [1, 0.5, 0, 0.5]
    with pytest.raises(ValueError, match="finite"):
        preference_credit([np.nan])
    values = np.array([[1, 1], [1, 1], [0, 0]])
    boot = cluster_bootstrap(values, ["a", "a", "b"], samples=200, seed=1)
    assert set(np.round(boot[:, 0], 6)) == {0, round(2 / 3, 6), 1}
    assert np.all(boot[:, 0] - boot[:, 1] == 0)


def test_lexical_scores_use_full_original_outputs():
    original = make_pair(
        "reddit_tldr", "p", "Prompt", "a b c d", "a z", "a b c", "ref", {}
    )
    scored = {
        **original,
        "preferred": "a",
        "rejected": "a",
        "reference": "a",
        "reference_prefix_overlaps_candidate": True,
    }
    scored["lengths"] = {
        field: {
            "original_words": len(original[field].split()),
            "scored_words": 1,
            "truncated": True,
        }
        for field in ("prompt", "preferred", "rejected", "reference")
    }
    rows, _ = Backend("lexical", {}, {}).evaluate(scored, original)
    values = {(row["reward"], row["side"]): row["score"] for row in rows}
    assert values["length", "preferred"] == 4
    assert values["length", "rejected"] == 2
    assert values["rouge_l", "preferred"] == pytest.approx(6 / 7)
    from non_verifiable_rewards.rewards import ChrFPlusPlusReward

    expected = ChrFPlusPlusReward().score(
        [original["preferred"], original["rejected"]],
        references=[original["reference"]] * 2,
    )
    chrf = [row for row in rows if row["reward"] == "chrf_pp"]
    assert [row["score"] for row in chrf] == expected
    assert expected[0] != 100  # The truncated inputs would be identical.
    assert all("nw:2" in row["statistics"]["sacrebleu_signature"] for row in chrf)
    assert all(row["statistics"]["beta"] == 2 for row in chrf)
    assert all(not row["preprocessing_truncated"] for row in rows)


@pytest.mark.parametrize("conditioned", [(-3.2, -3.4), (-1, -2)])
@pytest.mark.parametrize("dataset", (*DATASETS, "wmt14_de_en"))
def test_lm_backend_reuses_task_likelihoods_for_all_reference_rewards(
    tmp_path, conditioned, dataset
):
    pair = make_pair(
        dataset, "p", "Prompt", " ".join(["better"] * 100), "worse", "Reference", "ref", {}
    )
    if dataset == "wmt14_de_en":
        from non_verifiable_rewards.rewards.prompts import translation_prompt

        pair["prompt"] = translation_prompt("Ein deutscher Satz.")
    pair["lengths"] = {
        field: {
            "original_words": len(pair[field].split()),
            "scored_words": len(pair[field].split()),
            "truncated": False,
        }
        for field in ("prompt", "preferred", "rejected", "reference")
    }
    pair["reference_prefix_overlaps_candidate"] = False

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False, "add_generation_prompt": True, "enable_thinking": False
            }
            return f"USER:{messages[0]['content']}\nASSISTANT:"

        def encode(self, text, **kwargs):
            return text.split()

    class Scorer:
        tokenizer = Tokenizer()

        def __init__(self):
            self.calls = []

        def log_likelihood(self, contexts, references, *, reduction):
            assert reduction == "mean"
            assert references == ["Reference"] * len(contexts)
            assert all("Reference" not in context for context in contexts)
            self.calls.append(contexts)
            if len(contexts) == 1:
                return [-3]
            assert "Here is an example" in contexts[0]
            return list(conditioned)

        def self_certainty(self, contexts, candidates):
            assert contexts == [chat_context(self.tokenizer, pair["prompt"])] * 2
            return [0.4] * len(candidates)

        def label_logprobs(self, contexts, labels):
            assert labels == ("0", "1", "2", "3", "4")
            assert all("impartial evaluator" in context for context in contexts)
            return [[math.log(0.1)] * 5 for _ in contexts]

    engine = Backend.__new__(Backend)
    engine.name, engine.scorer = "lm", Scorer()
    engine.vr_cli = VRCLIReward(engine.scorer)
    with Journal(tmp_path / "baselines.jsonl") as baselines:
        rows, _ = engine.evaluate(pair, pair, baselines)
        assert len(engine.scorer.calls) == 2  # One baseline, one shared conditioned batch.
        repeated, timing = engine.evaluate(pair, pair, baselines)
        assert repeated == rows
        assert timing["baseline_cache_hit"]
        other = {
            **pair,
            "dataset": "litbench" if dataset == "reddit_tldr" else "reddit_tldr",
        }
        _, timing = engine.evaluate(other, other, baselines)
        assert not timing["baseline_cache_hit"]  # Different task needs its own baseline.
    vr = [row for row in rows if row["reward"] == "vr_cli"]
    likelihood = [row for row in rows if row["reward"] == "reference_likelihood"]
    expected = [100 * (1 - math.exp(-3 - value)) for value in conditioned]
    assert [row["score"] for row in vr] == pytest.approx(expected)
    assert vr[0]["score"] > vr[1]["score"]
    assert all(
        row["statistics"]["vr_cli_variant"] == "continuous_task_specific" for row in vr
    )
    assert all(
        row["score"] == row["statistics"]["perplexity_improvement_percent"] for row in vr
    )
    assert [row["score"] for row in likelihood] == list(conditioned)
    assert likelihood[0]["score"] > likelihood[1]["score"]
    for ref_row, vr_row in zip(likelihood, vr, strict=True):
        assert all(
            vr_row["statistics"][key] == value
            for key, value in ref_row["statistics"].items()
        )
        assert (
            ref_row["statistics"]["reference_prompt_template"] == REFERENCE_PROMPT_TEMPLATE
        )
    nover = [row for row in rows if row["reward"] == "nover"]
    assert [row["score"] for row in nover] == pytest.approx([conditioned[0] / 3, conditioned[1]])
    assert [row["statistics"]["candidate_tokens"] for row in nover] == [100, 1]
    assert all(row["statistics"]["length_log_base"] == 10 for row in nover)
    judge = [row for row in rows if row["reward"] == "llm_judge"]
    assert [row["score"] for row in judge] == pytest.approx([0.5, 0.5])
    assert all(row["statistics"]["judge_task"] == task_for_dataset(dataset) for row in judge)
    assert all(row["statistics"]["rating_token_mass"] == pytest.approx(0.5) for row in judge)
    assert len(engine.scorer.calls) == 5  # Two baselines, three shared batches.
    assert [row["statistics"]["conditioned_logp_mean"] for row in vr] == list(
        conditioned
    )
    builder = make_reference_context(
        task=task_for_dataset(dataset), tokenizer=engine.scorer.tokenizer
    )
    assert engine.scorer.calls[0] == [builder(pair["prompt"], None)]
    assert engine.scorer.calls[1] == [
        builder(pair["prompt"], pair[side]) for side in ("preferred", "rejected")
    ]
    context = engine.scorer.calls[1][0]
    assert {"reddit_tldr": "summary", "litbench": "story", "wmt14_de_en": "English translation"}[dataset] in context
    if dataset == "wmt14_de_en":
        assert context.count("Translate the following German text into English:") == 1
        assert "Ein deutscher Satz." in context


def test_reference_prompts_reject_unrecognized_dataset():
    with pytest.raises(ValueError, match="No reference task prompt"):
        task_for_dataset("unknown")


@pytest.fixture
def synthetic_run(tmp_path):
    for name in DATASETS:
        pairs = [
            make_pair(
                name,
                "prompt" + str(index // 2),
                "Prompt",
                f"preferred {index}",
                f"rejected {index}",
                "Reference",
                "ref",
                {},
            )
            for index in range(3)
        ]
        write_jsonl(tmp_path / "data" / f"{name}.jsonl", pairs)
        for backend, rewards in BACKEND_REWARDS.items():
            records = []
            for pair in pairs:
                rows = [
                    {
                        "dataset": name,
                        "pair_id": pair["pair_id"],
                        "prompt_id": pair["prompt_id"],
                        "side": side,
                        "reward": reward,
                        "score": 1.0
                        if reward == "vr_cli" or side == "preferred"
                        else 0.0,
                        "status": "ok",
                        "error": None,
                        "preprocessing_truncated": False,
                        "native_truncated": False,
                        "statistics": {},
                    }
                    for reward in rewards
                    for side in ("preferred", "rejected")
                ]
                records.append(
                    {
                        "pair_id": pair["pair_id"],
                        "rows": rows,
                        "elapsed_seconds": 0.1,
                        "timings": {},
                    }
                )
            write_jsonl(
                tmp_path / "scores" / backend / f"{name}-shard-00000-of-00001.jsonl",
                records,
            )
    return {
        "run_dir": str(tmp_path),
        "bootstrap_samples": 50,
        "seed": 1,
        "tie_tolerance": 1e-7,
    }


def test_analysis_preserves_ties_and_undefined_constant_correlations(synthetic_run):
    result = analyze(synthetic_run)
    assert result["vr_cli_variant"] == "thresholded"
    assert result["reference_prompt_template"] == "generic"
    vr = result["summary"].query("reward == 'vr_cli'")
    assert vr.accuracy.tolist() == [0.5, 0.5]
    assert vr.tie_rate.tolist() == [1, 1]
    assert result["correlations"]["reddit_tldr"].isna().all().all()
    assert (
        Path(synthetic_run["run_dir"]) / "tables" / "candidate_scores.parquet"
    ).is_file()


@pytest.fixture
def legacy_run(synthetic_run):
    run = Path(synthetic_run["run_dir"])
    write_json(run / "config.json", synthetic_run)
    write_json(run / "environment.json", {"schema_version": 1})
    for path in (run / "scores").glob("*/*.jsonl"):
        records = read_jsonl(path)
        for record in records:
            record["rows"] = [
                row for row in record["rows"] if row["reward"] in LEGACY_REWARDS
            ]
            for row in record["rows"]:
                if row["reward"] == "reference_likelihood":
                    row["score"] = -2 if row["side"] == "preferred" else -3
                    row["statistics"] = {
                        "conditioned_logp_mean": row["score"],
                        "candidate_tokens": 1 if row["side"] == "preferred" else 100,
                    }
        write_jsonl(path, records)
    return synthetic_run


def test_legacy_reporting_and_smoke_remain_readable_but_new_runs_require_nover(legacy_run):
    from non_verifiable_rewards.experiments.runner import score

    run = Path(legacy_run["run_dir"])
    result = analyze(legacy_run)
    assert result["rewards"] == LEGACY_REWARDS
    assert "nover" not in set(result["coverage"].reward)
    assert "chrf_pp" not in set(result["coverage"].reward)
    assert "meteor" not in set(result["coverage"].reward)
    record = read_jsonl(run / "scores" / "lm" / "reddit_tldr-shard-00000-of-00001.jsonl")[0]
    write_json(run / "qwen_smoke_check.json", {"validation": record})
    assert smoke_check(legacy_run)["validation"] == record
    with pytest.raises(ValueError, match="LM reward set changed"):
        score(legacy_run, "lm")
    with pytest.raises(ValueError, match="Lexical reward set changed"):
        score(legacy_run, "lexical")
    with pytest.raises(ValueError, match="METEOR reward set changed"):
        score(legacy_run, "meteor")
    write_json(run / "environment.json", {"rewards": REWARDS})
    with pytest.raises(ValueError, match="missing"):
        analyze(legacy_run)
    with pytest.raises(ValueError, match="incomplete"):
        smoke_check(legacy_run)


@pytest.mark.parametrize("new_reward,backend", [("chrf_pp", "lexical"), ("meteor", "meteor"), ("llm_judge", "lm")])
def test_historical_manifest_preserves_scope_but_new_runs_require_new_reward(
    synthetic_run, new_reward, backend
):
    run = Path(synthetic_run["run_dir"])
    for path in (run / "scores" / backend).glob("*.jsonl"):
        records = read_jsonl(path)
        for record in records:
            record["rows"] = [row for row in record["rows"] if row["reward"] != new_reward]
        write_jsonl(path, records)
    with pytest.raises(ValueError, match="missing"):
        analyze(synthetic_run)
    historical = tuple(reward for reward in REWARDS if reward != new_reward)
    write_json(run / "environment.json", {"rewards": historical})
    assert analyze(synthetic_run)["rewards"] == historical
    # Applying the local smoke-only scope must not demand a missing new metric.
    write_json(run / "config.json", synthetic_run)
    set_lm_scope(synthetic_run)
    assert set(analyze(synthetic_run)["rewards"]) == set(PILOT_REWARDS) - {new_reward}


def test_legacy_smoke_without_completed_pair_requires_new_run(legacy_run, monkeypatch):
    run = Path(legacy_run["run_dir"])
    for path in (run / "scores" / "lm").glob("*.jsonl"):
        write_jsonl(path, [])
    write_json(run / "preprocessing.json", {})
    monkeypatch.setattr(
        "non_verifiable_rewards.experiments.assets.model_paths", lambda config: {}
    )
    with pytest.raises(ValueError, match="LM reward set changed"):
        smoke_check(legacy_run)


@pytest.mark.parametrize("task_prompt", [False, True])
def test_nover_derivation_preserves_source_scores_and_reports_new_reward(
    legacy_run, tmp_path, task_prompt
):
    from non_verifiable_rewards.experiments.io import ensure_run, file_sha256
    from non_verifiable_rewards.experiments.nover import derive_nover

    source = Path(legacy_run["run_dir"])
    if task_prompt:
        for path in (source / "scores" / "lm").glob("*.jsonl"):
            records = read_jsonl(path)
            for record in records:
                for row in record["rows"]:
                    if row["reward"] == "reference_likelihood":
                        row["statistics"]["reference_prompt_template"] = REFERENCE_PROMPT_TEMPLATE
            write_jsonl(path, records)
    before = {str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path.parent / (tmp_path.name + "-derived")
    config = derive_nover(source, output)
    after = {str(p.relative_to(source)): file_sha256(p) for p in source.rglob("*") if p.is_file()}
    assert before == after
    for path in (source / "scores" / "lm").glob("*.jsonl"):
        old_records = read_jsonl(path)
        new_records = read_jsonl(output / path.relative_to(source))
        for old, new in zip(old_records, new_records, strict=True):
            new_rows = new.pop("rows")
            assert {**new, "rows": [r for r in new_rows if r["reward"] != "nover"]} == old
            assert [r["score"] for r in new_rows if r["reward"] == "nover"] == [-2, -1]
            if task_prompt:
                assert all(
                    row["statistics"]["reference_prompt_template"] == REFERENCE_PROMPT_TEMPLATE
                    for row in new_rows if row["reward"] == "nover"
                )
    result = analyze(config)
    assert result["summary"].query("reward == 'nover'").accuracy.tolist() == [0, 0]
    assert result["vr_cli_variant"] == "thresholded"
    assert result["reference_prompt_template"] == (
        REFERENCE_PROMPT_TEMPLATE if task_prompt else "generic"
    )
    provenance = json.loads((output / "derivation.json").read_text())
    assert provenance["derived_candidate_records"] == 12
    assert provenance["source_files_sha256"] == before
    assert not provenance["new_model_inference"]
    with pytest.raises(ValueError, match="report-only"):
        ensure_run(config)
    with pytest.raises(ValueError, match="already exists"):
        derive_nover(source, output)
    with pytest.raises(ValueError, match="separate"):
        derive_nover(source, source / "derived")


def test_nover_derivation_rejects_incomplete_likelihood_statistics(legacy_run, tmp_path):
    from non_verifiable_rewards.experiments.nover import derive_nover

    source = Path(legacy_run["run_dir"])
    path = source / "scores" / "lm" / "reddit_tldr-shard-00000-of-00001.jsonl"
    records = read_jsonl(path)
    row = next(r for r in records[0]["rows"] if r["reward"] == "reference_likelihood")
    row["statistics"]["candidate_tokens"] = 0
    write_jsonl(path, records)
    output = tmp_path.parent / (tmp_path.name + "-invalid")
    with pytest.raises(ValueError, match="positive integer"):
        derive_nover(source, output)
    assert not output.exists()


@pytest.mark.parametrize("mixed", [False, True])
def test_analysis_identifies_shared_reference_template_and_rejects_mixed_runs(
    synthetic_run, mixed
):
    root = Path(synthetic_run["run_dir"]) / "scores" / "lm"
    for path in root.glob("*.jsonl"):
        records = read_jsonl(path)
        for index, record in enumerate(records):
            for row in record["rows"]:
                if row["reward"] == "reference_likelihood" and (not mixed or index == 0):
                    row["statistics"]["reference_prompt_template"] = REFERENCE_PROMPT_TEMPLATE
        write_jsonl(path, records)
    if mixed:
        with pytest.raises(ValueError, match="Mixed or unknown reference prompt templates"):
            analyze(synthetic_run)
    else:
        assert analyze(synthetic_run)["reference_prompt_template"] == REFERENCE_PROMPT_TEMPLATE


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("variant", ["continuous", "continuous_task_specific"])
def test_analysis_identifies_continuous_vrcli_and_rejects_mixed_runs(
    synthetic_run, mixed, variant
):
    root = Path(synthetic_run["run_dir"]) / "scores" / "lm"
    for path in root.glob("*.jsonl"):
        records = read_jsonl(path)
        for index, record in enumerate(records):
            for row in record["rows"]:
                if row["reward"] == "vr_cli" and (not mixed or index == 0):
                    row["score"] = 10 if row["side"] == "preferred" else 5
                    row["statistics"]["vr_cli_variant"] = variant
        write_jsonl(path, records)
    if mixed:
        with pytest.raises(ValueError, match="Mixed or unknown VR-CLI variants"):
            analyze(synthetic_run)
    else:
        result = analyze(synthetic_run)
        assert result["vr_cli_variant"] == variant
        vr = result["summary"].query("reward == 'vr_cli'")
        assert vr.accuracy.tolist() == [1, 1]
        assert vr.tie_rate.tolist() == [0, 0]


def test_vrcli_diagnostics_check_its_own_prompt_likelihoods(synthetic_run):
    import pandas as pd

    from non_verifiable_rewards.experiments.supplementary import supplementary

    run = Path(synthetic_run["run_dir"])
    write_json(run / "config.json", synthetic_run)
    for path in (run / "scores" / "lm").glob("*.jsonl"):
        records = read_jsonl(path)
        for record in records:
            for row in record["rows"]:
                if row["reward"] == "vr_cli":
                    row["score"] = -20 if row["side"] == "preferred" else -10
                    row["statistics"] = {
                        "vr_cli_variant": "continuous_task_specific",
                        "conditioned_logp_mean": -3 - math.log1p(-row["score"] / 100),
                    }
        write_jsonl(path, records)
    analyze(synthetic_run)
    supplementary(run, run)
    diagnostics = pd.read_csv(run / "tables" / "vr_cli_information_loss.csv")
    assert diagnostics.ranking_inversions.eq(0).all()
    assert diagnostics.reference_likelihood_decision_agreement.eq(0).all()
    assert diagnostics.raw_likelihood_accuracy.eq(0).all()
    assert diagnostics.vr_cli_accuracy.eq(0).all()


def test_analysis_rejects_overlapping_shards(synthetic_run):
    root = Path(synthetic_run["run_dir"]) / "scores" / "lexical"
    source = root / "reddit_tldr-shard-00000-of-00001.jsonl"
    (root / "reddit_tldr-shard-00000-of-00002.jsonl").write_text(source.read_text())
    with pytest.raises(ValueError, match="Duplicate"):
        load_results(synthetic_run["run_dir"])


def test_supplementary_without_a_nested_pilot(synthetic_run):
    from non_verifiable_rewards.experiments.supplementary import supplementary

    run = Path(synthetic_run["run_dir"])
    write_json(run / "config.json", synthetic_run)
    analyze(synthetic_run)
    supplementary(run)
    import pandas as pd

    comparisons = pd.read_parquet(run / "tables" / "all_paired_comparisons.parquet")
    assert len(comparisons) == len(DATASETS) * len(REWARDS) * (len(REWARDS) - 1) // 2
    assert (run / "tables" / "prompt_weighting_sensitivity.parquet").exists()
    assert (run / "tables" / "encoder_truncation_diagnostics.parquet").exists()
    assert not (run / "tables" / "pilot_overlap.parquet").exists()


def test_analysis_rejects_incomplete_runs(synthetic_run):
    path = (
        Path(synthetic_run["run_dir"])
        / "scores"
        / "lexical"
        / "reddit_tldr-shard-00000-of-00001.jsonl"
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["rows"].pop()
    write_jsonl(path, records)
    with pytest.raises(ValueError, match="missing"):
        load_results(synthetic_run["run_dir"])


def test_partial_lm_requires_explicit_scope_and_is_retained_outside_summary(
    synthetic_run,
):
    run = Path(synthetic_run["run_dir"])
    for path in (run / "scores" / "lm").glob("*-shard-*.jsonl"):
        write_jsonl(path, read_jsonl(path)[:1])
    with pytest.raises(ValueError, match="missing"):
        analyze(synthetic_run)
    write_json(run / "config.json", synthetic_run)
    set_lm_scope(synthetic_run)
    assert lm_mode(synthetic_run) == "smoke"
    result = analyze(synthetic_run)
    assert set(result["summary"].reward) == set(PILOT_REWARDS)
    assert result["summary"].pairs.eq(3).all()
    partial = result["scores"].query("reward == 'reference_likelihood'")
    assert len(partial) == 4
    assert not partial.included_in_primary_benchmark.any()
    assert result["correlations"]["reddit_tldr"].shape == (len(PILOT_REWARDS),) * 2
    coverage = result["coverage"].query("reward == 'reference_likelihood'")
    assert coverage.complete_pairs.tolist() == [1, 1]
    assert coverage.common_pairs.tolist() == [0, 0]
    required_path = run / "scores" / "bleurt" / "reddit_tldr-shard-00000-of-00001.jsonl"
    write_jsonl(required_path, read_jsonl(required_path)[:1])
    with pytest.raises(ValueError, match="missing"):
        analyze(synthetic_run)


def test_smoke_reuses_success_without_loading_model(synthetic_run, monkeypatch):
    run = Path(synthetic_run["run_dir"])
    write_json(run / "config.json", synthetic_run)
    record = read_jsonl(
        run / "scores" / "lm" / "reddit_tldr-shard-00000-of-00001.jsonl"
    )[0]
    write_json(run / "qwen_smoke_check.json", {"validation": record})

    def forbidden(*args, **kwargs):
        pytest.fail("A saved successful smoke check must not load models")

    monkeypatch.setattr("non_verifiable_rewards.experiments.runner.Backend", forbidden)
    monkeypatch.setattr(
        "non_verifiable_rewards.experiments.assets.model_paths", forbidden
    )
    assert smoke_check(synthetic_run)["validation"] == record


def test_fresh_smoke_scores_only_one_pair_and_keeps_it_separate(
    synthetic_run, monkeypatch
):
    run = Path(synthetic_run["run_dir"])
    write_json(run / "config.json", synthetic_run)
    write_json(run / "preprocessing.json", {})
    templates = read_jsonl(
        run / "scores" / "lm" / "reddit_tldr-shard-00000-of-00001.jsonl"
    )
    for path in (run / "scores" / "lm").glob("*.jsonl"):
        write_jsonl(path, [])
    for name in DATASETS:
        write_jsonl(
            run / "scored_text" / f"{name}.jsonl",
            read_jsonl(run / "data" / f"{name}.jsonl"),
        )
    evaluated = []

    class FakeLM:
        def __init__(self, *args):
            pass

        def evaluate(self, pair, original, baselines):
            evaluated.append(pair["pair_id"])
            return templates[0]["rows"], {}

    monkeypatch.setattr("non_verifiable_rewards.experiments.runner.Backend", FakeLM)
    monkeypatch.setattr(
        "non_verifiable_rewards.experiments.assets.model_paths",
        lambda config: {"lm": {"id": "test"}},
    )
    saved = smoke_check(synthetic_run)
    assert evaluated == [templates[0]["pair_id"]]
    assert len(saved["validation"]["rows"]) == 2 * len(BACKEND_REWARDS["lm"])
    assert saved["scored_input"]["pair_id"] == saved["input"]["pair_id"]
    assert smoke_check(synthetic_run) == saved
    assert len(evaluated) == 1
    assert all(not read_jsonl(path) for path in (run / "scores" / "lm").glob("*.jsonl"))


def test_lm_scope_blocks_full_lm_scoring_on_rerun(synthetic_run, monkeypatch):
    from non_verifiable_rewards.experiments.cli import main

    run = Path(synthetic_run["run_dir"])
    config = {**synthetic_run, "threads": 1, "cache_dir": str(run / "cache")}
    write_json(run / "config.json", config)
    set_lm_scope(config, skip=True)
    assert lm_mode(config) == "skip"
    monkeypatch.setattr(
        "non_verifiable_rewards.experiments.cli.load_config", lambda path: config
    )
    monkeypatch.setattr("sys.argv", ["nvr", "score", "--backend", "lm"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
