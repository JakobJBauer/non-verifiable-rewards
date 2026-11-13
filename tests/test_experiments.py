import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from non_verifiable_rewards.experiments.analysis import (
    analyze,
    cluster_bootstrap,
    load_results,
    preference_credit,
)
from non_verifiable_rewards.experiments.data import (
    litbench_pairs,
    make_pair,
    reddit_pairs,
    sample_pairs,
)
from non_verifiable_rewards.experiments.io import (
    BACKEND_REWARDS,
    DATASETS,
    Journal,
    read_jsonl,
    write_json,
    write_jsonl,
)
from non_verifiable_rewards.experiments.runner import Backend
from non_verifiable_rewards.experiments.smoke import (
    PILOT_REWARDS,
    lm_mode,
    set_lm_scope,
    smoke_check,
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
    assert all(not row["preprocessing_truncated"] for row in rows)


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
    vr = result["summary"].query("reward == 'vr_cli'")
    assert vr.accuracy.tolist() == [0.5, 0.5]
    assert vr.tie_rate.tolist() == [1, 1]
    assert result["correlations"]["reddit_tldr"].isna().all().all()
    assert (
        Path(synthetic_run["run_dir"]) / "tables" / "candidate_scores.parquet"
    ).is_file()


def test_analysis_rejects_overlapping_shards(synthetic_run):
    root = Path(synthetic_run["run_dir"]) / "scores" / "lexical"
    source = root / "reddit_tldr-shard-00000-of-00001.jsonl"
    (root / "reddit_tldr-shard-00000-of-00002.jsonl").write_text(source.read_text())
    with pytest.raises(ValueError, match="Duplicate"):
        load_results(synthetic_run["run_dir"])


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
    assert result["correlations"]["reddit_tldr"].shape == (4, 4)
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
    assert len(saved["validation"]["rows"]) == 6
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
