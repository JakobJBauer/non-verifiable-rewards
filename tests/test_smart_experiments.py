"""Opt-in SMART workers and reporting alongside existing rewards."""

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from non_verifiable_rewards.data.pairs import make_pair
from non_verifiable_rewards.experiments.io import (
    ALL_REWARDS,
    BACKEND_REWARDS,
    DATASETS,
    REWARDS,
    SMART_REWARDS,
    configured_rewards,
    enabled_backends,
    ensure_run,
    run_rewards,
    write_json,
    write_jsonl,
)
from non_verifiable_rewards.experiments.runner import Backend, failure_rows, score


def test_smart_is_opt_in_and_recorded_in_run_identity(tmp_path):
    config = {"run_dir": str(tmp_path / "enabled"), "include_smart_rewards": True}
    assert configured_rewards({}) == REWARDS
    assert configured_rewards(config) == ALL_REWARDS
    assert not set(SMART_REWARDS) & set(enabled_backends({}))
    assert set(SMART_REWARDS) <= set(enabled_backends(config))
    assert run_rewards(ensure_run(config)) == ALL_REWARDS
    with pytest.raises(ValueError, match="boolean"):
        configured_rewards({"include_smart_rewards": "false"})
    with pytest.raises(ValueError, match="disabled"):
        score({"run_dir": str(tmp_path / "disabled")}, "smart1_bleurt")
    assert not (tmp_path / "disabled").exists()


@pytest.mark.parametrize("backend", SMART_REWARDS)
def test_smart_worker_scores_full_original_text_and_records_details(backend):
    original = make_pair("litbench", "p", "Prompt", "A. B.", "C. D.", "A. D.", "r", {})
    pair = {
        **original,
        "preferred": "A.",
        "rejected": "C.",
        "reference": "A.",
        "reference_prefix_overlaps_candidate": True,
    }
    pair["lengths"] = {
        field: {
            "original_words": len(original[field].split()),
            "scored_words": 1,
            "truncated": True,
        }
        for field in ("prompt", "preferred", "rejected", "reference")
    }
    calls = []

    def evaluate(candidates, *, references):
        calls.append((candidates, references))
        return [
            {
                "f1": value,
                "precision": value,
                "recall": value,
                "sentence_pairs": 4,
                "candidate_segments": [text],
            }
            for value, text in zip((0.8, 0.4), candidates, strict=True)
        ]

    engine = Backend.__new__(Backend)
    engine.name = backend
    engine.reward = SimpleNamespace(score_with_details=evaluate)
    rows, _ = engine.evaluate(pair, original)
    assert calls == [(["A. B.", "C. D."], ["A. D."] * 2)]
    assert [r["score"] for r in rows] == [0.8, 0.4]
    assert all(r["reward"] == backend for r in rows)
    assert all(r["scored_candidate_words"] == 2 for r in rows)
    assert all(
        not r["preprocessing_truncated"] and not r["native_truncated"] for r in rows
    )
    assert all(not r["reference_prefix_overlaps_candidate"] for r in rows)
    assert all(r["statistics"]["sentence_pairs"] == 4 for r in rows)
    assert [r["reward"] for r in failure_rows(pair, backend, "failure")] == [
        backend
    ] * 2


def test_punkt_asset_download_is_cached_and_tampering_is_detected(
    tmp_path, monkeypatch
):
    from non_verifiable_rewards.experiments import assets

    archive = tmp_path / "resource.zip"
    files = [
        "collocations.tab",
        "sent_starters.txt",
        "abbrev_types.txt",
        "ortho_context.tab",
    ]
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in files:
            bundle.writestr(f"punkt_tab/english/{name}", "")
    monkeypatch.setattr(assets, "download_file", lambda *args: archive)
    punkt = assets.download_punkt(tmp_path)
    assert assets.download_punkt(tmp_path) == punkt
    manifest = {key: {"path": str(tmp_path)} for key in ("lm", "bleurt", "bertscore")}
    manifest["punkt"] = punkt
    write_json(tmp_path / "models.json", manifest)
    config = {"run_dir": str(tmp_path), "include_smart_rewards": True}
    assert assets.model_paths(config) == manifest
    (Path(punkt["path"]) / files[0]).write_text("changed")
    with pytest.raises(ValueError, match="tokenizer changed"):
        assets.model_paths(config)


@pytest.mark.parametrize("enabled", [False, True])
def test_analysis_includes_smart_only_when_enabled(tmp_path, enabled):
    pytest.importorskip("scipy")
    from non_verifiable_rewards.experiments.analysis import analyze, load_results
    from non_verifiable_rewards.experiments.report import COLORS, LABELS, runtime_figure
    from non_verifiable_rewards.experiments.smoke import set_lm_scope

    config = {
        "run_dir": str(tmp_path),
        "include_smart_rewards": enabled,
        "bootstrap_samples": 10,
        "seed": 42,
        "tie_tolerance": 1e-7,
    }
    ensure_run(config)
    for name in DATASETS:
        pair = make_pair(name, "p", "Prompt", "A.", "B.", "C.", "r", {})
        write_jsonl(tmp_path / "data" / f"{name}.jsonl", [pair])
        for backend in enabled_backends(config):
            rows = [
                {
                    "dataset": name,
                    "pair_id": pair["pair_id"],
                    "prompt_id": "p",
                    "side": side,
                    "reward": reward,
                    "score": float(side == "preferred"),
                    "status": "ok",
                    "preprocessing_truncated": False,
                    "native_truncated": False,
                    "statistics": {},
                }
                for reward in BACKEND_REWARDS[backend]
                for side in ("preferred", "rejected")
            ]
            write_jsonl(
                tmp_path / "scores" / backend / f"{name}-shard-00000-of-00001.jsonl",
                [
                    {
                        "pair_id": pair["pair_id"],
                        "rows": rows,
                        "elapsed_seconds": 0.1,
                        "timings": {},
                    }
                ],
            )
    result = analyze(config)
    expected = configured_rewards(config)
    assert result["rewards"] == expected
    assert set(result["summary"].reward) == set(expected)
    assert set(result["coverage"].reward) == set(expected)
    assert all(reward in LABELS and reward in COLORS for reward in expected)
    figure_config = {
        **config,
        "name": "test",
        "pilot": False,
        "models": {
            "lm": {"id": "test"},
            "bertscore": {"id": "test"},
            "bleurt": {"name": "test"},
        },
    }
    runtime_figure(result, figure_config)
    assert (tmp_path / "figures" / "runtime.png").exists()
    set_lm_scope(config, skip=True)
    scoped = analyze(config)
    assert (
        set(SMART_REWARDS) <= set(scoped["rewards"])
        if enabled
        else not set(SMART_REWARDS) & set(scoped["rewards"])
    )
    if enabled:
        path = (
            tmp_path
            / "scores"
            / "smart1_bleurt"
            / "litbench-shard-00000-of-00001.jsonl"
        )
        write_jsonl(path, [])
        with pytest.raises(ValueError, match="missing"):
            load_results(tmp_path)
