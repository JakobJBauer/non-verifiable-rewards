import json

import numpy as np
import pandas as pd
import pytest

from non_verifiable_rewards._hashing import digest, file_sha256
from non_verifiable_rewards.experiments.io import write_json, write_jsonl
from non_verifiable_rewards.experiments.judge_report import (
    DATASETS,
    analyze,
    freeze_observed,
    load_completed,
    paired_statistics,
)


@pytest.fixture
def completed_run(tmp_path):
    candidates, pairs, journals = [], [], {"control": [], "thinking": []}
    for dataset in DATASETS:
        for i in range(3):
            key = f"{dataset}:{i}"
            candidates.append(
                dict(key=key, dataset=dataset, historical_context_sha256=key)
            )
            for condition in journals:
                probabilities = [0.0, 0.0, 0.0, 0.0, 0.0]
                probabilities[3 - i] = 1.0
                journals[condition].append(
                    dict(
                        key=key,
                        dataset=dataset,
                        thinking=condition == "thinking",
                        status="ok",
                        score=(3 - i) / 4,
                        prompt_sha256=key,
                        rating_probabilities=probabilities,
                        rating_token_mass=0.9,
                        thinking_tokens=2 if condition == "thinking" else 0,
                        thinking_token_ids=[7, 9] if condition == "thinking" else [],
                        generation_batch_seconds=2 if condition == "thinking" else 0,
                        generation_batch_size=3,
                        scoring_batch_seconds=1,
                        scoring_batch_size=3,
                    )
                )
        # Candidate 0 occurs twice, and both pairs belong to the same prompt.
        for i in (1, 2):
            pairs.append(
                dict(
                    dataset=dataset,
                    pair_id=str(i),
                    prompt_id="shared",
                    preferred_key=f"{dataset}:0",
                    rejected_key=f"{dataset}:{i}",
                    preferred_historical_score=0.75,
                    rejected_historical_score=(3 - i) / 4,
                    preferred_words=10,
                    rejected_words=5,
                )
            )
    manifest = dict(
        config=dict(
            bootstrap_samples=100,
            seed=1729,
            tie_tolerance=1e-7,
            max_thinking_tokens=4096,
        ),
        model=dict(revision="SYNTHETIC TEST FIXTURE"),
        pairs=len(pairs),
        unique_candidates=len(candidates),
        pairs_by_dataset={d: 2 for d in DATASETS},
        candidates_sha256=digest(candidates),
        pairs_sha256=digest(pairs),
        historical_prompt_hashes_checked=True,
    )
    write_json(tmp_path / "manifest.json", manifest)
    write_jsonl(tmp_path / "candidates.jsonl", candidates)
    write_jsonl(tmp_path / "pairs.jsonl", pairs)
    for name, rows in journals.items():
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    write_json(
        tmp_path / "scoring_identity.json",
        dict(
            manifest_sha256=file_sha256(tmp_path / "manifest.json"),
            packages={p: "test" for p in ("torch", "vllm", "transformers")},
        ),
    )
    write_json(tmp_path / "run_status.json", {"complete": True})
    return tmp_path


def test_report_expands_repeated_candidates_and_does_not_double_count_time(
    completed_run,
):
    frame, journals, provenance = load_completed(completed_run)
    assert len(frame) == 6 and len(journals["control"]) == 9
    assert (frame.thinking_credit == 1).all()
    result = analyze(frame, journals, provenance["manifest"]["config"])
    assert result["diagnostics"]["thinking"]["all"][
        "inference_seconds"
    ] == pytest.approx(9)
    assert result["diagnostics"]["control"]["all"][
        "inference_seconds"
    ] == pytest.approx(3)
    assert result["macro"]["thinking_minus_control"]["ci95"] == [0.0, 0.0]


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "duplicate",
        "wrong_condition",
        "wrong_prompt",
        "invalid_probability",
        "torn_tail",
        "changed_input",
    ],
)
def test_report_rejects_incomplete_or_inconsistent_results(completed_run, corruption):
    path = completed_run / "control.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "wrong_condition":
        rows[0]["thinking"] = True
    elif corruption == "wrong_prompt":
        rows[0]["prompt_sha256"] = "changed"
    elif corruption == "invalid_probability":
        rows[0]["rating_probabilities"] = [float("nan")] * 5
    elif corruption == "changed_input":
        with (completed_run / "pairs.jsonl").open("a") as stream:
            stream.write("{}\n")
    if corruption == "invalid_probability":
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        write_jsonl(path, rows)
    if corruption == "torn_tail":
        path.write_text(path.read_text().rstrip())
    with pytest.raises(ValueError):
        load_completed(completed_run)


def test_failed_unique_candidate_invalidates_every_affected_pair(completed_run):
    path = completed_run / "thinking.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(status="thinking_limit", score=None)
    write_jsonl(path, rows)
    frame, _, _ = load_completed(completed_run)
    assert frame.thinking_credit.isna().sum() == 2
    assert (frame.control_credit == 1).all()
    assert len(frame) == 6


def test_paired_bootstrap_and_missing_pair_bounds():
    frame = pd.DataFrame(
        {
            "prompt_id": ["a", "a", "b", "b"],
            "historical_credit": [1.0, 0.0, 1.0, 0.0],
            "control_credit": [1.0, 0.0, 1.0, 0.0],
            "thinking_credit": [1.0, 0.0, np.nan, 0.0],
            "historical_preferred": [0.8] * 4,
            "historical_rejected": [0.2] * 4,
            "control_preferred": [0.8] * 4,
            "control_rejected": [0.2] * 4,
        }
    )
    result, _ = paired_statistics(frame, samples=500, seed=17)
    # Marginal accuracy varies across bootstrap draws; paired delta is exactly zero.
    assert (
        result["conditions"]["control"]["ci95"][0]
        < result["conditions"]["control"]["ci95"][1]
    )
    assert result["thinking_minus_control"] == {"estimate": 0.0, "ci95": [0.0, 0.0]}
    assert result["matched_pairs"] == 3
    assert result["scope_delta_bounds"] == [-0.25, 0.0]
    assert result["scope_sample"]["thinking"]["accuracy_bounds"] == [0.25, 0.5]
    assert np.sum(result["transitions"]) == 3


def test_repeated_prompt_pairs_are_resampled_together():
    # A single prompt cluster cannot be split into independent observations.
    frame = pd.DataFrame(
        {
            "prompt_id": ["same", "same"],
            "historical_credit": [0.0, 1.0],
            "control_credit": [0.0, 1.0],
            "thinking_credit": [1.0, 1.0],
            "historical_preferred": [0.1, 0.9],
            "historical_rejected": [0.9, 0.1],
            "control_preferred": [0.1, 0.9],
            "control_rejected": [0.9, 0.1],
        }
    )
    result, _ = paired_statistics(frame, samples=100, seed=19)
    assert result["thinking_minus_control"] == {"estimate": 0.5, "ci95": [0.5, 0.5]}


def test_explicit_stopped_scope_distinguishes_unattempted_from_failed(completed_run):
    path = completed_run / "thinking.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()][:-1]
    rows[1].update(status="thinking_limit", score=None)
    write_jsonl(path, rows)
    write_json(completed_run / "run_status.json", {"complete": False})
    with pytest.raises(ValueError, match="incomplete"):
        load_completed(completed_run)
    scope = freeze_observed(completed_run, reason="User requested stopping inference")
    assert scope["observed_thinking_candidates"] == 8
    frame, journals, provenance = load_completed(completed_run)
    result = analyze(frame, journals, provenance["manifest"]["config"])
    assert len(frame) == 6
    assert frame.in_analysis_scope.sum() == 5
    assert result["unattempted_pairs"] == 1
    assert result["macro"]["matched_pairs"] == 4
    # Equal context sets are used for timing, despite the larger control journal.
    assert result["diagnostics"]["control"]["all"]["unique_candidates"] == 8
    assert result["diagnostics"]["control"]["all"][
        "inference_seconds"
    ] == pytest.approx(8 / 3)
    assert (
        sum(
            s["scope_sample"]["thinking"]["failed_pairs"]
            for s in result["datasets"].values()
        )
        == 1
    )
    with path.open("a") as stream:
        stream.write(json.dumps(rows[-1]) + "\n")
    with pytest.raises(ValueError, match="changed after freezing"):
        load_completed(completed_run)


def test_stopped_scope_rejects_arbitrary_missing_contexts(completed_run):
    path = completed_run / "thinking.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()][1:]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="not the frozen input prefix"):
        freeze_observed(completed_run, reason="Requested stop")
    assert not (completed_run / "analysis_scope.json").exists()
