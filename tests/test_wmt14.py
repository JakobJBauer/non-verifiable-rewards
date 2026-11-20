import csv
import io
import json
import tarfile
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from non_verifiable_rewards.data import (
    load_wmt14,
    load_wmt14_parallel,
    load_wmt14_preferences,
)
from non_verifiable_rewards.data.pairs import make_pair
from non_verifiable_rewards.data.wmt14 import (
    JUDGMENTS_URL,
    SUBMISSIONS_URL,
    wmt14_pairs,
)
from non_verifiable_rewards.experiments.io import (
    DATASETS,
    LEGACY_DATASETS,
    configured_datasets,
    read_jsonl,
    run_datasets,
    write_json,
)
from non_verifiable_rewards.rewards.prompts import reference_context, translation_prompt


def ranking(segment=1, ranks=(1, 2, 3, 4, 5), judge="annotator1"):
    return {
        "srclang": "German",
        "trglang": "English",
        "srcIndex": str(segment),
        "segmentId": str(segment),
        "judgeId": judge,
        **{f"system{i}Id": f"system{i}.de-en" for i in range(1, 6)},
        **{f"system{i}rank": str(rank) for i, rank in enumerate(ranks, 1)},
    }


def write_archive(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, text in members.items():
            data = text.encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


@pytest.fixture
def wmt_sources(tmp_path):
    sources, references = ["Ein Hund.", "Eine Katze."], ["A dog.", "A cat."]
    outputs = {
        f"system{i}.de-en": [f"Dog translation {i}.", f"Cat translation {i}."]
        for i in range(1, 6)
    }
    rows = [
        ranking(),
        ranking(judge="annotator2"),
        ranking(ranks=(5, 4, 3, 2, 1)),
        ranking(2),
    ]
    submissions, judgments = tmp_path / "submissions.tgz", tmp_path / "judgments.tgz"
    members = {
        "wmt14-data/txt/sources/newstest2014-deen-src.de": "\n".join(sources) + "\n",
        "wmt14-data/txt/references/newstest2014-deen-ref.en": "\n".join(references)
        + "\n",
        **{
            f"wmt14-data/txt/system-outputs/newstest2014/de-en/{name}": "\n".join(lines)
            + "\n"
            for name, lines in outputs.items()
        },
    }
    write_archive(submissions, members)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
    write_archive(
        judgments,
        {"wmt14-manual-evaluation/wmt14-judgments-anonymized.csv": stream.getvalue()},
    )
    return submissions, judgments, sources, references, outputs, rows, members


def test_official_archive_join_orientation_and_disagreement(wmt_sources):
    submissions, judgments, *_ = wmt_sources
    data, metadata = load_wmt14_preferences(
        submissions_archive=submissions,
        judgments_archive=judgments,
        return_metadata=True,
    )
    assert isinstance(data, DatasetDict)
    pairs = data["test"]
    assert len(pairs) == 30
    first = pairs[0]
    assert first["prompt"] == translation_prompt("Ein Hund.")
    assert first["reference"] == "A dog."
    assert first["preferred"] == "Dog translation 1."
    assert first["rejected"] == "Dog translation 2."
    assert len(first["metadata"]["judgments"]) == 2
    assert first["metadata"]["opposing_judgments"] == 1
    assert first["reference"] not in first["prompt"]
    audit = metadata["test"]["audit"]
    assert audit["eligible_judgments"] == 40
    assert audit["merged_duplicate_labeled_pair"] == 10
    assert audit["labeled_pairs_with_opposing_judgments"] == 20
    assert audit["eligible_prompts"] == 2
    assert all(
        len(source["sha256"]) == 64 for source in metadata["test"]["sources"].values()
    )


def test_preference_default_downloads_and_single_split(monkeypatch, wmt_sources):
    submissions, judgments, *_ = wmt_sources
    requested = []

    def download(self, url):
        requested.append(url)
        return {SUBMISSIONS_URL: submissions, JUDGMENTS_URL: judgments}[url]

    monkeypatch.setattr(
        "non_verifiable_rewards.data.wmt14.DownloadManager.download", download
    )
    assert len(load_wmt14_preferences(split="test")) == 30
    assert requested == [SUBMISSIONS_URL, JUDGMENTS_URL]
    with pytest.raises(ValueError, match="test-only"):
        load_wmt14_preferences(split="train")


def test_rank_filters_and_text_overlap(wmt_sources):
    _, _, sources, references, outputs, _, _ = wmt_sources
    rows = [ranking(ranks=(1, 2, 2, -1, 5))]
    pairs, audit = wmt14_pairs(sources, references, outputs, rows)
    assert len(pairs) == 5
    assert audit["excluded_unranked"] == 4
    assert audit["excluded_human_tie"] == 1
    outputs["system5.de-en"][0] = references[0]
    outputs["system3.de-en"][0] = outputs["system1.de-en"][0]
    pairs, audit = wmt14_pairs(sources, references, outputs, rows)
    assert len(pairs) == 1
    assert audit["excluded_empty_or_overlapping_texts"] == 4
    assert pairs[0]["preferred"] == "Dog translation 1."


@pytest.mark.parametrize(
    "change",
    [
        {"srcIndex": "0"},
        {"segmentId": "2"},
        {"system1Id": "missing"},
        {"system1rank": "0"},
    ],
)
def test_bad_alignment_and_judgments_fail(wmt_sources, change):
    _, _, sources, references, outputs, _, _ = wmt_sources
    with pytest.raises(ValueError):
        wmt14_pairs(sources, references, outputs, [{**ranking(), **change}])


def test_truncated_archive_and_empty_selection(wmt_sources):
    submissions, judgments, _, _, _, _, members = wmt_sources
    members["wmt14-data/txt/system-outputs/newstest2014/de-en/system1.de-en"] = (
        "Only one line\n"
    )
    write_archive(submissions, members)
    with pytest.raises(ValueError, match="align"):
        load_wmt14_preferences(
            submissions_archive=submissions, judgments_archive=judgments
        )
    # An all-tie source must still expose the complete pair schema.
    stream = io.StringIO()
    row = ranking(ranks=(1, 1, 1, 1, 1))
    writer = csv.DictWriter(stream, fieldnames=row)
    writer.writeheader()
    writer.writerow(row)
    write_archive(
        judgments,
        {"wmt14-manual-evaluation/wmt14-judgments-anonymized.csv": stream.getvalue()},
    )
    members["wmt14-data/txt/system-outputs/newstest2014/de-en/system1.de-en"] = "A\nB\n"
    write_archive(submissions, members)
    data = load_wmt14_preferences(
        split="test", submissions_archive=submissions, judgments_archive=judgments
    )
    assert len(data) == 0
    assert {"prompt", "reference", "preferred", "rejected"} <= set(data.column_names)


@pytest.mark.parametrize("streaming", [False, True])
def test_parallel_training_has_references_but_no_preferences(monkeypatch, streaming):
    def load(path, name, **kwargs):
        assert (path, name) == ("wmt/wmt14", "de-en")
        assert kwargs["split"] == "train"
        assert kwargs["streaming"] is streaming
        assert kwargs["revision"] == "pinned"
        data = Dataset.from_list(
            [
                {"translation": {"de": " Ein Hund. ", "en": " A dog. "}},
                {"translation": {"de": "", "en": "No source"}},
                {"translation": {"de": "No target", "en": " "}},
                {"translation": {"de": "Eine Katze.", "en": "A cat."}},
            ]
        )
        return data.to_iterable_dataset() if streaming else data

    monkeypatch.setattr("non_verifiable_rewards.data.wmt14.load_dataset", load)
    rows = list(load_wmt14_parallel(streaming=streaming, revision="pinned"))
    assert len(rows) == 2
    assert [r["metadata"]["source_row"] for r in rows] == [0, 3]
    assert rows[0]["reference"] == "A dog."
    assert rows[0]["prompt"] == translation_prompt("Ein Hund.")
    assert all("preferred" not in r and "rejected" not in r for r in rows)
    prompt = rows[0]["prompt"]
    conditioned = reference_context(prompt, "Candidate", task="translation_de_en")
    assert conditioned.count("Translate the following German text into English:") == 1
    assert conditioned == reference_context(
        "Ein Hund.", "Candidate", task="translation_de_en"
    )
    assert "A dog." not in conditioned


@pytest.mark.parametrize("streaming", [False, True])
def test_task_train_excludes_validation_and_official_test_sources(
    monkeypatch, wmt_sources, streaming
):
    submissions, _, *_ = wmt_sources

    def load(path, name, **kwargs):
        assert (path, name) == ("wmt/wmt14", "de-en")
        assert kwargs["revision"] == "pinned"
        rows = [
            {"translation": {"de": " Ein Hund. ", "en": "Official test overlap"}},
            {"translation": {"de": "Ein Vogel.", "en": "Validation source"}},
        ]
        if kwargs["split"] == "train":
            rows.append({"translation": {"de": "Eine Maus.", "en": "A mouse."}})
        data = Dataset.from_list(rows)
        return data.to_iterable_dataset() if kwargs["streaming"] else data

    monkeypatch.setattr("non_verifiable_rewards.data.wmt14.load_dataset", load)
    data, metadata = load_wmt14(
        streaming=streaming,
        revision="pinned",
        submissions_archive=submissions,
        return_metadata=True,
    )
    rows = list(data)
    assert len(rows) == 1
    assert rows[0]["prompt"] == translation_prompt("Eine Maus.")
    assert rows[0]["metadata"]["source_row"] == 2
    assert "preferred" not in rows[0]
    assert metadata["train"]["audit"]["excluded_heldout_source_overlap"] == (
        None if streaming else 2
    )
    validation = list(
        load_wmt14(
            split="validation",
            streaming=streaming,
            revision="pinned",
            submissions_archive=submissions,
        )
    )
    assert len(validation) == 1
    assert validation[0]["prompt"] == translation_prompt("Ein Vogel.")
    assert validation[0]["metadata"]["source_row"] == 1


def test_dataset_scope_preserves_historical_runs_and_rejects_invalid_values(tmp_path):
    assert configured_datasets({}) == LEGACY_DATASETS
    assert run_datasets(tmp_path) == LEGACY_DATASETS
    write_json(tmp_path / "data_manifest.json", {name: {} for name in LEGACY_DATASETS})
    assert run_datasets(tmp_path) == LEGACY_DATASETS
    write_json(tmp_path / "data_manifest.json", {"datasets": ["wmt14_de_en"]})
    assert run_datasets(tmp_path) == ("wmt14_de_en",)
    write_json(tmp_path / "config.json", {})
    with pytest.raises(ValueError, match="Dataset scope differs"):
        run_datasets(tmp_path)
    for value in ([], "wmt14_de_en", ["unknown"], ["litbench", "litbench"]):
        with pytest.raises(ValueError, match="distinct known"):
            configured_datasets({"datasets": value})


def test_public_test_split_is_ranked_pairs(wmt_sources):
    submissions, judgments, *_ = wmt_sources
    data, metadata = load_wmt14(
        split="test",
        submissions_archive=submissions,
        judgments_archive=judgments,
        return_metadata=True,
    )
    assert len(data) > 0
    assert {"preferred", "rejected", "reference"} <= set(data.column_names)
    assert set(metadata) == {"test"}
    assert "ranked_test" in metadata["test"]["split_policy"]
    for kwargs in ({"streaming": True}, {"revision": "main"}):
        with pytest.raises(ValueError, match="does not support"):
            load_wmt14(split="test", **kwargs)
    with pytest.raises(ValueError, match="only to annotated test"):
        load_wmt14(split="train", judgments_archive=judgments)
    with pytest.raises(ValueError, match="split must"):
        load_wmt14(split="invalid")


def test_unannotated_wmt_test_remains_explicitly_available(monkeypatch):
    def load(path, name, **kwargs):
        assert kwargs["split"] == "test"
        return Dataset.from_list([{"translation": {"de": "Hund", "en": "Dog"}}])

    monkeypatch.setattr("non_verifiable_rewards.data.wmt14.load_dataset", load)
    data = load_wmt14_parallel(split="test")
    assert data[0]["reference"] == "Dog"
    assert "preferred" not in data.column_names


@pytest.mark.parametrize("datasets", [None, ["wmt14_de_en"], list(DATASETS)])
def test_prepare_score_and_report_selected_datasets(
    tmp_path, monkeypatch, wmt_sources, datasets
):
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    from non_verifiable_rewards.experiments.data import prepare, preprocess
    from non_verifiable_rewards.experiments.report import generate_report
    from non_verifiable_rewards.experiments.runner import score

    config = {
        "run_dir": str(tmp_path / "run"),
        "cache_dir": str(tmp_path / "cache"),
        "name": "Dataset integration",
        "seed": 7,
        "pairs_per_dataset": 2,
        "max_prompt_tokens": 0,
        "max_candidate_tokens": 0,
        "max_reference_tokens": 0,
        "bootstrap_samples": 10,
        "tie_tolerance": 1e-7,
        "pilot": False,
        "models": {
            "lm": {"id": "test"},
            "bertscore": {"id": "test"},
            "bleurt": {"name": "test"},
        },
    }
    if datasets is not None:
        config["datasets"] = datasets
    names = configured_datasets(config)

    def other_loader(name, split):
        def load(**kwargs):
            assert name in names
            assert kwargs["split"] == split
            pairs = [
                make_pair(
                    name,
                    str(i),
                    f"Prompt {i}",
                    "A dog",
                    "Dog two",
                    "The dog",
                    "r",
                    {"source_row": i},
                )
                for i in range(2)
            ]
            return Dataset.from_list(pairs), {
                split: {
                    "audit": {"eligible_pairs": 2},
                    "sources": {},
                    "split_policy": "fixture",
                }
            }

        return load

    monkeypatch.setattr(
        "non_verifiable_rewards.experiments.data.load_sff",
        other_loader("reddit_tldr", "test"),
    )
    monkeypatch.setattr(
        "non_verifiable_rewards.experiments.data.load_litbench",
        other_loader("litbench", "test"),
    )
    submissions, judgments, *_ = wmt_sources
    prepare(config, wmt14_submissions=submissions, wmt14_judgments=judgments)
    run = Path(config["run_dir"])
    assert set(path.stem for path in (run / "data").glob("*.jsonl")) == set(names)
    assert run_datasets(run) == names
    assert json.loads((run / "environment.json").read_text())["datasets"] == list(names)
    prepare(config)  # Reuse frozen files without downloading.

    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

    models = {"lm": {"id": "test", "revision": "fixed"}}
    write_json(run / "models.json", models)
    preprocess(config, Tokenizer(), models)
    monkeypatch.setattr(
        "non_verifiable_rewards.experiments.runner.model_paths", lambda config: models
    )
    score(config, "lexical")
    for name in names:
        records = read_jsonl(
            run / "scores" / "lexical" / f"{name}-shard-00000-of-00001.jsonl"
        )
        assert len(records) == 2
        assert all(r["status"] == "ok" for record in records for r in record["rows"])
    with pytest.raises(ValueError, match="not enabled"):
        score(config, "lexical", dataset="unknown")
    write_json(
        run / "evaluation_scope.json",
        {"benchmark_rewards": ["rouge_l", "chrf_pp", "length"], "lm_mode": "skip"},
    )
    generate_report(config)
    report = (run / "report.html").read_text()
    assert ("WMT14 German" in report) == ("wmt14_de_en" in names)
    assert len(list((run / "figures").glob("*.png"))) == 8


@pytest.mark.parametrize("names", [("wmt14_de_en",), DATASETS])
def test_vrcli_plot_supports_one_or_three_datasets(monkeypatch, names):
    pd = pytest.importorskip("pandas")
    plt = pytest.importorskip("matplotlib.pyplot")
    from non_verifiable_rewards.experiments.report import vrcli_figure

    result = {
        "datasets": names,
        "vr_cli_variant": "continuous_task_specific",
        "scores": pd.DataFrame(
            [
                {
                    "dataset": name,
                    "reward": "vr_cli",
                    "status": "ok",
                    "statistics.perplexity_improvement_percent": 20,
                }
                for name in names
            ]
        ),
        "margins": pd.DataFrame(
            [
                {"dataset": name, "reference_likelihood": 0.1, "vr_cli": 1.0}
                for name in names
            ]
        ),
        "summary": pd.DataFrame(
            [{"dataset": name, "reward": "vr_cli", "tie_rate": 0} for name in names]
        ),
    }
    saved = []

    def capture(fig, name, config):
        assert len(fig.axes) == 2 * len(names)
        saved.append(name)
        plt.close(fig)

    monkeypatch.setattr(
        "non_verifiable_rewards.experiments.report.save_figure", capture
    )
    vrcli_figure(result, {})
    assert saved == ["vr_cli_diagnostics"]
