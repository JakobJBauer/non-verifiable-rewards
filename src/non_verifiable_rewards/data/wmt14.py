"""WMT14 German-to-English parallel training data and human-ranked test pairs."""

import csv
import io
import tarfile
from collections import Counter
from itertools import combinations
from pathlib import Path

from datasets import DatasetDict, DownloadConfig, DownloadManager, load_dataset

from .._hashing import digest, file_sha256
from ..rewards.prompts import translation_prompt
from .loaders import _pair_dataset
from .pairs import make_pair

SUBMISSIONS_URL = "https://www.statmt.org/wmt14/submissions.tgz"
JUDGMENTS_URL = "https://www.statmt.org/wmt14/wmt14-manual-evaluation.tgz"
PARALLEL_DATASET = "wmt/wmt14"
_SUBMISSIONS_ROOT = "wmt14-data/txt/"
_SYSTEM_ROOT = _SUBMISSIONS_ROOT + "system-outputs/newstest2014/de-en/"


def _parallel_row(row, index, *, split):
    source, reference = (
        row["translation"]["de"].strip(),
        row["translation"]["en"].strip(),
    )
    prompt_id = digest(["wmt14_de_en", source])[:24]
    return {
        "dataset": "wmt14_de_en",
        "prompt_id": prompt_id,
        "prompt": translation_prompt(source) if source else "",
        "reference": reference,
        "reference_id": digest([prompt_id, reference])[:24],
        "metadata": {"split": split, "source_row": index},
    }


def load_wmt14(
    *,
    split="train",
    cache_dir=None,
    streaming=False,
    revision=None,
    submissions_archive=None,
    judgments_archive=None,
    return_metadata=False,
):
    """Load parallel train/validation prompts or human-ranked test pairs.

    Train starts from 4,508,785 upstream translations without preferences,
    excluding exact German sources in validation or official newstest2014.
    Test uses the official human rankings, with separate references. Test takes
    precedence over validation. Streaming/revision apply to parallel data;
    submissions_archive also supplies the source exclusion set for training.
    Use ``load_wmt14_parallel`` for the original unpartitioned parallel rows.
    """
    if split not in ("train", "validation", "test"):
        raise ValueError("WMT14 split must be 'train', 'validation', or 'test'")
    if split == "test":
        if streaming or revision is not None:
            raise ValueError("WMT14 annotated test does not support streaming/revision")
        return load_wmt14_preferences(
            split="test",
            cache_dir=cache_dir,
            submissions_archive=submissions_archive,
            judgments_archive=judgments_archive,
            return_metadata=return_metadata,
        )
    if judgments_archive is not None:
        raise ValueError("WMT14 judgment overrides apply only to annotated test")
    downloader = DownloadManager(download_config=DownloadConfig(cache_dir=cache_dir))
    submissions = Path(submissions_archive or downloader.download(SUBMISSIONS_URL))
    with tarfile.open(submissions) as archive:
        _check_archive(archive)
        heldout = set(
            _read_member(
                archive, _SUBMISSIONS_ROOT + "sources/newstest2014-deen-src.de"
            ).splitlines()
        )
    heldout = {source.strip() for source in heldout if source.strip()}
    if not heldout:
        raise ValueError("Missing official WMT14 test sources for split exclusion")
    sources = {
        "wmt14_heldout_submissions": {
            "path": str(submissions.resolve()),
            "sha256": file_sha256(submissions),
            "url": SUBMISSIONS_URL,
        }
    }
    if split == "train":
        validation = load_dataset(
            PARALLEL_DATASET,
            "de-en",
            split="validation",
            cache_dir=cache_dir,
            streaming=False,
            revision=revision,
        )
        heldout.update(row["translation"]["de"].strip() for row in validation)
        sources["wmt14_heldout_validation"] = {
            "dataset": PARALLEL_DATASET,
            "config": "de-en",
            "split": "validation",
            "revision": revision,
            "fingerprint": validation._fingerprint,
        }
    data = load_wmt14_parallel(
        split=split, cache_dir=cache_dir, streaming=streaming, revision=revision
    )
    before = None if streaming else len(data)
    heldout_prompts = {translation_prompt(source) for source in heldout if source}
    data = data.filter(lambda row: row["prompt"] not in heldout_prompts)
    metadata = {
        split: {
            "sources": {
                **sources,
                "wmt14_parallel": {
                    "dataset": PARALLEL_DATASET,
                    "config": "de-en",
                    "split": split,
                    "revision": revision,
                    "streaming": streaming,
                },
            },
            "audit": {
                "eligible_rows": None if streaming else len(data),
                "excluded_heldout_source_overlap": None
                if streaming
                else before - len(data),
            },
            "split_policy": "wmt14_parallel_train_validation_ranked_test_exclude_heldout_sources_v1",
        }
    }
    return (data, metadata) if return_metadata else data


def load_wmt14_parallel(
    *, split="train", cache_dir=None, streaming=False, revision=None
):
    """Load German prompts/English references without preference labels.

    The upstream de-en corpus has 4,508,785 train, 3,000 validation and 3,003
    test rows before blank-text filtering. Train is the RL source; reserve
    validation/test for evaluation. ``streaming=True`` avoids materializing
    the full training corpus. ``load_wmt14(split='test')`` instead loads
    human-ranked evaluation pairs.
    """
    if split not in ("train", "validation", "test"):
        raise ValueError(
            "WMT14 parallel split must be 'train', 'validation', or 'test'"
        )
    raw = load_dataset(
        PARALLEL_DATASET,
        "de-en",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        revision=revision,
    )
    # Map before filtering so provenance indexes the original upstream row.
    data = raw.map(
        _parallel_row,
        with_indices=True,
        fn_kwargs={"split": split},
        remove_columns=["translation"],
    )
    return data.filter(lambda row: bool(row["reference"]) and bool(row["prompt"]))


def _read_member(archive, name):
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"Expected a regular archive member: {name}")
    return archive.extractfile(member).read().decode("utf-8")


def _check_archive(archive):
    names = archive.getnames()
    if len(set(names)) != len(names):
        raise ValueError("Duplicate archive member names")


def _load_sources(submissions, judgments):
    # Read exact members in place, without extracting untrusted archive paths.
    with tarfile.open(submissions) as archive:
        _check_archive(archive)
        sources = _read_member(
            archive, _SUBMISSIONS_ROOT + "sources/newstest2014-deen-src.de"
        ).splitlines()
        references = _read_member(
            archive, _SUBMISSIONS_ROOT + "references/newstest2014-deen-ref.en"
        ).splitlines()
        outputs = {
            member.name.removeprefix(_SYSTEM_ROOT): _read_member(
                archive, member.name
            ).splitlines()
            for member in archive.getmembers()
            if member.isfile() and member.name.startswith(_SYSTEM_ROOT)
        }
    if not sources or len(references) != len(sources) or not outputs:
        raise ValueError(
            "Missing or misaligned WMT14 sources/references/system outputs"
        )
    if any(len(lines) != len(sources) for lines in outputs.values()):
        raise ValueError("WMT14 system outputs must align with source sentence numbers")
    with tarfile.open(judgments) as archive:
        _check_archive(archive)
        text = _read_member(
            archive, "wmt14-manual-evaluation/wmt14-judgments-anonymized.csv"
        )
    return sources, references, outputs, list(csv.DictReader(io.StringIO(text)))


def wmt14_pairs(sources, references, outputs, judgments):
    """Expand strict human ranks; merge repeated labeled texts, retain disagreement."""
    audit, pairs = Counter(), {}
    for row_index, row in enumerate(judgments):
        if (row["srclang"], row["trglang"]) != ("German", "English"):
            continue
        audit["source_rankings"] += 1
        segment = int(row["srcIndex"])
        if str(segment) != row["segmentId"] or not 1 <= segment <= len(sources):
            raise ValueError("Invalid or mismatched WMT14 source sentence number")
        systems = [row[f"system{i}Id"] for i in range(1, 6)]
        ranks = [int(row[f"system{i}rank"]) for i in range(1, 6)]
        if len(set(systems)) != 5 or any(system not in outputs for system in systems):
            raise ValueError("Unknown or repeated WMT14 system in a human ranking")
        if any(rank not in (-1, 1, 2, 3, 4, 5) for rank in ranks):
            raise ValueError("Invalid WMT14 rank; expected -1 (unranked) or 1..5")
        for first, second in combinations(range(5), 2):
            audit["pairwise_observations"] += 1
            if min(ranks[first], ranks[second]) < 1:
                audit["excluded_unranked"] += 1
                continue
            if ranks[first] == ranks[second]:
                audit["excluded_human_tie"] += 1
                continue
            better, worse = sorted((first, second), key=lambda i: ranks[i])
            prompt_id = f"newstest2014:de-en:{segment}"
            judgment = {
                "source_row": row_index,
                "judge_id": row["judgeId"],
                "preferred_system": systems[better],
                "rejected_system": systems[worse],
                "preferred_rank": ranks[better],
                "rejected_rank": ranks[worse],
            }
            try:
                pair = make_pair(
                    "wmt14_de_en",
                    prompt_id,
                    translation_prompt(sources[segment - 1]),
                    outputs[systems[better]][segment - 1],
                    outputs[systems[worse]][segment - 1],
                    references[segment - 1],
                    prompt_id + ":human_translation",
                    {
                        "segment_id": segment,
                        "judgments": [judgment],
                        "opposing_judgments": 0,
                        "reference_rule": "Official English reference aligned by 1-based source sentence number",
                    },
                )
                if not sources[segment - 1].strip():
                    raise ValueError("Empty German source")
            except ValueError:
                audit["excluded_empty_or_overlapping_texts"] += 1
                continue
            audit["eligible_judgments"] += 1
            if pair["pair_id"] in pairs:
                audit["merged_duplicate_labeled_pair"] += 1
                pairs[pair["pair_id"]]["metadata"]["judgments"].append(judgment)
            else:
                pairs[pair["pair_id"]] = pair
    for pair in pairs.values():
        reverse_id = digest(
            ["wmt14_de_en", pair["prompt_id"], pair["rejected"], pair["preferred"]]
        )[:24]
        if reverse_id in pairs:
            pair["metadata"]["opposing_judgments"] = len(
                pairs[reverse_id]["metadata"]["judgments"]
            )
            audit["labeled_pairs_with_opposing_judgments"] += 1
    audit["eligible_pairs"] = len(pairs)
    audit["eligible_prompts"] = len({p["prompt_id"] for p in pairs.values()})
    return list(pairs.values()), dict(audit)


def load_wmt14_preferences(
    *,
    split=None,
    cache_dir=None,
    submissions_archive=None,
    judgments_archive=None,
    return_metadata=False,
):
    """Load official newstest2014 human preferences with separate references.

    Returns DatasetDict(test=...), or a Dataset for ``split='test'``. Repeated
    observations of an oriented text pair are merged with all judgments retained
    in metadata. Opposing strict labels remain separate, as for SFF; they are
    flagged rather than silently majority-voted. Ties/unranked comparisons and
    empty/identical/reference-overlapping texts are excluded.
    """
    if split not in (None, "test"):
        raise ValueError(
            "WMT14 human preferences are test-only; use load_wmt14 for RL train"
        )
    downloader = DownloadManager(download_config=DownloadConfig(cache_dir=cache_dir))
    paths = {
        "wmt14_submissions": Path(
            submissions_archive or downloader.download(SUBMISSIONS_URL)
        ),
        "wmt14_judgments": Path(
            judgments_archive or downloader.download(JUDGMENTS_URL)
        ),
    }
    pairs, audit = wmt14_pairs(*_load_sources(*paths.values()))
    data = _pair_dataset(pairs, "wmt14_de_en")
    metadata = {
        "test": {
            "split_policy": "wmt14_parallel_train_validation_ranked_test_exclude_heldout_sources_v1",
            "audit": audit,
            "sources": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "url": url,
                }
                for (name, path), url in zip(
                    paths.items(), (SUBMISSIONS_URL, JUDGMENTS_URL), strict=True
                )
            },
        }
    }
    result = data if split else DatasetDict(test=data)
    return (result, metadata) if return_metadata else result
