"""Artifact formats, run identities, and recoverable per-pair journals."""

import contextlib
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
import platform
import sys
import tarfile
import tomllib
from pathlib import Path

from .._hashing import digest, file_sha256

SCHEMA_VERSION = 1
LEGACY_DATASETS = ("reddit_tldr", "litbench")
DATASETS = (*LEGACY_DATASETS, "wmt14_de_en")
SMART_REWARDS = ("smart1_bertscore", "smart1_bleurt")
BACKENDS = ("lexical", "meteor", "bertscore", "bleurt", "lm", *SMART_REWARDS)
REWARDS = (
    "rouge_l",
    "chrf_pp",
    "meteor",
    "bertscore",
    "bleurt",
    "self_certainty",
    "reference_likelihood",
    "vr_cli",
    "nover",
    "length",
)
LEGACY_REWARDS = tuple(
    reward for reward in REWARDS if reward not in ("nover", "chrf_pp", "meteor")
)
ALL_REWARDS = (*REWARDS, *SMART_REWARDS)
BACKEND_REWARDS = {
    "lexical": ("rouge_l", "chrf_pp", "length"),
    "meteor": ("meteor",),
    "bertscore": ("bertscore",),
    "bleurt": ("bleurt",),
    "smart1_bertscore": ("smart1_bertscore",),
    "smart1_bleurt": ("smart1_bleurt",),
    "lm": ("self_certainty", "reference_likelihood", "vr_cli", "nover"),
}


def configured_datasets(config):
    """Explicit dataset scope; omitted scope preserves historical two-task runs."""
    names = config.get("datasets", LEGACY_DATASETS)
    if isinstance(names, (list, tuple)) and "supergpqa" in names:
        raise ValueError(
            "SuperGPQA has answer keys, not preference pairs; use load_supergpqa "
            "and python -m non_verifiable_rewards.reasoning for generated-answer evaluation"
        )
    if (
        not isinstance(names, (list, tuple))
        or not names
        or any(not isinstance(name, str) or name not in DATASETS for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("datasets must contain distinct known dataset names")
    return tuple(names)


def run_datasets(run):
    """Read the frozen dataset scope instead of assuming every known task exists."""
    run = Path(run)
    path = run / "data_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        names = manifest.get(
            "datasets", [name for name in DATASETS if name in manifest]
        )
        names = configured_datasets({"datasets": names})
        config_path = run / "config.json"
        if config_path.exists() and names != configured_datasets(
            json.loads(config_path.read_text())
        ):
            raise ValueError(
                "Dataset scope differs between the saved config and data manifest"
            )
        return names
    path = run / "config.json"
    return configured_datasets(json.loads(path.read_text()) if path.exists() else {})


def configured_rewards(config):
    enabled = config.get("include_smart_rewards", False)
    if not isinstance(enabled, bool):
        raise ValueError("include_smart_rewards must be a boolean")
    return ALL_REWARDS if enabled else REWARDS


def enabled_backends(config):
    rewards = configured_rewards(config)
    return tuple(name for name in BACKENDS if set(BACKEND_REWARDS[name]) & set(rewards))


def run_rewards(run):
    """Use the recorded reward set; historical environments predate NOVER."""
    path = Path(run) / "environment.json"
    if path.exists():
        return tuple(json.loads(path.read_text()).get("rewards", LEGACY_REWARDS))
    return REWARDS


def utc_now():
    return dt.datetime.now(dt.UTC).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_config(path):
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    for key in (
        "threads",
        "pairs_per_dataset",
        "bootstrap_samples",
        "batch_size",
        "lm_batch_size",
        "max_context_tokens",
    ):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("max_prompt_tokens", "max_candidate_tokens", "max_reference_tokens"):
        if not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    configured_rewards(config)
    configured_datasets(config)
    return config


def code_files():
    root = Path(__file__).resolve().parents[3]
    files = [
        *root.glob("src/**/*.py"),
        *root.glob("configs/*.toml"),
        *root.glob("scripts/*.sh"),
    ]
    files += [root / "pyproject.toml", root / "uv.lock"]
    return root, sorted(path for path in files if path.is_file())


def code_digest():
    root, files = code_files()
    return digest({str(path.relative_to(root)): file_sha256(path) for path in files})


def ensure_run(config):
    run = Path(config["run_dir"])
    if (run / "derivation.json").exists():
        raise ValueError(
            "Derived runs are report-only; choose a new run_dir for scoring"
        )
    run.mkdir(parents=True, exist_ok=True)
    path = run / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f"Configuration differs from {path}; choose a new run_dir")
        return run
    write_json(path, config)
    write_json(
        run / "environment.json",
        {
            "created_at": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "schema_version": SCHEMA_VERSION,
            "rewards": configured_rewards(config),
            "datasets": configured_datasets(config),
            "config_sha256": digest(config),
            "code_sha256_at_creation": code_digest(),
            "packages": {
                item.metadata["Name"]: item.version
                for item in importlib.metadata.distributions()
            },
        },
    )
    root, files = code_files()
    with tarfile.open(run / "source_at_creation.tar.gz", "w:gz") as archive:
        for file in files:
            archive.add(file, arcname=file.relative_to(root))
    return run


class Journal:
    """One complete pair per line; exclusive writer lock and torn-tail recovery.

    Existing complete records are immutable. Errors are retained, not silently
    retried; choose a new run for changed settings or implementations.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = None
        self.completed = {}

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.stream.seek(0)
            while True:
                offset = self.stream.tell()
                line = self.stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    self.stream.truncate(offset)
                    break
                record = json.loads(line)
                key = record["pair_id"]
                if key in self.completed:
                    raise ValueError(f"Duplicate pair {key} in {self.path}")
                self.completed[key] = record
            self.stream.seek(0, os.SEEK_END)
            return self
        except BaseException:
            self.stream.close()
            raise

    def append(self, record):
        if record["pair_id"] in self.completed:
            raise ValueError("A pair cannot be appended twice")
        self.stream.write(
            (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
        )
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.completed[record["pair_id"]] = record

    def __exit__(self, *args):
        with contextlib.suppress(OSError):
            fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()
