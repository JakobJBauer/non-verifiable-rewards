"""Artifact formats, run identities, and recoverable per-pair journals."""

import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tarfile
import tomllib
from pathlib import Path

SCHEMA_VERSION = 1
DATASETS = ("reddit_tldr", "litbench")
BACKENDS = ("lexical", "bertscore", "bleurt", "lm")
REWARDS = (
    "rouge_l",
    "bertscore",
    "bleurt",
    "self_certainty",
    "reference_likelihood",
    "vr_cli",
    "length",
)
BACKEND_REWARDS = {
    "lexical": ("rouge_l", "length"),
    "bertscore": ("bertscore",),
    "bleurt": ("bleurt",),
    "lm": ("self_certainty", "reference_likelihood", "vr_cli"),
}


def utc_now():
    return dt.datetime.now(dt.UTC).isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
