"""Immutable sweep identities, exclusive execution, and persisted checkpoints."""

from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import tarfile

from .._hashing import file_sha256
from ..experiments.io import write_json


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {
        str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*.py"))
    }


def ensure_sweep(config):
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "source": source_identity(), "schema_version": 1}
    path = root / "sweep.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(
                "Sweep configuration or source changed; use a new output_dir"
            )
    else:
        write_json(path, identity)
        write_json(
            root / "environment.json",
            {
                "packages": {
                    d.metadata["Name"]: d.version
                    for d in importlib.metadata.distributions()
                },
                "reward_policy": "Frozen reward models, independent of the trained policy",
            },
        )
        package = Path(__file__).resolve().parents[1]
        with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
            for file in sorted(package.rglob("*.py")):
                archive.add(file, arcname=str(file.relative_to(package.parent)))
    return root


@contextmanager
def exclusive(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is using {root}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_data(root):
    root = Path(root)
    manifest = json.loads((root / "data_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if file_sha256(root / name) != sha:
            raise ValueError(f"Frozen data changed: {name}")
    return manifest


def checkpoint_manifest(path):
    path = Path(path)
    return {
        str(p.relative_to(path)): file_sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def verify_model(run_dir):
    run_dir = Path(run_dir)
    done = json.loads((run_dir / "trained.json").read_text())
    if done["model_files"] != checkpoint_manifest(run_dir / "model"):
        raise ValueError("Saved model is missing or changed after training")
    return done


def assert_single_process():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This launcher runs one GPU/process per run; do not use torchrun/accelerate launch"
        )
