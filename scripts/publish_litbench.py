#!/usr/bin/env python3
"""Export a verified LitBench adaptation, then publish it as a separate action."""

import argparse
import gzip
import io
import json
import subprocess
import tarfile
from pathlib import Path

from non_verifiable_rewards.data import load_litbench
from non_verifiable_rewards.data.publication import (
    SOURCE_REVISION,
    export_bundle,
    publish_bundle,
)

SOURCE_PATHS = [
    "src",
    "scripts/publish_litbench.py",
    "pyproject.toml",
    "uv.lock",
    "docs/litbench_dataset_card.md",
    "docs/litbench_citation.bib",
]


def source_snapshot(root):
    if (root / ".git").exists():

        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args])

        # Unrelated worktree edits do not enter the committed source archive.
        if git("status", "--porcelain", "--", *SOURCE_PATHS).strip():
            raise ValueError(
                "Commit the exporter, loader and card changes before exporting"
            )
        revision = git("rev-parse", "HEAD").decode().strip()
        buffer = io.BytesIO(
            git("archive", "--format=tar", "HEAD", *SOURCE_PATHS, "README.md")
        )
        with tarfile.open(fileobj=buffer, mode="a") as archive:
            content = (revision + "\n").encode()
            member = tarfile.TarInfo("SOURCE_COMMIT")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    else:
        # The published source archive can reproduce the export without Git.
        revision = (root / "SOURCE_COMMIT").read_text().strip()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name in [*SOURCE_PATHS, "README.md", "SOURCE_COMMIT"]:
                archive.add(
                    root / name,
                    arcname=name,
                    filter=lambda member: (
                        None if "__pycache__" in Path(member.name).parts else member
                    ),
                )
    return revision, gzip.compress(buffer.getvalue(), mtime=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser(
        "export", help="Build and verify local release files"
    )
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--repo-id", required=True)
    export.add_argument("--cache-dir", default="artifacts/cache/litbench-publication")
    upload = subparsers.add_parser("upload", help="Upload an existing verified release")
    upload.add_argument("--output", type=Path, required=True)
    upload.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    if args.command == "upload":
        result = publish_bundle(args.output, args.repo_id)
        print(
            json.dumps(
                {
                    "dataset": f"https://huggingface.co/datasets/{args.repo_id}",
                    "commit": result.oid,
                    "commit_url": result.commit_url,
                }
            )
        )
        return
    root = Path(__file__).resolve().parents[1]
    revision, archive = source_snapshot(root)
    data, metadata = load_litbench(
        revision=SOURCE_REVISION, cache_dir=args.cache_dir, return_metadata=True
    )
    actual = {
        name: (len(rows), len(set(rows["prompt"]))) for name, rows in data.items()
    }
    if actual != {"train": (9935, 9935), "test": (7774, 1304)}:
        raise ValueError(f"Pinned release cohort changed: {actual}")
    result = export_bundle(
        data,
        metadata,
        args.output,
        repo_id=args.repo_id,
        code_revision=revision,
        source_archive=archive,
        card_template=(root / "docs/litbench_dataset_card.md").read_text(),
        citation=(root / "docs/litbench_citation.bib").read_text(),
    )
    print(
        json.dumps({"output": str(args.output), "splits": result["splits"]}, indent=2)
    )


if __name__ == "__main__":
    main()
