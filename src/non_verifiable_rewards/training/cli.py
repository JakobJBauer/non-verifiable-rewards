"""Plan, prepare, train, benchmark and report reproducible GRPO sweeps."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .config import load_config
from .state import assert_single_process, ensure_sweep, exclusive


def selected_runs(config, names):
    if not names:
        return config["runs"]
    available = {run["name"]: run for run in config["runs"]}
    if len(set(names)) != len(names) or not set(names) <= set(available):
        raise ValueError(f"Select distinct configured runs: {list(available)}")
    return [available[name] for name in names]


def child(config_path, config, command, name=None):
    command_line = [
        sys.executable,
        "-m",
        "non_verifiable_rewards.training",
        command,
        "--config",
        str(Path(config_path).resolve()),
    ]
    if name is not None:
        command_line += ["--run", name]
    logs = Path(config["output_dir"]) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{name or 'sweep'}-{command}.log"
    print(f"{command}: {name or 'sweep'} (log: {path})", flush=True)
    with path.open("a") as stream:
        result = subprocess.run(command_line, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(
            f"Stage {command} failed for {name}; see {path}. Completed stages are resumable."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "prepare",
            "train",
            "generate",
            "score",
            "benchmark",
            "run",
            "report",
        ),
    )
    parser.add_argument("--config", default="configs/grpo_qwen3_4b.toml")
    parser.add_argument(
        "--run",
        action="append",
        dest="names",
        help="Configured run name; repeat for run/report. 'base' is allowed for benchmark/generate/score.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    assert_single_process()
    if args.command in ("generate", "score", "benchmark") and args.names == ["base"]:
        runs = []
    else:
        runs = selected_runs(config, args.names)
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "output_dir": config["output_dir"],
                    "policy": config["policy"],
                    "inference": config["inference"],
                    "runs": runs,
                    "optimization": config["optimization"],
                    "wandb": config["wandb"],
                    "data": config["data"],
                    "benchmark": config["benchmark"],
                    "reward_models": "Frozen independently of the optimized policy; weighted sum before group normalization",
                },
                indent=2,
            )
        )
        return
    root = Path(config["output_dir"])
    if args.command == "run":
        with exclusive(root / "launcher_lock"):
            child(args.config, config, "prepare")
            if (
                config["benchmark"]["enabled"]
                and config["benchmark"]["include_base_model"]
            ):
                child(args.config, config, "generate", "base")
                child(args.config, config, "score", "base")
            for run in runs:
                child(args.config, config, "train", run["name"])
                if config["benchmark"]["enabled"]:
                    child(args.config, config, "generate", run["name"])
                    child(args.config, config, "score", run["name"])
            if config["benchmark"]["enabled"]:
                from .report import report

                with exclusive(root):
                    report(config, [r["name"] for r in runs])
        return
    if args.command == "benchmark":
        names = args.names or [run["name"] for run in runs]
        with exclusive(root / "launcher_lock"):
            if config["benchmark"]["include_base_model"] and "base" not in names:
                names.insert(0, "base")
            for name in names:
                child(args.config, config, "generate", name)
                child(args.config, config, "score", name)
            from .report import report

            with exclusive(root):
                report(config, names)
        return
    with exclusive(root):
        ensure_sweep(config)
        if args.command == "prepare":
            from .assets import policy_path
            from .data import prepare
            from .train import load_tokenizer

            prepare(config, load_tokenizer(policy_path(config)))
        elif args.command == "report":
            from .report import report

            report(config, [r["name"] for r in runs])
        else:
            if not args.names or len(args.names) != 1:
                raise ValueError(f"{args.command} requires exactly one --run name")
            name = args.names[0]
            if args.command == "train":
                from .train import train_run

                train_run(config, runs[0])
            elif args.command == "generate":
                from .benchmark import generate

                generate(config, name)
            elif args.command == "score":
                from .benchmark import score_generations

                score_generations(config, name)


if __name__ == "__main__":
    main()
