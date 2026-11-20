"""JSON-line scorer; stdout is reserved exclusively for protocol responses."""

import argparse
from contextlib import redirect_stdout
import glob
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--family", choices=("torch", "bleurt"), required=True)
    args = parser.parse_args()
    if args.family == "bleurt" and not os.environ.get("NVR_TF_LIBS_READY"):
        # TensorFlow's pip CUDA libraries live in its own environment. Set the
        # dynamic-loader path before importing TF, without contaminating Torch.
        libraries = glob.glob(sys.prefix + "/lib/python*/site-packages/nvidia/*/lib")
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            libraries + [os.environ.get("LD_LIBRARY_PATH", "")]
        )
        os.environ["NVR_TF_LIBS_READY"] = "1"
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "non_verifiable_rewards.training.reward_worker",
                *sys.argv[1:],
            ],
        )
    from .rewards import RewardFactory

    config = json.loads(Path(args.sweep).read_text())["config"]
    factory = RewardFactory(config)
    output = sys.stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            is_bleurt = request["reward"] in ("bleurt", "smart1_bleurt")
            if is_bleurt != (args.family == "bleurt"):
                raise ValueError("Reward routed to the wrong worker")
            with redirect_stdout(sys.stderr):
                scores = factory.score(
                    request["reward"],
                    request["rows"],
                    request["completions"],
                    request["view"],
                )
            response = {"scores": scores}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {"error": f"{type(error).__name__}: {error}"}
        output.write(json.dumps(response, allow_nan=False) + "\n")
        output.flush()


if __name__ == "__main__":
    main()
