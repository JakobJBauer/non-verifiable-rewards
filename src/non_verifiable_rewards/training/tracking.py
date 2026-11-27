"""One resumable Weights & Biases run per independently trained reward policy."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid

from ..experiments.io import write_json


@contextmanager
def tracking_run(config, run, target):
    """Own the W&B lifecycle while Transformers logs the trainer's metrics.

    Persist the ID before connecting, so retries and optimizer-checkpoint
    resumes attach to the same online run. Separate sweep directories get
    separate IDs, even when they reuse reward/run names.
    """
    options = config["wandb"]
    if not options["enabled"]:
        yield None
        return
    import wandb

    if wandb.run is not None:
        raise RuntimeError("Finish the active W&B run before starting a GRPO run")
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "wandb.json"
    if path.exists():
        identity = json.loads(path.read_text())
        if identity["project"] != options["project"] or (
            options["entity"] and identity.get("entity") != options["entity"]
        ):
            raise ValueError("W&B project/entity changed for an existing training run")
    else:
        identity = {
            "id": uuid.uuid4().hex[:16],
            "project": options["project"],
            "entity": options["entity"] or None,
        }
        write_json(path, identity)
    # The trainer integration would otherwise honor global artifact/watch
    # settings. Keep this integration focused on scalar training diagnostics.
    previous = {key: os.environ.get(key) for key in ("WANDB_LOG_MODEL", "WANDB_WATCH")}
    os.environ.update(WANDB_LOG_MODEL="false", WANDB_WATCH="false")
    tracked = None
    exit_code = 1
    try:
        tracked = wandb.init(
            project=identity["project"],
            entity=identity["entity"],
            id=identity["id"],
            name=run["name"],
            group=Path(config["output_dir"]).name,
            job_type="grpo",
            dir=str(target),
            mode=options["mode"],
            resume="allow" if options["mode"] == "online" else None,
            config={"experiment": config, "reward_run": run},
            save_code=False,
            settings=wandb.Settings(init_timeout=60, disable_code=True),
        )
        identity.update(entity=tracked.entity or identity["entity"], url=tracked.url)
        write_json(path, identity)
        yield tracked
        exit_code = 0
    finally:
        try:
            if tracked is not None:
                tracked.finish(exit_code=exit_code)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
