"""Single-GPU TRL switching with non-mutating LoRA synchronization."""

import json
import os
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path


@contextmanager
def engine_dtype(dtype):
    """TRL does not forward the policy dtype to its colocated LLM constructor.

    Scope this override to construction in our single-process launcher; engine
    auto-detection from the *base checkpoint* can choose a different dtype.
    """
    from trl.generation import vllm_generation

    constructor = vllm_generation.LLM
    invariant = os.environ.get("VLLM_BATCH_INVARIANT")
    # vLLM's invariant mode replaces process-wide aten::linear with an
    # inference-only kernel. In a colocated trainer that breaks CUDA backward.
    # Use normal kernels for sampled rollouts. Separate frozen reward workers
    # and held-out generation can safely retain invariant inference.
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    vllm_generation.LLM = partial(constructor, dtype=dtype)
    try:
        yield
    finally:
        vllm_generation.LLM = constructor
        if invariant is None:
            os.environ.pop("VLLM_BATCH_INVARIANT", None)
        else:
            os.environ["VLLM_BATCH_INVARIANT"] = invariant


def policy_weights(model):
    """Stream merged tensors without modifying the optimizer's parameters.

    Repeated BF16 merge/unmerge is not reversible and can slowly corrupt the
    frozen base. Only a single layer's temporary merged weight is live here.
    TRL 1.18.2's normal sync/cache invalidation handles the resulting iterator.
    """
    import torch

    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if any(part.startswith("lora_") for part in name.split(".")):
                continue
            if "modules_to_save" in name or "original_module" in name:
                raise ValueError(
                    "modules_to_save is not supported by the single-GPU switch"
                )
            value = parameter.detach()
            if ".base_layer." in name:
                parent, suffix = name.rsplit(".base_layer.", 1)
                layer = modules[parent]
                if layer.merged:
                    raise RuntimeError("Training adapters must remain unmerged")
                if suffix == "weight" and not layer.disable_adapters:
                    value = value.clone()
                    for adapter in layer.active_adapters:
                        value.add_(layer.get_delta_weight(adapter))
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            yield name, value


class TrainingSwitch:
    """Instrument the pinned TRL lifecycle and fail closed on stale weights."""

    def __init__(self, trainer, trace_path):
        self.trainer = trainer
        self.backend = trainer.vllm_generation
        self.trace_path = Path(trace_path)
        self.synced_step = None
        self.sync_count = 0
        self._sync = self.backend.sync_weights
        self._generate = self.backend.generate
        self._parameters = self.backend._iter_named_params
        self._closed = False
        # This project supports one GPU, standard LoRA or full fine-tuning.
        # Keep TRL's wake/copy/reset-prefix-cache logic, replacing only the
        # iterator that would otherwise merge into the live training weights.
        self.backend._iter_named_params = lambda: policy_weights(self.backend.model)
        self.backend.sync_weights = self.sync_weights
        self.backend.generate = self.generate

    def record(self, event, elapsed):
        import torch

        row = dict(
            event=event,
            step=self.trainer.state.global_step,
            synced_step=self.synced_step,
            sync_count=self.sync_count,
            seconds=elapsed,
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
        )
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def sync_weights(self):
        import torch

        start = time.perf_counter()
        self.synced_step = None
        try:
            self._sync()
            torch.cuda.synchronize()
        except BaseException:
            self.sleep()
            raise
        self.synced_step = self.trainer.state.global_step
        self.sync_count += 1
        self.record("sync", time.perf_counter() - start)

    def sleep(self):
        if not self.backend._llm_weights_sleeping:
            self.backend.llm.sleep(level=2)
            self.backend._llm_weights_sleeping = True

    def generate(self, *args, **kwargs):
        # A checkpoint resume changes parameters before the first rollout;
        # synced_step starts at None, so even step zero must synchronize.
        if (
            self.synced_step != self.trainer.state.global_step
            or self.backend._llm_weights_sleeping
        ):
            self.sync_weights()
        start = time.perf_counter()
        try:
            return self._generate(*args, **kwargs)
        finally:
            self.sleep()
            self.record("generate_and_sleep", time.perf_counter() - start)

    def close(self):
        if self._closed:
            return
        self.sleep()
        self.backend.llm.llm_engine.engine_core.shutdown()
        self.backend.llm = None
        # Break bound-method cycles, including trainer -> switch -> trainer.
        self.backend.sync_weights = self._sync
        self.backend.generate = self._generate
        self.backend._iter_named_params = self._parameters
        self.trainer = None
        self._closed = True
