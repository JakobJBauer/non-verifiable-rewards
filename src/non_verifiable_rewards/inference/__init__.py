"""Explicit inference selection; importing this module does not load CUDA."""


def configure_vllm():
    import os

    # The EIDF image has a driver, not a CUDA compiler. vLLM's native sampler
    # avoids FlashInfer's optional nvcc JIT dependency. Respect explicit tuning.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # Async batching can change BF16 greedy decisions near ties between calls.
    # Use batch-invariant kernels for repeatable inference across switches.
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def use_vllm(config, *, device=None):
    backend = config["inference"]["backend"]
    device = device or config["policy"]["device"]
    if backend == "vllm" and device != "cuda":
        raise ValueError("vLLM requires CUDA")
    return backend == "vllm" or (backend == "auto" and device == "cuda")


def grpo_inference_options(config):
    enabled = use_vllm(config)
    return {
        "use_vllm": enabled,
        "vllm_mode": "colocate",
        "vllm_enable_sleep_mode": enabled,
        "vllm_gpu_memory_utilization": config["inference"]["gpu_memory_utilization"],
        "vllm_max_model_length": config["data"]["max_prompt_tokens"]
        + config["optimization"]["max_completion_length"],
        "vllm_tensor_parallel_size": 1,
        # Preserve the original unconstrained top-p sampler and GRPO objective.
        # TRL's extra vLLM correction requires a bounded top-k candidate mask,
        # which would change the sampling distribution.
        "vllm_importance_sampling_correction": False,
    }
