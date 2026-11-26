"""Fixed-policy backends used by held-out generation and preference scoring."""

import gc
from contextlib import contextmanager

from ..rewards.causal_lm import CausalLMScorer
from . import use_vllm


class TransformersPolicy(CausalLMScorer):
    def generate_ids(self, prompts, *, max_new_tokens):
        import torch

        results = []
        with torch.inference_mode():
            for prompt in prompts:
                ids = torch.tensor([prompt], device=self.model.device)
                output = self.model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
                results.append(output[0, len(prompt) :].tolist())
        return results

    @property
    def generation_config(self):
        return self.model.generation_config


@contextmanager
def policy_backend(config, path):
    import torch

    from ..training.train import load_policy, load_tokenizer

    if use_vllm(config):
        from .scorer import VLLMScorer

        backend = VLLMScorer(
            path,
            dtype=config["policy"]["dtype"],
            max_length=config["reward_models"]["max_context_tokens"],
            batch_size=config["inference"]["batch_size"],
            gpu_memory_utilization=config["inference"]["gpu_memory_utilization"],
            sleep=False,
        )
    else:
        model = load_policy(config, path).to(config["policy"]["device"])
        model.requires_grad_(False)
        model.eval()
        backend = TransformersPolicy(
            model,
            load_tokenizer(path),
            batch_size=config["reward_models"]["lm_batch_size"],
        )
    try:
        yield backend
    finally:
        if use_vllm(config):
            backend.close()
        else:
            backend.model.to("cpu")
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
