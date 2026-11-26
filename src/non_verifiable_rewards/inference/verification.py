"""Opt-in vLLM worker extension for the GPU acceptance audit."""


class WeightVerification:
    def weight_digest(self):
        import hashlib

        import torch

        digest = hashlib.sha256()
        for value in self.model_runner.model.parameters():
            digest.update(
                value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        return digest.hexdigest()
