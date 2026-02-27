import torch
import torch.nn as nn

class ZeroSubModel(nn.Module):
    """
    Early-step placeholder that outputs large negative logits (PD≈0).
    Used for ZERO_WARMSTART so that initial PDs are near-zero, not 0.5.
    """
    def __init__(self, n: int, neg_logit: float = -50.0):
        super().__init__()
        self.n = int(n)
        self.neg_logit = float(neg_logit)

    def forward(self, x: torch.Tensor):
        B, n, _ = x.shape
        return x.new_full((B, n, 1), self.neg_logit)
