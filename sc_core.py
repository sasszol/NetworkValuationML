import math
import profile
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from line_profiler_pycharm import profile

def approx_min_ce(t: torch.Tensor) -> float:
    t = torch.clamp(t, 1e-6, 1 - 1e-6)
    e = -(t * torch.log(t) + (1 - t) * torch.log(1 - t))
    return float(e.mean().item())

def squeeze_last(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(-1) if x.dim() == 3 and x.shape[-1] == 1 else x

class BaseMLP(nn.Module):
    """Flat MLP that outputs logits (no sigmoid)."""
    def __init__(self, n: int, in_features_per_node: int = 1):
        super().__init__()
        self.n = int(n)
        self.F = int(in_features_per_node)
        in_dim = self.n * self.F
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, n)
        with torch.no_grad():
            self.out.bias.fill_(math.log(0.02 / 0.98))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class DeepSetsPD(nn.Module):
    """Permutation-equivariant predictor in leave-one-out DeepSets form.

    This A-only-friendly version normalizes the leave-one-out pooled context by
    (n-1), so changing the number of banks does not automatically rescale the
    input seen by g.
    """
    def __init__(self, n: int, in_features_per_node: int = 1,
                 d_embed: int = 64, h_hidden: int = 128):
        super().__init__()
        self.n = int(n)
        self.F = int(in_features_per_node)
        self.d_embed = int(d_embed)
        self.h_hidden = int(h_hidden)
        self.phi = nn.Sequential(
            nn.Linear(self.F, self.d_embed),
            nn.ReLU(),
            nn.Linear(self.d_embed, self.d_embed),
        )
        self.g = nn.Sequential(
            nn.Linear(self.F + self.d_embed, self.h_hidden),
            nn.ReLU(),
            nn.Linear(self.h_hidden, self.h_hidden),
            nn.ReLU(),
            nn.Linear(self.h_hidden, 1),
        )
        with torch.no_grad():
            self.g[-1].bias.fill_(math.log(0.02 / 0.98))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.phi(x)
        S = e.sum(dim=1, keepdim=True)
        n = int(x.shape[1])
        if n > 1:
            others = (S - e) / float(n - 1)
        else:
            others = torch.zeros_like(e)
        h = torch.cat([x, others], dim=-1)
        return self.g(h).squeeze(-1)

class PDToMaturityNet(nn.Module):
    """Predicts till-maturity PD logits using a swappable backbone."""
    def __init__(self, n: int, in_features_per_node: int = 1,
                 use_deepsets: bool = False,
                 d_embed: int = 64, h_hidden: int = 128):
        super().__init__()
        self.in_features_per_node = int(in_features_per_node)
        self.use_deepsets = bool(use_deepsets)
        if self.use_deepsets:
            self.mlp = DeepSetsPD(
                n,
                in_features_per_node=in_features_per_node,
                d_embed=d_embed,
                h_hidden=h_hidden,
            )
        else:
            self.mlp = BaseMLP(n, in_features_per_node=in_features_per_node)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).unsqueeze(-1)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(squeeze_last(self.forward(x)))

# ----------------------- clearing post-processing --------------------------

def _apply_L(spay: torch.Tensor,
             L: Optional[torch.Tensor],
             ell: Optional[float]) -> torch.Tensor:
    if ell is not None: #TODO matrix operation instead of ifs
        Sigma = spay.sum(dim=-1, keepdim=True)
        return float(ell) * (Sigma - spay)
    return spay @ L

@profile
@torch.no_grad()
def hard_clear_pd(pd: torch.Tensor,
                  A: torch.Tensor,
                  L: Optional[torch.Tensor],
                  liab: torch.Tensor,
                  n_it: int,
                  a_dead: float,
                  *,
                  ell: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8
    s = (A > (a_dead + eps)).float()

    for _ in range(int(n_it)):
        spay = s * (1. - pd)
        recv = _apply_L(spay, L, ell) + A
        s = torch.minimum(s, (recv > liab).float())

    pd_final = torch.clamp((1. - s) + s * pd, 1e-8, 1 - 1e-8)
    return pd_final, s

class _STEClearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pd, A, L, liab, n_it: int, a_dead: float, ell: Optional[float] = None):
        with torch.no_grad():
            s = (A > (a_dead + 1e-8)).float()
            for _ in range(int(n_it)):
                spay = s * (1. - pd)
                recv = _apply_L(spay, L, ell) + A
                s = torch.minimum(s, (recv > liab).float())
            out = torch.clamp((1. - s) + s * pd, 1e-8, 1 - 1e-8)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None, None, None, None, None

def ste_clear_pd(pd: torch.Tensor,
                 A: torch.Tensor,
                 L: Optional[torch.Tensor],
                 liab: torch.Tensor,
                 n_it: int,
                 a_dead: float,
                 *,
                 ell: Optional[float] = None) -> torch.Tensor:
    return _STEClearFn.apply(pd, A, L, liab, int(n_it), float(a_dead), ell)

class LastStepPDIter(nn.Module):
    def __init__(self, L: torch.Tensor, liab: torch.Tensor, a_dead: float,
                 *, ell: Optional[float] = None):
        super().__init__()
        self.register_buffer('L', L.clone())
        self.register_buffer('liab', liab.clone())
        self.n_it = int(L.shape[0])
        self.a_dead = float(a_dead)
        self.ell = None if ell is None else float(ell)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = x[..., 0]
        s = (A > (self.a_dead + 1e-8)).float()
        for _ in range(self.n_it):
            recv = _apply_L(s, self.L, self.ell) + A
            s = torch.minimum(s, (recv > self.liab).float())
        pd = torch.clamp(1 - s, 1e-8, 1 - 1e-8)
        return torch.logit(pd).unsqueeze(-1)
