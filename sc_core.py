import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def approx_min_ce(t: torch.Tensor) -> float:
    t = torch.clamp(t, 1e-6, 1 - 1e-6)
    e = -(t * torch.log(t) + (1 - t) * torch.log(1 - t))
    return float(e.mean().item())

def squeeze_last(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(-1) if x.dim() == 3 and x.shape[-1] == 1 else x

# ---------------------------- predictor -------------------------------

class BaseMLP(nn.Module):
    """Flat MLP that outputs logits (no sigmoid).

    Not permutation-equivariant: swapping two agents' inputs yields a different
    output. Equivariance must therefore be learned from data (or enforced via
    permutation augmentation in the training loop).
    """
    def __init__(self, n: int, in_features_per_node: int = 2):
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
            # bias to ~2% PD prior
            self.out.bias.fill_(math.log(0.02 / 0.98))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, F), F=1 → [A]; F=2 → [A, ψ]
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)  # (B, n)


class DeepSetsPD(nn.Module):
    """Permutation-equivariant predictor in leave-one-out DeepSets form.

    For input x of shape (B, n, F):
        e   = phi(x)                    # (B, n, d)   per-agent embedding
        S   = e.sum(dim=1, keepdim=True)# (B, 1, d)   total over agents
        oth = S - e                     # (B, n, d)   leave-one-out summary
        h   = cat([x, oth], dim=-1)     # (B, n, F+d) own + others
        out = g(h).squeeze(-1)          # (B, n)      per-agent logit

    Same phi and g are shared across agents → output is S_n-equivariant by
    construction. Parameter count is independent of n; the same trained model
    can be evaluated at any n.
    """
    def __init__(self, n: int, in_features_per_node: int = 2,
                 d_embed: int = 32, h_hidden: int = 64):
        super().__init__()
        self.n = int(n)
        self.F = int(in_features_per_node)
        self.d_embed = int(d_embed)
        self.phi = nn.Sequential(
            nn.Linear(self.F, self.d_embed),
            nn.ReLU(),
            nn.Linear(self.d_embed, self.d_embed),
        )
        self.g = nn.Sequential(
            nn.Linear(self.F + self.d_embed, h_hidden),
            nn.ReLU(),
            nn.Linear(h_hidden, h_hidden),
            nn.ReLU(),
            nn.Linear(h_hidden, 1),
        )
        with torch.no_grad():
            # match BaseMLP's ~2% PD prior on the final bias
            self.g[-1].bias.fill_(math.log(0.02 / 0.98))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, F)
        e = self.phi(x)                                 # (B, n, d)
        S = e.sum(dim=1, keepdim=True)                  # (B, 1, d)
        others = S - e                                  # (B, n, d)
        h = torch.cat([x, others], dim=-1)              # (B, n, F+d)
        return self.g(h).squeeze(-1)                    # (B, n)


class PDToMaturityNet(nn.Module):
    """Predicts till-maturity PD logits using a swappable backbone.

    Backbone is BaseMLP (flat) or DeepSetsPD (equivariant), selected via
    `use_deepsets`.
    """
    def __init__(self, n: int, in_features_per_node: int = 2,
                 use_deepsets: bool = False):
        super().__init__()
        self.in_features_per_node = int(in_features_per_node)
        self.use_deepsets = bool(use_deepsets)
        if self.use_deepsets:
            self.mlp = DeepSetsPD(n, in_features_per_node=in_features_per_node)
        else:
            self.mlp = BaseMLP(n, in_features_per_node=in_features_per_node)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).unsqueeze(-1)  # (B, n, 1)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(squeeze_last(self.forward(x)))  # (B, n)

# ----------------------- clearing post-processing --------------------------

def _apply_L(spay: torch.Tensor,
             L: Optional[torch.Tensor],
             ell: Optional[float]) -> torch.Tensor:
    """Compute spay @ L, optionally exploiting the symmetric rank-1 form.

    In the symmetric/fully-connected setting of the paper,
        L = ell * (1 1^T - I)
    so
        (spay @ L)_i = ell * (sum_j spay_j - spay_i).

    Cost drops from O(n^2) to O(n) per scenario, which (combined with the
    n-iteration Picard loop in clearing) takes the per-scenario cost from
    O(n^3) to O(n^2). Numerics are exact up to floating-point reordering.

    If `ell` is None we fall back to the dense matrix multiply.
    """
    if ell is not None:
        Sigma = spay.sum(dim=-1, keepdim=True)
        return float(ell) * (Sigma - spay)
    # dense fallback (general L)
    return spay @ L


@torch.no_grad()
def hard_clear_pd(pd: torch.Tensor,
                  A: torch.Tensor,
                  L: Optional[torch.Tensor],
                  liab: torch.Tensor,
                  n_it: int,
                  a_dead: float,
                  *,
                  ell: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hard clearing projection for additive assets.
    Agents with A <= a_dead are already defaulted (PD=1 in peers' books).
    Returns (pd_final, surv_final), both (B, n).

    If `ell` is provided the symmetric rank-1 form of L is exploited
    (L = ell*(11^T - I)); `L` is then unused and may be passed as None.
    """
    eps = 1e-8
    s = (A > (a_dead + eps)).float()

    for _ in range(int(n_it)):
        # expected payment fraction: alive banks pay (1 - pd)
        spay = s * (1. - pd)
        recv = _apply_L(spay, L, ell) + A
        s = torch.minimum(s, (recv > liab).float())

    pd_final = torch.clamp((1. - s) + s * pd, 1e-8, 1 - 1e-8)
    return pd_final, s

class _STEClearFn(torch.autograd.Function):
    """Straight‑Through Estimator: forward clears; backward passes grad to 'pd'."""
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

# ----------------------- last deterministic step ---------------------------

class LastStepPDIter(nn.Module):
    """
    Deterministic final step: clear purely from A (hard), then PD = 1 - S.

    If `ell` is provided the rank-1 form of L is used in the inner loop.
    """
    def __init__(self, L: torch.Tensor, liab: torch.Tensor, a_dead: float,
                 *, ell: Optional[float] = None):
        super().__init__()
        self.register_buffer("L", L.clone())
        self.register_buffer("liab", liab.clone())
        self.n_it = int(L.shape[0])
        self.a_dead = float(a_dead)
        self.ell = None if ell is None else float(ell)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, n, 1) [A]
        A = x[..., 0]
        s = (A > (self.a_dead + 1e-8)).float()
        for _ in range(self.n_it):
            recv = _apply_L(s, self.L, self.ell) + A
            s = torch.minimum(s, (recv > self.liab).float())
        pd = torch.clamp(1 - s, 1e-8, 1 - 1e-8)
        return torch.logit(pd).unsqueeze(-1)
