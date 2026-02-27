import math
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
    """MLP that outputs logits (no sigmoid)."""
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

class PDToMaturityNet(nn.Module):
    """Predicts till‑maturity PD logits; we always post‑process by clearing."""
    def __init__(self, n: int, in_features_per_node: int = 2):
        super().__init__()
        self.mlp = BaseMLP(n, in_features_per_node=in_features_per_node)
        self.in_features_per_node = int(in_features_per_node)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).unsqueeze(-1)  # (B, n, 1)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(squeeze_last(self.forward(x)))  # (B, n)

# ----------------------- clearing post-processing --------------------------

@torch.no_grad()
def hard_clear_pd(pd: torch.Tensor,
                  A: torch.Tensor,
                  L: torch.Tensor,
                  liab: torch.Tensor,
                  n_it: int,
                  a_dead: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hard clearing projection for additive assets.
    Agents with A <= a_dead are already defaulted (PD=1 in peers' books).
    Returns (pd_final, surv_final), both (B, n).
    """
    eps = 1e-8
    s = (A > (a_dead + eps)).float()

    for _ in range(int(n_it)):
        # expected payment fraction: alive banks pay (1 - pd)
        spay = s * (1. - pd)
        recv = (spay @ L) + A
        s = torch.minimum(s, (recv > liab).float())

    pd_final = torch.clamp((1. - s) + s * pd, 1e-8, 1 - 1e-8)
    return pd_final, s

class _STEClearFn(torch.autograd.Function):
    """Straight‑Through Estimator: forward clears; backward passes grad to 'pd'."""
    @staticmethod
    def forward(ctx, pd, A, L, liab, n_it: int, a_dead: float):
        with torch.no_grad():
            s = (A > (a_dead + 1e-8)).float()
            for _ in range(int(n_it)):
                spay = s * (1. - pd)
                recv = (spay @ L) + A
                s = torch.minimum(s, (recv > liab).float())
            out = torch.clamp((1. - s) + s * pd, 1e-8, 1 - 1e-8)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None, None, None, None

def ste_clear_pd(pd: torch.Tensor,
                 A: torch.Tensor,
                 L: torch.Tensor,
                 liab: torch.Tensor,
                 n_it: int,
                 a_dead: float) -> torch.Tensor:
    return _STEClearFn.apply(pd, A, L, liab, int(n_it), float(a_dead))

# ----------------------- last deterministic step ---------------------------

class LastStepPDIter(nn.Module):
    """
    Deterministic final step: clear purely from A (hard), then PD = 1 - S.
    """
    def __init__(self, L: torch.Tensor, liab: torch.Tensor, a_dead: float):
        super().__init__()
        self.register_buffer("L", L.clone())
        self.register_buffer("liab", liab.clone())
        self.n_it = int(L.shape[0])
        self.a_dead = float(a_dead)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, n, 1) [A]
        A = x[..., 0]
        s = (A > (self.a_dead + 1e-8)).float()
        for _ in range(self.n_it):
            recv = (s.unsqueeze(1) @ self.L.unsqueeze(0)).squeeze(1) + A
            s = torch.minimum(s, (recv > self.liab).float())
        pd = torch.clamp(1 - s, 1e-8, 1 - 1e-8)
        return torch.logit(pd).unsqueeze(-1)
