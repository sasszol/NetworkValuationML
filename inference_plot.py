# inference_plot.py
"""
Clean plotting script that matches training diagnostics exactly (ABM, S-less).

What it guarantees
------------------
• Loads each "step0_corr_*.pt" checkpoint (pure state_dict saved by the sweep)   [matches sc_train]
• Reconstructs the exact MLP dimensions from the state_dict                      [robust to hidden sizes]
• Builds inputs *exactly like training diagnostics*:
      F=1 -> [A]
      F=2 -> [A, ψ] where ψ is the iterative breach feature                      [matches sc_utils.iterative_breach_feature]
• Applies the same HARD clearing projection as the trainer                       [matches sc_core.hard_clear_pd]
• Optional one-line diagnostic print per ρ at a chosen A                         [like _diag_print_step0 in sc_train]

How to run
----------
python inference_plot.py
(or adjust the USER CONFIG block below)

If you try to load an *old* checkpoint trained with S+A+ψ (F=3), the script will
raise a friendly error. Use new checkpoints trained with the A-only pipeline.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- use the same utilities the trainer uses ---
from sc_utils import build_L, iterative_breach_feature  # ψ and exposures, ABM version  [training uses this]
from sc_core import hard_clear_pd                       # hard clearing projection       [training uses this]


# =============================== USER CONFIG =============================== #
MODEL_DIR = "10_steps_3_DD_0_5_kL/five_banks_orig"  # folder with files like: step0_corr_{rho:.4f}.pt
A_INPUT  = [2.0]            # scalar (broadcast) or length-n vector of A (distance-to-default)
A_DEAD   = -4.0            # sentinel for “already-defaulted”; only used if your clearing uses a_dead
kL = 0.5
L_MATRIX = None            # optional custom exposure matrix (list[list[float]]), overrides kL
LIAB_VECTOR = None         # optional custom liabilities; default = row sums of L_MATRIX
# parameters that must match your training setup
SIGMA = 1.0
T_TOTAL = 1.0
DT = 0.1
STEP = 0 #leave this on 0
K_SURV_ITERS = 5            # iterations in ψ fixed point

# plotting / diagnostics
AGGREGATE = "mean"          # "none" for per-bank curves, or "mean" for averaged curve
SAVE_PATH = None            # e.g. "runs/try/pd_vs_rho.png"
SHOW_FIG  = True
PRINT_DIAG = True           # print a one-line diagnostic per ρ at A_INPUT (post-clearing PD)
# ========================================================================== #


# ----------------------------- state-dict helpers ---------------------------

def _normalize_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalize keys so they live under 'mlp.*'.
    Accepts the current saves ('mlp.*') and older wrapper saves.
    """
    if any(k.startswith("mlp.") for k in sd.keys()):
        return {k: v for k, v in sd.items() if k.startswith("mlp.")}
    if any(k.startswith("clearing.base_net.") for k in sd.keys()):
        return {k.replace("clearing.base_net.", "mlp."): v
                for k, v in sd.items() if k.startswith("clearing.base_net.")}
    if any(k.startswith("base_net.") for k in sd.keys()):
        return {k.replace("base_net.", "mlp."): v
                for k, v in sd.items() if k.startswith("base_net.")}
    # Otherwise return as-is; loader will error if incompatible
    return sd


def _infer_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int, int]:
    """
    Infer (n, F, H1, H2, H3) from:
      mlp.fc1.weight: (H1, n*F)
      mlp.fc2.weight: (H2, H1)
      mlp.fc3.weight: (H3, H2)
      mlp.out.weight: (n,  H3)
    """
    try:
        w1 = sd["mlp.fc1.weight"]; w2 = sd["mlp.fc2.weight"]
        w3 = sd["mlp.fc3.weight"]; wo = sd["mlp.out.weight"]
    except KeyError as e:
        raise RuntimeError(f"checkpoint missing expected key: {e}")

    H1, in_dim = w1.shape
    H2, H1_ = w2.shape
    H3, H2_ = w3.shape
    n,  H3_ = wo.shape

    if H1_ != H1 or H2_ != H2 or H3_ != H3:
        raise RuntimeError("inconsistent hidden sizes in checkpoint layers")

    if in_dim % n != 0:
        raise RuntimeError(f"cannot infer features-per-node: in_dim={in_dim} not divisible by n={n}")
    F = in_dim // n
    return int(n), int(F), int(H1), int(H2), int(H3)


# ----------------------------- dynamic model --------------------------------

class _DynamicMLP(nn.Module):
    """Dynamically-sized MLP matching checkpoint shapes: (n*F)→H1→H2→H3→n."""
    def __init__(self, n: int, F: int, H1: int, H2: int, H3: int):
        super().__init__()
        self.n, self.F = int(n), int(F)
        in_dim = self.n * self.F
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, H3)
        self.out = nn.Linear(H3, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)  # (B, n)


class _PDInfer(nn.Module):
    """Minimal PD head: returns logits with shape (B, n, 1), like training."""
    def __init__(self, n: int, F: int, H1: int, H2: int, H3: int):
        super().__init__()
        self.mlp = _DynamicMLP(n, F, H1, H2, H3)
        self.n = int(n)
        self.in_features_per_node = int(F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).unsqueeze(-1)  # (B, n, 1)


# ---- DeepSets variant (matches sc_core.DeepSetsPD; loaded from checkpoints
#      saved by training runs with USE_DEEPSETS=True) ------------------------

class _DynamicDeepSets(nn.Module):
    """DeepSets backbone whose dimensions are inferred from a checkpoint.

    Mirrors sc_core.DeepSetsPD. The same trained weights can be evaluated on
    any number of agents N (parameter count is N-independent).
    """
    def __init__(self, F: int, d_embed: int, h_hidden: int):
        super().__init__()
        self.F = int(F)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.phi(x)
        S = e.sum(dim=1, keepdim=True)
        others = S - e
        h = torch.cat([x, others], dim=-1)
        return self.g(h).squeeze(-1)


class _PDInferDeepSets(nn.Module):
    """PD head wrapping the DeepSets backbone."""
    def __init__(self, n: int, F: int, d_embed: int, h_hidden: int):
        super().__init__()
        self.mlp = _DynamicDeepSets(F, d_embed, h_hidden)
        self.n = int(n)  # the n the model was *trained* at; not enforced at eval
        self.in_features_per_node = int(F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).unsqueeze(-1)


def _is_deepsets_sd(sd: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("mlp.phi.") or k.startswith("mlp.g.") for k in sd.keys())


def _infer_deepsets_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """Infer (F, d_embed, h_hidden) from a DeepSets checkpoint.

    Layout (sc_core.DeepSetsPD):
      mlp.phi.0.weight: (d_embed, F)
      mlp.phi.2.weight: (d_embed, d_embed)
      mlp.g.0.weight  : (h_hidden, F + d_embed)
      mlp.g.4.weight  : (1, h_hidden)
    """
    try:
        wp0 = sd["mlp.phi.0.weight"]
        wg0 = sd["mlp.g.0.weight"]
        wg_last = sd["mlp.g.4.weight"]
    except KeyError as e:
        raise RuntimeError(f"DeepSets checkpoint missing expected key: {e}")
    d_embed, F = wp0.shape
    h_hidden_a, in_g = wg0.shape
    one, h_hidden_b = wg_last.shape
    if h_hidden_a != h_hidden_b or one != 1:
        raise RuntimeError("inconsistent DeepSets g-MLP shapes in checkpoint")
    if in_g != F + d_embed:
        raise RuntimeError(
            f"DeepSets g-MLP input dim {in_g} != F+d_embed = {F + d_embed}"
        )
    return int(F), int(d_embed), int(h_hidden_a)


# ----------------------------- loading & inference --------------------------

@torch.no_grad()
def _parse_rho(path: Path) -> Optional[float]:
    m = re.search(r"step0_corr_([+-]?\d+(?:\.\d+)?)\.pt$", path.name)
    return float(m.group(1)) if m else None


@torch.no_grad()
def _load_model_from_ckpt(ckpt: Path):
    """Load one 'step0_corr_{rho}.pt' and return (rho, model).

    Returns either a _PDInfer (flat MLP) or a _PDInferDeepSets, transparently
    dispatched on the saved state-dict keys.
    """
    rho = _parse_rho(ckpt)
    if rho is None:
        raise ValueError(f"filename does not match step0_corr_*.pt pattern: {ckpt}")

    raw = torch.load(ckpt, map_location="cpu")
    if not isinstance(raw, dict):
        raise RuntimeError(f"checkpoint must be a state_dict (dict), got {type(raw)}")

    sd = _normalize_state_dict(raw)

    if _is_deepsets_sd(sd):
        F, d_embed, h_hidden = _infer_deepsets_dims(sd)
        if F not in (1, 2):
            raise RuntimeError(
                f"DeepSets checkpoint expects F={F} features per node; "
                f"only F=1 or F=2 are supported by this plotter."
            )
        # n is not stored in the DeepSets state-dict (the model is N-agnostic);
        # downstream code passes A explicitly, so we just record a placeholder.
        n_placeholder = 1
        model = _PDInferDeepSets(n_placeholder, F, d_embed, h_hidden)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return rho, model

    n, F, H1, H2, H3 = _infer_dims(sd)

    # only support the new ABM pipeline (A-only or A+ψ)
    if F not in (1, 2):
        raise RuntimeError(
            f"Checkpoint expects F={F} features per node. "
            f"This plotter supports the ABM S-less pipeline only (F=1 or F=2). "
            f"Please re-train with the new code or use the legacy plotter for S+A+ψ runs."
        )

    model = _PDInfer(n, F, H1, H2, H3)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return rho, model


@torch.no_grad()
def _broadcast_np(x: List[float], n: int, name: str) -> np.ndarray:
    """Allow single scalar or length-n values."""
    if len(x) == 1:
        return np.full(n, float(x[0]), dtype=np.float32)
    if len(x) == n:
        return np.array(x, dtype=np.float32)
    raise ValueError(f"{name} must be length 1 or {n} (got {len(x)})")


@torch.no_grad()
def _eval_L_liab(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluation-time clearing matrices.

    By default we mirror the original homogeneous complete-network setup
    L_ij = kL/(n-1), i != j. If L_MATRIX is provided in the USER CONFIG
    block, it is used verbatim so diagnostics can be run without any hidden
    symmetry assumption.
    """
    if L_MATRIX is not None:
        L = torch.as_tensor(L_MATRIX, dtype=torch.float32)
        if tuple(L.shape) != (int(n), int(n)):
            raise ValueError(f"L_MATRIX must have shape ({n}, {n}) (got {tuple(L.shape)}).")
        if LIAB_VECTOR is not None:
            liab = torch.as_tensor(LIAB_VECTOR, dtype=torch.float32)
            if tuple(liab.shape) != (int(n),):
                raise ValueError(f"LIAB_VECTOR must have shape ({n},) (got {tuple(liab.shape)}).")
        else:
            liab = L.sum(1)
        return L, liab
    L = build_L(n, offdiag=kL/(n-1))
    liab = L.sum(1)
    return L, liab


@torch.no_grad()
def predict_pd_postclearing(
    model: _PDInfer,
    A: torch.Tensor,  # (n,) or (B,n)
    *,
    sigma: float,
    T_total: float,
    dt: float,
    step: int,
    k_surv_iters: int,
    a_dead: float,
    L: Optional[torch.Tensor] = None,
    liab: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build ψ via iterative breach, run the model, then hard-clear to get PDs.
    Returns (n,) if input was (n,), else (B,n).
    Mirrors the training diagnostic path.
    """
    if A.dim() == 1:
        A = A.unsqueeze(0)
    B, n = A.shape

    if L is None or liab is None:
        # training exposures/liabilities are built from L.sum(1)                 [sc_utils.build_matrices]
        L, liab = _eval_L_liab(n)  # outside liabilities were merged into A in this ABM setup

    # same time index as in diagnostics: residual horizon at this step
    T_now = max(float(T_total) - step * float(dt), 1e-8)

    # Assemble features exactly like training diagnostics:
    if getattr(model, "in_features_per_node", 1) == 1:
        x = A.unsqueeze(-1)                          # (B, n, 1)  -> [A]
    else:
        psi = iterative_breach_feature(A, L, liab,   # (B, n)     -> ψ(A | counterparty)
                                       sigma=float(sigma), T=T_now,
                                       k=int(k_surv_iters), a_dead=float(a_dead))
        x = torch.stack([A, psi], dim=-1)           # (B, n, 2)  -> [A, ψ]

    # logits -> prob -> hard clearing (post-processing)
    logits = model(x).squeeze(-1)                   # (B, n)
    pd_raw = torch.sigmoid(logits)
    pd_proj, _ = hard_clear_pd(pd_raw, A, L, liab, n_it=int(n), a_dead=float(a_dead))

    return pd_proj.squeeze(0) if B == 1 else pd_proj


# ----------------------------- correlation sweep plot -----------------------

def plot_pd_vs_rho(
    model_dir: str | Path,
    A_vals: List[float],
    *,
    sigma: float,
    T_total: float,
    dt: float,
    step: int,
    k_surv_iters: int,
    a_dead: float,
    aggregate: str = "none",    # "none" or "mean"
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all step0_corr_*.pt files, compute post-clearing PDs at (A), and plot vs ρ.
    Returns:
      rhos:   (M,)
      pd_mat: (M, n) post-clearing PD per bank per ρ
    """
    model_dir = Path(model_dir)
    ckpts = sorted(model_dir.glob("step0_corr_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No 'step0_corr_*.pt' files in {model_dir}")

    # First model defines n (and F) for broadcasting input.
    # For DeepSets the model is N-agnostic; in that case n is inferred from
    # the length of `A_vals` (or set to 1 if a scalar was passed, which then
    # also implies n=1 — usually you'd pass a vector for DeepSets eval).
    rho0, model0 = _load_model_from_ckpt(ckpts[0])
    is_deepsets = isinstance(model0, _PDInferDeepSets)
    if is_deepsets:
        n = max(int(len(A_vals)), 1)
    else:
        n = model0.n
    A_vec = _broadcast_np(A_vals, n, "A")

    # Build clearing matrices once (ρ affects training dR, not L here)
    L, liab = _eval_L_liab(n)

    rhos: List[float] = []
    rows: List[np.ndarray] = []

    # Evaluate first checkpoint
    A_t = torch.tensor(A_vec, dtype=torch.float32)
    pd = predict_pd_postclearing(model0, A_t, sigma=sigma, T_total=T_total,
                                 dt=dt, step=step, k_surv_iters=k_surv_iters,
                                 a_dead=a_dead, L=L, liab=liab).numpy()
    if PRINT_DIAG:
        print(f"[plotter‑diag] rho={rho0:.4f}  A={A_vec.tolist()}  post‑clear PD={np.round(pd,4).tolist()}  mean={pd.mean():.4f}")
    rhos.append(rho0); rows.append(pd)

    # Remaining checkpoints
    for f in ckpts[1:]:
        rho, model = _load_model_from_ckpt(f)
        if (not isinstance(model, _PDInferDeepSets)) and model.n != n:
            raise RuntimeError(f"Mixed n across checkpoints: expected {n}, found {model.n} in {f.name}")
        pd = predict_pd_postclearing(model, A_t, sigma=sigma, T_total=T_total,
                                     dt=dt, step=step, k_surv_iters=k_surv_iters,
                                     a_dead=a_dead, L=L, liab=liab).numpy()
        if PRINT_DIAG:
            print(f"[plotter‑diag] rho={rho:.4f}  A={A_vec.tolist()}  post‑clear PD={np.round(pd,4).tolist()}  mean={pd.mean():.4f}")
        rhos.append(rho)
        rows.append(pd)

    rhos_arr = np.array(rhos, dtype=np.float64)
    pd_mat = np.stack(rows, axis=0)  # (M, n)

    # Sort by rho for a clean curve
    idx = np.argsort(rhos_arr)
    rhos_arr = rhos_arr[idx]
    pd_mat = pd_mat[idx]

    # ---- plot
    plt.figure(figsize=(8, 5))
    if aggregate == "mean":
        plt.plot(rhos_arr, pd_mat.mean(axis=1), marker="o")
        plt.ylabel("mean PD across banks")
    else:
        for j in range(n):
            plt.plot(rhos_arr, pd_mat[:, j], marker="o", label=f"bank {j}")
        if n <= 16:
            plt.legend()
        plt.ylabel("PD")

    plt.xlabel("asset correlation ρ")
    plt.title("Post-clearing default probability vs correlation (step-0)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"[saved] {out}")

    if show:
        plt.show()
    else:
        plt.close()

    return rhos_arr, pd_mat


# ----------------------------------- main -----------------------------------

if __name__ == "__main__":
    plot_pd_vs_rho(
        model_dir=MODEL_DIR,
        A_vals=A_INPUT,
        sigma=SIGMA,
        T_total=T_TOTAL,
        dt=DT,
        step=STEP,
        k_surv_iters=K_SURV_ITERS,
        a_dead=A_DEAD,
        aggregate=AGGREGATE,
        save_path=SAVE_PATH,
        show=SHOW_FIG,
    )
