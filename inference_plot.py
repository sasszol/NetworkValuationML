# inference_plot.py
"""A-only inference/plotting script.

Loads step0 checkpoints, builds inputs as ``A.unsqueeze(-1)`` with shape
``(B, n, 1)``, applies the same hard clearing projection as training, and plots
post-clearing PD versus asset correlation.

DeepSets note: the internal module name ``phi`` is kept because it is the
standard DeepSets per-node embedding network, not an engineered input feature.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sc_utils import build_L, infer_homogeneous_complete_ell
from sc_core import hard_clear_pd


# =============================== USER CONFIG =============================== #
MODEL_DIR = "C:/git/NetworkValuationML/kL_1_DD_2/20_banks_longer_3"
A_INPUT  = [2.0]            # scalar (broadcast) or length-n vector of A
A_DEAD   = -4.0
kL = 1.0
L_MATRIX = None
LIAB_VECTOR = None
N_EVAL = 20                 # required for DeepSets when A_INPUT is scalar

SIGMA = 1.0                 # kept for config compatibility; A-only inference does not use it
T_TOTAL = 1.0               # kept for config compatibility; A-only inference does not use it
DT = 0.1                    # kept for config compatibility; A-only inference does not use it
STEP = 0                    # leave this at 0 for step-0 nets

AGGREGATE = "mean"          # "none" for per-bank curves, or "mean" for averaged curve
SAVE_PATH = None
SHOW_FIG  = True
PRINT_DIAG = True
# ========================================================================== #


# ----------------------------- state-dict helpers ---------------------------

def _normalize_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("mlp.") for k in sd.keys()):
        return {k: v for k, v in sd.items() if k.startswith("mlp.")}
    if any(k.startswith("clearing.base_net.") for k in sd.keys()):
        return {k.replace("clearing.base_net.", "mlp."): v
                for k, v in sd.items() if k.startswith("clearing.base_net.")}
    if any(k.startswith("base_net.") for k in sd.keys()):
        return {k.replace("base_net.", "mlp."): v
                for k, v in sd.items() if k.startswith("base_net.")}
    return sd


def _infer_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int, int]:
    try:
        w1 = sd["mlp.fc1.weight"]
        w2 = sd["mlp.fc2.weight"]
        w3 = sd["mlp.fc3.weight"]
        wo = sd["mlp.out.weight"]
    except KeyError as e:
        raise RuntimeError(f"checkpoint missing expected key: {e}")

    H1, in_dim = w1.shape
    H2, H1_ = w2.shape
    H3, H2_ = w3.shape
    n, H3_ = wo.shape

    if H1_ != H1 or H2_ != H2 or H3_ != H3:
        raise RuntimeError("inconsistent hidden sizes in checkpoint layers")
    if in_dim % n != 0:
        raise RuntimeError(f"cannot infer features-per-node: in_dim={in_dim} not divisible by n={n}")

    F = in_dim // n
    return int(n), int(F), int(H1), int(H2), int(H3)


class _DynamicMLP(nn.Module):
    def __init__(self, n: int, F: int, H1: int, H2: int, H3: int):
        super().__init__()
        self.n, self.F = int(n), int(F)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.n * self.F, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, H3)
        self.out = nn.Linear(H3, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)


class _PDInfer(nn.Module):
    def __init__(self, n: int, F: int, H1: int, H2: int, H3: int):
        super().__init__()
        self.mlp = _DynamicMLP(n, F, H1, H2, H3)
        self.n = int(n)
        self.in_features_per_node = int(F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).unsqueeze(-1)


class _DynamicDeepSets(nn.Module):
    """Matches sc_core.DeepSetsPD.  The name ``phi`` is the DeepSets embedding."""
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
        n = int(x.shape[1])
        others = (S - e) / float(n - 1) if n > 1 else torch.zeros_like(e)
        h = torch.cat([x, others], dim=-1)
        return self.g(h).squeeze(-1)


class _PDInferDeepSets(nn.Module):
    def __init__(self, n: int, F: int, d_embed: int, h_hidden: int):
        super().__init__()
        self.mlp = _DynamicDeepSets(F, d_embed, h_hidden)
        self.n = int(n)
        self.in_features_per_node = int(F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).unsqueeze(-1)


def _is_deepsets_sd(sd: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("mlp.phi.") or k.startswith("mlp.g.") for k in sd.keys())


def _infer_deepsets_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
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
        raise RuntimeError(f"DeepSets g-MLP input dim {in_g} != F+d_embed = {F + d_embed}")
    return int(F), int(d_embed), int(h_hidden_a)


@torch.no_grad()
def _parse_rho(path: Path) -> Optional[float]:
    m = re.search(r"step0_corr_([+-]?\d+(?:\.\d+)?)\.pt$", path.name)
    return float(m.group(1)) if m else None


@torch.no_grad()
def _load_model_from_ckpt(ckpt: Path):
    rho = _parse_rho(ckpt)
    if rho is None:
        raise ValueError(f"filename does not match step0_corr_*.pt pattern: {ckpt}")

    raw = torch.load(ckpt, map_location="cpu")
    if not isinstance(raw, dict):
        raise RuntimeError(f"checkpoint must be a state_dict (dict), got {type(raw)}")

    sd = _normalize_state_dict(raw)

    if _is_deepsets_sd(sd):
        F, d_embed, h_hidden = _infer_deepsets_dims(sd)
        if F != 1:
            raise RuntimeError(f"This A-only plotter expects DeepSets F=1, but checkpoint has F={F}.")
        model = _PDInferDeepSets(1, F, d_embed, h_hidden)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return rho, model

    n, F, H1, H2, H3 = _infer_dims(sd)
    if F != 1:
        raise RuntimeError(f"This A-only plotter expects F=1, but checkpoint has F={F}.")

    model = _PDInfer(n, F, H1, H2, H3)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return rho, model


@torch.no_grad()
def _broadcast_np(x: Sequence[float], n: int, name: str) -> np.ndarray:
    x_list = list(x)
    if len(x_list) == 1:
        return np.full(n, float(x_list[0]), dtype=np.float32)
    if len(x_list) == n:
        return np.array(x_list, dtype=np.float32)
    raise ValueError(f"{name} must be length 1 or {n} (got {len(x_list)})")


@torch.no_grad()
def _resolve_eval_n(model,
                    A_vals: Sequence[float],
                    *,
                    n_eval: Optional[int] = None,
                    L_matrix=None,
                    liab_vector=None) -> int:
    if not isinstance(model, _PDInferDeepSets):
        return int(model.n)

    candidates: list[tuple[str, int]] = []
    if L_matrix is not None:
        L = torch.as_tensor(L_matrix)
        if L.ndim != 2 or int(L.shape[0]) != int(L.shape[1]):
            raise ValueError(f"L_matrix must be square (got shape {tuple(L.shape)}).")
        candidates.append(("L_matrix", int(L.shape[0])))
    if liab_vector is not None:
        liab = torch.as_tensor(liab_vector)
        if liab.ndim != 1:
            raise ValueError(f"liab_vector must be 1-D (got shape {tuple(liab.shape)}).")
        candidates.append(("liab_vector", int(liab.shape[0])))
    if n_eval is not None:
        candidates.append(("n_eval", int(n_eval)))
    A_vals_list = list(A_vals)
    if len(A_vals_list) > 1:
        candidates.append(("A_vals", int(len(A_vals_list))))

    if not candidates:
        raise ValueError(
            "DeepSets checkpoints are N-agnostic, so scalar A_INPUT=[a] needs "
            "N_EVAL, a length-n A_INPUT vector, or L_MATRIX/LIAB_VECTOR."
        )

    n = candidates[0][1]
    for name, value in candidates[1:]:
        if value != n:
            raise ValueError(f"Inconsistent evaluation bank counts: {candidates}.")
    if n <= 0:
        raise ValueError(f"Evaluation bank count must be positive (got {n}).")
    return int(n)


@torch.no_grad()
def _eval_L_liab(n: int,
                 *,
                 kL: float,
                 L_matrix=None,
                 liab_vector=None) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(n)
    if n <= 0:
        raise ValueError(f"evaluation bank count must be positive (got {n}).")

    if L_matrix is not None:
        L = torch.as_tensor(L_matrix, dtype=torch.float32)
        if tuple(L.shape) != (n, n):
            raise ValueError(f"L_matrix must have shape ({n}, {n}) (got {tuple(L.shape)}).")
        if liab_vector is not None:
            liab = torch.as_tensor(liab_vector, dtype=torch.float32)
            if tuple(liab.shape) != (n,):
                raise ValueError(f"liab_vector must have shape ({n},) (got {tuple(liab.shape)}).")
        else:
            liab = L.sum(1)
        return L, liab

    if liab_vector is not None:
        raise ValueError("liab_vector without L_matrix is ambiguous. Provide L_matrix too.")
    if n == 1:
        return torch.zeros(1, 1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)

    L = build_L(n, offdiag=float(kL) / float(n - 1))
    liab = L.sum(1)
    return L, liab


@torch.no_grad()
def predict_pd_postclearing(model,
                            A: torch.Tensor,
                            *,
                            a_dead: float,
                            L: Optional[torch.Tensor] = None,
                            liab: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Run the A-only model and hard-clear the resulting PDs."""
    if A.dim() == 1:
        A = A.unsqueeze(0)
    B, n = A.shape

    if L is None or liab is None:
        L, liab = _eval_L_liab(n, kL=kL, L_matrix=L_MATRIX, liab_vector=LIAB_VECTOR)
    ell = infer_homogeneous_complete_ell(L)

    x = A.unsqueeze(-1)
    logits = model(x).squeeze(-1)
    pd_raw = torch.sigmoid(logits)
    pd_proj, _ = hard_clear_pd(pd_raw, A, L, liab, n_it=int(n), a_dead=float(a_dead), ell=ell)
    return pd_proj.squeeze(0) if B == 1 else pd_proj


def plot_pd_vs_rho(model_dir: str | Path,
                   A_vals: List[float],
                   *,
                   a_dead: float,
                   aggregate: str = "none",
                   save_path: Optional[str | Path] = None,
                   show: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    model_dir = Path(model_dir)
    ckpts = sorted(model_dir.glob("step0_corr_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No 'step0_corr_*.pt' files in {model_dir}")

    rho0, model0 = _load_model_from_ckpt(ckpts[0])
    n = _resolve_eval_n(model0, A_vals, n_eval=N_EVAL, L_matrix=L_MATRIX, liab_vector=LIAB_VECTOR)
    A_vec = _broadcast_np(A_vals, n, "A")
    L, liab = _eval_L_liab(n, kL=kL, L_matrix=L_MATRIX, liab_vector=LIAB_VECTOR)
    A_t = torch.tensor(A_vec, dtype=torch.float32)

    rhos: List[float] = []
    rows: List[np.ndarray] = []

    for rho, model in [(rho0, model0)] + [_load_model_from_ckpt(f) for f in ckpts[1:]]:
        if (not isinstance(model, _PDInferDeepSets)) and model.n != n:
            raise RuntimeError(f"Mixed n across checkpoints: expected {n}, found {model.n}.")
        pd = predict_pd_postclearing(model, A_t, a_dead=a_dead, L=L, liab=liab).numpy()
        if PRINT_DIAG:
            print(f"[plotter-diag] rho={rho:.4f}  A={A_vec.tolist()}  post-clear PD={np.round(pd,4).tolist()}  mean={pd.mean():.4f}")
        rhos.append(float(rho))
        rows.append(pd)

    rhos_arr = np.array(rhos, dtype=np.float64)
    pd_mat = np.stack(rows, axis=0)
    idx = np.argsort(rhos_arr)
    rhos_arr = rhos_arr[idx]
    pd_mat = pd_mat[idx]

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

    plt.xlabel("asset correlation rho")
    plt.title("Post-clearing default probability vs correlation (step-0, A-only)")
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


if __name__ == "__main__":
    plot_pd_vs_rho(
        model_dir=MODEL_DIR,
        A_vals=A_INPUT,
        a_dead=A_DEAD,
        aggregate=AGGREGATE,
        save_path=SAVE_PATH,
        show=SHOW_FIG,
    )
