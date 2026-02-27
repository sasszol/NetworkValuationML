"""
Dump correlation ↦ averaged PD and correlation ↦ per-step MSEs to CSV,
reusing the exact inference & clearing path from inference_plot.py.

How to run
----------
python dump_metrics.py

Outputs (defaults under MODEL_DIR):
  • avg_pd_vs_corr.csv          with columns: corr, mean_pd
  • step_mse_vs_corr.csv        with columns: corr, step0, step1, ...

Notes
-----
• Uses inference_plot._load_model_from_ckpt(), predict_pd_postclearing(), etc.,
  to reconstruct network shapes and to compute ψ and hard-clearing consistently.
• MSE vectors are read from the same .pt checkpoints (best effort key search).
"""

from __future__ import annotations

from pathlib import Path
import csv
import math
from typing import List, Optional, Tuple, Sequence

import numpy as np
import torch

# --- reuse the exact diagnostic/inference path from your plotter ---
#     (underscored names are fine to import; they're just a naming convention)
from inference_plot import (                      # reuses the exact loader & inference path
    _load_model_from_ckpt,
    _parse_rho,
    _broadcast_np,
    predict_pd_postclearing,
)
from sc_utils import build_L                      # for L/liab reuse (same as in plotter)

# =============================== USER CONFIG =============================== #
MODEL_DIR = "steps_check_5_banks_2_DD_1_kL/50"      # folder with files like: step0_corr_{rho:.4f}.pt
A_INPUT  = [2.0]                                  # scalar (broadcast) or length-n vector of A (distance-to-default)
A_DEAD   = -4.0                                   # sentinel for “already-defaulted”
kL = 1.0

# parameters that must match your training setup
SIGMA = 1.0
T_TOTAL = 1.0
DT = 0.1
STEP = 0  # keep at 0 for step-0 nets
K_SURV_ITERS = 5

# outputs (default to MODEL_DIR)
OUT_PD_CSV  = None  # e.g. "runs/avg_pd_vs_corr.csv"
OUT_MSE_CSV = None  # e.g. "runs/step_mse_vs_corr.csv"
# ========================================================================== #


def _list_ckpts(model_dir: Path) -> List[Path]:
    ckpts = sorted(model_dir.glob("step0_corr_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No 'step0_corr_*.pt' files in {model_dir}")
    return ckpts


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _extract_step_mse(raw: dict) -> Optional[List[float]]:
    """
    Best-effort extractor for per-step MSE vector that you said is attached
    to the saved checkpoint. Accepts list / np.ndarray / torch.Tensor.
    Returns None if nothing plausible is found.
    """
    candidate_keys = [
        "step_mse", "step_mse_list", "step_mses", "mse_by_step",
        "val_mse_by_step", "eval_mse", "mse_vec",
    ]

    def _to_list(v):
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return [float(x) for x in v.detach().cpu().flatten().tolist()]
        if isinstance(v, np.ndarray):
            return [float(x) for x in v.flatten().tolist()]
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                try:
                    out.append(float(x))
                except Exception:
                    return None
            return out
        return None

    # direct lookup
    for k in candidate_keys:
        if k in raw:
            out = _to_list(raw[k])
            if out is not None and len(out) > 0:
                return out

    # fallback: nested dict(s) commonly named 'meta' or 'trainer'
    for nest_key in ["meta", "trainer", "info", "extras"]:
        nested = raw.get(nest_key, None)
        if isinstance(nested, dict):
            for k in candidate_keys:
                if k in nested:
                    out = _to_list(nested[k])
                    if out is not None and len(out) > 0:
                        return out
    return None


def export_avg_pd_csv(
    model_dir: str | Path,
    A_vals: Sequence[float],
    *,
    sigma: float,
    T_total: float,
    dt: float,
    step: int,
    k_surv_iters: int,
    a_dead: float,
    out_csv: str | Path,
) -> Path:
    """
    For each step0_corr_*.pt in model_dir, compute post-clearing PDs at (A_vals)
    and write a CSV with columns: corr, mean_pd.
    """
    model_dir = Path(model_dir)
    ckpts = _list_ckpts(model_dir)

    # First model defines n (and F) for broadcasting input
    rho0, model0 = _load_model_from_ckpt(ckpts[0])
    n = int(model0.n)
    A_vec = _broadcast_np(list(A_vals), n, "A")
    A_t = torch.tensor(A_vec, dtype=torch.float32)

    # Clearing matrices (ρ affects training shocks, not L here)
    L = build_L(n, offdiag=kL / (n - 1))
    liab = L.sum(1)

    rows = []
    # Evaluate first checkpoint
    pd0 = predict_pd_postclearing(model0, A_t, sigma=sigma, T_total=T_total,
                                  dt=dt, step=step, k_surv_iters=k_surv_iters,
                                  a_dead=a_dead, L=L, liab=liab).numpy()
    rows.append((float(rho0), float(pd0.mean())))

    # Remaining checkpoints
    for f in ckpts[1:]:
        rho, model = _load_model_from_ckpt(f)
        if int(model.n) != n:
            raise RuntimeError(f"Mixed n across checkpoints: expected {n}, found {model.n} in {f.name}")
        pd = predict_pd_postclearing(model, A_t, sigma=sigma, T_total=T_total,
                                     dt=dt, step=step, k_surv_iters=k_surv_iters,
                                     a_dead=a_dead, L=L, liab=liab).numpy()
        rows.append((float(rho), float(pd.mean())))

    # Sort by correlation for a clean table
    rows.sort(key=lambda t: t[0])

    out_csv = Path(out_csv)
    _ensure_dir(out_csv)
    with out_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["corr", "mean_pd"])
        w.writerows(rows)
    print(f"[saved] {out_csv}  ({len(rows)} rows)")
    return out_csv


def export_step_mse_csv(
    model_dir: str | Path,
    *,
    out_csv: str | Path,
) -> Path:
    """
    For each step0_corr_*.pt in model_dir, read the attached per-step MSE vector
    and write a CSV with columns: corr, step0, step1, ...
    If a checkpoint lacks MSEs, its row is filled with NaNs and a warning is printed.
    """
    model_dir = Path(model_dir)
    ckpts = _list_ckpts(model_dir)

    # First pass: collect (rho, mse_list or None), track maximum length
    raw_rows: List[Tuple[float, Optional[List[float]]]] = []
    max_len = 0
    for f in ckpts:
        rho = _parse_rho(f)
        if rho is None:
            print(f"[warn] skip file with nonstandard name: {f.name}")
            continue

        raw_obj = torch.load(f, map_location="cpu")
        if not isinstance(raw_obj, dict):
            print(f"[warn] checkpoint is not a dict (got {type(raw_obj)}): {f.name}")
            mvec = None
        else:
            mvec = _extract_step_mse(raw_obj)

        if mvec is None:
            print(f"[warn] no attached step-MSE vector found in {f.name}; will write NaNs")
        else:
            max_len = max(max_len, len(mvec))

        raw_rows.append((float(rho), mvec))

    if max_len == 0:
        raise RuntimeError("No MSE vectors found in any checkpoint; nothing to write.")

    # Build header and padded rows
    header = ["corr"] + [f"step{i}" for i in range(max_len)]
    rows = []
    for rho, mvec in raw_rows:
        if mvec is None:
            pad = [math.nan] * max_len
        else:
            pad = list(mvec) + [math.nan] * (max_len - len(mvec))
        rows.append([rho] + [float(x) for x in pad])

    # Sort by correlation
    rows.sort(key=lambda r: r[0])

    out_csv = Path(out_csv)
    _ensure_dir(out_csv)
    with out_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        w.writerows(rows)
    print(f"[saved] {out_csv}  ({len(rows)} rows, {max_len} step columns)")
    return out_csv


def main():
    model_dir = Path(MODEL_DIR)
    # default outputs under the model folder
    out_pd = Path(OUT_PD_CSV) if OUT_PD_CSV is not None else (model_dir / "avg_pd_vs_corr.csv")
    out_mse = Path(OUT_MSE_CSV) if OUT_MSE_CSV is not None else (model_dir / "step_mse_vs_corr.csv")

    export_avg_pd_csv(
        model_dir=model_dir,
        A_vals=A_INPUT,
        sigma=SIGMA,
        T_total=T_TOTAL,
        dt=DT,
        step=STEP,
        k_surv_iters=K_SURV_ITERS,
        a_dead=A_DEAD,
        out_csv=out_pd,
    )

    export_step_mse_csv(
        model_dir=model_dir,
        out_csv=out_mse,
    )


if __name__ == "__main__":
    main()
