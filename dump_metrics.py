"""
Dump correlation -> averaged PD and correlation -> per-step MSEs to CSV,
reusing the exact inference & clearing path from inference_plot.py.

DeepSets note
-------------
DeepSets checkpoints are N-agnostic, so if A_INPUT is scalar you must tell the
exporter which evaluation size to use. For your current 20-bank runs the default
below is N_EVAL = 20.
"""

from __future__ import annotations

from pathlib import Path
import csv
import math
from typing import List, Optional, Tuple, Sequence

import numpy as np
import torch

from inference_plot import (
    _load_model_from_ckpt,
    _parse_rho,
    _broadcast_np,
    predict_pd_postclearing,
    _PDInferDeepSets,
)
from sc_utils import build_L

# =============================== USER CONFIG =============================== #
MODEL_DIR = "C:/git/NetworkValuationML/kL_1_DD_2/5_banks"   # folder with files like step0_corr_{rho:.4f}.pt
A_INPUT = [2.0]                                  # scalar (broadcast) or length-n vector of A
N_EVAL = 5                                       # used for DeepSets when A_INPUT is scalar; change if needed
A_DEAD = -4.0
kL = 1.0
L_MATRIX = None                                  # optional custom exposure matrix (list[list[float]])
LIAB_VECTOR = None                               # optional custom liabilities; default = row sums of L_MATRIX

SIGMA = 1.0
T_TOTAL = 1.0
DT = 0.1
STEP = 0
K_SURV_ITERS = 5

OUT_PD_CSV = None
OUT_MSE_CSV = None
# ========================================================================== #


def _infer_eval_n(model, A_vals: Sequence[float]) -> int:
    """Infer the evaluation bank count.

    Flat MLP checkpoints store n directly. DeepSets checkpoints are N-agnostic,
    so n must come from the evaluation request or user config.
    """
    if not isinstance(model, _PDInferDeepSets):
        return int(model.n)

    explicit_n: Optional[int] = None

    if N_EVAL is not None:
        explicit_n = int(N_EVAL)
        if explicit_n <= 0:
            raise ValueError(f"N_EVAL must be positive when set (got {N_EVAL}).")

    if L_MATRIX is not None:
        L_arr = np.asarray(L_MATRIX)
        if L_arr.ndim != 2 or L_arr.shape[0] != L_arr.shape[1]:
            raise ValueError(f"L_MATRIX must be square (got shape {L_arr.shape}).")
        n_from_L = int(L_arr.shape[0])
        if explicit_n is not None and explicit_n != n_from_L:
            raise ValueError(f"N_EVAL={explicit_n} disagrees with L_MATRIX shape {L_arr.shape}.")
        explicit_n = n_from_L

    if LIAB_VECTOR is not None:
        n_from_liab = int(len(LIAB_VECTOR))
        if explicit_n is not None and explicit_n != n_from_liab:
            raise ValueError(f"N_EVAL={explicit_n} disagrees with LIAB_VECTOR length {n_from_liab}.")
        explicit_n = n_from_liab

    if len(A_vals) > 1:
        n_from_A = int(len(A_vals))
        if explicit_n is not None and explicit_n != n_from_A:
            raise ValueError(
                f"A_INPUT length {n_from_A} disagrees with N_EVAL/custom-L size {explicit_n}."
            )
        return n_from_A

    if explicit_n is not None:
        return explicit_n

    raise ValueError(
        "DeepSets checkpoint does not encode the evaluation bank count. "
        "With scalar A_INPUT, set N_EVAL (for example 5), or pass A_INPUT as a "
        "length-n vector, or provide L_MATRIX/LIAB_VECTOR."
    )


def _eval_L_liab(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluation-time clearing matrices, optionally custom."""
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

    if int(n) <= 1:
        raise ValueError(
            "Default homogeneous clearing matrix build_L(n, kL/(n-1)) requires n >= 2. "
            "For DeepSets with scalar A_INPUT, set N_EVAL >= 2 or provide L_MATRIX."
        )

    L = build_L(n, offdiag=kL / (n - 1))
    liab = L.sum(1)
    return L, liab


def _list_ckpts(model_dir: Path) -> List[Path]:
    ckpts = sorted(model_dir.glob("step0_corr_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No 'step0_corr_*.pt' files in {model_dir}")
    return ckpts


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _extract_step_mse(raw: dict) -> Optional[List[float]]:
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

    for k in candidate_keys:
        if k in raw:
            out = _to_list(raw[k])
            if out is not None and len(out) > 0:
                return out

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
    """For each step0_corr_*.pt, compute post-clearing PDs and write corr,mean_pd."""
    model_dir = Path(model_dir)
    ckpts = _list_ckpts(model_dir)

    rho0, model0 = _load_model_from_ckpt(ckpts[0])
    n = _infer_eval_n(model0, A_vals)
    A_vec = _broadcast_np(list(A_vals), n, "A")
    A_t = torch.tensor(A_vec, dtype=torch.float32)

    L, liab = _eval_L_liab(n)

    rows = []
    pd0 = predict_pd_postclearing(
        model0, A_t, sigma=sigma, T_total=T_total, dt=dt, step=step,
        k_surv_iters=k_surv_iters, a_dead=a_dead, L=L, liab=liab,
    ).numpy()
    rows.append((float(rho0), float(pd0.mean())))

    for f in ckpts[1:]:
        rho, model = _load_model_from_ckpt(f)
        n_this = _infer_eval_n(model, A_vals)
        if int(n_this) != int(n):
            raise RuntimeError(f"Mixed evaluation n across checkpoints: expected {n}, found {n_this} in {f.name}")
        pd = predict_pd_postclearing(
            model, A_t, sigma=sigma, T_total=T_total, dt=dt, step=step,
            k_surv_iters=k_surv_iters, a_dead=a_dead, L=L, liab=liab,
        ).numpy()
        rows.append((float(rho), float(pd.mean())))

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
    """For each step0_corr_*.pt, read the attached per-step MSE vector and write CSV."""
    model_dir = Path(model_dir)
    ckpts = _list_ckpts(model_dir)

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

    header = ["corr"] + [f"step{i}" for i in range(max_len)]
    rows = []
    for rho, mvec in raw_rows:
        if mvec is None:
            pad = [math.nan] * max_len
        else:
            pad = list(mvec) + [math.nan] * (max_len - len(mvec))
        rows.append([rho] + [float(x) for x in pad])

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
