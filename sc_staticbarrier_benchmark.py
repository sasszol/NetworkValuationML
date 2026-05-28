"""sc_staticbarrier_benchmark.py

Dynamic "state-dependent static-barrier" (SDB) benchmark and related utilities.

The original purpose of this module is a dynamic benchmark: replace the
neural-net PD predictor with a *static flat-barrier* ABM crossing proxy, but
keep the SAME multi-step network revaluation / clearing dynamics (path-by-path)
used during TD training.

At each time step t_k we solve a *single* blended fixed point (Picard) that
matches the LaTeX snippet (Section "Training buffer via a State--dependent
static--barrier benchmark"):

    p^(0) = S^-(t)   ("alive pays in full")
    H(p)  = b - p L
    phi   = S^-(t) * [2 Φ((A-H)/σ√T_rem) - 1]
    p_new = phi if A + (phi L) - b >= 0 else 0

and we iterate until convergence (||p_new - p||_∞ < tol).

We then evolve external assets with correlated ABM increments (Euler step;
absorbing defaults at A_DEAD), and finally run the deterministic terminal
clearing at maturity.

Correlation sweeps
------------------
This module also provides a correlation sweep utility. In addition to the SDB
benchmark dynamics (method="static_barrier"), the sweep can be run with the
"rollout" naive path generator in sc_data (method="sc_data"). The sweep output
format matches the original benchmark: a CSV with NO header and one line per
correlation value:

    corr,avg_pd_T_total

where avg_pd_T_total is the empirical default frequency by maturity, averaged
across scenarios AND banks.

Run
---
python sc_staticbarrier_benchmark.py --cfg-module batch_process --out-csv out.csv --seed 43

Optional:
  --method static_barrier|sc_data
  --S 50000                 override number of scenarios
  --corr-grid-lin "0,1,11"  override correlation grid
  --tol 1e-6                 convergence tolerance for the SDB Picard map
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from sc_core import hard_clear_pd
from sc_utils import build_matrices, draw_shocks_factor, sample_A0_band


# ----------------------------- math helpers -----------------------------

def _phi_normal(z: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF Φ(z) via erf (works on CPU/GPU)."""
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


@torch.no_grad()
def survival_abm_flat_barrier(
    A: torch.Tensor,
    H: torch.Tensor,
    *,
    sigma: float,
    T: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """ABM survival above a flat barrier.

    Arithmetic Brownian motion X_s = A + σ W_s with absorbing flat barrier at H.

    Survival (no down-crossing) probability by time T:
        φ = 2 Φ((A-H)/(σ√T)) - 1

    Output is clipped to [0, 1].
    """
    T_eff = max(float(T), eps)
    denom = max(float(sigma) * math.sqrt(T_eff), eps)
    d = (A - H) / denom
    out = 2.0 * _phi_normal(d) - 1.0
    return torch.clamp(out, 0.0, 1.0)


# -------------------------- sampling: A(0) --------------------------

@torch.no_grad()
def sample_A0(cfg: Dict[str, Any], *, S: int, device: torch.device, rho: float) -> torch.Tensor:
    """Initial A(0) sampler shared with the rollout generator.

    This wrapper exists for backward compatibility; the implementation lives in
    sc_utils.sample_A0_band.
    """
    n = int(cfg["N_BANKS"])
    T_steps = int(cfg["TOTAL_STEPS"])
    dt = float(cfg["DT"])

    init_dd = float(cfg.get("INIT_DD", 1.0))
    sigma = float(cfg["SIGMA"])

    band_mult = float(cfg.get("BAND_MULT", 1.0))
    init_jitter = float(cfg.get("INIT_JITTER", 0.0))
    stress_prob = float(cfg.get("SCENARIO_STRESS_PROB", 0.0))
    stress_shift = float(cfg.get("SCENARIO_STRESS_SHIFT", 0.0))

    return sample_A0_band(
        int(S),
        int(n),
        init_dd=init_dd,
        sigma=sigma,
        rho=float(rho),
        T_steps=int(T_steps),
        dt=float(dt),
        device=device,
        band_mult=band_mult,
        init_jitter=init_jitter,
        stress_prob=stress_prob,
        stress_shift=stress_shift,
    )


# --------------------- blended Picard map (until convergence) ---------------------

def _apply_L_sb(spay: torch.Tensor,
                L: "torch.Tensor | None",
                ell: "float | None") -> torch.Tensor:
    """Compute spay @ L, optionally exploiting the symmetric rank-1 form.

    Mirror of sc_core._apply_L; duplicated here to avoid a circular import.
    """
    if ell is not None:
        Sigma = spay.sum(dim=-1, keepdim=True)
        return float(ell) * (Sigma - spay)
    return spay @ L


@torch.no_grad()
def projected_static_barrier_fp(
    A: torch.Tensor,  # (S,n)
    S_minus: torch.Tensor,  # (S,n) in {0,1}
    L: "torch.Tensor | None",  # (n,n) or None when ell is provided
    liab: torch.Tensor,  # (n,)
    *,
    sigma: float,
    T_rem: float,
    tol: float = 1e-6,
    max_iters_guard: int = 20000,
    ell: "float | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Blended fixed-point iteration used by the SDB benchmark.

    Implements the Picard map until convergence:

        p^(0) = S^-(t)
        H(p)  = b - p L
        phi   = S^-(t) * [2 Φ((A-H)/σ√T_rem) - 1]
        p_new = phi if A + (phi L) - b >= 0 else 0

    If `ell` is provided, the rank-1 form L = ell*(11^T - I) is used and the
    `L` argument is ignored.

    Returns
    -------
    p_star : (S,n) converged expected payment fractions / valuations
    s_plus : (S,n) post-projection survival indicator (1 iff p_star > 0)
    iters  : Picard iterations used
    """
    p = S_minus.clone()
    liab_row = liab.unsqueeze(0)

    it = 0
    while True:
        it += 1

        # Effective barrier H(p) = b - (p L)
        incoming_p = _apply_L_sb(p, L, ell)
        H = liab_row - incoming_p

        # ABM static-barrier survival under H, clipped and masked by S^-
        phi = S_minus * survival_abm_flat_barrier(A, H, sigma=float(sigma), T=float(T_rem))

        # Projection / solvency test uses phi (per the LaTeX snippet)
        recv = A + _apply_L_sb(phi, L, ell)
        solvent = (S_minus > 0.5) & (recv >= liab_row)

        p_new = torch.where(solvent, phi, torch.zeros_like(phi))

        max_diff = (p_new - p).abs().max().item()
        p = p_new

        if max_diff < float(tol):
            break

        if it >= int(max_iters_guard):
            raise RuntimeError(
                f"Picard map did not converge within {max_iters_guard} iterations "
                f"(last ||Δp||_∞={max_diff:.3e}, tol={tol:.3e})."
            )

    s_plus = (p > 0.0).float()
    return p, s_plus, it


# -------------------------- terminal clearing --------------------------

@torch.no_grad()
def terminal_clearing(
    A: torch.Tensor,  # (S,n) at maturity
    L: "torch.Tensor | None",
    liab: torch.Tensor,
    *,
    a_dead: float,
    n_it: int,
    ell: "float | None" = None,
) -> torch.Tensor:
    """Deterministic terminal clearing (same logic as LastStepPDIter).

    Returned value is the survival indicator z in {0,1}^n (as float tensor).

    Implementation uses sc_core.hard_clear_pd with zero PDs ("pays in full").
    If `ell` is provided, the rank-1 form of L is exploited inside.
    """
    pd0 = torch.zeros_like(A)
    _pd_final, s = hard_clear_pd(pd0, A, L, liab, n_it=int(n_it), a_dead=float(a_dead),
                                 ell=ell)
    return s


# ----------------------- buffer generation for training -----------------------

@torch.no_grad()
def multistep_asset_buffers_sdb(
    S: int,
    n: int,
    T: int,
    dt: float,
    device: torch.device,
    *,
    init_dd: float,
    sigma: float,
    rho: float,
    a_dead: float,
    band_mult: float = 1.0,
    init_jitter: float = 0.0,
    step_jitter_sd: float = 0.0,
    stress_prob: float = 0.0,
    stress_shift: float = 0.0,
    seed: int = 0,
    L: torch.Tensor,
    liab: torch.Tensor,
    T_total: float,
    tol_fp: float = 1e-6,
    ell: "float | None" = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Generate a multi-step training buffer using the SDB (static-barrier) proxy.

    This matches the LaTeX description in the SDB buffer section:
      1) evolve external assets with correlated ABM increments
      2) at each time t, compute p(t) via the converged Picard iteration
      3) project / clear at t to obtain S^+(t)
      4) propagate survivors to t+1 via the next ABM increment, carry defaults at A_DEAD

    Returns the same data structure as sc_data.multistep_asset_buffers:
      - A_buf_list: list[T] of (S,n) pre-clearing assets at times t=0..T-1
      - dR_buf_list: list[T] of (S,n) ABM increments used for the propagation
      - alive_path: (S,T+1,n) indicator inferred from the propagated A(t) values

    Notes
    -----
    * This generator intentionally does NOT use Brownian-bridge within-step hits.
    * This generator intentionally does NOT apply EXTRA_DEFAULTS overlay.
    """
    torch.manual_seed(int(seed))

    S_ = int(S)
    n_ = int(n)
    T_ = int(T)
    dt_ = float(dt)
    sig = float(sigma)
    rho_ = float(rho)
    a_dead_ = float(a_dead)

    # Initial assets (same family as the rollout sampler)
    A = sample_A0_band(
        S_,
        n_,
        init_dd=float(init_dd),
        sigma=sig,
        rho=rho_,
        T_steps=T_,
        dt=dt_,
        device=device,
        band_mult=float(band_mult),
        init_jitter=float(init_jitter),
        stress_prob=float(stress_prob),
        stress_shift=float(stress_shift),
    )

    A_buf_list: List[torch.Tensor] = []
    dR_buf_list: List[torch.Tensor] = []

    alive_path = torch.empty(S_, T_ + 1, n_, device=device)
    alive_path[:, 0, :] = (A > (a_dead_ + 1e-8)).float()

    for t in range(T_):
        # Buffer uses the *pre-clearing* state at time t
        A_buf_list.append(A.clone())

        T_rem = max(float(T_total) - t * dt_, 1e-8)
        S_minus = (A > (a_dead_ + 1e-8)).float()

        _p_star, s_plus, _iters = projected_static_barrier_fp(
            A,
            S_minus,
            L,
            liab,
            sigma=sig,
            T_rem=T_rem,
            tol=float(tol_fp),
            ell=ell,
        )

        dR = draw_shocks_factor(S_, n_, rho_, sigma=sig, dt=dt_, device=device)
        if float(step_jitter_sd) > 0.0:
            dR = dR + torch.randn_like(dR) * float(step_jitter_sd)

        dR_buf_list.append(dR)

        A = torch.where(s_plus > 0.5, A + dR, torch.full_like(A, a_dead_))
        alive_path[:, t + 1, :] = (A > (a_dead_ + 1e-8)).float()

    A_buf_list = [x.contiguous() for x in A_buf_list]
    dR_buf_list = [x.contiguous() for x in dR_buf_list]
    return A_buf_list, dR_buf_list, alive_path


# -------------------------- simulation drivers --------------------------


def _auto_device() -> torch.device:
    # Auto device, but do not expose it to CLI.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def simulate_static_barrier_once(
    cfg: Dict[str, Any],
    *,
    rho: float,
    seed: int,
    S_override: Optional[int] = None,
    tol_fp: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    """Run the full T-step SDB dynamics and return avg terminal default frequency."""
    device = _auto_device()

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    n = int(cfg["N_BANKS"])
    T_steps = int(cfg["TOTAL_STEPS"])
    dt = float(cfg["DT"])
    T_total = float(cfg["T_TOTAL"])
    sigma = float(cfg["SIGMA"])
    a_dead = float(cfg.get("A_DEAD", -10.0))

    S = int(S_override) if S_override is not None else int(cfg.get("VAL_SAMPLES", cfg.get("BUFFER_SIZE", 10000)))

    # Matrices exactly as in training
    L, liab = build_matrices(cfg, device)

    # Optional rank-1 acceleration when the symmetric/homogeneous network is assumed
    use_sym = bool(cfg.get("USE_SYMMETRIC_ASSUMPTION", False))
    ell = (float(cfg.get("kL", 1.0)) / max(1, n - 1)) if use_sym else None

    # Initial assets
    A = sample_A0(cfg, S=S, device=device, rho=float(rho))

    fp_iters_total = 0

    # Forward simulate path-by-path
    for t in range(T_steps):
        T_rem = max(T_total - t * dt, 1e-8)

        S_minus = (A > (a_dead + 1e-8)).float()

        # Blended Picard map -> valuations + post-projection survivals
        _p_star, s_plus, it_used = projected_static_barrier_fp(
            A,
            S_minus,
            L,
            liab,
            sigma=sigma,
            T_rem=T_rem,
            tol=tol_fp,
            ell=ell,
        )
        fp_iters_total += int(it_used)

        # Evolve external assets with correlated ABM increments
        # NOTE: draw_shocks_factor assumes rho in [0,1].
        dR = draw_shocks_factor(S, n, float(rho), sigma=sigma, dt=dt, device=device)

        # Absorbing defaults
        A = torch.where(s_plus > 0.5, A + dR, torch.full_like(A, float(a_dead)))

    # Terminal deterministic clearing at maturity
    z = terminal_clearing(A, L, liab, a_dead=a_dead, n_it=n, ell=ell)
    avg_pd = float((1.0 - z).mean().item())

    diag = {
        "fp_iters_avg_per_step": fp_iters_total / max(1, T_steps),
    }
    return avg_pd, diag


# Backward compatible name (older scripts might import it)
simulate_benchmark_once = simulate_static_barrier_once


@torch.no_grad()
def simulate_sc_data_once(
    cfg: Dict[str, Any],
    *,
    rho: float,
    seed: int,
    S_override: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """Simulate using sc_data's rollout sampler and count default frequency at maturity.

    This follows the same external-asset evolution and barrier-hitting logic as
    sc_data.build_rollout_paths (including optional EXTRA_DEFAULTS overlay).

    For comparability with the SDB benchmark, we apply the deterministic terminal
    clearing at maturity (which can only ADD defaults).
    """
    device = _auto_device()

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    from sc_data import build_rollout_paths, apply_extra_defaults_per_step

    n = int(cfg["N_BANKS"])
    T_steps = int(cfg["TOTAL_STEPS"])
    dt = float(cfg["DT"])
    T_total = float(cfg["T_TOTAL"])
    sigma = float(cfg["SIGMA"])
    a_dead = float(cfg.get("A_DEAD", -10.0))

    S = int(S_override) if S_override is not None else int(cfg.get("VAL_SAMPLES", cfg.get("BUFFER_SIZE", 10000)))

    L, liab = build_matrices(cfg, device)

    # Optional rank-1 acceleration when the symmetric/homogeneous network is assumed.
    # Derived early so the data-generation overlay also benefits from the speedup.
    use_sym = bool(cfg.get("USE_SYMMETRIC_ASSUMPTION", False))
    ell = (float(cfg.get("kL", 1.0)) / max(1, n - 1)) if use_sym else None

    A_path, dR_all, alive0 = build_rollout_paths(
        S,
        n,
        T_steps,
        dt,
        device,
        init_dd=float(cfg.get("INIT_DD", 1.0)),
        sigma=float(sigma),
        rho=float(rho),
        a_dead=float(a_dead),
        band_mult=float(cfg.get("BAND_MULT", 1.0)),
        init_jitter=float(cfg.get("INIT_JITTER", 0.0)),
        step_jitter_sd=float(cfg.get("STEP_JITTER_SD", 0.0)),
        stress_prob=float(cfg.get("SCENARIO_STRESS_PROB", 0.0)),
        stress_shift=float(cfg.get("SCENARIO_STRESS_SHIFT", 0.0)),
        seed=int(seed),
    )

    if bool(cfg.get("EXTRA_DEFAULTS_ON", False)):
        A_path, alive_path = apply_extra_defaults_per_step(
            A_path,
            dR_all,
            L,
            liab,
            sigma=float(sigma),
            dt=float(dt),
            T_total=float(T_total),
            rho=float(rho),
            a_dead=float(a_dead),
            k_surv_iters=int(cfg.get("K_SURV_ITERS", 5)),
            extra_on=True,
            extra_intensity=float(cfg.get("EXTRA_INTENSITY", 0.20)),
            extra_cap=float(cfg.get("EXTRA_CAP", 0.25)),
            extra_rho=cfg.get("EXTRA_RHO", None),
            seed=int(seed),
            ell=ell,
        )
    else:
        alive_path = alive0

    A_T = A_path[:, -1, :]

    # Default rate before terminal clearing (i.e. purely from the rollout generator)
    pre_term_surv = (A_T > (a_dead + 1e-8)).float()
    pre_term_pd = float((1.0 - pre_term_surv).mean().item())

    # Terminal deterministic clearing at maturity (can only reduce survivals)
    z = terminal_clearing(A_T, L, liab, a_dead=a_dead, n_it=n, ell=ell)
    avg_pd = float((1.0 - z).mean().item())

    diag = {
        "pre_terminal_avg_pd": pre_term_pd,
    }
    return avg_pd, diag


def _normalize_method(method: str) -> str:
    m = str(method).strip().lower()
    if m in {"static_barrier", "static", "sdb", "benchmark"}:
        return "static_barrier"
    if m in {"sc_data", "rollout", "rollout_paths", "data"}:
        return "sc_data"
    raise ValueError(f"Unknown method={method!r}. Expected one of: static_barrier|sc_data")


@torch.no_grad()
def simulate_once(
    cfg: Dict[str, Any],
    *,
    method: str,
    rho: float,
    seed: int,
    S_override: Optional[int] = None,
    tol_fp: float = 1e-6,
) -> Tuple[float, Dict[str, float]]:
    """Dispatch to one of the supported forward simulators."""
    m = _normalize_method(method)
    if m == "static_barrier":
        return simulate_static_barrier_once(cfg, rho=rho, seed=seed, S_override=S_override, tol_fp=tol_fp)
    if m == "sc_data":
        return simulate_sc_data_once(cfg, rho=rho, seed=seed, S_override=S_override)
    raise AssertionError("unreachable")


# -------------------------- correlation sweep --------------------------


def _default_corr_grid() -> np.ndarray:
    """Default grid (nonnegative, because the one-factor shock generator assumes rho >= 0)."""
    part1 = np.linspace(0.0, 0.8, 20, endpoint=True)
    part2 = np.linspace(0.8, 1.0, 10, endpoint=True)[1:]
    return np.concatenate([part1, part2])


def _parse_corr_grid_lin(spec: str) -> np.ndarray:
    toks = [t.strip() for t in spec.split(",")]
    if len(toks) != 3:
        raise ValueError(f"--corr-grid-lin expects 'start,end,num', got: {spec!r}")
    a = float(toks[0])
    b = float(toks[1])
    num = int(toks[2])
    return np.linspace(a, b, num, endpoint=True)


def sweep_correlation(
    cfg_base: Dict[str, Any],
    corr_values: Iterable[float],
    *,
    seed: int,
    out_csv: str,
    method: str = "static_barrier",
    S_override: Optional[int] = None,
    tol_fp: float = 1e-6,
) -> None:
    """Run a correlation sweep and write a 2-column CSV: corr,avg_pd_T_total."""
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    method_norm = _normalize_method(method)

    rows: List[Tuple[float, float]] = []

    for c in corr_values:
        c = float(c)
        cfg = dict(cfg_base)
        cfg["ASSET_CORR"] = c

        avg_pd, diag = simulate_once(cfg, method=method_norm, rho=c, seed=seed, S_override=S_override, tol_fp=tol_fp)

        if method_norm == "static_barrier":
            extra = f" (fp iters/step≈{diag.get('fp_iters_avg_per_step', float('nan')):.2f})"
        else:
            extra = ""
            if "pre_terminal_avg_pd" in diag:
                extra = f" (pre-terminal≈{diag['pre_terminal_avg_pd']:.6f})"

        print(f"[{method_norm} | corr={c: .4f}] avg_pd_T_total={avg_pd:.6f}{extra}")
        rows.append((c, float(avg_pd)))

    with open(out_path, "w", encoding="utf-8") as f:
        for c, pdv in rows:
            f.write(f"{c},{pdv}\n")

    print(f"[saved] {out_path}")


# -------------------------- CLI --------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg-module",
        type=str,
        required=True,
        help="Python module that exposes CFG dict, e.g. batch_process",
    )
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out-csv", type=str, required=True)
    ap.add_argument(
        "--method",
        type=str,
        default="static_barrier",
        help="Which simulator to use: static_barrier (SDB) or sc_data (rollout generator)",
    )
    ap.add_argument("--S", type=int, default=None, help="Override number of scenarios")
    ap.add_argument(
        "--corr-grid-lin",
        type=str,
        default=None,
        help='Override correlation grid as "start,end,num" (use nonnegative rho)',
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for the blended Picard map (SDB only)",
    )
    args = ap.parse_args()

    mod = importlib.import_module(args.cfg_module)
    cfg = dict(getattr(mod, "CFG"))

    if args.corr_grid_lin is not None:
        corr_values = _parse_corr_grid_lin(args.corr_grid_lin)
    else:
        corr_values = _default_corr_grid()

    sweep_correlation(
        cfg,
        corr_values,
        seed=int(args.seed),
        out_csv=str(args.out_csv),
        method=str(args.method),
        S_override=args.S,
        tol_fp=float(args.tol),
    )


if __name__ == "__main__":
    main()
