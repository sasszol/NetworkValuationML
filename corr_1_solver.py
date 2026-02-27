# rho1_exact_solver.py
# ---------------------------------------------------------------
# Exact, neural-net-free computation of self-consistent default
# probabilities under rho=1 for your symmetric clearing model.
#
# It solves p(A, tau) backward in time:
#   r(A, tau) = E_Z[ p(A + sigma*sqrt(dt)*Z, tau - dt) ]
#   p(A, tau) = 1                  if A + (1 - r)*c <= ell
#             = r(A, tau)          if A + (1 - r)*c >  ell
#
# Terminal condition (deterministic final clearing):
#   p(A, 0) = 1{ A + c <= ell }      → with your default L: p(A,0)=1{A<=0}.
#
# This matches your teacher + hard_clear_pd under symmetry:
#   - "raw PD" at step t is r(A, tau)
#   - clearing at t uses expected-pay fraction (1 - r)
#   - next step reuses p(., tau-dt)
#
# References in your repo this mirrors:
#   • draw_shocks_factor (ABM increments)                 [sc_utils.py]   (rho→1)  :contentReference[oaicite:8]{index=8}
#   • hard_clear_pd (expected-pay clearing)               [sc_core.py]             :contentReference[oaicite:9]{index=9}
#   • LastStepPDIter (deterministic final clearing)       [sc_core.py]             :contentReference[oaicite:10]{index=10}
#   • simulate_one_step_with_projection (teacher layout)  [sc_train.py]            :contentReference[oaicite:11]{index=11}
# ---------------------------------------------------------------

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np

# ------------------------ network / clearing params -------------------------

@dataclass
class NetSpec:
    n_banks: int = 10
    # symmetric complete network with equal off-diagonal weights 1/(n-1)
    # row sum (own interbank liabilities) ell = 1, column sum c = 1 by construction
    colsum_c: float = 1.0  # ∑_j L_{j,i}, receipts capacity per bank
    liab_ell: float = 1.0  # ∑_k L_{i,k}, interbank liabilities per bank

@dataclass
class ModelSpec:
    sigma: float = 1.0       # ABM volatility (per sqrt(year), say)
    T_total: float = 1.0     # total horizon
    dt: float = 0.1          # time step
    a_dead: float = -4.0    # absorbing asset level (unused here; we stop via clearing)
    gh_nodes: int = 32       # Gauss–Hermite nodes for the one-step expectation

@dataclass
class GridSpec:
    A0: float = 1.5          # common starting asset level (distance-to-default)
    A_min: Optional[float] = None
    A_max: Optional[float] = None
    nA: int = 801            # dense 1D grid (odd helps monotone checks)

# --------------------------- utilities --------------------------------------

def _hermgauss(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nodes (x_k) and weights (w_k) for ∫ e^{-x^2} f(x) dx ≈ Σ w_k f(x_k).
    Expectation over N(0,1):  E[f(Z)] = (1/√π) Σ w_k f(√2 x_k).
    """
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(int(n))
    return x.astype(np.float64), w.astype(np.float64)

def _interp_clamped(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    1D linear interpolation with clamped tails:
      left  → fp[0]
      right → fp[-1]
    """
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1])

# --------------------------- main solver ------------------------------------

@dataclass
class Rho1Result:
    A_grid: np.ndarray
    p_grid: np.ndarray  # shape (N_tau+1, nA); p_grid[0,:] is p(., tau=0)
    tau_grid: np.ndarray
    p0: float           # p(A0, T_total)

def solve_self_consistent_pd_rho1(
    net: NetSpec,
    mdl: ModelSpec,
    grd: GridSpec,
) -> Rho1Result:
    """
    Exact rho=1 self-consistent PD under your clearing rules.

    Returns the whole surface p(A, tau) on a regular A-grid and time grid
    (tau = 0, dt, 2dt, ... , T_total), plus the step-0 PD at A0.
    """
    sigma, T_total, dt = float(mdl.sigma), float(mdl.T_total), float(mdl.dt)
    c, ell = float(net.colsum_c), float(net.liab_ell)

    # --- time grid (allow a non-integer last step if needed)
    if T_total <= 0:
        raise ValueError("T_total must be positive.")
    n_steps = max(1, int(math.floor(T_total / dt)))
    tail = T_total - n_steps * dt
    if tail > 1e-12:
        deltas = [dt] * n_steps + [tail]
    else:
        deltas = [dt] * n_steps
    tau_grid = np.concatenate(([0.0], np.cumsum(deltas)[::-1]))[::-1]  # 0..T_total

    # --- asset grid
    A0 = float(grd.A0)
    # choose wide bounds so p(A, tau) saturates at the ends
    width = 6.0 * sigma * math.sqrt(T_total + 1e-12)
    A_min = grd.A_min if grd.A_min is not None else min(-10.0, A0 - width) - 1.0
    # right end: beyond ell - c (terminal barrier) and typical upward moves
    A_max = grd.A_max if grd.A_max is not None else max(5.0, A0 + width) + 1.0
    nA = int(grd.nA)
    A = np.linspace(A_min, A_max, nA, dtype=np.float64)

    # --- terminal condition: deterministic final step clearing
    # LastStepPDIter under symmetry sets p(A,0)=1{ A + c <= ell }.
    p_next = (A + c <= ell).astype(np.float64)

    # --- prepare Gauss–Hermite nodes/weights for one-step expectation
    x, w = _hermgauss(mdl.gh_nodes)                      # for e^{-x^2}
    z_fac = sigma * math.sqrt(2.0)                       # √2 σ
    sqrt = np.sqrt  # minor speed

    # --- backward induction in tau
    # p_grid[k,:] will store p(A, tau_k) with tau_0 = 0.
    p_slices = [p_next.copy()]  # start with tau=0 slice

    tau_now = 0.0
    for dt_k in deltas:  # build up to T_total
        tau_now += dt_k

        # raw PD: r(A) = E[ p_next(A + σ√dt_k Z) ]
        shifts = z_fac * sqrt(dt_k) * x[None, :]       # (1, M)
        A_plus = A[:, None] + shifts                   # (nA, M)
        pvals = _interp_clamped(A_plus, A, p_next)     # (nA, M): linear interp
        r = (pvals * w[None, :]).sum(axis=1) / math.sqrt(math.pi)  # (nA,)

        # clearing at time t under expected payments: s = 1{ A + (1 - r) c > ell }
        s_one = (A + (1.0 - r) * c > ell).astype(np.float64)

        # post-clearing PD at current tau: p(A, tau) = (1 - s) + s * r
        p_now = (1.0 - s_one) + s_one * r

        p_slices.append(p_now)
        p_next = p_now  # for the next outer loop

    # assemble p_grid in ascending tau order: [tau=0, dt, 2dt, ..., T_total]
    p_grid = np.stack(p_slices, axis=0)
    # evaluate p at (A0, T_total)
    p0 = float(_interp_clamped(np.array([A0]), A, p_grid[-1, :])[0])

    return Rho1Result(A_grid=A, p_grid=p_grid, tau_grid=tau_grid, p0=p0)

# --------------------------- a small driver ---------------------------------

if __name__ == "__main__":
    # Example: mimic your defaults (two banks or many — symmetric L gives c=ell=1).
    kL = 1.0
    net = NetSpec(n_banks=5, colsum_c=kL, liab_ell=kL)

    mdl = ModelSpec(sigma=1.0, T_total=1.0, dt=0.001, gh_nodes=32)
    grd = GridSpec(A0=2.0, nA=801)

    out = solve_self_consistent_pd_rho1(net, mdl, grd)
    print(f"rho=1 self-consistent PD at A0={grd.A0}, T={mdl.T_total}:  {out.p0:.6f}")
    # You can also inspect out.A_grid, out.tau_grid, out.p_grid for diagnostics.
