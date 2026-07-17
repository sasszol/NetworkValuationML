# sc_data.py
import math
from typing import Tuple, List, Optional

import torch
from sc_utils import draw_shocks_factor, iterative_breach_probability, sample_A0_band
# draw_shocks_factor: one-factor ABM increments with variance sigma^2 * dt.
# iterative_breach_probability is used only for the optional extra-default overlay.

# -------------------- Original helpers (kept for backward compatibility) --------------------

@torch.no_grad()
def create_asset_buffer(sz: int, n: int, device: torch.device,
                        init_dd: float, sigma: float, ttm: float,
                        p0_alive: float = 1.0, a_dead: float = -10.0, seed: int = 0):
    """
    Original sampler: wide uniform in [a_min, a_max] plus optional independent 'dead' overlay.
    Kept unchanged so older scripts continue to run.
    """
    torch.manual_seed(seed)
    width = 0.4 + 0.1 * float(sigma) * math.sqrt(max(ttm, 1e-8))
    a_min = float(init_dd) - width
    a_max = float(init_dd) + width
    # Historical override in repo:
    a_min = 0
    a_max = 7
    A0 = torch.rand(sz, n, device=device) * (a_max - a_min) + a_min
    if p0_alive < 1.0:
        dead_mask = (torch.rand(sz, n, device=device) >= float(p0_alive))
        A0 = torch.where(dead_mask, torch.full_like(A0, float(a_dead)), A0)
    return A0

@torch.no_grad()
def make_dR_buffer_old(sz: int, n: int, chol: torch.Tensor,
                       sigma: float, dt: float, device: torch.device):
    """Legacy (unused here): ABM increments via Cholesky."""
    z = torch.randn(sz, n, device=device)
    return (z @ chol.T) * (sigma * math.sqrt(dt))

def make_dR_buffer(sz: int, n: int, rho: float, sigma: float, dt: float, device: torch.device):
    """Legacy helper: single‑step increments (kept for compatibility)."""
    return draw_shocks_factor(sz, n, rho, sigma, dt, device)

# -------------------- New: multi‑step rollout with stochastic overlay ------------------------

@torch.no_grad()
def build_rollout_paths(
    S: int,                   # scenarios
    n: int,                   # banks
    T: int,                   # number of steps   (grid times t=0..T)
    dt: float,
    device: torch.device,
    *,
    init_dd: float,
    sigma: float,
    rho: float,
    a_dead: float = -10.0,
    band_mult: float = 1.0,     # width around init_dd (scenario‑level)
    init_jitter: float = 0.05,  # per‑bank jitter as fraction of width, scaled by √(1-ρ)
    step_jitter_sd: float = 0.0,# optional extra tiny idiosyncratic jitter per step (absolute units)
    stress_prob: float = 0.0,   # scenario mixture
    stress_shift: float = -1.0, # shift applied to init_dd if stressed
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      A_path : (S, T+1, n)  assets (distance‑to‑default); a_dead for defaulted names
      dR_all : (S, T,   n)  ABM increments used to evolve A (before overlay)
      alive  : (S, T+1, n)  1 if alive, else 0

    Model:
      • ABM increments with one‑factor correlation ρ (your draw_shocks_factor).
      • Row‑coherent initial A0 around init_dd; dispersion shrinks as ρ→1.
      • First‑passage within each step using Brownian‑bridge crossing prob.
    """
    torch.manual_seed(int(seed))
    barrier = 0.0
    S_, n_, T_ = int(S), int(n), int(T)
    dt = float(dt)
    sig = float(sigma)

    # Shared initial-state sampler (kept consistent with the static-barrier benchmark).
    A = sample_A0_band(
        S_, n_,
        init_dd=float(init_dd),
        sigma=sig,
        rho=float(rho),
        T_steps=T_,
        dt=dt,
        device=device,
        band_mult=float(band_mult),
        init_jitter=float(init_jitter),
        stress_prob=float(stress_prob),
        stress_shift=float(stress_shift),
    )

    A_path = torch.empty(S_, T_ + 1, n_, device=device)
    dR_all = torch.empty(S_, T_, n_, device=device)
    alive  = torch.empty(S_, T_ + 1, n_, device=device)

    A_path[:, 0, :] = A
    alive[:, 0, :]  = (A > barrier).float()

    # precompute denominator for the Brownian‑bridge hit probability
    eps_den = max(1e-12, (sig ** 2) * dt)

    alive_t = alive[:, 0, :].clone()
    for t in range(T_):
        # ABM increments + optional tiny extra idio jitter (does not depend on ρ)
        dR = draw_shocks_factor(S_, n_, float(rho), sig, dt, device)
        if step_jitter_sd > 0.0:
            dR = dR + torch.randn_like(dR) * float(step_jitter_sd)
        dR_all[:, t, :] = dR

        A_free = A + dR                     # proposed next asset (no overlay yet)

        # obvious hits at the step end
        end_hit = (alive_t > 0.5) & (A_free <= barrier)

        # Brownian‑bridge hit if both endpoints are > 0
        both_pos = (alive_t > 0.5) & (A > barrier) & (A_free > barrier)
        x = torch.clamp(A, min=0.0); y = torch.clamp(A_free, min=0.0)
        p_hit = torch.zeros_like(A)
        p_hit[both_pos] = torch.exp(-2.0 * x[both_pos] * y[both_pos] / eps_den)
        bb_hit = (torch.rand_like(A) < p_hit) & both_pos

        new_dead = end_hit | bb_hit
        alive_t = alive_t * (~new_dead).float()
        A = torch.where(new_dead, torch.full_like(A, float(a_dead)), A_free)

        A_path[:, t + 1, :] = A
        alive[:,  t + 1, :] = alive_t

    return A_path, dR_all, alive

@torch.no_grad()
def apply_extra_defaults_per_step(
    A_path: torch.Tensor,           # (S, T+1, n)
    dR_all: torch.Tensor,           # (S, T,   n) – not used in overlay, kept for training
    L: torch.Tensor,                # (n, n)
    liab: torch.Tensor,             # (n,)
    *,
    sigma: float,
    dt: float,
    T_total: float,
    rho: float,
    a_dead: float,
    k_surv_iters: int = 5,
    # ---- stochastic overlay knobs (all optional) ----
    extra_on: bool = True,
    extra_intensity: float = 0.30,  # scales breach probability to get extra default probability
    extra_cap: float = 0.35,        # ceiling for per‑name extra default probability
    extra_rho: Optional[float] = None,  # correlation for the copula (defaults to ASSET_CORR)
    seed: int = 0,
    ell: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adds optional per-step stochastic defaults correlated with A and the network
    through a counterparty-aware breach proxy. Sampling uses a one-factor Gaussian
    copula with correlation extra_rho.

    If ``ell`` is provided (homogeneous symmetric network: L_ij = ell off-diagonal,
    L_ii = 0), the breach proxy is computed via the rank-1 fast path. Pass
    ``ell = kL / (N - 1)`` when ``USE_SYMMETRIC_ASSUMPTION`` is True.

    Returns (A_path_with_overlay, alive_path_with_overlay).
    """
    if not extra_on or extra_intensity <= 0.0 or extra_cap <= 0.0:
        # no overlay -> return unchanged
        S, T1, n = A_path.shape
        alive = (A_path > 0.0).float()
        return A_path, alive

    torch.manual_seed(int(seed))
    S, T1, n = A_path.shape
    T = T1 - 1
    device = A_path.device
    barrier = 0.0
    erho = float(extra_rho) if extra_rho is not None else float(rho)

    A = A_path[:, 0, :].clone()
    alive_t = (A > barrier).float()
    outA = A_path.clone()
    outAlive = (A_path > barrier).float()

    for t in range(T):
        T_now = max(float(T_total) - t * float(dt), 1e-8)
        A_t = outA[:, t, :]

        # Only compute the overlay breach probability for still-alive names
        breach_t = iterative_breach_probability(A_t, L, liab,
                                         sigma=float(sigma), T=T_now,
                                         k=int(k_surv_iters), a_dead=float(a_dead),
                                         ell=ell)
        # map breach probability to extra default probability; clip to avoid 0/1
        p_extra = torch.clamp(breach_t * float(extra_intensity), 0.0, float(extra_cap))
        p_extra = torch.where(A_t <= barrier, torch.zeros_like(p_extra), p_extra)

        # One‑factor Gaussian copula sampling for correlation across banks
        mkt = torch.randn(S, 1, device=device)
        idio = torch.randn(S, n, device=device)
        Z = math.sqrt(erho) * mkt + math.sqrt(max(0.0, 1.0 - erho)) * idio
        U = 0.5 * (1.0 + torch.erf(Z / math.sqrt(2.0)))  # Φ(Z)

        new_extra = ((U < p_extra) & (A_t > barrier)).float()

        # Apply overlay defaults *after* the ABM step (t→t+1) already stored in A_path
        A_tp1 = outA[:, t + 1, :]
        A_tp1 = torch.where(new_extra > 0.5, torch.full_like(A_tp1, float(a_dead)), A_tp1)

        outA[:, t + 1, :] = A_tp1
        outAlive[:, t + 1, :] = (A_tp1 > barrier).float()

    return outA, outAlive

@torch.no_grad()
def multistep_asset_buffers(
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
    init_jitter: float = 0.05,
    step_jitter_sd: float = 0.0,
    stress_prob: float = 0.0,
    stress_shift: float = -1.0,
    seed: int = 0,
    # overlay defaults (stochastic, asset/network‑correlated)
    use_extra_defaults: bool = True,
    extra_intensity: float = 0.30,
    extra_cap: float = 0.35,
    extra_rho: Optional[float] = None,
    k_surv_iters: int = 5,
    # network needed only for overlay
    L: Optional[torch.Tensor] = None,
    liab: Optional[torch.Tensor] = None,
    T_total: Optional[float] = None,
    ell: Optional[float] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    High‑level convenience: build path‑consistent per‑step buffers for training.
    Returns:
      A_buf_list : list[T] of (S, n) assets at times t=0..T-1
      dR_buf_list: list[T] of (S, n) ABM increments (underlying shocks)
      alive_path : (S, T+1, n) alive mask after overlay
    """
    A_path, dR_all, alive0 = build_rollout_paths(
        S, n, T, dt, device,
        init_dd=init_dd, sigma=sigma, rho=rho, a_dead=a_dead,
        band_mult=band_mult, init_jitter=init_jitter, step_jitter_sd=step_jitter_sd,
        stress_prob=stress_prob, stress_shift=stress_shift, seed=seed
    )

    if use_extra_defaults:
        assert L is not None and liab is not None and T_total is not None, \
            "Overlay defaults require (L, liab, T_total)."
        A_path, alive_path = apply_extra_defaults_per_step(
            A_path, dR_all, L, liab,
            sigma=sigma, dt=dt, T_total=T_total, rho=rho, a_dead=a_dead,
            k_surv_iters=k_surv_iters,
            extra_on=True, extra_intensity=extra_intensity, extra_cap=extra_cap,
            extra_rho=extra_rho if extra_rho is not None else rho,
            seed=seed,
            ell=ell,
        )
    else:
        alive_path = alive0

    # Per‑step buffers (inputs for your training at time t)
    A_buf_list  = [A_path[:, t, :].contiguous() for t in range(T)]
    dR_buf_list = [dR_all[:, t, :].contiguous() for t in range(T)]
    return A_buf_list, dR_buf_list, alive_path
