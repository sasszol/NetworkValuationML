"""
sc_baseline.py

Baseline / "crude under-estimate" default probabilities implied purely by the
*sample generator* (ABM barrier hits + optional extra-default overlay).

This is meant to be compared against the ML-fitted, self-consistent PD
functions learned by sc_train.py.

Key idea
--------
For a given config (same as used in sweep_correlation / train_all_and_get_step0),
we generate the same rollout paths (A_path) used to populate the training buffer
and then compute empirical default probabilities (by time and to maturity)
directly from those generated paths.

Default in the generator can occur via:
  (1) hitting the external-asset barrier (0) within a step (Brownian-bridge hit),
      which is encoded by overwriting A with A_DEAD ("a_dead sentinel");
  (2) the stochastic "extra defaults" overlay driven by ψ(A_t) (iterated breach
      proxy) and sampled with a 1-factor Gaussian copula.

This file does NOT use the neural networks at all.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from sc_utils import build_matrices, iterative_breach_feature
from sc_data import build_rollout_paths, apply_extra_defaults_per_step


# ------------------------------- dataclasses -------------------------------

@dataclass
class BaselineStats:
    """
    Container for baseline statistics. Everything is NumPy arrays for convenience.
    """
    corr: float
    n_banks: int
    total_steps: int
    dt: float
    t_total: float
    seed: int

    # time grid (0..T)
    times: np.ndarray                 # (T+1,)

    # cumulative default probabilities P(default by time t)
    pd_cum_total: np.ndarray          # (T+1, n)
    pd_cum_boundary: np.ndarray       # (T+1, n)

    # per-step hazard probabilities P(default in (t,t+1] | alive at t)
    hazard_total: np.ndarray          # (T, n)
    hazard_boundary: np.ndarray       # (T, n)
    hazard_overlay: np.ndarray        # (T, n)  # incremental overlay-only hazards

    # to-maturity PDs (same as pd_cum_*(T))
    pd_T_total: np.ndarray            # (n,)
    pd_T_boundary: np.ndarray         # (n,)
    pd_T_overlay_increment: np.ndarray # (n,)

    # optional crude conditional curves (binning) at t=0
    # Each is dict with: edges (nbins+1,), centers (nbins,), mean (nbins,), count (nbins,)
    by_A0: Optional[Dict[str, Any]] = None
    by_psi0: Optional[Dict[str, Any]] = None

    # raw sample-level t=0 data (optional; can be big)
    A0: Optional[np.ndarray] = None            # (S, n)
    psi0: Optional[np.ndarray] = None          # (S, n)
    default_T: Optional[np.ndarray] = None     # (S, n) 1 if defaulted by maturity
    default_time: Optional[np.ndarray] = None  # (S, n) first default step index in {0..T-1}, -1 if survives
    default_cause: Optional[np.ndarray] = None # (S, n) 0 survive, 1 boundary, 2 overlay


# ------------------------------- utilities ---------------------------------

def _device_from_cfg(cfg: Dict[str, Any]) -> torch.device:
    # mimic sc_train: prefer CUDA if available
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bin_curve(x: np.ndarray, y: np.ndarray, nbins: int = 40, *, quantile_bins: bool = True) -> Dict[str, Any]:
    """
    Bin y by x and compute the mean in each bin.

    Parameters
    ----------
    x : (S,) values
    y : (S,) outcomes in [0,1] (here: default indicator)
    nbins : number of bins
    quantile_bins : if True, use quantile edges so bins have similar mass

    Returns dict with edges, centers, mean, count.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    if x.size == 0:
        return {"edges": np.array([]), "centers": np.array([]), "mean": np.array([]), "count": np.array([])}

    if np.allclose(x.min(), x.max()):
        edges = np.array([x.min() - 1e-6, x.max() + 1e-6])
        centers = np.array([(edges[0] + edges[1]) / 2])
        return {"edges": edges, "centers": centers, "mean": np.array([float(y.mean())]), "count": np.array([int(x.size)])}

    if quantile_bins:
        qs = np.linspace(0.0, 1.0, nbins + 1)
        edges = np.quantile(x, qs)
        # make edges strictly increasing if needed
        edges[0] -= 1e-6
        edges[-1] += 1e-6
    else:
        edges = np.linspace(x.min() - 1e-6, x.max() + 1e-6, nbins + 1)

    # digitize into bins 0..nbins-1
    bin_idx = np.digitize(x, edges[1:-1], right=False)

    means = np.full(nbins, np.nan, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)
    for b in range(nbins):
        m = (bin_idx == b)
        counts[b] = int(m.sum())
        if counts[b] > 0:
            means[b] = float(y[m].mean())

    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"edges": edges, "centers": centers, "mean": means, "count": counts}


def _first_default_step(alive: torch.Tensor) -> torch.Tensor:
    """
    alive: (S, T+1, n) in {0,1}
    returns: (S, n) int64, first default step t in {0..T-1} such that alive[t]=1 and alive[t+1]=0,
             or -1 if no default.
    """
    S, T1, n = alive.shape
    T = T1 - 1
    # event mask (S,T,n)
    event = (alive[:, :-1, :] > 0.5) & (alive[:, 1:, :] < 0.5)
    idx = torch.arange(T, device=alive.device, dtype=torch.int64).view(1, T, 1).expand(S, T, n)
    idx_event = torch.where(event, idx, torch.full_like(idx, T))
    first = idx_event.min(dim=1).values  # (S,n), T if no event
    out = torch.where(first < T, first, torch.full_like(first, -1))
    return out


def _gather_time(alive: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
    """
    alive: (S, T+1, n)
    t_idx: (S, n) int64, time index in [0..T] (inclusive)
    returns: alive_at: (S, n)
    """
    S, T1, n = alive.shape
    alive_snT = alive.permute(0, 2, 1)  # (S,n,T+1)
    gidx = t_idx.unsqueeze(-1).clamp(0, T1 - 1)
    out = torch.gather(alive_snT, 2, gidx).squeeze(-1)
    return out


# -------------------------- core baseline computation -----------------------

@torch.no_grad()
def baseline_from_generator(
    cfg: Dict[str, Any],
    *,
    seed: int = 0,
    device: Optional[torch.device] = None,
    nbins: int = 40,
    store_raw: bool = True,
) -> BaselineStats:
    """
    Generate rollout paths and compute empirical PDs implied by the generator.

    Notes
    -----
    - Uses the same generator logic as sc_data.multistep_asset_buffers, but we keep
      the full alive paths to compute default probabilities.
    - "Boundary" refers to the ABM barrier hits (Brownian-bridge + endpoint <= 0).
    - "Total" refers to boundary + extra overlay (if enabled).

    Returns
    -------
    BaselineStats dataclass (mostly NumPy arrays).
    """
    if device is None:
        device = _device_from_cfg(cfg)

    # -------- unpack config with safe defaults ----------
    n = int(cfg["N_BANKS"])
    T = int(cfg["TOTAL_STEPS"])
    dt = float(cfg["DT"])
    T_total = float(cfg.get("T_TOTAL", T * dt))
    S = int(cfg.get("BUFFER_SIZE", 50_000))

    rho = float(cfg["ASSET_CORR"])
    sigma = float(cfg["SIGMA"])
    init_dd = float(cfg.get("INIT_DD", 1.0))
    a_dead = float(cfg.get("A_DEAD", -10.0))

    band_mult = float(cfg.get("BAND_MULT", 1.0))
    init_jitter = float(cfg.get("INIT_JITTER", 0.05))
    step_jitter_sd = float(cfg.get("STEP_JITTER_SD", 0.0))
    stress_prob = float(cfg.get("SCENARIO_STRESS_PROB", 0.0))
    stress_shift = float(cfg.get("SCENARIO_STRESS_SHIFT", -1.0))

    extra_on = bool(cfg.get("EXTRA_DEFAULTS_ON", True))
    extra_intensity = float(cfg.get("EXTRA_INTENSITY", 0.30))
    extra_cap = float(cfg.get("EXTRA_CAP", 0.35))
    extra_rho_cfg = cfg.get("EXTRA_RHO", None)
    extra_rho = None if extra_rho_cfg is None else float(extra_rho_cfg)
    k_surv = int(cfg.get("K_SURV_ITERS", 5))

    # -------- network matrices (needed only for overlay / psi) ---------------
    L, liab = build_matrices(cfg, device)

    # Optional rank-1 acceleration when the symmetric/homogeneous network is assumed.
    # Derived once and forwarded to every clearing-related primitive below so that
    # the overlay path and the t=0 ψ diagnostic both see the speedup.
    use_sym = bool(cfg.get("USE_SYMMETRIC_ASSUMPTION", False))
    ell = (float(cfg.get("kL", 1.0)) / max(1, n - 1)) if use_sym else None

    # -------- base ABM + Brownian-bridge barrier defaults --------------------
    A_path, dR_all, alive0 = build_rollout_paths(
        S=S, n=n, T=T, dt=dt, device=device,
        init_dd=init_dd, sigma=sigma, rho=rho, a_dead=a_dead,
        band_mult=band_mult, init_jitter=init_jitter, step_jitter_sd=step_jitter_sd,
        stress_prob=stress_prob, stress_shift=stress_shift, seed=seed,
    )

    # -------- overlay defaults (optional) ------------------------------------
    if extra_on and (extra_intensity > 0.0) and (extra_cap > 0.0):
        A_path2, alive = apply_extra_defaults_per_step(
            A_path, dR_all, L, liab,
            sigma=sigma, dt=dt, T_total=T_total, rho=rho, a_dead=a_dead,
            k_surv_iters=k_surv,
            extra_on=True,
            extra_intensity=extra_intensity,
            extra_cap=extra_cap,
            extra_rho=extra_rho,  # None -> use rho inside
            seed=seed,
            ell=ell,
        )
        # keep the modified path
        A_path = A_path2
    else:
        alive = alive0

    # -------- empirical PD curves --------------------------------------------
    times = (np.arange(T + 1, dtype=np.float64) * dt)

    pd_cum_total = (1.0 - alive.float().mean(dim=0)).detach().cpu().numpy()    # (T+1,n)
    pd_cum_boundary = (1.0 - alive0.float().mean(dim=0)).detach().cpu().numpy()

    # per-step hazard: P(default in step t | alive at t)
    alive_t = alive[:, :-1, :]  # (S,T,n)
    alive_tp1 = alive[:, 1:, :]
    step_default = (alive_t > 0.5) & (alive_tp1 < 0.5)

    denom = alive_t.float().mean(dim=0).clamp_min(1e-12)  # (T,n) fraction alive at t
    hazard_total = (step_default.float().mean(dim=0) / denom).detach().cpu().numpy()  # (T,n)

    # boundary hazard (same but for alive0)
    alive0_t = alive0[:, :-1, :]
    alive0_tp1 = alive0[:, 1:, :]
    step_default0 = (alive0_t > 0.5) & (alive0_tp1 < 0.5)
    denom0 = alive0_t.float().mean(dim=0).clamp_min(1e-12)
    hazard_boundary = (step_default0.float().mean(dim=0) / denom0).detach().cpu().numpy()

    # overlay-only hazard (defaults that happen *only* because of overlay)
    # classification uses shared underlying base path: if alive0 survives to t+1 but alive dies, it's overlay
    overlay_event = step_default & (alive0_tp1 > 0.5)  # (S,T,n)
    hazard_overlay = (overlay_event.float().mean(dim=0) / denom).detach().cpu().numpy()

    pd_T_total = pd_cum_total[-1, :]
    pd_T_boundary = pd_cum_boundary[-1, :]
    pd_T_overlay_increment = np.clip(pd_T_total - pd_T_boundary, 0.0, 1.0)

    # -------- sample-level t=0 state and maturity outcomes -------------------
    # Only needed if you want to (a) save raw samples or (b) build crude conditional curves.
    need_samples = bool(store_raw) or (nbins is not None and int(nbins) > 0)

    A0 = None
    default_T_np = None
    first_step_np = None
    cause_np = None

    by_A0 = None
    by_psi0 = None
    psi0_np = None

    if need_samples:
        A0 = A_path[:, 0, :].detach().cpu().numpy()
        default_T = (alive[:, -1, :] < 0.5).to(torch.int8)
        default_T_np = default_T.detach().cpu().numpy()

        # default time & cause (0 survive, 1 boundary, 2 overlay)
        first_step = _first_default_step(alive)  # (S,n) in {-1,0..T-1}
        defaulted = (first_step >= 0)

        # cause boundary vs overlay determined at first default step
        # look at alive0 at time t+1
        t_plus1 = torch.where(defaulted, first_step + 1, torch.zeros_like(first_step))
        alive0_at_tp1 = _gather_time(alive0, t_plus1)  # (S,n)
        boundary_cause = defaulted & (alive0_at_tp1 < 0.5)
        overlay_cause = defaulted & (~boundary_cause)

        cause = torch.zeros_like(first_step, dtype=torch.int64)
        cause = torch.where(boundary_cause, torch.full_like(cause, 1), cause)
        cause = torch.where(overlay_cause, torch.full_like(cause, 2), cause)

        first_step_np = first_step.detach().cpu().numpy().astype(np.int64)
        cause_np = cause.detach().cpu().numpy().astype(np.int64)

        # -------- optional conditional curves at t=0 -----------------------------
        if nbins is not None and int(nbins) > 0:
            # ψ0 is often useful for diagnostics even if the ML net doesn't use it
            psi0 = iterative_breach_feature(
                torch.tensor(A0, device=device, dtype=torch.float32),
                L, liab,
                sigma=sigma,
                T=T_total,
                k=k_surv,
                a_dead=a_dead,
                ell=ell,
            ).detach().cpu().numpy()
            psi0_np = psi0

            # by-bank 1D curves (own A_i or ψ_i vs default indicator)
            by_A0 = {}
            by_psi0 = {}
            for i in range(n):
                by_A0[f"bank_{i}"] = _bin_curve(A0[:, i], default_T_np[:, i], nbins=nbins, quantile_bins=True)
                by_psi0[f"bank_{i}"] = _bin_curve(psi0_np[:, i], default_T_np[:, i], nbins=nbins, quantile_bins=True)

    stats = BaselineStats(
        corr=float(rho),
        n_banks=n,
        total_steps=T,
        dt=dt,
        t_total=T_total,
        seed=int(seed),
        times=times,
        pd_cum_total=pd_cum_total,
        pd_cum_boundary=pd_cum_boundary,
        hazard_total=hazard_total,
        hazard_boundary=hazard_boundary,
        hazard_overlay=hazard_overlay,
        pd_T_total=pd_T_total,
        pd_T_boundary=pd_T_boundary,
        pd_T_overlay_increment=pd_T_overlay_increment,
        by_A0=by_A0,
        by_psi0=by_psi0,
        A0=(A0 if store_raw else None),
        psi0=(psi0_np if store_raw else None),
        default_T=(default_T_np if store_raw else None),
        default_time=(first_step_np if store_raw else None),
        default_cause=(cause_np if store_raw else None),
    )
    return stats


def baseline_to_dict(stats: BaselineStats) -> Dict[str, Any]:
    """Convert BaselineStats (dataclass) to a plain dict (for torch.save / json)."""
    d = stats.__dict__.copy()
    return d


def print_summary(stats: BaselineStats) -> None:
    n = stats.n_banks
    T = stats.total_steps
    print(f"[baseline] corr={stats.corr:.4f}  n={n}  T={T}  dt={stats.dt:.4g}  T_total={stats.t_total:.4g}  seed={stats.seed}")
    print(f"  PD(T) total   : {np.round(stats.pd_T_total, 6).tolist()}   (mean={float(np.mean(stats.pd_T_total)):.6f})")
    print(f"  PD(T) boundary: {np.round(stats.pd_T_boundary, 6).tolist()}   (mean={float(np.mean(stats.pd_T_boundary)):.6f})")
    print(f"  PD(T) overlayΔ: {np.round(stats.pd_T_overlay_increment, 6).tolist()}   (mean={float(np.mean(stats.pd_T_overlay_increment)):.6f})")

    # quick check: default timing mix
    if stats.default_time is not None and stats.default_cause is not None:
        dtm = stats.default_time.reshape(-1)
        cause = stats.default_cause.reshape(-1)
        total_default = (dtm >= 0).mean()
        boundary_share = ((cause == 1).mean() / max(total_default, 1e-12)) if total_default > 0 else 0.0
        overlay_share = ((cause == 2).mean() / max(total_default, 1e-12)) if total_default > 0 else 0.0
        print(f"  defaulted-by-maturity frequency (all names pooled): {float(total_default):.6f}")
        print(f"    cause shares among defaults: boundary={boundary_share:.3f}, overlay={overlay_share:.3f}")


# ------------------------------ sweep helper --------------------------------

@torch.no_grad()
def sweep_correlation_baseline(
    cfg_base: Dict[str, Any],
    corr_values: Iterable[float],
    *,
    save_dir: Optional[str] = None,
    seed: int = 0,
    nbins: int = 40,
    store_raw: bool = False,
) -> Dict[float, Dict[str, Any]]:
    """
    Analogous to sc_train.sweep_correlation, but only computes generator-implied PDs.

    Returns dict keyed by corr with plain-python dict payloads (torch-saveable).
    """
    out: Dict[float, Dict[str, Any]] = {}
    save_path = Path(save_dir) if save_dir is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    for c in corr_values:
        cfg = dict(cfg_base)
        cfg["ASSET_CORR"] = float(c)
        stats = baseline_from_generator(cfg, seed=seed, nbins=nbins, store_raw=store_raw)
        payload = baseline_to_dict(stats)
        out[float(c)] = payload

        print_summary(stats)

        if save_path is not None:
            tag = f"{float(c):.4f}"
            torch.save(payload, save_path / f"baseline_corr_{tag}.pt")

    return out


# ---------------------------------- CLI ------------------------------------

def _load_cfg_from_module(module_name: str, var_name: str = "CFG") -> Dict[str, Any]:
    mod = importlib.import_module(module_name)
    if not hasattr(mod, var_name):
        raise AttributeError(f"Module '{module_name}' has no variable '{var_name}'.")
    cfg = getattr(mod, var_name)
    if not isinstance(cfg, dict):
        raise TypeError(f"{module_name}.{var_name} must be a dict (got {type(cfg)}).")
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline PDs implied by the rollout data generator.")
    p.add_argument("--cfg-module", type=str, default="batch_process",
                   help="Python module that defines a config dict (default: batch_process).")
    p.add_argument("--cfg-var", type=str, default="CFG",
                   help="Variable name of the config dict in that module (default: CFG).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (passed into the generator).")
    p.add_argument("--corr", type=float, default=None,
                   help="Override ASSET_CORR for a single run (no sweep).")
    p.add_argument("--sweep", action="store_true",
                   help="Run a correlation sweep (uses --corr-grid-npy or --corr-grid-lin).")
    p.add_argument("--corr-grid-npy", type=str, default=None,
                   help="Path to a .npy file containing an array of correlation values.")
    p.add_argument("--corr-grid-lin", type=str, default=None,
                   help="Linspace spec 'start,stop,num' (e.g. '-1,1,51').")
    p.add_argument("--save-dir", type=str, default=None, help="Directory to save results (.pt).")
    p.add_argument("--nbins", type=int, default=40, help="Bins for crude 1D conditional curves.")
    p.add_argument("--store-raw", action="store_true",
                   help="Store raw per-scenario arrays (A0, psi0, default indicators) in saved payloads.")

    args = p.parse_args()

    cfg = _load_cfg_from_module(args.cfg_module, args.cfg_var)

    if args.sweep:
        if args.corr_grid_npy is None and args.corr_grid_lin is None:
            raise ValueError("For --sweep you must provide --corr-grid-npy or --corr-grid-lin.")
        if args.corr_grid_npy is not None:
            corr_grid = np.load(args.corr_grid_npy).astype(np.float64).tolist()
        else:
            start, stop, num = [float(x) for x in args.corr_grid_lin.split(",")]
            corr_grid = np.linspace(start, stop, int(num), endpoint=True).tolist()

        sweep_correlation_baseline(cfg, corr_grid, save_dir=args.save_dir, seed=args.seed,
                                   nbins=args.nbins, store_raw=args.store_raw)
        return

    if args.corr is not None:
        cfg = dict(cfg)
        cfg["ASSET_CORR"] = float(args.corr)

    stats = baseline_from_generator(cfg, seed=args.seed, nbins=args.nbins, store_raw=args.store_raw)
    print_summary(stats)

    if args.save_dir is not None:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{float(stats.corr):.4f}"
        torch.save(baseline_to_dict(stats), out_dir / f"baseline_corr_{tag}.pt")


if __name__ == "__main__":
    main()
