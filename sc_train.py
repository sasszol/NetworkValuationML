# sc_train.py
import math, time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Any, Optional, Sequence

import torch
import torch.nn as nn
from tqdm import trange

from sc_core import (
    PDToMaturityNet, LastStepPDIter, approx_min_ce, squeeze_last,
    hard_clear_pd, ste_clear_pd
)
from sc_utils import build_matrices, iterative_breach_feature, draw_shocks_factor
from sc_data import create_asset_buffer, make_dR_buffer, multistep_asset_buffers
from sc_staticbarrier_benchmark import multistep_asset_buffers_sdb
from sc_zero import ZeroSubModel
from sc_diagnostics import TgtBufDiagnostics

# -------------------------- helpers (MC replicas) ---------------------------

def _k_mc_train(cfg: Dict[str, Any]) -> int:
    """Single knob for ALL training sims (tgt0 + refresh + first)."""
    return int(cfg.get("K_MC", cfg.get("K_MC_MAX", 1)))

def _k_mc_valid(cfg: Dict[str, Any]) -> int:
    """Separate knob for validation sims."""
    return int(cfg.get("K_MC_VALID", cfg.get("K_MC", 1)))

# -------------------------- schedules (LR / refresh) ------------------------

def _scheduled_value(schedule: Optional[Dict[int, float]], default: float, epoch: int) -> float:
    if not schedule:
        return float(default)
    keys = sorted(int(k) for k in schedule.keys())
    val = float(default)
    for k in keys:
        if epoch >= k:
            val = float(schedule[k])
        else:
            break
    return val

def _apply_lr_schedule(opt: torch.optim.Optimizer, cfg: Dict[str, Any], epoch: int, *, key: str = "LR_SCHEDULE"):
    sched = cfg.get(key, None)
    base_lr = float(cfg["LR_SMALL"])
    lr = _scheduled_value(sched, base_lr, epoch)
    for g in opt.param_groups:
        g["lr"] = lr

def _refresh_every(cfg: Dict[str, Any], epoch: int) -> int:
    return int(_scheduled_value(cfg.get("REFRESH_EVERY_SCHEDULE", None), cfg["REFRESH_EVERY"], epoch))

def _refresh_fract(cfg: Dict[str, Any], epoch: int) -> int:
    return int(_scheduled_value(cfg.get("REFRESH_FRACT_SCHEDULE", None), cfg["REFRESH_FRACT"], epoch))

# ---------- teacher simulator: post-clearing at t and t+1 -------------------

@torch.no_grad()
def simulate_one_step_with_projection(nets,
                                      step: int,
                                      A_t: torch.Tensor,
                                      dR_t: torch.Tensor,
                                      L: torch.Tensor,
                                      liab: torch.Tensor,
                                      n_it: int,
                                      *,
                                      sigma: float,
                                      T_total: float,
                                      dt: float,
                                      k_surv_iters: int,
                                      a_dead: float,
                                      use_phi: bool):
    """
    Teacher: compute PD_t; CLEAR at t; simulate t→t+1; compute PD_{t+1}; CLEAR at t+1.
    TD target: (1 - S^+(t)) + S^+(t) * E[PD^{post}(t+1)].
    """
    # --- step t
    T_now = max(T_total - step * dt, 1e-8)
    if use_phi:
        psi_t = iterative_breach_feature(A_t, L, liab, sigma, T_now, k_surv_iters, a_dead)
        inp_t = torch.stack([A_t, psi_t], dim=-1)  # (B, n, 2)
    else:
        inp_t = A_t.unsqueeze(-1)                  # (B, n, 1)

    logits_t = squeeze_last(nets[step](inp_t))
    pd_t_raw = torch.sigmoid(logits_t)
    pd_t, s_t = hard_clear_pd(pd_t_raw, A_t, L, liab, n_it, a_dead)  # post‑clearing (B, n)

    # --- advance assets with additive ABM; lock new defaults at a_dead
    A_next_free = A_t + dR_t
    A_tp1 = torch.where(s_t > 0.5, A_next_free, torch.full_like(A_t, float(a_dead)))

    # --- step t+1
    T_next = max(T_total - (step + 1) * dt, 1e-8)
    next_net = nets[step + 1]
    if isinstance(next_net, LastStepPDIter):
        inp_tp1 = A_tp1.unsqueeze(-1)
    else:
        if getattr(next_net, "in_features_per_node", 1) >= 2:
            psi_tp1 = iterative_breach_feature(A_tp1, L, liab, sigma, T_next, k_surv_iters, a_dead)
            inp_tp1 = torch.stack([A_tp1, psi_tp1], dim=-1)
        else:
            inp_tp1 = A_tp1.unsqueeze(-1)

    logits_tp1 = squeeze_last(next_net(inp_tp1))
    pd_tp1_raw = torch.sigmoid(logits_tp1)
    pd_tp1, _ = hard_clear_pd(pd_tp1_raw, A_tp1, L, liab, n_it, a_dead)

    tgt = (1. - s_t) + s_t * pd_tp1
    return pd_t, tgt, s_t

# ------------------------------ diagnostics ---------------------------------

@torch.no_grad()
def _diag_print_step0(step0: nn.Module, cfg: Dict[str, Any], A_list: Sequence[float]) -> None:
    """
    Print the post‑clearing PDs of the trained step‑0 net at chosen A.
    Uses the *same* ψ and hard‑clearing as training (fully consistent).
    """
    device = next(step0.parameters()).device
    n = int(cfg["N_BANKS"])
    L, liab = build_matrices(cfg, device)
    a_dead = float(cfg.get("A_DEAD", -10.0))
    use_phi = bool(cfg.get("USE_PHI_FEATURE", True))
    k_phi = int(cfg.get("K_SURV_ITERS", 5))
    T_now = float(cfg["T_TOTAL"])
    sigma = float(cfg["SIGMA"])

    if len(A_list) == 1:
        A = torch.full((1, n), float(A_list[0]), device=device)
    elif len(A_list) == n:
        A = torch.tensor(A_list, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        raise ValueError(f"DIAG_A_LIST must be length 1 or {n} (got {len(A_list)})")

    if use_phi and getattr(step0, "in_features_per_node", 1) >= 2:
        psi = iterative_breach_feature(A, L, liab, sigma=sigma, T=T_now, k=k_phi, a_dead=a_dead)
        xin = torch.stack([A, psi], dim=-1)
    else:
        xin = A.unsqueeze(-1)

    logits = squeeze_last(step0(xin))
    pd_raw = torch.sigmoid(logits)
    pd_post, _ = hard_clear_pd(pd_raw, A, L, liab, n_it=int(n), a_dead=a_dead)

    arr = pd_post.squeeze(0).detach().cpu().numpy()
    msg = (f"[diag] corr={cfg['ASSET_CORR']:.4f}  A={list(A_list)}  "
           f"post‑clear PD mean={arr.mean():.4f}")
    print(msg)

def np_round(x, k: int = 4):
    try:
        import numpy as _np
        return _np.round(x, k).tolist()
    except Exception:
        return x

# ----------------------------- training step --------------------------------
def _train_one_step(cfg: Dict[str, Any],
                    nets, step: int,
                    A_buf, dR_buf,
                    L, liab,
                    device: torch.device,
                    log: List[str],
                    diag: "TgtBufDiagnostics | None" = None) -> None:

    net = nets[step]; net.train()
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["LR_SMALL"]))
    bce_elem = nn.BCELoss(reduction="none")
    mse = nn.MSELoss()

    n_it = int(L.shape[0])
    use_ste = bool(cfg.get("USE_STE_CLEARING", True))
    a_dead = float(cfg.get("A_DEAD", -10.0))
    use_phi = bool(cfg.get("USE_PHI_FEATURE", True))
    mask_surv = bool(cfg.get("MASK_LOSS_TO_SURVIVORS", True))

    BUF = int(cfg["BUFFER_SIZE"])
    VAL = int(min(cfg.get("VAL_SAMPLES", BUF // 5), BUF))
    train_pool_idx = torch.arange(VAL, BUF, device=device)
    val_idx = torch.arange(0, VAL, device=device)

    k_phi = int(cfg.get("K_SURV_ITERS", 5))
    T_now = max(cfg["T_TOTAL"] - step * cfg["DT"], 1e-8)
    if use_phi:
        psi_buf = iterative_breach_feature(A_buf, L, liab, sigma=cfg["SIGMA"], T=T_now, k=k_phi, a_dead=a_dead)
    else:
        psi_buf = None

    with torch.no_grad():
        if cfg.get("ZERO_WARMSTART", False):
            zeros = [ZeroSubModel(int(cfg["N_BANKS"])).to(device) for _ in range(cfg["TOTAL_STEPS"])]
            zeros.append(nets[-1])
            nets_for_tgt0 = zeros
        else:
            nets_for_tgt0 = nets

        k_mc = _k_mc_train(cfg)
        if k_mc > 1:
            A_rep  = A_buf.repeat_interleave(k_mc, dim=0)
            rho = float(cfg["ASSET_CORR"])
            dR_rep = draw_shocks_factor(A_rep.shape[0], A_rep.shape[1],
                                        rho, cfg["SIGMA"], cfg["DT"], device=A_rep.device)
            _, td_many, s_now = simulate_one_step_with_projection(nets_for_tgt0, step, A_rep, dR_rep,
                                                                  L, liab, n_it,
                                                                  sigma=cfg["SIGMA"], T_total=cfg["T_TOTAL"],
                                                                  dt=cfg["DT"], k_surv_iters=k_phi,
                                                                  a_dead=a_dead, use_phi=use_phi)
            tgt0 = td_many.view(-1, k_mc, int(cfg["N_BANKS"])).mean(dim=1)
        else:
            _, tgt0, s_now = simulate_one_step_with_projection(nets_for_tgt0, step, A_buf, dR_buf,
                                                               L, liab, n_it,
                                                               sigma=cfg["SIGMA"], T_total=cfg["T_TOTAL"],
                                                               dt=cfg["DT"], k_surv_iters=k_phi,
                                                               a_dead=a_dead, use_phi=use_phi)
    tgt_buf = tgt0.detach().clamp_(0., 1.)

    if diag is not None:
        diag.record(step, epoch=0, tgt_buf=tgt_buf)

    best_val_mse = float("inf")
    best_state = None
    eval_bad = 0
    EVAL_PATIENCE = int(cfg.get("EVAL_PATIENCE", 10))
    MIN_DELTA = float(cfg.get("EVAL_MIN_DELTA", 0.0))

    last_refresh_epoch = -10**9
    next_eval_epoch = int(cfg.get("EVAL_EVERY", 1000))
    MAX_EPOCHS = int(cfg["MAX_EPOCHS"])
    BATCH = max(1, int(cfg["BUFFER_SIZE"] // max(1, int(cfg["BATCH_FRACT"]))))

    bar = trange(MAX_EPOCHS, desc=f"step {step}", leave=False)

    for ep in bar:
        _apply_lr_schedule(opt, cfg, ep + 1)
        refresh_every = _refresh_every(cfg, ep + 1)

        ridx = torch.randint(0, train_pool_idx.numel(), (BATCH,), device=device)
        idx = train_pool_idx[ridx]
        A_b = A_buf[idx]

        opt.zero_grad(set_to_none=True)
        if use_phi:
            psi_b = psi_buf[idx]
            x_b = torch.stack([A_b, psi_b], dim=-1)
        else:
            x_b = A_b.unsqueeze(-1)

        logits = squeeze_last(net(x_b))
        pd_raw = torch.sigmoid(logits)

        if use_ste:
            pd_pred = ste_clear_pd(pd_raw, A_b, L, liab, n_it, a_dead)
        else:
            with torch.no_grad():
                pd_pred, _ = hard_clear_pd(pd_raw, A_b, L, liab, n_it, a_dead)

        per_elem = bce_elem(pd_pred, tgt_buf[idx])
        if mask_surv:
            with torch.no_grad():
                _, s_curr = hard_clear_pd(pd_raw, A_b, L, liab, n_it, a_dead)
            w = s_curr
            denom = w.sum().clamp_min(1.0)
            loss = (per_elem * w).sum() / denom
        else:
            loss = per_elem.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["GRAD_CLIP"])
        opt.step()

        # refresh teacher targets
        if (ep + 1) % int(refresh_every) == 0:
            with torch.no_grad():
                sel_train = train_pool_idx[
                    torch.randperm(train_pool_idx.numel(), device=device)[
                        : max(1, train_pool_idx.numel() // max(1, int(_refresh_fract(cfg, ep + 1))))
                    ]
                ]
                k_mc = _k_mc_train(cfg)
                if k_mc > 1:
                    A_rep  = A_buf[sel_train].repeat_interleave(k_mc, dim=0)
                    rho = float(cfg["ASSET_CORR"])
                    dR_rep = draw_shocks_factor(A_rep.shape[0], A_rep.shape[1],
                                                rho, cfg["SIGMA"], cfg["DT"], device=A_rep.device)
                    _, td_many, _ = simulate_one_step_with_projection(nets, step, A_rep, dR_rep,
                                                                      L, liab, n_it,
                                                                      sigma=cfg["SIGMA"], T_total=cfg["T_TOTAL"],
                                                                      dt=cfg["DT"], k_surv_iters=k_phi,
                                                                      a_dead=a_dead, use_phi=use_phi)
                    tgt_buf[sel_train] = td_many.view(-1, k_mc, int(cfg["N_BANKS"])).mean(dim=1)
                else:
                    _, td_once, _ = simulate_one_step_with_projection(nets, step,
                                                                      A_buf[sel_train], dR_buf[sel_train],
                                                                      L, liab, n_it,
                                                                      sigma=cfg["SIGMA"], T_total=cfg["T_TOTAL"],
                                                                      dt=cfg["DT"], k_surv_iters=k_phi,
                                                                      a_dead=a_dead, use_phi=use_phi)
                    tgt_buf[sel_train] = td_once

            last_refresh_epoch = ep + 1

            if diag is not None:
                diag.record(step, epoch=ep + 1, tgt_buf=tgt_buf,
                            n_refreshed=sel_train.numel())

        should_eval = ((ep + 1) == (last_refresh_epoch + 1)) and ((ep + 1) >= next_eval_epoch)
        if should_eval:
            with torch.no_grad():
                if use_phi:
                    psi_val = iterative_breach_feature(A_buf[val_idx], L, liab,
                                                       sigma=cfg["SIGMA"],
                                                       T=max(cfg["T_TOTAL"] - step * cfg["DT"], 1e-8),
                                                       k=k_phi, a_dead=a_dead)
                    x_v = torch.stack([A_buf[val_idx], psi_val], dim=-1)
                else:
                    x_v = A_buf[val_idx].unsqueeze(-1)

                logits_v = squeeze_last(net(x_v))
                pd_raw_v = torch.sigmoid(logits_v)
                pd_pred_v, _ = hard_clear_pd(pd_raw_v, A_buf[val_idx], L, liab, n_it, a_dead)

                k_mc_v = _k_mc_valid(cfg)
                A_rep  = A_buf[val_idx].repeat_interleave(k_mc_v, dim=0)
                rho = float(cfg["ASSET_CORR"])
                dR_rep = draw_shocks_factor(A_rep.shape[0], A_rep.shape[1],
                                            rho, cfg["SIGMA"], cfg["DT"], device=A_rep.device)
                _, td_many, _ = simulate_one_step_with_projection(nets, step, A_rep, dR_rep,
                                                                  L, liab, n_it,
                                                                  sigma=cfg["SIGMA"], T_total=cfg["T_TOTAL"],
                                                                  dt=cfg["DT"], k_surv_iters=k_phi,
                                                                  a_dead=a_dead, use_phi=use_phi)
                true_v = td_many.view(-1, k_mc_v, int(cfg["N_BANKS"])).mean(dim=1)

                val_mse = mse(pd_pred_v, true_v).item()
                val_ce  = nn.BCELoss()(pd_pred_v, true_v).item()
                ce_min  = approx_min_ce(true_v)

                msg = (f"[eval] step {step} ep {ep + 1}: "
                       f"MSE={val_mse:.4e}  CE={val_ce:.4e}  minCE≈{ce_min:.4e}")
                print(msg); log.append(msg)

                next_eval_epoch = (ep + 1) + int(cfg["EVAL_EVERY"])

                if val_mse + float(cfg.get("EVAL_MIN_DELTA", 0.0)) < best_val_mse:
                    best_val_mse = val_mse
                    best_state = deepcopy(net.state_dict())
                    eval_bad = 0
                else:
                    eval_bad += 1

                if eval_bad >= int(cfg.get("EVAL_PATIENCE", 10)):
                    log.append(f"[early stop] step {step} at epoch {ep + 1} (best MSE={best_val_mse:.3e})")
                    print(f"[early stop] step {step} at epoch {ep + 1} (best MSE={best_val_mse:.3e})")
                    break

    if best_state is not None:
        net.load_state_dict(best_state)

    return best_val_mse

# ----------------------------- public APIs ----------------------------------
def train_all_and_get_step0(cfg: Dict[str, Any],
                            work_dir=None,
                            seed: int = 10,
                            *,
                            prev_state_dicts: Optional[List[Dict[str, Any]]] = None,
                            return_all_states: bool = False
                            ) -> Tuple[nn.Module, List[str]] | Tuple[nn.Module, List[str], List[Dict[str, Any]]]:
    torch.manual_seed(seed)
    import numpy as _np; _np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = int(cfg["N_BANKS"])
    T = int(cfg["TOTAL_STEPS"])
    dt = float(cfg["DT"])

    L, liab = build_matrices(cfg, device)

    use_phi = bool(cfg.get("USE_PHI_FEATURE", True))
    in_F = 1 + int(use_phi)

    nets = [PDToMaturityNet(n, in_features_per_node=in_F).to(device)
            for _ in range(T)]
    nets.append(LastStepPDIter(L, liab, a_dead=float(cfg.get("A_DEAD", -10.0))).to(device))

    if prev_state_dicts is not None:
        for i in range(min(T, len(prev_state_dicts))):
            nets[i].load_state_dict(prev_state_dicts[i])

    log: List[str] = []

    # --------- NEW: build path‑consistent per‑step buffers ----------
    if bool(cfg.get("USE_ROLLOUT_PATHS", True)):
        buf_method = str(cfg.get("NAIVE_PATHS_METHOD", "sc_data")).lower().strip()

        if buf_method in {"sc_data", "rollout", "rollout_paths"}:
            A_buf_list, dR_buf_list, alive_paths = multistep_asset_buffers(
                S=cfg["BUFFER_SIZE"], n=n, T=T, dt=dt, device=device,
                init_dd=cfg.get("INIT_DD", 1.0), sigma=cfg["SIGMA"], rho=cfg["ASSET_CORR"],
                a_dead=cfg.get("A_DEAD", -10.0),
                band_mult=float(cfg.get("BAND_MULT", 1.0)),
                init_jitter=float(cfg.get("INIT_JITTER", 0.05)),
                step_jitter_sd=float(cfg.get("STEP_JITTER_SD", 0.0)),
                stress_prob=float(cfg.get("SCENARIO_STRESS_PROB", 0.0)),
                stress_shift=float(cfg.get("SCENARIO_STRESS_SHIFT", -1.0)),
                seed=seed,
                use_extra_defaults=bool(cfg.get("EXTRA_DEFAULTS_ON", True)),
                extra_intensity=float(cfg.get("EXTRA_INTENSITY", 0.30)),
                extra_cap=float(cfg.get("EXTRA_CAP", 0.35)),
                extra_rho=cfg.get("EXTRA_RHO", None),
                k_surv_iters=int(cfg.get("K_SURV_ITERS", 5)),
                L=L, liab=liab, T_total=float(cfg["T_TOTAL"]),
            )

        elif buf_method in {"static_barrier", "sdb", "static"}:
            # SDB (state-dependent static-barrier) buffer generator.
            # This uses the same network revaluation / projection dynamics as training,
            # but replaces the neural PD predictor by the ABM flat-barrier proxy.
            A_buf_list, dR_buf_list, alive_paths = multistep_asset_buffers_sdb(
                S=cfg["BUFFER_SIZE"], n=n, T=T, dt=dt, device=device,
                init_dd=cfg.get("INIT_DD", 1.0), sigma=cfg["SIGMA"], rho=cfg["ASSET_CORR"],
                a_dead=cfg.get("A_DEAD", -10.0),
                band_mult=float(cfg.get("BAND_MULT", 1.0)),
                init_jitter=float(cfg.get("INIT_JITTER", 0.05)),
                step_jitter_sd=float(cfg.get("STEP_JITTER_SD", 0.0)),
                stress_prob=float(cfg.get("SCENARIO_STRESS_PROB", 0.0)),
                stress_shift=float(cfg.get("SCENARIO_STRESS_SHIFT", -1.0)),
                seed=seed,
                L=L, liab=liab, T_total=float(cfg["T_TOTAL"]),
                tol_fp=float(cfg.get("SDB_TOL", 1e-6)),
            )

        else:
            raise ValueError(
                f"Unknown NAIVE_PATHS_METHOD={cfg.get('NAIVE_PATHS_METHOD')!r}. "
                "Use 'sc_data' or 'static_barrier'."
            )

    else:
        # Fallback: old behaviour, per‑step independent buffers

        dR_buf = make_dR_buffer(cfg["BUFFER_SIZE"], n, cfg["ASSET_CORR"],
                                sigma=cfg["SIGMA"], dt=cfg["DT"], device=device)
        A_buf_list = []
        dR_buf_list = []
        for step in range(T):
            ttm = step * cfg["DT"]
            A_buf = create_asset_buffer(cfg["BUFFER_SIZE"], n, device,
                                        init_dd=cfg.get("INIT_DD", 1.0),
                                        sigma=cfg["SIGMA"], ttm=ttm,
                                        p0_alive=cfg.get("P0_ALIVE", 1.0),
                                        a_dead=cfg.get("A_DEAD", -10.0), seed=seed)
            A_buf_list.append(A_buf)
            dR_buf_list.append(dR_buf)
    # ---------- diagnostics logger ----------
    diag_dir = None
    if work_dir is not None:
        diag_dir = str(Path(work_dir) / "diag")
    elif cfg.get("DIAG_DIR", None) is not None:
        diag_dir = str(cfg["DIAG_DIR"])
    diag = TgtBufDiagnostics(output_dir=diag_dir) if diag_dir is not None else None

    step_mse_list = []
    for step in reversed(range(T)):
        t0 = time.time()
        A_buf = A_buf_list[step]
        dR_buf = dR_buf_list[step]

        last_mse = _train_one_step(cfg, nets, step, A_buf, dR_buf, L, liab, device, log,
                                   diag=diag)

        if diag is not None:
            csv_path = Path(diag_dir) / f"tgt_diag_step_{step}.csv"
            diag.flush_step(step, csv_path)

        log.append(f"[step {step}] finished in {(time.time() - t0):.1f}s")
        print(f"step {step} finished in {(time.time() - t0)/60:.2f} min")
        step_mse_list.append(last_mse)

    step0 = nets[0].eval()

    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        torch.save(step0.state_dict(), Path(work_dir) / "step0.pt")

    if return_all_states:
        all_states = [nets[i].state_dict() for i in range(T)]
        return step0, log, all_states, step_mse_list
    else:
        return step0, log, step_mse_list

def sweep_correlation(cfg_base: Dict[str, Any],
                      corr_values: Iterable[float],
                      save_dir: str | None = None,
                      seed: int = 10,
                      work_dir: str | None = None) -> Dict[float, Dict[str, Any]]:
    results = {}
    for c in corr_values:
        cfg = dict(cfg_base)
        cfg["ASSET_CORR"] = float(c)
        print(f"[sweep] corr={c:.4f}")
        t0 = time.time()
        diag_work_dir = None
        if save_dir is not None:
            diag_work_dir = str(Path(save_dir) / "diag" / f"corr_{c:.4f}")
        model0, log, step_mse_list = train_all_and_get_step0(cfg, work_dir=diag_work_dir, seed=seed)
        state = model0.state_dict()
        results[c] = {"state_dict": state, "log": log, "seconds": time.time() - t0}
        results[c]["step_mse"] = step_mse_list

        diag_A = cfg.get("DIAG_A_LIST", None)
        if diag_A is not None:
            _diag_print_step0(model0, cfg, diag_A)

        if save_dir:
            tag = f"{c:.4f}"
            out = Path(save_dir) / f"step0_corr_{tag}.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            state_for_disk = dict(state)
            state_for_disk["mse_vec"] = torch.tensor(step_mse_list, dtype=torch.float32)
            torch.save(state_for_disk, out)
    return results

def sweep_correlation_warmstart(cfg_base: Dict[str, Any],
                                corr_values: Iterable[float],
                                save_dir: str | None = None,
                                seed: int = 10) -> Dict[float, Dict[str, Any]]:
    results: Dict[float, Dict[str, Any]] = {}

    prev_states: Optional[List[Dict[str, Any]]] = None

    for idx, c in enumerate(corr_values):
        cfg = dict(cfg_base)
        cfg["ASSET_CORR"] = float(c)
        print(f"[sweep (warm-start)] corr={c:.4f}")
        t0 = time.time()

        ret = train_all_and_get_step0(cfg, work_dir=None, seed=seed + idx,
                                      prev_state_dicts=prev_states, return_all_states=True)

        step0, log, states_this = ret
        state0 = step0.state_dict()
        results[c] = {"state_dict": state0, "log": log, "seconds": time.time() - t0}

        diag_A = cfg.get("DIAG_A_LIST", None)
        if diag_A is not None:
            _diag_print_step0(step0, cfg, diag_A)

        if save_dir:
            tag = f"{c:.4f}"
            out = Path(save_dir) / f"step0_corr_{tag}.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state0, out)

        prev_states = states_this

    return results
