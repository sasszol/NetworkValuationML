# sc_train.py
import math, time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Any, Optional, Sequence

from line_profiler_pycharm import profile

import torch
import torch.nn as nn
from tqdm import trange

from sc_core import (
    PDToMaturityNet, LastStepPDIter, approx_min_ce, squeeze_last,
    hard_clear_pd, ste_clear_pd
)
from sc_utils import build_matrices, draw_shocks_factor
from sc_data import create_asset_buffer, make_dR_buffer, multistep_asset_buffers
from sc_staticbarrier_benchmark import multistep_asset_buffers_sdb
from sc_zero import ZeroSubModel
from sc_diagnostics import TgtBufDiagnostics


# -------------------------- symmetric-assumption helpers --------------------

def _sym_ell(cfg: Dict[str, Any], n: int) -> Optional[float]:
    """Return the homogeneous off-diagonal exposure ell = kL/(n-1) when the
    symmetric assumption is enabled in the config; otherwise None.

    Returning a non-None ell switches all clearing primitives in sc_core/
    sc_utils/sc_staticbarrier_benchmark to the rank-1 fast path.
    """
    if not bool(cfg.get("USE_SYMMETRIC_ASSUMPTION", False)):
        return None
    if n <= 1:
        return None
    return float(cfg.get("kL", 1.0)) / float(n - 1)


def _permute_per_scenario(*tensors: torch.Tensor) -> tuple:
    """Apply an independent random permutation along the agent axis (-1) to
    each scenario, jointly across all input tensors.

    All inputs must share the same leading shape (B, n) — they may carry
    additional trailing dims (e.g. (B, n, F)).
    Returns the permuted tensors in the same order.
    """
    if len(tensors) == 0:
        return tuple()
    B, n = tensors[0].shape[0], tensors[0].shape[1]
    device = tensors[0].device
    perm = torch.argsort(torch.rand(B, n, device=device), dim=-1)  # (B, n)
    out = []
    for t in tensors:
        if t.dim() == 2:
            out.append(t.gather(dim=-1, index=perm))
        else:
            # gather along axis 1 with broadcast over trailing dims
            idx = perm.unsqueeze(-1).expand(B, n, t.shape[-1])
            out.append(t.gather(dim=1, index=idx))
    return tuple(out)

# -------------------------- helpers (MC replicas) ---------------------------

def _k_mc_train(cfg: Dict[str, Any]) -> int:
    """Single knob for ALL training sims (tgt0 + refresh + first)."""
    return int(cfg.get("K_MC", cfg.get("K_MC_MAX", 1)))

def _k_mc_valid(cfg: Dict[str, Any]) -> int:
    """Separate knob for validation sims."""
    return int(cfg.get("K_MC_VALID", cfg.get("K_MC", 1)))


def _mc_sim_max_rows(cfg: Dict[str, Any]) -> int:
    """Maximum replicated rows processed in one teacher-simulation chunk.

    This caps the temporary `(B * K_MC, n, F)` activations created by the
    teacher / validation Monte-Carlo path. It is purely a memory-control knob:
    the Monte-Carlo estimator itself is unchanged.
    """
    return max(1, int(cfg.get("MC_SIM_MAX_ROWS", 32768)))


def _mc_replica_block(cfg: Dict[str, Any], k_total: int) -> int:
    """Maximum number of MC replicas per scenario handled in one chunk."""
    return max(1, int(cfg.get("MC_REPLICA_BLOCK", k_total)))


@torch.no_grad()
def _current_step_pd_and_survival(nets,
                                  step: int,
                                  A_t: torch.Tensor,
                                  L,
                                  liab: torch.Tensor,
                                  n_it: int,
                                  *,
                                  a_dead: float,
                                  ell: Optional[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the time-t teacher part once on unrepeated base scenarios.

    A-only version: the network input is always ``A_t.unsqueeze(-1)``, with
    shape ``(B, n, 1)``.  No engineered breach/survival feature is constructed.
    """
    inp_t = A_t.unsqueeze(-1)
    logits_t = squeeze_last(nets[step](inp_t))
    pd_t_raw = torch.sigmoid(logits_t)
    pd_t, s_t = hard_clear_pd(pd_t_raw, A_t, L, liab, int(n_it), float(a_dead), ell=ell)
    return pd_t, s_t


@torch.no_grad()
def _next_step_post_clear_pd(nets,
                             step: int,
                             A_tp1: torch.Tensor,
                             L,
                             liab: torch.Tensor,
                             n_it: int,
                             *,
                             a_dead: float,
                             ell: Optional[float]) -> torch.Tensor:
    """Evaluate the shock-dependent time-(t+1) teacher part on flat rows.

    A-only version: every trainable future net receives ``A_tp1.unsqueeze(-1)``.
    The terminal net already consumes the same A-only shape.
    """
    inp_tp1 = A_tp1.unsqueeze(-1)
    logits_tp1 = squeeze_last(nets[step + 1](inp_tp1))
    pd_tp1_raw = torch.sigmoid(logits_tp1)
    pd_tp1, _ = hard_clear_pd(pd_tp1_raw, A_tp1, L, liab, int(n_it), float(a_dead), ell=ell)
    return pd_tp1


@torch.no_grad()
@profile
def _mc_average_td_target(cfg: Dict[str, Any],
                          nets,
                          step: int,
                          A_base: torch.Tensor,
                          L,
                          liab: torch.Tensor,
                          n_it: int,
                          *,
                          sigma: float,
                          dt: float,
                          a_dead: float,
                          ell: Optional[float],
                          k_mc_total: int,
                          use_sym: bool,
                          dR_single: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Chunked Monte-Carlo average of the TD teacher target.

    This A-only/vectorized implementation computes the current-time clearing
    once per base-scenario chunk, broadcasts the resulting survival mask across
    MC shock replicas, and evaluates only the shock-dependent next-time part on
    the expanded ``B * K`` rows.
    """
    _ = use_sym  # kept for call-site compatibility; sharing is valid for A-only deterministic nets.

    B, n = A_base.shape
    K = max(1, int(k_mc_total))
    rho = float(cfg["ASSET_CORR"])

    max_rows = _mc_sim_max_rows(cfg)
    rep_cap = min(K, _mc_replica_block(cfg, K))
    base_blk = max(1, max_rows // max(1, rep_cap))

    accum = torch.zeros(B, n, device=A_base.device, dtype=A_base.dtype)

    for start in range(0, B, base_blk):
        stop = min(start + base_blk, B)
        B0 = stop - start
        A_chunk = A_base[start:stop]

        _, s_t = _current_step_pd_and_survival(
            nets, step, A_chunk, L, liab, n_it,
            a_dead=a_dead, ell=ell,
        )

        s_t_3 = s_t.unsqueeze(1)           # (B0, 1, n), broadcasts over replicas
        A_chunk_3 = A_chunk.unsqueeze(1)   # (B0, 1, n), broadcasts over replicas

        done = 0
        while done < K:
            k_blk = min(rep_cap, K - done)

            if K == 1 and dR_single is not None:
                dR_3 = dR_single[start:stop].unsqueeze(1)
            else:
                dR_flat = draw_shocks_factor(B0 * k_blk, n, rho, float(sigma), float(dt),
                                             device=A_base.device)
                dR_3 = dR_flat.view(B0, k_blk, n)

            A_tp1_3 = torch.where(
                s_t_3 > 0.5,
                A_chunk_3 + dR_3,
                torch.full((B0, k_blk, n), float(a_dead), device=A_base.device, dtype=A_base.dtype),
            )
            A_tp1_flat = A_tp1_3.reshape(B0 * k_blk, n)

            pd_tp1_flat = _next_step_post_clear_pd(
                nets, step, A_tp1_flat, L, liab, n_it,
                a_dead=a_dead, ell=ell,
            )
            pd_tp1_3 = pd_tp1_flat.view(B0, k_blk, n)

            tgt_3 = (1.0 - s_t_3) + s_t_3 * pd_tp1_3
            accum[start:stop] += tgt_3.sum(dim=1)

            del dR_3, A_tp1_3, A_tp1_flat, pd_tp1_flat, pd_tp1_3, tgt_3
            done += k_blk

    return accum / float(K)

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
                                      L: "torch.Tensor | None",
                                      liab: torch.Tensor,
                                      n_it: int,
                                      *,
                                      a_dead: float,
                                      ell: Optional[float] = None,
                                      k_mc_share: int = 1):
    """A-only post-clearing one-step TD teacher.

    The neural input is always ``A.unsqueeze(-1)``.  If ``k_mc_share > 1``, the
    input rows must be grouped as consecutive replicas of each base scenario,
    allowing the current-time clearing to be evaluated once per base scenario
    and broadcast across replicas.
    """
    K = max(1, int(k_mc_share))

    if K > 1:
        A_t_unique = A_t[::K].contiguous()
        pd_t_u, s_t_u = _current_step_pd_and_survival(
            nets, step, A_t_unique, L, liab, n_it,
            a_dead=a_dead, ell=ell,
        )
        pd_t = pd_t_u.repeat_interleave(K, dim=0)
        s_t = s_t_u.repeat_interleave(K, dim=0)
    else:
        pd_t, s_t = _current_step_pd_and_survival(
            nets, step, A_t, L, liab, n_it,
            a_dead=a_dead, ell=ell,
        )

    A_next_free = A_t + dR_t
    A_tp1 = torch.where(s_t > 0.5, A_next_free, torch.full_like(A_t, float(a_dead)))

    pd_tp1 = _next_step_post_clear_pd(
        nets, step, A_tp1, L, liab, n_it,
        a_dead=a_dead, ell=ell,
    )

    tgt = (1. - s_t) + s_t * pd_tp1
    return pd_t, tgt, s_t

# ------------------------------ diagnostics ---------------------------------

@torch.no_grad()
def _diag_print_step0(step0: nn.Module, cfg: Dict[str, Any], A_list: Sequence[float]) -> None:
    """Print the post-clearing PDs of the trained step-0 net at chosen A.

    A-only diagnostic: input is always ``A.unsqueeze(-1)``.
    """
    device = next(step0.parameters()).device
    n = int(cfg["N_BANKS"])
    L, liab = build_matrices(cfg, device)
    a_dead = float(cfg.get("A_DEAD", -10.0))

    if len(A_list) == 1:
        A = torch.full((1, n), float(A_list[0]), device=device)
    elif len(A_list) == n:
        A = torch.tensor(A_list, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        raise ValueError(f"DIAG_A_LIST must be length 1 or {n} (got {len(A_list)})")

    ell = _sym_ell(cfg, n)
    xin = A.unsqueeze(-1)

    logits = squeeze_last(step0(xin))
    pd_raw = torch.sigmoid(logits)
    pd_post, _ = hard_clear_pd(pd_raw, A, L, liab, n_it=int(n), a_dead=a_dead, ell=ell)

    arr = pd_post.squeeze(0).detach().cpu().numpy()
    msg = (f"[diag] corr={cfg['ASSET_CORR']:.4f}  A={list(A_list)}  "
           f"post-clear PD mean={arr.mean():.4f}")
    print(msg)

def np_round(x, k: int = 4):
    try:
        import numpy as _np
        return _np.round(x, k).tolist()
    except Exception:
        return x

# ----------------------------- training step --------------------------------
@profile
def _train_one_step(cfg: Dict[str, Any],
                    nets, step: int,
                    A_buf, dR_buf,
                    L, liab,
                    device: torch.device,
                    log: List[str],
                    diag: "TgtBufDiagnostics | None" = None) -> None:

    net = nets[step]
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["LR_SMALL"]))
    bce_elem = nn.BCELoss(reduction="none")
    mse = nn.MSELoss()

    n_it = int(L.shape[0])
    use_ste = bool(cfg.get("USE_STE_CLEARING", True))
    a_dead = float(cfg.get("A_DEAD", -10.0))
    mask_surv = bool(cfg.get("MASK_LOSS_TO_SURVIVORS", True))

    n = int(cfg["N_BANKS"])
    use_sym = bool(cfg.get("USE_SYMMETRIC_ASSUMPTION", False))
    ell = _sym_ell(cfg, n)

    BUF = int(cfg["BUFFER_SIZE"])
    VAL = int(min(cfg.get("VAL_SAMPLES", BUF // 5), BUF))
    train_pool_idx = torch.arange(VAL, BUF, device=device)
    val_idx = torch.arange(0, VAL, device=device)

    with torch.no_grad():
        if cfg.get("ZERO_WARMSTART", False):
            zeros = [ZeroSubModel(int(cfg["N_BANKS"])).to(device) for _ in range(cfg["TOTAL_STEPS"])]
            zeros.append(nets[-1])
            nets_for_tgt0 = zeros
        else:
            nets_for_tgt0 = nets

        k_mc = _k_mc_train(cfg)
        tgt0 = _mc_average_td_target(
            cfg, nets_for_tgt0, step, A_buf,
            L, liab, n_it,
            sigma=cfg["SIGMA"], dt=cfg["DT"],
            a_dead=a_dead, ell=ell,
            k_mc_total=k_mc,
            use_sym=use_sym,
            dR_single=dR_buf,
        )
    tgt_buf = tgt0.detach().clamp_(0., 1.)

    if diag is not None:
        diag.record(step, epoch=0, tgt_buf=tgt_buf)

    best_val_mse = float("inf")
    best_state = None
    eval_bad = 0

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
        tgt_b = tgt_buf[idx]

        # Under the symmetric assumption, the flat MLP benefits from random
        # per-scenario bank permutations.  DeepSets is already equivariant, so
        # this augmentation is redundant there.
        if use_sym and (not getattr(net, "use_deepsets", False)):
            A_b, tgt_b = _permute_per_scenario(A_b, tgt_b)

        opt.zero_grad(set_to_none=True)
        x_b = A_b.unsqueeze(-1)

        logits = squeeze_last(net(x_b))
        pd_raw = torch.sigmoid(logits)

        if use_ste:
            pd_pred = ste_clear_pd(pd_raw, A_b, L, liab, n_it, a_dead, ell=ell)
        else:
            with torch.no_grad():
                pd_pred, _ = hard_clear_pd(pd_raw, A_b, L, liab, n_it, a_dead, ell=ell)

        per_elem = bce_elem(pd_pred, tgt_b)
        if mask_surv: #TODO do we need this?
            with torch.no_grad():
                _, s_curr = hard_clear_pd(pd_raw, A_b, L, liab, n_it, a_dead, ell=ell)
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
                tgt_buf[sel_train] = _mc_average_td_target(
                    cfg, nets, step, A_buf[sel_train],
                    L, liab, n_it,
                    sigma=cfg["SIGMA"], dt=cfg["DT"],
                    a_dead=a_dead, ell=ell,
                    k_mc_total=k_mc,
                    use_sym=use_sym,
                    dR_single=dR_buf[sel_train],
                )

            last_refresh_epoch = ep + 1

            if diag is not None:
                diag.record(step, epoch=ep + 1, tgt_buf=tgt_buf,
                            n_refreshed=sel_train.numel())

        should_eval = ((ep + 1) == (last_refresh_epoch + 1)) and ((ep + 1) >= next_eval_epoch)
        if should_eval:
            with torch.no_grad():
                x_v = A_buf[val_idx].unsqueeze(-1)

                logits_v = squeeze_last(net(x_v))
                pd_raw_v = torch.sigmoid(logits_v)
                pd_pred_v, _ = hard_clear_pd(pd_raw_v, A_buf[val_idx], L, liab,
                                             n_it, a_dead, ell=ell)

                k_mc_v = _k_mc_valid(cfg)
                true_v = _mc_average_td_target(
                    cfg, nets, step, A_buf[val_idx],
                    L, liab, n_it,
                    sigma=cfg["SIGMA"], dt=cfg["DT"],
                    a_dead=a_dead, ell=ell,
                    k_mc_total=k_mc_v,
                    use_sym=use_sym,
                    dR_single=None,
                )

                val_mse = mse(pd_pred_v, true_v).item()
                val_ce  = nn.BCELoss()(pd_pred_v, true_v).item()
                ce_min  = approx_min_ce(true_v)

                msg = (f"[eval] step {step} ep {ep + 1}: "
                       f"MSE={val_mse:.4e}  CE={val_ce:.4e}  minCE≈{ce_min:.4e}")
                print(msg)
                log.append(msg)

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

    in_F = 1

    use_deepsets = bool(cfg.get("USE_DEEPSETS", False))
    ell = _sym_ell(cfg, n)

    nets = [PDToMaturityNet(
                n,
                in_features_per_node=in_F,
                use_deepsets=use_deepsets,
                d_embed=int(cfg.get("DEEPSETS_D_EMBED", 64)),
                h_hidden=int(cfg.get("DEEPSETS_H_HIDDEN", 128)),
            ).to(device)
            for _ in range(T)]
    nets.append(LastStepPDIter(L, liab,
                               a_dead=float(cfg.get("A_DEAD", -10.0)),
                               ell=ell).to(device))

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
                k_surv_iters=int(cfg.get("EXTRA_K_SURV_ITERS", 5)),
                L=L, liab=liab, T_total=float(cfg["T_TOTAL"]),
                ell=ell,
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
                ell=ell,
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

# not used in ayn experiments
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
