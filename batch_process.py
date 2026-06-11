"""batch_process.py

Central configuration + convenience entry point.

This file is used in two ways:
  1) as a config module (CFG dict) consumed by training/benchmark scripts
  2) as an executable to run a quick correlation sweep

Key switches
------------
NAIVE_PATHS_METHOD
  Controls which naive multi-step buffer generator is used when
  USE_ROLLOUT_PATHS=True in sc_train:

    - "sc_data"         : rollout sampler from sc_data (Brownian-bridge
                           barrier hits + optional EXTRA_DEFAULTS overlay)
    - "static_barrier"  : SDB benchmark buffer generator (Picard fixed-point
                           + projection each step)

USE_DEEPSETS
  False -> original flat MLP predictor.
  True  -> DeepSets predictor (permutation-equivariant architecture).

Important modelling note
------------------------
We keep the *initial / structural* homogeneous-complete network assumptions of
the benchmark setup (equal off-diagonal L, common initial setup, one-factor
asset shocks). But we do NOT apply any post-shock/pathwise symmetrisation once
assets become heterogeneous.

Concretely:
  * The old data-side permutation augmentation / pathwise symmetrisation is not
    used.
  * Full bank-level clearing / ψ evaluation is retained after shocks.
  * Chunked MC target evaluation is used to avoid CUDA OOM during training /
    validation without changing the estimator.
"""

import numpy as np

from sc_train import sweep_correlation


CFG = dict(
    N_BANKS=20,
    TOTAL_STEPS=10,        # number of revaluation dates (final settlement after last propagation)
    T_TOTAL=1.0,           # overall horizon
    DT=0.1,                # step size
    SIGMA=1.0,
    INIT_DD=2.0,
    ASSET_CORR=0.80,       # correlation of increments (one-factor)
    kL=1.0,
    BUFFER_SIZE=30_000,

    # ----------------- NEW: choose naive buffer generator -----------------
    # "sc_data" or "static_barrier" (SDB)
    NAIVE_PATHS_METHOD="static_barrier",

    # MC knobs
    K_MC=50,
    K_MC_VALID=2_000,
    MC_SIM_MAX_ROWS=16_384,   # max replicated rows per teacher/validation chunk
    MC_REPLICA_BLOCK=32,      # max MC replicas per chunk

    BATCH_FRACT=1,

    # Refresh schedule
    REFRESH_EVERY=40,
    REFRESH_FRACT=1,
    REFRESH_EVERY_SCHEDULE={0: 40},
    REFRESH_FRACT_SCHEDULE={0: 1},

    MAX_EPOCHS=4_000,
    LR_SMALL=2e-4,
    LR_SCHEDULE={0: 6e-4, 4_000: 1e-4, 7_000: 6e-5},
    GRAD_CLIP=1.0,

    EVAL_EVERY=4000,
    EVAL_PATIENCE=10,
    EVAL_MIN_DELTA=0.0,
    VAL_SAMPLES=5_000,

    # ----------------- Architecture switch -----------------
    # False -> original BaseMLP (Flatten -> 64 -> 64 -> 32 -> N)
    # True  -> DeepSets (permutation-equivariant; parameter count independent of N)
    USE_DEEPSETS=True,

    # Legacy compatibility key; retained only so older scripts do not break.

    # Final-step / clearing params
    A_DEAD=-4.0,
    ZERO_WARMSTART=True,
    USE_STE_CLEARING=True,

    # feature: breach ψ (counterparty-aware survival complement)
    K_SURV_ITERS=5,
    USE_PHI_FEATURE=False,       # use [A, ψ] as inputs

    MASK_LOSS_TO_SURVIVORS=True,

    # ---------------- NEW: rollout sampler knobs ----------------
    USE_ROLLOUT_PATHS=True,  # turn on the multi-step rollout buffers
    BAND_MULT=1.5,  # width around INIT_DD for scenario base
    INIT_JITTER=0.05,  # per-bank jitter (shrinks as ρ→1)
    STEP_JITTER_SD=0.0,  # tiny extra idio jitter added to dR each step
    SCENARIO_STRESS_PROB=0.15,  # optional stress mixture
    SCENARIO_STRESS_SHIFT=-1.0,

    # stochastic extra defaults per step (asset/network-correlated)
    EXTRA_DEFAULTS_ON=True,
    EXTRA_INTENSITY=0.20,  # scales ψ → extra default prob
    EXTRA_CAP=0.25,  # cap for extra default probability
    EXTRA_RHO=None,  # None → use ASSET_CORR

    # SDB-only solver tolerance (Picard fixed point)
    SDB_TOL=1e-6,

    # Diagnostics
    DIAG_A_LIST=[2.0],
)


def _default_corr_grid() -> np.ndarray:
    part1 = np.linspace(0.0, 0.8, 20, endpoint=True)
    part2 = np.linspace(0.8, 1.0, 10, endpoint=True)[1:]
    return np.concatenate([part1, part2])


if __name__ == "__main__":
    corr_grid = _default_corr_grid()
    results = sweep_correlation(CFG, corr_grid[corr_grid>0.87], save_dir="C:/git/NetworkValuationML/kL_1_DD_2/20_banks_longer_2", seed=43)
    # Run a correlation sweep using either naive simulator.
    # Output: CSV with NO header and one line per corr value: corr,avg_pd_T_total
    #sweep_correlation(
    #    CFG,
    #    corr_grid,
    #    seed=43,
    #    out_csv=str("baseline.csv"),
    #    method="sc_data",
    #)
