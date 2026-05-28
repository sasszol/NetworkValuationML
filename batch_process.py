"""batch_process.py

Central configuration + convenience entry point.

This file is used in two ways:
  1) as a config module (CFG dict) consumed by training/benchmark scripts
  2) as an executable to run a quick correlation sweep

Key switches
------------
NAIVE_PATHS_METHOD
  Controls which *naive multi-step buffer generator* is used when
  USE_ROLLOUT_PATHS=True in sc_train:

    - "sc_data"         : rollout sampler from sc_data (Brownian-bridge
                           barrier hits + optional EXTRA_DEFAULTS overlay)
    - "static_barrier"  : SDB benchmark buffer generator (Picard fixed-point
                           + projection each step)

  The same value can also be used as the sweep method when running this file
  as a script.

USE_DEEPSETS  (architecture switch)
  False -> use the original BaseMLP predictor (Flatten -> 64 -> 64 -> 32 -> N).
  True  -> use the DeepSets predictor (permutation-equivariant by construction;
           parameter count independent of N; supports zero-shot transfer to
           other N values). See deepsets_vs_mlp.tex for the full discussion.

USE_SYMMETRIC_ASSUMPTION  (computational switch; valid iff the homogeneous
                            symmetric network of the paper's Assumption is in
                            force, i.e. L_ij = kL/(N-1) off-diagonal, b_i =
                            kL, common a_0).
  When True, the following optimisations are enabled jointly:

    * Layer-1 rank-1 / diagonal factorisation of L in every clearing-related
      primitive (hard_clear_pd, ste_clear_pd, LastStepPDIter,
      projected_static_barrier_fp, iterative_survival_feature). Brings the
      per-scenario clearing cost from O(N^3) to O(N^2). Numerics are exact
      up to floating-point ordering.

    * Layer-4 sharing of the "pre-shock" time-t clearing across MC replicas
      inside simulate_one_step_with_projection: replicas of the same scenario
      share the same A_t, hence the same pd_t / s_t.

    * Option B (data-side symmetrisation):
        - permutation augmentation of training batches (joint S_n permutation
          of A, psi, target). For a non-equivariant predictor this teaches the
          symmetry directly from data and improves sample efficiency, so
          fewer K_MC may be sufficient at the same target MSE.

  Note: an earlier draft also collapsed per-agent TD targets to the
  within-scenario mean over alive agents ("Rao-Blackwell"). That step was
  removed because it is biased: under the symmetric assumption the joint law
  of A_t is exchangeable, but a single realisation is heterogeneous from t>0
  onward (different idiosyncratic shocks per bank), so each per-agent target
  estimates a different conditional expectation and pathwise averaging biases
  the TD target.

  The two flags are independent: you may enable the architecture refactor
  without the symmetric-assumption optimisations and vice versa.
"""

import numpy as np

from sc_train import sweep_correlation


CFG = dict(
    N_BANKS=5,
    TOTAL_STEPS=50,        # number of revaluation dates (final settlement after last propagation)
    T_TOTAL=1.0,           # overall horizon
    DT=0.02,                # step size
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

    BATCH_FRACT=1,

    # Refresh schedule
    REFRESH_EVERY=40,
    REFRESH_FRACT=1,
    REFRESH_EVERY_SCHEDULE={0: 40},
    REFRESH_FRACT_SCHEDULE={0: 1},

    MAX_EPOCHS=7_000,
    LR_SMALL=2e-4,
    LR_SCHEDULE={0: 7e-4, 4_000: 5e-4, 7_000: 2e-4},
    GRAD_CLIP=1.0,

    EVAL_EVERY=1_000,
    EVAL_PATIENCE=10,
    EVAL_MIN_DELTA=0.0,
    VAL_SAMPLES=5_000,

    # ----------------- Architecture switch -----------------
    # False -> original BaseMLP (Flatten -> 64 -> 64 -> 32 -> N)
    # True  -> DeepSets (permutation-equivariant; parameter count independent of N)
    USE_DEEPSETS=False,

    # ----------------- Symmetric-assumption switch -----------------
    # Enables the rank-1 clearing fast path AND permutation augmentation of
    # training batches AND sharing of the pre-shock clearing across MC
    # replicas. Only valid when the homogeneous symmetric network assumption
    # is in force.
    USE_SYMMETRIC_ASSUMPTION=False,

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
    results = sweep_correlation(CFG, corr_grid, save_dir="steps_check_5_banks_2_DD_1_kL/50", seed=43)
    # Run a correlation sweep using either naive simulator.
    # Output: CSV with NO header and one line per corr value: corr,avg_pd_T_total
    #sweep_correlation(
    #    CFG,
    #    corr_grid,
    #    seed=43,
    #    out_csv=str("baseline.csv"),
    #    method="sc_data",
    #)
