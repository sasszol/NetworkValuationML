import math
from typing import Any, Dict, Tuple, Optional

import torch


def build_L(n: int, offdiag: float = 1.0) -> torch.Tensor:
    """All-ones off-diagonal exposure matrix (diag=0)."""
    L = torch.ones(n, n, dtype=torch.float32) * float(offdiag)
    L.fill_diagonal_(0.0)
    return L


def infer_homogeneous_complete_ell(
    L: torch.Tensor,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> Optional[float]:
    """Detect the homogeneous complete-network form ``L = ell * (11^T - I)``.

    This is a structural check on the fixed liability matrix. When it succeeds,
    the rank-1 formulas used in clearing / ψ evaluation are exact even after the
    realised asset vector becomes heterogeneous.
    """
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        return None

    n = int(L.shape[0])
    if n <= 1:
        return None

    diag = torch.diagonal(L)
    if not torch.allclose(diag, torch.zeros_like(diag), atol=atol, rtol=rtol):
        return None

    mask = ~torch.eye(n, dtype=torch.bool, device=L.device)
    off = L[mask]
    ell_t = off.mean()
    if not torch.allclose(off, torch.full_like(off, ell_t), atol=atol, rtol=rtol):
        return None

    return float(ell_t.item())


def build_matrices(cfg: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the interbank exposure matrix and implied liabilities.

    Returns
    -------
    (L, liab)
      L    : (n,n) exposure matrix
      liab : (n,)  interbank liabilities

    Behaviour
    ---------
    * If ``cfg["L_MATRIX"]`` is provided, it is used verbatim. This allows
      asymmetric / heterogeneous network structures without touching the rest of
      the codebase.
    * Otherwise we fall back to the paper's homogeneous fully-connected default
      ``L_ij = kL/(n-1)`` for ``i != j`` and ``L_ii = 0``.

    Notes
    -----
    Outside liabilities are assumed to be absorbed into the external asset
    variable A ("distance-to-default") at t=0.
    """
    n = int(cfg["N_BANKS"])

    if n <= 1:
        raise ValueError(f"N_BANKS must be >= 2 (got {n}).")

    if cfg.get("L_MATRIX", None) is not None:
        L = torch.as_tensor(cfg["L_MATRIX"], dtype=torch.float32, device=device)
        if L.shape != (n, n):
            raise ValueError(
                f"L_MATRIX must have shape ({n}, {n}) (got {tuple(L.shape)})."
            )
        liab_cfg = cfg.get("LIAB_VECTOR", None)
        if liab_cfg is not None:
            liab = torch.as_tensor(liab_cfg, dtype=torch.float32, device=device)
            if liab.shape != (n,):
                raise ValueError(
                    f"LIAB_VECTOR must have shape ({n},) (got {tuple(liab.shape)})."
                )
        else:
            liab = L.sum(1)
        return L, liab

    kL = float(cfg.get("kL", 1.0))
    L = build_L(n, offdiag=kL / (n - 1))
    liab = L.sum(1)
    return L.to(device), liab.to(device)


# -------------------------- shared initial sampling --------------------------

@torch.no_grad()
def sample_A0_band(
    S: int,
    n: int,
    *,
    init_dd: float,
    sigma: float,
    rho: float,
    T_steps: int,
    dt: float,
    device: torch.device,
    band_mult: float = 1.0,
    init_jitter: float = 0.0,
    stress_prob: float = 0.0,
    stress_shift: float = 0.0,
) -> torch.Tensor:
    """Scenario-band sampler for the initial asset state A(0).

    This is the shared implementation used by:
      - sc_data.build_rollout_paths
      - sc_staticbarrier_benchmark (state-dependent static-barrier benchmark)

    The construction follows the LaTeX description:
      - scenario-level base drawn uniformly in a band around INIT_DD
      - optional stressed mixture (scenario-level shift)
      - optional per-bank jitter, scaled by sqrt(1-rho) so dispersion shrinks as rho -> 1

    Parameters
    ----------
    S : number of scenarios
    n : number of banks
    init_dd : center of the band
    sigma : ABM volatility (only used to set the band width)
    rho : one-factor correlation in [0,1]
    T_steps, dt : used to set the band width via sqrt(T_steps*dt)
    device : torch device

    Returns
    -------
    A0 : (S, n) float tensor
    """
    S_ = int(S)
    n_ = int(n)
    T_steps_ = int(T_steps)
    dt_ = float(dt)

    # Same functional form as in the original rollout sampler.
    base_width = float(band_mult) * (0.4 + 0.1 * float(sigma) * math.sqrt(max(T_steps_ * dt_, 1e-8)))

    base = torch.empty(S_, 1, device=device).uniform_(float(init_dd) - base_width, float(init_dd) + base_width)

    if float(stress_prob) > 0.0 and float(stress_shift) != 0.0:
        stress_mask = (torch.rand(S_, 1, device=device) < float(stress_prob)).float()
        base = base + stress_mask * float(stress_shift)

    if float(init_jitter) > 0.0:
        jitter_sd = float(init_jitter) * base_width * math.sqrt(max(0.0, 1.0 - float(rho)))
        A0 = base + torch.randn(S_, n_, device=device) * jitter_sd
    else:
        A0 = base.expand(S_, n_).clone()

    return A0


# ---------- ABM survival / breach proxy (counterparty-aware, iterated) ----------

@torch.no_grad()
def draw_shocks_factor(
    sz: int,
    n: int,
    rho: float,
    sigma: float,
    dt: float,
    device: torch.device,
) -> torch.Tensor:
    """Correlated ABM increments via a single common-factor decomposition.

    Output shape: (sz, n)

    Note: this construction assumes rho in [0,1].
    """
    scale = float(sigma) * math.sqrt(float(dt))
    mkt = torch.randn(int(sz), 1, device=device)  # common market factor M
    idio = torch.randn(int(sz), int(n), device=device)  # idiosyncratic eps_i
    return scale * (math.sqrt(float(rho)) * mkt + math.sqrt(1.0 - float(rho)) * idio)


@torch.no_grad()
def _survival_abm(
    A: torch.Tensor,
    H: torch.Tensor,
    sigma: float,
    T: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """ABM survival above a flat barrier.

    Arithmetic Brownian motion: X_t = A + sigma * W_t, flat barrier H.

    Survival (no down-crossing) by horizon T:
        P_surv = 2 * Phi((A - H) / (sigma * sqrt(T))) - 1.
    """
    T_eff = max(float(T), eps)
    denom = max(float(sigma) * math.sqrt(T_eff), eps)
    d = (A - H) / denom
    cdf = 0.5 * (1.0 + torch.erf(d / math.sqrt(2.0)))
    out = 2.0 * cdf - 1.0
    return torch.clamp(out, 0.0, 1.0)


def _apply_L_util(spay: torch.Tensor,
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
def iterative_survival_feature(
    A: torch.Tensor,
    L: "torch.Tensor | None",
    liab: torch.Tensor,
    sigma: float,
    T: float,
    k: int = 5,
    a_dead: float = -10.0,
    eps: float = 1e-8,
    *,
    ell: "float | None" = None,
) -> torch.Tensor:
    """Counterparty-aware survival phi via fixed point on expected payments.

    Alive is inferred from assets via the sentinel: A > a_dead.

    If `ell` is provided, uses the symmetric rank-1 form of L
    (L = ell*(11^T - I)); `L` may then be passed as None.
    """
    liab_row = liab.unsqueeze(0)
    S = (A > (float(a_dead) + eps)).float()

    spay = S  # initial guess: survivors pay 1
    H = liab_row - _apply_L_util(spay, L, ell)
    phi = _survival_abm(A, H, float(sigma), float(T), eps)
    phi = torch.minimum(phi, S)

    for _ in range(int(max(0, k))):
        spay = torch.minimum(S, phi)
        H = liab_row - _apply_L_util(spay, L, ell)
        phi = _survival_abm(A, H, float(sigma), float(T), eps)
        phi = torch.minimum(phi, S)

    return phi


@torch.no_grad()
def iterative_breach_feature(
    A: torch.Tensor,
    L: "torch.Tensor | None",
    liab: torch.Tensor,
    sigma: float,
    T: float,
    k: int = 5,
    a_dead: float = -10.0,
    eps: float = 1e-8,
    *,
    ell: "float | None" = None,
) -> torch.Tensor:
    """Barrier breach probability psi = 1 - phi_survival (ABM, iterated)."""
    surv = iterative_survival_feature(A, L, liab, sigma, T, k, a_dead, eps, ell=ell)
    return 1.0 - surv
