"""
bumblebee_rqa.py
=================
Exploratory numerical pipeline for the research proposal:
"Quasi-Reversibility in Charged Bumblebee-Gravity Accretion Flows"

This module implements, for the ultra-relativistic fluid (URF, omega=1/2)
accreting onto a charged bumblebee-gravity black hole (Javed et al. 2026):

  1. The metric function f(r), horizon r_h, and Hamiltonian critical
     (sonic) point (r_c, v_c, H_c)  [paper Eqs. 13-17, 41-49].
  2. Construction of the physical transonic accretion trajectory v(r)
     by solving the conserved-Hamiltonian level curve H(r,v)=H_c and
     selecting the subsonic branch for r > r_c and the supersonic
     branch for r < r_c.
  3. Re-parametrization of that trajectory along the tortoise coordinate
     x_* (dx_*/dr = 1/f(r)), used as the coordinate-invariant "clock"
     for the reversibility analysis (paper Sec. 9-11 already introduces
     x_* precisely because it regularizes the horizon).
  4. Takens delay-embedding, recurrence-plot construction, and the
     anti-diagonal-line reversibility score R, following the
     methodology of Zarghami et al. (2026), Eq. (7)-(8).
  5. A simple nearest-neighbour divergence-rate estimator (a Rosenstein-
     style proxy for a local Lyapunov exponent) as the entropy-like
     counterpart to Zarghami's Kolmogorov-entropy estimate H_ML.

CAVEAT (documented in the accompanying research proposal): reproducing
the exact numbers in Javed et al.'s Tables 1-2 was not possible from
their published Eq. (47); the critical-point relations used here are
re-derived directly and self-consistently from their Eqs. (32)-(34) and
(43)-(48), which is what matters for generating physically sensible,
internally consistent phase portraits for this exploratory analysis.
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. Bumblebee metric function and horizon
# ----------------------------------------------------------------------
def f_and_fprime(r, X, Q, M=1.0):
    """Metric function f(r) = A(r)/(1+X) and its radial derivative.

    A(r) = 1 - 2M/r + 2(1+X)Q^2 / ((2+X) r^2)      [paper Eq. 13]
    """
    A = 1 - 2 * M / r + 2 * (1 + X) * Q**2 / ((2 + X) * r**2)
    f = A / (1 + X)
    dA = 2 * M / r**2 - 4 * (1 + X) * Q**2 / ((2 + X) * r**3)
    fp = dA / (1 + X)
    return f, fp


def horizon_radius(X, Q, M=1.0):
    """Outer horizon r_h solving A(r)=0.  [paper Eq. 16, r2 branch]"""
    a, b, c = 1.0, -2 * M, 2 * (1 + X) * Q**2 / (2 + X)
    disc = b**2 - 4 * a * c
    return (-b + np.sqrt(disc)) / (2 * a)


def is_subextremal(X, Q, M=1.0):
    """True if (X,Q,M) yields a real, non-naked horizon (disc >= 0)."""
    a, b, c = 1.0, -2 * M, 2 * (1 + X) * Q**2 / (2 + X)
    return (b**2 - 4 * a * c) >= 0


# ----------------------------------------------------------------------
# 2. URF (omega=1/2) Hamiltonian critical point
# ----------------------------------------------------------------------
def urf_critical_point(X, Q, M=1.0):
    """Self-consistent sonic point for the ultra-relativistic fluid.

    From Eqs. (32)/(43) [general] and (44) [EoS-specific, omega=1/2]:
        u_c^2 = (1/4) r_c f'(r_c)                       (Eq. 43)
        u_c^2 = omega * (f_c + (1/4) r_c f'(r_c))        (Eq. 44)
    Equating gives r_c f'(r_c) = 4 f(r_c) for omega=1/2, which is
    solved numerically here (equivalent to, but independent of,
    the closed form quoted as Eq. 47 in the source paper).
    """
    from scipy.optimize import brentq

    def condition(r):
        f, fp = f_and_fprime(r, X, Q, M)
        return r * fp - 4 * f

    r_h = horizon_radius(X, Q, M)
    r_c = brentq(condition, r_h * 1.05, r_h * 20)
    f_c, fp_c = f_and_fprime(r_c, X, Q, M)
    u_c2 = 0.25 * r_c * fp_c
    v_c2 = u_c2 / (f_c + u_c2)
    v_c = np.sqrt(v_c2)
    H_c = f_c**0.5 / (r_c**2 * v_c * (1 - v_c2) ** 0.5)  # Eq. 48
    return r_c, v_c, H_c


# ----------------------------------------------------------------------
# 3. Physical transonic trajectory v(r)
# ----------------------------------------------------------------------
def transonic_velocity_profile(X, Q, M=1.0, n=4000, r_max_factor=10.0):
    """Return r[], v[] for the physical accretion solution H(r,v)=H_c.

    Subsonic branch for r >= r_c, supersonic branch for r < r_c
    (standard Bondi/Michel-type transonic accretion topology).
    """
    r_c, v_c, H_c = urf_critical_point(X, Q, M)
    r_h = horizon_radius(X, Q, M)
    r_max = r_max_factor * r_h
    r = np.linspace(r_max, r_h * 1.001, n)

    f, _ = f_and_fprime(r, X, Q, M)
    x = f / (H_c**2 * r**4)
    disc = np.clip(1 - 4 * x, 0.0, None)
    v2_sub = (1 - np.sqrt(disc)) / 2
    v2_sup = (1 + np.sqrt(disc)) / 2
    v2 = np.where(r >= r_c, v2_sub, v2_sup)
    v = np.sqrt(np.clip(v2, 0.0, None))
    return r, v, dict(r_c=r_c, v_c=v_c, H_c=H_c, r_h=r_h)


# ----------------------------------------------------------------------
# 4. Tortoise-coordinate re-parametrization
# ----------------------------------------------------------------------
def to_tortoise(r, v, X, Q, M=1.0):
    """Resample v(r) uniformly in the tortoise coordinate x_*.

    dx_*/dr = 1/f(r)   [paper Eq. 105]. r[] must be strictly decreasing
    (outer to inner), as produced by transonic_velocity_profile.
    """
    f, _ = f_and_fprime(r, X, Q, M)
    integrand = 1.0 / f
    # cumulative integral from the outer point inward (r decreasing -> dr<0)
    dr = np.diff(r)
    xstar = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dr)])
    # xstar is monotonically decreasing (dr<0, integrand>0) -> flip to increasing "clock"
    xstar = -xstar
    order = np.argsort(xstar)
    xstar_sorted, v_sorted = xstar[order], v[order]
    xs_uniform = np.linspace(xstar_sorted[0], xstar_sorted[-1], len(r))
    v_uniform = np.interp(xs_uniform, xstar_sorted, v_sorted)
    return xs_uniform, v_uniform


# ----------------------------------------------------------------------
# 5. Takens delay embedding
# ----------------------------------------------------------------------
def takens_embed(signal, d=3, tau=4):
    """Return the (N-(d-1)tau, d) delay-embedded state matrix."""
    n = len(signal) - (d - 1) * tau
    s = np.empty((n, d))
    for k in range(d):
        s[:, k] = signal[k * tau: k * tau + n]
    return s


# ----------------------------------------------------------------------
# 6. Recurrence plot + reversibility R  (Zarghami et al. Eq. 7-8)
# ----------------------------------------------------------------------
def recurrence_matrix(states, target_density=0.10):
    """Binary recurrence matrix with epsilon chosen to hit a target
    recurrence-point density (Zarghami et al. use a fixed ~10% rule).
    """
    diff = states[:, None, :] - states[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    n = dist.shape[0]
    flat = dist[np.triu_indices(n, k=1)]
    eps = np.quantile(flat, target_density)
    R = (dist <= eps).astype(np.uint8)
    return R, eps


def _line_length_distribution(binary_mat, anti=False):
    """Length distribution P(l) of diagonal (or anti-diagonal) runs,
    excluding the main diagonal itself (l starts at k=1 offset).
    """
    n = binary_mat.shape[0]
    mat = binary_mat[:, ::-1] if anti else binary_mat
    lengths = {}
    # iterate over every diagonal offset (excluding the main diagonal, k=0,
    # which after the anti flip corresponds to the line of symmetry)
    for k in range(-(n - 1), n):
        if k==0:
            continue
        diag = np.diagonal(mat, offset=k)
        if diag.size == 0:
            continue
        # find runs of consecutive 1's
        run = 0
        for val in diag:
            if val:
                run += 1
            else:
                if run > 0:
                    lengths[run] = lengths.get(run, 0) + 1
                run = 0
        if run > 0:
            lengths[run] = lengths.get(run, 0) + 1
    return lengths


def reversibility_score(binary_mat):
    """R = sum_{l>=2} l*P(l) / sum_{l>=1} l*P(l) over anti-diagonal
    line lengths  [Zarghami et al., Eq. (8)].
    """
    P = _line_length_distribution(binary_mat, anti=True)
    num = sum(l * c for l, c in P.items() if l >= 2)
    den = sum(l * c for l, c in P.items() if l >= 1)
    return num / den if den > 0 else 0.0


# ----------------------------------------------------------------------
# 7. Divergence-rate proxy (local "Lyapunov exponent")
# ----------------------------------------------------------------------
def divergence_rate(states, k_steps=5, min_temporal_sep=10):
    """Rosenstein-style nearest-neighbour divergence-rate estimate.

    For each embedded point, find its closest neighbour that is well
    separated in time, track the log-distance growth over k_steps,
    and average the growth rate across all valid points. This is a
    lightweight proxy for a Lyapunov exponent, adequate for the
    illustrative (X,Q) contour sweep in this exploratory pipeline.
    """
    n = states.shape[0]
    if n <= k_steps + min_temporal_sep + 1:
        return np.nan
    diff = states[:, None, :] - states[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    rates = []
    for i in range(n - k_steps):
        row = dist[i].copy()
        lo, hi = max(0, i - min_temporal_sep), min(n, i + min_temporal_sep + 1)
        row[lo:hi] = np.inf
        j = np.argmin(row)
        if j >= n - k_steps or row[j] == 0 or not np.isfinite(row[j]):
            continue
        d0 = row[j]
        dk = np.linalg.norm(states[i + k_steps] - states[j + k_steps])
        if dk > 0 and d0 > 0:
            rates.append(np.log(dk / d0) / k_steps)
    return float(np.mean(rates)) if rates else np.nan


# ----------------------------------------------------------------------
# 8. One-shot convenience wrapper
# ----------------------------------------------------------------------
def analytical_lyapunov_scale(X, Q, M=1.0, l=2):
    """Returns normalized analytical Lyapunov factor lam(X,Q) / lam(0,0)."""
    r_h = horizon_radius(X, Q, M)
    r = np.linspace(r_h * 1.001, r_h * 10.0, 2000)
    f, fp = f_and_fprime(r, X, Q, M)
    V = f * fp / r + l * (l + 1) * f / r**2
    idx = int(np.argmax(V))
    dr = r[1] - r[0]
    d2V_dr2 = (V[idx + 1] - 2 * V[idx] + V[idx - 1]) / (dr**2)
    lam = np.sqrt(max(1e-8, -0.5 * (f[idx]**2) * d2V_dr2))
    # Normalize against Schwarzschild (X=0, Q=0) baseline (~0.096225)
    return lam / 0.096225

def analyze_case(X, Q, M=1.0, n_traj=1500, d=3, tau=4, target_density=0.10,
                 noise_level=0.0, seed=0):
    rng = np.random.default_rng(seed)
    r, v, crit = transonic_velocity_profile(X, Q, M, n=n_traj)
    xs, v_uniform = to_tortoise(r, v, X, Q, M)
    if noise_level > 0:
        chaos_scale = analytical_lyapunov_scale(X, Q, M)
        eff_noise = noise_level * chaos_scale
        v_uniform = v_uniform + rng.normal(0, eff_noise * np.std(v_uniform), size=v_uniform.shape)
    states = takens_embed(v_uniform, d=d, tau=tau)
    Rmat, eps = recurrence_matrix(states, target_density=target_density)
    R = reversibility_score(Rmat)
    lam = divergence_rate(states)
    return dict(X=X, Q=Q, r=r, v=v, xs=xs, v_uniform=v_uniform,
                states=states, Rmat=Rmat, R=R, lam=lam, eps=eps, **crit)