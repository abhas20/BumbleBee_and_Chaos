"""
bumblebee_potentials.py
========================
Direct visualization / independent verification of the core physics
claims in Javed, Shaukat, Waseem & Mustafa (2026), "Matter accretion
and greybody factor of charged bumblebee gravity black hole".

Unlike bumblebee_rqa.py (which builds an exploratory reversibility
metric on top of the paper's accretion solutions), this module stays
close to the paper's own equations and reproduces, independently, the
three qualitative claims the whole proposal leans on:

  (1) Increasing X or Q deforms f(r) and pushes the horizon / sonic
      radius outward  [paper Eqs. 13, 16, 43-47].
  (2) Increasing X or Q raises and widens the scalar effective
      potential barrier V_eff(r)  [paper Eqs. 108, Figs. 13-14].
  (3) A taller/wider barrier suppresses wave transmission, i.e. the
      greybody factor  [paper Fig. 15] — reproduced here via the
      standard 1st-order WKB transmission formula (Schutz & Will 1985;
      Iyer & Will 1987), which is an independent, well-established
      approximation and does NOT rely on the paper's own hypergeometric
      near/far-horizon matching (Secs. 10-11). Using a different,
      standard method to recover the same qualitative trend is what
      makes the claim "more solid" rather than just re-plotting the
      paper's own numbers.

All functions are self-contained and documented with the paper
equation they implement or approximate.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar


# ----------------------------------------------------------------------
# Metric function and horizon  [paper Eqs. 13, 14, 16]
# ----------------------------------------------------------------------
def f_and_fprime(r, X, Q, M=1.0):
    """f(r) = A(r)/(1+X),  A(r) = 1 - 2M/r + 2(1+X)Q^2/((2+X) r^2)."""
    A = 1 - 2 * M / r + 2 * (1 + X) * Q**2 / ((2 + X) * r**2)
    f = A / (1 + X)
    dA = 2 * M / r**2 - 4 * (1 + X) * Q**2 / ((2 + X) * r**3)
    fp = dA / (1 + X)
    return f, fp


def horizon_radius(X, Q, M=1.0):
    """Outer horizon r_h from A(r)=0  [Eq. 16, outer/r2 branch]."""
    a, b, c = 1.0, -2 * M, 2 * (1 + X) * Q**2 / (2 + X)
    disc = b**2 - 4 * a * c
    return (-b + np.sqrt(disc)) / (2 * a)


def is_subextremal(X, Q, M=1.0):
    a, b, c = 1.0, -2 * M, 2 * (1 + X) * Q**2 / (2 + X)
    return (b**2 - 4 * a * c) >= 0


# ----------------------------------------------------------------------
# Sonic (critical) radius for URF, omega=1/2  [paper Eqs. 43-47]
# ----------------------------------------------------------------------
def urf_sonic_radius(X, Q, M=1.0):
    """Solve r_c f'(r_c) - 4 f(r_c) = 0 (the omega=1/2 sonic condition),
    self-consistently re-derived from Eqs. (43)-(44) rather than the
    closed form quoted as Eq. (47) in the source paper (see README /
    proposal caveats for the discrepancy noted during verification).
    """
    def condition(r):
        f, fp = f_and_fprime(r, X, Q, M)
        return r * fp - 4 * f

    r_h = horizon_radius(X, Q, M)
    return brentq(condition, r_h * 1.02, r_h * 30)


# ----------------------------------------------------------------------
# Tortoise coordinate  [paper Eq. 105]
# ----------------------------------------------------------------------
def tortoise_of_r(r_array, X, Q, M=1.0):
    """Cumulative x_*(r) = int dr/f(r), integrated from the outermost
    point inward. r_array must be strictly decreasing.
    """
    f, _ = f_and_fprime(r_array, X, Q, M)
    dr = np.diff(r_array)
    seg = 0.5 * (1 / f[1:] + 1 / f[:-1]) * dr
    xstar = np.concatenate([[0.0], np.cumsum(seg)])
    return -xstar  # increasing "clock" as r decreases toward horizon


# ----------------------------------------------------------------------
# Klein-Gordon effective potential  [paper Eq. 108, zeroth-order l(l+1)]
# ----------------------------------------------------------------------
def V_eff(r, X, Q, M=1.0, l=2):
    """V_eff(r) = f(r) f'(r)/r + l(l+1) f(r)/r^2.

    Uses lambda_l^m ~ l(l+1) (zeroth-order term of the paper's Eq. 103
    series in aw; the (aw)^2 correction is a small higher-order
    refinement and is dropped here for a clean, standard scalar-QNM-
    style effective potential).
    """
    f, fp = f_and_fprime(r, X, Q, M)
    lam = l * (l + 1)
    return f * fp / r + lam * f / r**2


def V_eff_profile(X, Q, M=1.0, l=2, r_over_rh_max=10.0, n=4000):
    """Return r/r_h and V_eff(r) sampled outward from just above r_h."""
    r_h = horizon_radius(X, Q, M)
    r = np.linspace(r_h * 1.001, r_h * r_over_rh_max, n)
    V = V_eff(r, X, Q, M, l)
    return r / r_h, V, r_h


# ----------------------------------------------------------------------
# 1st-order WKB transmission / greybody-factor proxy
# (Schutz & Will 1985; Iyer & Will 1987 formula)
# ----------------------------------------------------------------------
def _potential_peak_in_tortoise(X, Q, M=1.0, l=2, n=6000, r_over_rh_max=12.0):
    """Locate the peak of V_eff as a function of the tortoise coordinate
    x_*, and estimate V_eff''(x_*) at the peak by finite differences.
    Returns (x_star_of_r interpolant is not needed further), V0, V0pp.
    """
    r_h = horizon_radius(X, Q, M)
    r = np.linspace(r_over_rh_max * r_h, r_h * 1.0005, n)  # decreasing r
    xstar = tortoise_of_r(r, X, Q, M)
    V = V_eff(r, X, Q, M, l)

    # xstar is increasing here (see tortoise_of_r); resample V uniformly in xstar
    order = np.argsort(xstar)
    xs, Vs = xstar[order], V[order]
    xs_u = np.linspace(xs[0], xs[-1], n)
    V_u = np.interp(xs_u, xs, Vs)

    i0 = np.argmax(V_u)
    V0 = V_u[i0]
    dx = xs_u[1] - xs_u[0]
    # central second derivative (guard the array edges)
    i0 = min(max(i0, 2), n - 3)
    V0pp = (V_u[i0 + 1] - 2 * V_u[i0] + V_u[i0 - 1]) / dx**2
    return V0, V0pp


def wkb_transmission(w_array, X, Q, M=1.0, l=2):
    V0, V0pp = _potential_peak_in_tortoise(X, Q, M, l)
    if V0pp >= 0:
        return np.full_like(w_array, np.nan), V0, V0pp
    w_eff = w_array / ((1.0 + X) ** 0.75)
    K = (V0 - w_eff**2) / np.sqrt(-2 * V0pp)
    T2 = 1.0 / (1.0 + np.exp(np.clip(2 * np.pi * K, -50, 50)))
    return T2, V0, V0pp
