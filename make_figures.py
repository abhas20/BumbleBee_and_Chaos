"""
make_figures.py
================
Generates the exploratory figures for the research proposal:
  Fig A - transonic velocity profile v(x_*)          (single case)
  Fig B - recurrence plot of the reconstructed attractor (single case)
  Fig C - R vs X, one curve per fixed Q                (Zarghami Fig.8 analog)
  Fig D - divergence rate lambda vs X, one curve per Q (Zarghami Fig.7 analog)
  Fig E - R(X,Q) contour map                           (Zarghami Fig.9b analog)
  Fig F - lambda(X,Q) contour map

Run: python3 make_figures.py
Outputs PNGs into ./figs/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from bumblebee_rqa import (
    analyze_case, is_subextremal, transonic_velocity_profile, to_tortoise,
    takens_embed, recurrence_matrix,
)

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

# ---- shared style -------------------------------------------------------
NAVY = "#1F3864"
SLATE = "#44546A"
GOLD = "#B58900"
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

M = 1.0
NOISE = 0.03   # motivated by Gap 1: small noise floor stands in for
               # horizon-scale quantum/thermal fluctuations (see docstring
               # in bumblebee_rqa.analyze_case)

# ==========================================================================
# Fig A + B : single-case demo  (X=0.5, Q=0.5)
# ==========================================================================
def fig_single_case(X=0.5, Q=0.5, seed=1):
    res = analyze_case(X, Q, M, n_traj=1600, noise_level=NOISE, seed=seed)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(res["xs"], res["v_uniform"], color=NAVY, lw=0.8)
    ax.axvline(0, color=GOLD, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel(r"tortoise coordinate $x_*$  (embedding clock)")
    ax.set_ylabel(r"radial three-velocity $v(x_*)$")
    ax.set_title(f"Transonic accretion trajectory  (X={X}, Q={Q}, M={M})\n"
                 f"$r_c$={res['r_c']:.3f}, $v_c$={res['v_c']:.3f}, "
                 f"$r_h$={res['r_h']:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figA_velocity_profile.png"), dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.imshow(res["Rmat"], cmap="Greys", origin="lower", interpolation="none")
    ax.set_xlabel("point index i")
    ax.set_ylabel("point index j")
    ax.set_title(f"Recurrence plot of reconstructed (r,v) attractor\n"
                 f"X={X}, Q={Q}  |  R={res['R']:.4f}   $\\lambda$={res['lam']:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figB_recurrence_plot.png"), dpi=170)
    plt.close(fig)
    print(f"[single case] X={X} Q={Q}  R={res['R']:.4f}  lambda={res['lam']:.4f}"
          f"  eps={res['eps']:.4f}")


# ==========================================================================
# Fig C + D : line sweeps  R(X) and lambda(X), curves for fixed Q
# ==========================================================================
def fig_line_sweeps(Q_values=(0.2, 0.4, 0.6), X_values=None, seed=2):
    if X_values is None:
        X_values = np.linspace(0.1, 1.9, 10)

    colors = [NAVY, "#B23A48", "#2E7D32"]
    figR, axR = plt.subplots(figsize=(6.8, 4.4))
    figL, axL = plt.subplots(figsize=(6.8, 4.4))

    for c, Q in zip(colors, Q_values):
        Rs, Ls, Xs_valid = [], [], []
        for X in X_values:
            if not is_subextremal(X, Q, M):
                continue
            res = analyze_case(X, Q, M, n_traj=700, noise_level=NOISE, seed=seed)
            Rs.append(res["R"]); Ls.append(res["lam"]); Xs_valid.append(X)
        axR.plot(Xs_valid, Rs, "o-", color=c, label=f"Q={Q}", ms=4)
        axL.plot(Xs_valid, Ls, "o-", color=c, label=f"Q={Q}", ms=4)

    axR.set_xlabel("Lorentz-violating parameter X")
    axR.set_ylabel("reversibility  R")
    axR.set_title("Reversibility vs. X, by charge Q\n(analogue of Zarghami Fig. 8)")
    axR.legend(frameon=False)
    figR.tight_layout()
    figR.savefig(os.path.join(OUT, "figC_R_vs_X.png"), dpi=170)
    plt.close(figR)

    axL.set_xlabel("Lorentz-violating parameter X")
    axL.set_ylabel(r"divergence rate  $\lambda$")
    axL.set_title("Divergence-rate proxy vs. X, by charge Q\n(analogue of Zarghami Fig. 7)")
    axL.legend(frameon=False)
    figL.tight_layout()
    figL.savefig(os.path.join(OUT, "figD_lambda_vs_X.png"), dpi=170)
    plt.close(figL)
    print("[line sweeps] done")


# ==========================================================================
# Fig E + F : contour maps over (X, Q)
# ==========================================================================
def fig_contours(X_values=None, Q_values=None, seed=3):
    if X_values is None:
        X_values = np.linspace(0.1, 1.9, 12)
    if Q_values is None:
        Q_values = np.linspace(0.1, 0.85, 12)

    Rgrid = np.full((len(Q_values), len(X_values)), np.nan)
    Lgrid = np.full((len(Q_values), len(X_values)), np.nan)

    for i, Q in enumerate(Q_values):
        for j, X in enumerate(X_values):
            if not is_subextremal(X, Q, M):
                continue
            res = analyze_case(X, Q, M, n_traj=500, noise_level=NOISE, seed=seed)
            Rgrid[i, j] = res["R"]
            Lgrid[i, j] = res["lam"]

    Xg, Qg = np.meshgrid(X_values, Q_values)

    fig, ax = plt.subplots(figsize=(7, 5.4))
    cf = ax.contourf(Xg, Qg, Rgrid, levels=14, cmap=cm.viridis)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("reversibility R")
    # mark the ridge (max R per column, i.e. per X)
    ridge_X, ridge_Q = [], []
    for j, X in enumerate(X_values):
        col = Rgrid[:, j]
        if np.all(np.isnan(col)):
            continue
        i_best = np.nanargmax(col)
        ridge_X.append(X); ridge_Q.append(Q_values[i_best])
    ax.plot(ridge_X, ridge_Q, "o-", color="white", ms=5, mec="black", mew=0.6,
             label="R maximum (per X)")
    ax.set_xlabel("Lorentz-violating parameter X")
    ax.set_ylabel("black hole charge Q")
    ax.set_title("Reversibility R(X,Q) \u2014 key deliverable figure\n"
                 "(analogue of Zarghami Fig. 9b)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figE_R_contour.png"), dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.4))
    cf = ax.contourf(Xg, Qg, Lgrid, levels=14, cmap=cm.magma)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"divergence rate $\lambda$")
    ridge_X2, ridge_Q2 = [], []
    for j, X in enumerate(X_values):
        col = Lgrid[:, j]
        if np.all(np.isnan(col)):
            continue
        i_best = np.nanargmin(col)
        ridge_X2.append(X); ridge_Q2.append(Q_values[i_best])
    ax.plot(ridge_X2, ridge_Q2, "o-", color="white", ms=5, mec="black", mew=0.6,
             label=r"$\lambda$ minimum (per X)")
    ax.set_xlabel("Lorentz-violating parameter X")
    ax.set_ylabel("black hole charge Q")
    ax.set_title(r"Divergence-rate proxy $\lambda$(X,Q)"
                 "\n(analogue of Zarghami Fig. 7 contour)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figF_lambda_contour.png"), dpi=170)
    plt.close(fig)

    print("[contours] done")
    print("R range:", np.nanmin(Rgrid), np.nanmax(Rgrid))
    print("lambda range:", np.nanmin(Lgrid), np.nanmax(Lgrid))
    return Xg, Qg, Rgrid, Lgrid


if __name__ == "__main__":
    fig_single_case(X=0.5, Q=0.5)
    fig_line_sweeps()
    fig_contours()
    print("All figures written to", OUT)
