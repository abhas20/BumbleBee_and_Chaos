"""
make_physics_figures.py
=========================
Independent visual verification of the bumblebee-gravity paper's core
claims, using bumblebee_potentials.py. Produces:

  figG_metric_function.png   - f(r) deformation & horizon shift
  figH_Veff_vs_Q.png         - effective potential, varying Q (fixed X)
  figI_Veff_vs_X.png         - effective potential, varying X (fixed Q)
  figJ_sonic_radius.png      - r_c and r_c/r_h vs X and vs Q
  figK_wkb_transmission.png  - independent WKB greybody-factor proxy

Run: python3 make_physics_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bumblebee_potentials import (
    f_and_fprime, horizon_radius, urf_sonic_radius, is_subextremal,
    V_eff_profile, wkb_transmission,
)

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

M = 1.0
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
})
PALETTE = ["#1F3864", "#B23A48", "#2E7D32", "#B58900", "#6A3D9A"]


# ==========================================================================
# Fig G: metric function f(r) and horizon shift
# ==========================================================================
def fig_metric_function():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # left: vary X at fixed Q
    Q = 0.5
    for c, X in zip(PALETTE, [0.1, 0.5, 1.0, 1.5, 1.9]):
        r_h = horizon_radius(X, Q, M)
        r = np.linspace(r_h * 0.9, 6, 400)
        f, _ = f_and_fprime(r, X, Q, M)
        axes[0].plot(r, f, color=c, label=f"X={X}")
        axes[0].axvline(r_h, color=c, ls=":", lw=1, alpha=0.6)
    axes[0].axhline(0, color="grey", lw=0.7)
    axes[0].set_xlabel("r / M"); axes[0].set_ylabel("f(r)")
    axes[0].set_title(f"Metric deformation vs. X  (Q={Q})")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_ylim(-0.3, 1.05)

    # right: vary Q at fixed X
    X = 0.5
    for c, Q in zip(PALETTE, [0.1, 0.3, 0.5, 0.7, 0.85]):
        r_h = horizon_radius(X, Q, M)
        r = np.linspace(r_h * 0.9, 6, 400)
        f, _ = f_and_fprime(r, X, Q, M)
        axes[1].plot(r, f, color=c, label=f"Q={Q}")
        axes[1].axvline(r_h, color=c, ls=":", lw=1, alpha=0.6)
    axes[1].axhline(0, color="grey", lw=0.7)
    axes[1].set_xlabel("r / M"); axes[1].set_ylabel("f(r)")
    axes[1].set_title(f"Metric deformation vs. Q  (X={X})")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_ylim(-0.3, 1.05)

    fig.suptitle("Bumblebee metric function f(r)  (dotted lines = horizon r_h)",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figG_metric_function.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[figG] done")


# ==========================================================================
# Fig H / I: effective potential  (paper Fig. 13 / 14 analogues)
# ==========================================================================
def fig_Veff():
    # vary Q at fixed X  (paper Fig. 13 style)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    X = 1.0
    for c, Q in zip(PALETTE, [0.1, 0.3, 0.5, 0.7, 0.85]):
        rr, V, rh = V_eff_profile(X, Q, M, l=2, r_over_rh_max=8, n=2000)
        ax.plot(rr, V, color=c, label=f"Q={Q}")
    ax.set_xlabel(r"$r/r_h$"); ax.set_ylabel(r"$V_{eff}$")
    ax.set_title(f"Effective potential vs. Q  (X={X}, l=2)\n"
                 "independent re-derivation of paper Fig. 13")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figH_Veff_vs_Q.png"), dpi=170)
    plt.close(fig)

    # vary X at fixed Q  (paper Fig. 14 style)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    Q = 0.5
    for c, X in zip(PALETTE, [0.1, 0.5, 1.0, 1.5, 1.9]):
        rr, V, rh = V_eff_profile(X, Q, M, l=2, r_over_rh_max=8, n=2000)
        ax.plot(rr, V, color=c, label=f"X={X}")
    ax.set_xlabel(r"$r/r_h$"); ax.set_ylabel(r"$V_{eff}$")
    ax.set_title(f"Effective potential vs. X  (Q={Q}, l=2)\n"
                 "independent re-derivation of paper Fig. 14")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figI_Veff_vs_X.png"), dpi=170)
    plt.close(fig)
    print("[figH/I] done")


# ==========================================================================
# Fig J: sonic radius vs X and vs Q (absolute and relative to horizon)
# ==========================================================================
def fig_sonic_radius():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))

    X_values = np.linspace(0.1, 1.9, 10)
    Q_values_for_X = [0.2, 0.4, 0.6]
    for c, Q in zip(PALETTE, Q_values_for_X):
        rc_list, ratio_list, Xs = [], [], []
        for X in X_values:
            if not is_subextremal(X, Q, M):
                continue
            rh = horizon_radius(X, Q, M)
            rc = urf_sonic_radius(X, Q, M)
            rc_list.append(rc); ratio_list.append(rc / rh); Xs.append(X)
        axes[0, 0].plot(Xs, rc_list, "o-", color=c, ms=4, label=f"Q={Q}")
        axes[0, 1].plot(Xs, ratio_list, "o-", color=c, ms=4, label=f"Q={Q}")

    axes[0, 0].set_xlabel("X"); axes[0, 0].set_ylabel(r"$r_c$  (absolute)")
    axes[0, 0].set_title("Sonic radius vs. X"); axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].set_xlabel("X"); axes[0, 1].set_ylabel(r"$r_c/r_h$  (relative to horizon)")
    axes[0, 1].set_title("Sonic radius / horizon radius vs. X"); axes[0, 1].legend(frameon=False, fontsize=9)

    Q_values = np.linspace(0.1, 0.85, 10)
    X_values_for_Q = [0.2, 0.6, 1.2]
    for c, X in zip(PALETTE, X_values_for_Q):
        rc_list, ratio_list, Qs = [], [], []
        for Q in Q_values:
            if not is_subextremal(X, Q, M):
                continue
            rh = horizon_radius(X, Q, M)
            rc = urf_sonic_radius(X, Q, M)
            rc_list.append(rc); ratio_list.append(rc / rh); Qs.append(Q)
        axes[1, 0].plot(Qs, rc_list, "o-", color=c, ms=4, label=f"X={X}")
        axes[1, 1].plot(Qs, ratio_list, "o-", color=c, ms=4, label=f"X={X}")

    axes[1, 0].set_xlabel("Q"); axes[1, 0].set_ylabel(r"$r_c$  (absolute)")
    axes[1, 0].set_title("Sonic radius vs. Q"); axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 1].set_xlabel("Q"); axes[1, 1].set_ylabel(r"$r_c/r_h$  (relative to horizon)")
    axes[1, 1].set_title("Sonic radius / horizon radius vs. Q"); axes[1, 1].legend(frameon=False, fontsize=9)

    fig.suptitle("Independent check: does the sonic radius really shift 'outward'?\n"
                 "(absolute $r_c$ can fall even while $r_c/r_h$ rises, because the "
                 "horizon itself shrinks faster)", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figJ_sonic_radius.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[figJ] done")


# ==========================================================================
# Fig K: independent WKB greybody-factor proxy (paper Fig. 15 analogue)
# ==========================================================================
def fig_wkb_transmission():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    wrh = np.linspace(0.01, 1.8, 200)

    # left: vary X at fixed Q
    Q = 0.3
    for c, X in zip(PALETTE, [0.1, 0.5, 1.0, 1.5, 1.9]):
        r_h = horizon_radius(X, Q, M)
        w = wrh / r_h
        T2, V0, V0pp = wkb_transmission(w, X, Q, M, l=2)
        axes[0].plot(wrh, T2, color=c, label=f"X={X}")
    axes[0].set_xlabel(r"$\omega r_h$"); axes[0].set_ylabel(r"$|T|^2$  (WKB greybody proxy)")
    axes[0].set_title(f"Transmission vs. X  (Q={Q}, l=2)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_ylim(-0.02, 1.02)

    # right: vary Q at fixed X
    X = 0.5
    for c, Q in zip(PALETTE, [0.1, 0.3, 0.5, 0.7, 0.85]):
        r_h = horizon_radius(X, Q, M)
        w = wrh / r_h
        T2, V0, V0pp = wkb_transmission(w, X, Q, M, l=2)
        axes[1].plot(wrh, T2, color=c, label=f"Q={Q}")
    axes[1].set_xlabel(r"$\omega r_h$"); axes[1].set_ylabel(r"$|T|^2$  (WKB greybody proxy)")
    axes[1].set_title(f"Transmission vs. Q  (X={X}, l=2)")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_ylim(-0.02, 1.02)

    fig.suptitle("Independent 1st-order WKB transmission (Schutz-Will formula)\n"
                 "cross-check of paper Fig. 15's suppression trend, via a different method",
                 y=1.06, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figK_wkb_transmission.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("[figK] done")


if __name__ == "__main__":
    fig_metric_function()
    fig_Veff()
    fig_sonic_radius()
    fig_wkb_transmission()
    print("All physics-verification figures written to", OUT)
