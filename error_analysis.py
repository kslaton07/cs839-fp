"""
Numerical error analysis — local truncation error vs dt for each integrator.

For a fixed initial state, steps forward one dt with each integrator and
compares to a high-accuracy RK4 reference (REFERENCE_DT).

Produces a log-log plot: error vs dt with reference slope lines showing
theoretical convergence rates (O(dt¹), O(dt²), O(dt⁴)).

Results saved to: {RESULTS_DIR}/plots/integrator_error_vs_dt.png

Usage:
    python error_analysis.py    (run from project root)
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from integrators import rk4, rk2, feuler, seuler, ieuler, vverlet

# ─────────────────────────────────────────────────────────────────────────────
# ★ Configure this variable to switch environment
# ─────────────────────────────────────────────────────────────────────────────
ENV_NAME    = "acrobot"  # "acrobot"  or  "cartpole"

# Output directory — adjust to match where you want plots saved
RESULTS_DIR = f"results_{ENV_NAME}"


# ─────────────────────────────────────────────────────────────────────────────
# Integrators
# ─────────────────────────────────────────────────────────────────────────────
INTEGRATORS = {
    "rk4":     rk4,
    "rk2":     rk2,
    "feuler":  feuler,
    "seuler":  seuler,
    "ieuler":  ieuler,
    "vverlet": vverlet,
}

INTEGRATOR_COLORS = {
    "rk4":     "#5b3f8c",
    "rk2":     "#7d5ba6",
    "feuler":  "#9b72cf",
    "seuler":  "#b388d8",
    "ieuler":  "#6d28d9",
    "vverlet": "#c084fc",
}

# dt range to sweep (log-spaced, 30 points from 0.01 to 2.0)
DT_VALUES     = np.logspace(-2, np.log10(2), 30)
REFERENCE_DT  = 1e-5      # RK4 sub-step size for ground truth


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics functions — standalone, no env instantiation needed
# ─────────────────────────────────────────────────────────────────────────────
def _dsdt_acrobot(s_aug):
    """Acrobot equations of motion. s_aug = [theta1, theta2, dtheta1, dtheta2, torque]"""
    from numpy import cos, sin, pi
    m1, m2   = 1.0, 1.0
    l1       = 1.0
    lc1, lc2 = 0.5, 0.5
    I1, I2   = 1.0, 1.0
    g        = 9.8

    a        = s_aug[-1]
    th1, th2 = s_aug[0], s_aug[1]
    dth1, dth2 = s_aug[2], s_aug[3]

    d1   = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*cos(th2)) + I1 + I2
    d2   = m2*(lc2**2 + l1*lc2*cos(th2)) + I2
    phi2 = m2*lc2*g*cos(th1 + th2 - pi/2.0)
    phi1 = (-m2*l1*lc2*dth2**2*sin(th2)
            - 2*m2*l1*lc2*dth2*dth1*sin(th2)
            + (m1*lc1 + m2*l1)*g*cos(th1 - pi/2)
            + phi2)
    ddth2 = (a + d2/d1*phi1 - m2*l1*lc2*dth1**2*sin(th2) - phi2) / \
            (m2*lc2**2 + I2 - d2**2/d1)
    ddth1 = -(d2*ddth2 + phi1) / d1

    return dth1, dth2, ddth1, ddth2, 0.0


def _dsdt_cartpole(s_aug):
    """CartPole equations of motion. s_aug = [x, theta, x_dot, theta_dot, force]"""
    gravity         = 9.8
    masscart        = 1.0
    masspole        = 0.1
    total_mass      = masscart + masspole
    length          = 0.5
    polemass_length = masspole * length

    force     = s_aug[-1]
    x         = s_aug[0];   theta     = s_aug[1]
    x_dot     = s_aug[2];   theta_dot = s_aug[3]

    costh  = math.cos(theta);  sinth = math.sin(theta)
    temp   = (force + polemass_length * theta_dot**2 * sinth) / total_mass
    th_dd  = (gravity * sinth - costh * temp) / (
              length * (4.0/3.0 - masspole * costh**2 / total_mass))
    x_dd   = temp - polemass_length * th_dd * costh / total_mass

    return x_dot, theta_dot, x_dd, th_dd, 0.0


def _get_dsdt(env_name):
    return _dsdt_acrobot if env_name == "acrobot" else _dsdt_cartpole


def _default_state(env_name):
    """A representative initial state with a small perturbation and nonzero control."""
    if env_name == "acrobot":
        # [theta1, theta2, dtheta1, dtheta2, torque]
        return np.array([0.1, 0.1, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        # [x, theta, x_dot, theta_dot, force]
        return np.array([0.05, 0.05, 0.0, 0.0, 10.0], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth: RK4 with many tiny sub-steps
# ─────────────────────────────────────────────────────────────────────────────
def _reference(dsdt, y0, dt_target):
    n_sub    = max(1, int(dt_target / REFERENCE_DT))
    sub_dt   = dt_target / n_sub
    y        = y0.copy()
    torque   = y0[-1]
    for _ in range(n_sub):
        result = rk4(dsdt, y, [0, sub_dt])   # returns first 4 elements
        y      = np.append(result, torque)
    return y[:4]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    plots_dir = f"{RESULTS_DIR}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    dsdt = _get_dsdt(ENV_NAME)
    y0   = _default_state(ENV_NAME)

    print(f"Error analysis: env={ENV_NAME}")
    print(f"  Output: {plots_dir}/integrator_error_vs_dt.png\n")

    fig, ax = plt.subplots(figsize=(10, 6))

    all_errors = {}
    for integ_name, integrator_fn in INTEGRATORS.items():
        errors = []
        for dt in DT_VALUES:
            ref = _reference(dsdt, y0, dt)
            try:
                pred  = integrator_fn(dsdt, y0, [0, dt])
                err   = float(np.linalg.norm(pred[:4] - ref))
            except Exception:
                err   = np.nan
            errors.append(err)
        errors = np.array(errors)
        all_errors[integ_name] = errors
        valid  = np.isfinite(errors) & (errors > 0)
        ax.loglog(DT_VALUES[valid], errors[valid],
                  marker="o", markersize=4, linewidth=2,
                  color=INTEGRATOR_COLORS[integ_name], label=integ_name)

    # ── Reference slope lines ─────────────────────────────────────────────────
    anchor_dt  = 0.1
    anchor_idx = np.argmin(np.abs(DT_VALUES - anchor_dt))

    def slope_line(order, ref_integ, scale, label):
        anchor_err = all_errors[ref_integ][anchor_idx]
        if np.isnan(anchor_err) or anchor_err <= 0:
            return
        y = anchor_err * scale * (DT_VALUES / anchor_dt) ** order
        ax.loglog(DT_VALUES, y, linestyle="--", linewidth=1, color="gray", alpha=0.5)
        ax.text(DT_VALUES[-1] * 1.05, y[-1], label,
                fontsize=9, color="gray", va="center")

    slope_line(1, "feuler",  3.0,  "O(dt¹)")
    slope_line(2, "rk2",     3.0,  "O(dt²)")
    slope_line(4, "rk4",     10.0, "O(dt⁴)")

    ax.set_xlabel("dt (log scale)", fontsize=13)
    ax.set_ylabel("Local truncation error ||pred − ref||₂  (log scale)", fontsize=13)
    ax.set_title(f"Integrator accuracy vs timestep — {ENV_NAME.capitalize()} dynamics\n"
                 "Dashed lines show theoretical convergence rates", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()

    fname = f"{plots_dir}/integrator_error_vs_dt.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Plot saved: {fname}")


if __name__ == "__main__":
    main()