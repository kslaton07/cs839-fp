"""
Numerical error analysis — local truncation error vs dt for each integrator.

For a fixed initial Acrobot state, steps forward one dt with each integrator
and compares to a high-accuracy RK4 reference (dt=1e-5).

Produces a log-log plot showing error vs dt — straight lines with slopes
matching integration order (1st, 2nd, 4th order).

Results saved to:
  - results_acrobot/plots/integrator_error_vs_dt.png

Usage:
    python error_analysis.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, sin

from acrobot import AcrobotEnv, rk4, rk2, feuler, seuler, ieuler, vverlet


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
INTEGRATORS = {
    "rk4":     rk4,
    "rk2":     rk2,
    "feuler":  feuler,
    "seuler":  seuler,
    "ieuler":  ieuler,
    "vverlet": vverlet,
}

# dt values to test — use finer resolution than training for smoother curves
DT_VALUES = np.logspace(-2, np.log10(2), 30)

# Reference: RK4 at tiny dt — this is our "ground truth"
REFERENCE_DT = 1e-5

# Fixed initial state: [theta1, theta2, dtheta1, dtheta2, torque]
# Starting near hanging position with a small perturbation and torque applied
INITIAL_STATE = np.array([0.1, 0.1, 0.0, 0.0, 1.0], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Acrobot dynamics — standalone so we don't need a full env instance
# ─────────────────────────────────────────────────────────────────────────────
def dsdt(s_augmented):
    """Acrobot equations of motion — same as AcrobotEnv._dsdt."""
    m1, m2 = 1.0, 1.0
    l1 = 1.0
    lc1, lc2 = 0.5, 0.5
    I1, I2 = 1.0, 1.0
    g = 9.8

    a       = s_augmented[-1]
    s       = s_augmented[:-1]
    theta1  = s[0]
    theta2  = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]

    d1  = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*cos(theta2)) + I1 + I2
    d2  = m2*(lc2**2 + l1*lc2*cos(theta2)) + I2
    phi2 = m2*lc2*g*cos(theta1 + theta2 - pi/2.0)
    phi1 = (-m2*l1*lc2*dtheta2**2*sin(theta2)
            - 2*m2*l1*lc2*dtheta2*dtheta1*sin(theta2)
            + (m1*lc1 + m2*l1)*g*cos(theta1 - pi/2)
            + phi2)

    ddtheta2 = (a + d2/d1*phi1 - m2*l1*lc2*dtheta1**2*sin(theta2) - phi2) / \
               (m2*lc2**2 + I2 - d2**2/d1)
    ddtheta1 = -(d2*ddtheta2 + phi1) / d1

    return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Compute reference solution using RK4 at very small dt
# ─────────────────────────────────────────────────────────────────────────────
def compute_reference(y0, dt_target):
    """
    Step from y0 to y0 + dt_target using many tiny RK4 steps.
    This is our ground truth.
    """
    n_substeps = max(1, int(dt_target / REFERENCE_DT))
    actual_sub_dt = dt_target / n_substeps

    y = y0.copy()
    for _ in range(n_substeps):
        result = rk4(dsdt, y, [0, actual_sub_dt])
        # rk4 returns first 4 elements, re-attach torque
        y = np.append(result, y0[-1])

    return y[:4]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results_acrobot/plots", exist_ok=True)

    y0 = INITIAL_STATE.copy()

    # Colors for each integrator
    colors = {
        "rk4":     "#1D9E75",
        "rk2":     "#7F77DD",
        "feuler":  "#D85A30",
        "seuler":  "#BA7517",
        "ieuler":  "#185FA5",
        "vverlet": "#993556",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for integ_name, integrator_fn in INTEGRATORS.items():
        errors = []

        for dt in DT_VALUES:
            # Ground truth: RK4 at tiny substeps
            ref = compute_reference(y0, dt)

            # Integrator prediction: one step of size dt
            try:
                pred = integrator_fn(dsdt, y0, [0, dt])
                error = float(np.linalg.norm(pred[:4] - ref))
            except Exception:
                error = np.nan

            errors.append(error)

        errors = np.array(errors)
        valid  = np.isfinite(errors) & (errors > 0)

        ax.loglog(DT_VALUES[valid], errors[valid],
                  marker="o", markersize=4, linewidth=2,
                  color=colors[integ_name], label=integ_name)

    # ── Reference slope lines ─────────────────────────────────────────────────
    dt_ref = DT_VALUES[valid]
    # Anchor all reference lines at dt=0.1 for visual clarity
    anchor_dt  = 0.1
    anchor_idx = np.argmin(np.abs(DT_VALUES - anchor_dt))

    # Get approximate error at anchor for each order to position lines
    # Use feuler as anchor for 1st order, rk2 for 2nd, rk4 for 4th
    def slope_line(order, anchor_error, label):
        y = anchor_error * (DT_VALUES / anchor_dt) ** order
        ax.loglog(DT_VALUES, y, linestyle="--", linewidth=1,
                  color="gray", alpha=0.5)
        # Label at right end
        ax.text(DT_VALUES[-1]*1.05, y[-1], label,
                fontsize=9, color="gray", va="center")

    # Compute anchor errors for reference lines
    ref_1st = float(np.linalg.norm(
        feuler(dsdt, y0, [0, anchor_dt])[:4] - compute_reference(y0, anchor_dt)
    ))
    ref_2nd = float(np.linalg.norm(
        rk2(dsdt, y0, [0, anchor_dt])[:4] - compute_reference(y0, anchor_dt)
    ))
    ref_4th = float(np.linalg.norm(
        rk4(dsdt, y0, [0, anchor_dt])[:4] - compute_reference(y0, anchor_dt)
    ))

    slope_line(1, ref_1st * 3,  "O(dt¹)")
    slope_line(2, ref_2nd * 3,  "O(dt²)")
    slope_line(4, ref_4th * 10, "O(dt⁴)")

    ax.set_xlabel("dt (log scale)", fontsize=13)
    ax.set_ylabel("Local truncation error (log scale)", fontsize=13)
    ax.set_title("Integrator accuracy vs timestep — Acrobot dynamics\n"
                 "Dashed lines show theoretical convergence rates",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()

    fname = "results_acrobot/plots/integrator_error_vs_dt.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Plot saved: {fname}")


if __name__ == "__main__":
    main()