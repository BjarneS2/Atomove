import sys
from pathlib import Path

import numpy as np
import scipy.constants
import matplotlib.pyplot as plt

SEED = 42

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "VisualizationForThesis")
)

from ReleaseRecapture import ReleaseRecapture
from style_thesis import savefig_thesis, style_axes

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "ResultsForThesis"

time_exp = np.load(HERE / "time.npy")
survival_exp = np.load(HERE / "survival.npy")
error_exp = np.load(HERE / "error_survival.npy")

w_0 = 1.2e-6
m = 2.2069393e-25
Gamma_D2 = 2 * np.pi * 5.2227e6
omega_D2 = 2 * np.pi * scipy.constants.c / 852.34727582e-9
omega_trap = 2 * np.pi * scipy.constants.c / 933e-9
delta_D2 = omega_trap - omega_D2

T_vec = np.linspace(10e-6, 50e-6, 300)
dt_vec = (
    np.array(
        [
            0.0,
            5.0,
            7.5,
            10.0,
            12.5,
            15.0,
            17.5,
            20.0,
            22.5,
            25.0,
            30.0,
            35.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
        ]
    )
    * 1e-6
)

powers_mW = [0.6, 0.9, 1.2, 1.5]
colors = ["crimson", "royalblue", "darkgreen", "darkorange"]
linestyles = ["-", "--", "-.", ":"]

fig, ax = plt.subplots(figsize=(7, 5))

ax.errorbar(
    time_exp,
    survival_exp / survival_exp[0],
    error_exp,
    marker="o",
    ms=5,
    ls="none",
    color="black",
    capsize=3,
    label="Experimental data",
    zorder=5,
)

def trap_params(P_mW):
    P = P_mW * 1e-3
    U_0 = -3 * scipy.constants.c**2 * P * Gamma_D2 / (w_0**2 * omega_D2**3 * delta_D2)
    z_R = np.pi * w_0**2 / 933e-9
    omega_r = np.sqrt(4 * U_0 / (m * w_0**2))
    omega_ax = np.sqrt(2 * U_0 / (m * z_R**2))
    return U_0, z_R, omega_r, omega_ax


def run_chi_sq(P_mW, n, seed=SEED):
    U_0, z_R, omega_r, omega_ax = trap_params(P_mW)
    np.random.seed(seed)
    sim = ReleaseRecapture(
        T_vec=T_vec,
        delta_t_vec=dt_vec,
        mass=m,
        omega_r=omega_r,
        omega_axial=omega_ax,
        w_0=w_0,
        z_R=z_R,
        U_0=U_0,
        n=n,
        plots=False,
    )
    survival_matrix = sim.run()
    chi_sq = np.array(
        [sim.chi_square(survival_matrix[i], survival_exp) for i in range(len(T_vec))]
    )
    return survival_matrix, chi_sq


chi_sq_by_power = {}

for P_mW, color, ls in zip(powers_mW, colors, linestyles):
    U_0, z_R, omega_r, omega_ax = trap_params(P_mW)
    survival_matrix, chi_sq = run_chi_sq(P_mW, n=50000)
    chi_sq_by_power[P_mW] = chi_sq
    best_idx = np.argmin(chi_sq)
    T_best = T_vec[best_idx]

    ax.plot(
        dt_vec * 1e6,
        survival_matrix[best_idx],
        color=color,
        ls=ls,
        lw=2,
        label=f"P = {P_mW:.1f} mW, T = {T_best * 1e6:.1f} µK",
    )

ax.set_xlabel("Release time [µs]")
ax.set_ylabel("Survival fraction")
style_axes(ax)
ax.legend(loc="best", fontsize=9, framealpha=0.9)
plt.tight_layout()

OUTPUT_DIR.mkdir(exist_ok=True)
savefig_thesis(fig, OUTPUT_DIR / "release_recapture_best_fits_thesis.png")

P_chi2 = 0.9
T_lo, T_hi = 25e-6, 47e-6
region_mask = (T_vec >= T_lo) & (T_vec <= T_hi)

chi_sq_50k = chi_sq_by_power[P_chi2]
_, chi_sq_70k = run_chi_sq(P_chi2, n=70000)

runs = [
    ("n = 50 000", chi_sq_50k, "royalblue"),
    ("n = 70 000", chi_sq_70k, "crimson"),
]

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(11, 5))

for label, chi_sq, color in runs:
    best_idx = np.argmin(chi_sq)
    T_best = T_vec[best_idx]
    ax2.plot(T_vec * 1e6, chi_sq, color=color, lw=1.5, label=f"{label}, T = {T_best * 1e6:.1f} µK")
    ax2.axvline(T_best * 1e6, color=color, ls=":", lw=1)
    ax3.plot(T_vec[region_mask] * 1e6, chi_sq[region_mask], color=color, lw=1.5)
    ax3.axvline(T_best * 1e6, color=color, ls=":", lw=1)

ax2.axvspan(T_lo * 1e6, T_hi * 1e6, color="gray", alpha=0.15)
ax2.set_xlabel("Temperature [µK]")
ax2.set_ylabel(r"$\chi^2$")
style_axes(ax2)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

ax3.set_xlim(T_lo * 1e6, T_hi * 1e6)
ax3.set_xlabel("Temperature [µK]")
ax3.set_ylabel(r"$\chi^2$")
style_axes(ax3)

plt.tight_layout()
savefig_thesis(fig2, OUTPUT_DIR / "release_recapture_chi2_0p9mW_thesis.png")

plt.show()
