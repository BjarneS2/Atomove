import sys

sys.path.insert(
    0,
    r"c:\dev\GitHub\Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays\scripts\Miscellaneous",
)

import numpy as np
import scipy.constants
from ReleaseRecapture import ReleaseRecapture

time_data_path = r"c:\dev\GitHub\Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays\scripts\Miscellaneous\time.npy"
survival_data_path = r"c:\dev\GitHub\Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays\scripts\Miscellaneous\survival.npy"
survival_error_data_path = r"c:\dev\GitHub\Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays\scripts\Miscellaneous\error_survival.npy"

survival_exp = np.load(survival_data_path)
time_exp = np.load(time_data_path)
error_exp = np.load(survival_error_data_path)

w_0 = 1.2e-6
m = 2.2069393e-25
Gamma_D2 = 2 * np.pi * 5.2227e6
omega_D2 = 2 * np.pi * scipy.constants.c / 852.34727582e-9
omega_trap = 2 * np.pi * scipy.constants.c / 933e-9
delta_D2 = omega_trap - omega_D2

T_vec = np.linspace(0.1e-6, 150e-6, 1000)
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

powers_mW = np.round(np.arange(0.6, 2.21, 0.1), 1)

results = []
for P_mW in powers_mW:
    P = P_mW * 1e-3
    U_0 = -3 * scipy.constants.c**2 * P * Gamma_D2 / (w_0**2 * omega_D2**3 * delta_D2)
    z_R = np.pi * w_0**2 / 933e-9
    omega_r = np.sqrt(4 * U_0 / (m * w_0**2))
    omega_ax = np.sqrt(2 * U_0 / (m * z_R**2))

    sim = ReleaseRecapture(
        T_vec=T_vec,
        delta_t_vec=dt_vec,
        mass=m,
        omega_r=omega_r,
        omega_axial=omega_ax,
        w_0=w_0,
        z_R=z_R,
        U_0=U_0,
        n=10000,
        plots=False,
    )

    survival_matrix = sim.run()

    chi_sq = [
        sim.chi_square(survival_matrix[i], survival_exp) for i in range(len(T_vec))
    ]
    best_idx = np.argmin(chi_sq)
    T_best = T_vec[best_idx]
    chi_best = chi_sq[best_idx]

    trap_depth_uK = U_0 / scipy.constants.k * 1e6

    results.append(
        (
            P_mW,
            T_best * 1e6,
            chi_best,
            -trap_depth_uK,
            omega_r / (2 * np.pi) / 1e3,
            omega_ax / (2 * np.pi) / 1e3,
        )
    )
    print(
        f"P = {P_mW:5.1f} mW  ->  T_best = {T_best * 1e6:7.2f} uK   chi2_min = {chi_best:8.4f}   U0 = {-trap_depth_uK:8.1f} uK   f_r = {omega_r / (2 * np.pi) / 1e3:7.1f} kHz  f_ax = {omega_ax / (2 * np.pi) / 1e3:6.2f} kHz"
    )

print()
print("P [mW] | T_best [uK] | chi2_min | U0 [uK] | f_radial [kHz] | f_axial [kHz]")
for r in results:
    print(
        f"{r[0]:6.1f} | {r[1]:11.2f} | {r[2]:8.4f} | {r[3]:7.1f} | {r[4]:14.1f} | {r[5]:12.2f}"
    )
