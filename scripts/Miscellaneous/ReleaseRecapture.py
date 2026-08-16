import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


class ReleaseRecapture:
    def __init__(
        self,
        T_vec: np.ndarray,
        delta_t_vec: np.ndarray,
        mass: float,
        omega_r: float,
        omega_axial: float,
        w_0: float,
        z_R: float,
        U_0: float,
        n: int,
        plots: bool,
    ):
        self.T_vec = T_vec
        self.delta_t_vec = delta_t_vec
        self.mass = mass
        self.omega_r = omega_r
        self.omega_axial = omega_axial
        self.w_0 = w_0
        self.z_R = z_R
        self.U_0 = U_0
        self.n = n
        self.plots = plots

    def initial_conditions(
        self, mean_velocity: float, Temperature: float, threshold_energy: float
    ) -> tuple:
        x_r = np.random.normal(
            0,
            np.sqrt(scipy.constants.k * Temperature / (self.mass * self.omega_r**2)),
            self.n,
        )
        x_axial = np.random.normal(
            0,
            np.sqrt(
                scipy.constants.k * Temperature / (self.mass * self.omega_axial**2)
            ),
            self.n,
        )
        theta_x = np.random.uniform(0, 2 * np.pi, self.n)

        x_0 = x_r * np.cos(theta_x)
        y_0 = x_r * np.sin(theta_x)
        z_0 = x_axial

        v_0 = np.random.normal(
            mean_velocity,
            np.sqrt(3 * scipy.constants.k * Temperature / self.mass),
            self.n,
        )
        theta = np.random.uniform(0, np.pi, self.n)
        phi = np.random.uniform(0, 2 * np.pi, self.n)

        v_x = v_0 * np.cos(phi) * np.sin(theta)
        v_y = v_0 * np.sin(phi) * np.sin(theta)
        v_z = v_0 * np.cos(theta)

        kin_en = 0.5 * self.mass * (v_x**2 + v_y**2 + v_z**2)
        pot_en = -threshold_energy * (
            1 - 2 * (x_0**2 + y_0**2) / self.w_0**2 - (z_0 / self.z_R) ** 2
        )

        E_in = kin_en + pot_en

        return x_0, y_0, z_0, v_x, v_y, v_z, v_0, E_in

    def time_evolution(
        self,
        x_0: float | np.ndarray,
        y_0: float | np.ndarray,
        z_0: float | np.ndarray,
        v_x_0: float | np.ndarray,
        v_y_0: float | np.ndarray,
        v_z_0: float | np.ndarray,
        delta_t: float,
    ) -> tuple:
        g = 9.81
        x_fin = x_0 + v_x_0 * delta_t
        y_fin = y_0 + v_y_0 * delta_t - 0.5 * g * delta_t**2
        z_fin = z_0 + v_z_0 * delta_t

        v_x_fin = v_x_0
        v_y_fin = v_y_0 - g * delta_t
        v_z_fin = v_z_0
        return x_fin, y_fin, z_fin, v_x_fin, v_y_fin, v_z_fin

    def recapture_probability(
        self,
        x_fin: float | np.ndarray,
        y_fin: float | np.ndarray,
        z_fin: float | np.ndarray,
        v_x_fin: float | np.ndarray,
        v_y_fin: float | np.ndarray,
        v_z_fin: float | np.ndarray,
        threshold_energy: float = 0,
    ) -> np.ndarray:
        kin_en_fin = 0.5 * self.mass * (v_x_fin**2 + v_y_fin**2 + v_z_fin**2)
        pot_en_fin = -self.U_0 * (
            1 - 2 * (x_fin**2 + y_fin**2) / self.w_0**2 - (z_fin / self.z_R) ** 2
        )

        E_fin = kin_en_fin + pot_en_fin
        return np.sum(E_fin < threshold_energy) / self.n

    def run(self, threshold_energy: float = 0) -> np.ndarray:
        recapture_probabilities = np.zeros((len(self.T_vec), len(self.delta_t_vec)))
        for i, T in enumerate(self.T_vec):
            mean_velocity = np.sqrt(8 * scipy.constants.k * T / (np.pi * self.mass))
            for j, delta_t in enumerate(self.delta_t_vec):
                x_0, y_0, z_0, v_x_0, v_y_0, v_z_0, _, _ = self.initial_conditions(
                    mean_velocity, T, self.U_0
                )

                x_fin, y_fin, z_fin, v_x_fin, v_y_fin, v_z_fin = self.time_evolution(
                    x_0, y_0, z_0, v_x_0, v_y_0, v_z_0, delta_t
                )

                recapture_probabilities[i, j] = self.recapture_probability(
                    x_fin, y_fin, z_fin, v_x_fin, v_y_fin, v_z_fin, threshold_energy
                )
        loading = recapture_probabilities[:, 0:1]
        survival = recapture_probabilities / loading
        self.plot(survival)
        return survival

    def plot(self, survival: np.ndarray) -> None:
        if self.plots:
            _, ax = plt.subplots()
            colormap = plt.colormaps["plasma"]
            norm = Normalize(vmin=self.T_vec[0] * 1e6, vmax=self.T_vec[-1] * 1e6)
            for i, T in enumerate(self.T_vec):
                ax.plot(
                    self.delta_t_vec * 1e6,
                    survival[i],
                    color=colormap(norm(T * 1e6)),
                    label=f"{T * 1e6:.1f} µK",
                )
            sm = ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Temperature [µK]")
            ax.set_xlabel("Release time [µs]")
            ax.set_ylabel("Survival fraction")
            plt.tight_layout()
            plt.show()

    def chi_square(self, survival: np.ndarray, experimental_data: np.ndarray) -> float:
        exp_norm = experimental_data / experimental_data[0]
        return np.sum((survival - exp_norm) ** 2 / exp_norm)


if __name__ == "__main__":
    time_data_path = r"C:\\dev\\GitHub\\Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays\\scripts\\Miscellaneous\\time.npy"
    survival_data_path = r"C:\\dev\\GitHub\\Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays\\scripts\\Miscellaneous\\survival.npy"
    survival_error_data_path = r"C:\\dev\\GitHub\\Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays\\scripts\\Miscellaneous\\error_survival.npy"

    if time_data_path and survival_data_path and survival_error_data_path:
        survival = np.load(survival_data_path)
        time = np.load(time_data_path)
        error_survival = np.load(survival_error_data_path)

    P = 4e-3
    w_0 = 1.2e-6
    m = 2.2069393e-25
    Gamma_D2 = 2 * np.pi * 5.2227e6
    omega_D2 = 2 * np.pi * scipy.constants.c / 852.34727582e-9
    omega_trap = 2 * np.pi * scipy.constants.c / 933e-9
    delta_D2 = omega_trap - omega_D2

    U_0 = -3 * scipy.constants.c**2 * P * Gamma_D2 / (w_0**2 * omega_D2**3 * delta_D2)
    z_R = np.pi * w_0**2 / 933e-9
    omega_r = np.sqrt(4 * U_0 / (m * w_0**2))
    omega_ax = np.sqrt(2 * U_0 / (m * z_R**2))

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
        plots=True,
    )

    survival_matrix = sim.run()

    if time_data_path and survival_data_path and survival_error_data_path:
        delta_t_exp = time[:] * 1e-6
        survival_exp = survival[:]
        error_exp = error_survival[:]

        chi_sq = [
            sim.chi_square(survival_matrix[i], survival_exp) for i in range(len(T_vec))
        ]

        T_best = T_vec[np.argmin(chi_sq)]
        print(f"T = {T_best * 1e6:.2f} µK")
