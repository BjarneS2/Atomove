import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "Visualization"))

import h5py
import numpy as np
from constants3d import DEFAULTS, compute_scales
from plot_control_protocol_3d import plot_control_protocol_3d
from plot_energy_3d import plot_energy_distributions_3d
from plot_phase_space_3d import animate_phase_space_3d
from plot_survival_3d import plot_survival_3d
from plot_trajectories_3d import plot_trajectories_3d

from plot_transport_3d import create_transport_animation_3d
from utils3d import compute_loss_mask, total_energy

ENABLE_TRANSPORT_ANIMATION = False
ENABLE_SURVIVAL = True
ENABLE_PHASE_SPACE = False
ENABLE_ENERGY = True
ENABLE_TRAJECTORIES = True
ENABLE_CONTROL_PROTOCOL = True

FPS_TRANSPORT = 20
FPS_PHASE_SPACE = 15
DPI_TRANSPORT = 120
DPI_PHASE_SPACE = 180


def load_forward3d(file_path: Path) -> dict:
    with h5py.File(file_path, "r") as f:
        t = f["t"][:]
        ux = f["ux"][:]
        uy = f["uy"][:]
        ua = f["ua"][:]
        x = np.array(
            f["x"]
        ).T
        y = np.array(f["y"]).T
        z = np.array(f["z"]).T
        vx = np.array(f["vx"]).T
        vy = np.array(f["vy"]).T
        vz = np.array(f["vz"]).T

        atr = f.attrs

        def ga(k, default=None):
            return atr[k] if k in atr else default

        w = float(ga("w", 1.2))
        w_aux_factor = float(ga("w_aux_factor", 1.1 / 1.2))
        zR = float(ga("zR", 15.0))
        zR_aux = float(ga("zR_aux", 13.0))
        x_start = float(ga("x_start", 0.0))
        y_start = float(ga("y_start", 0.0))
        x_stop = float(ga("x_stop", 4.6))
        y_stop = float(ga("y_stop", 0.0))
        T_atom = float(ga("T_atom", 40e-6))
        T_tweezer = float(ga("T_tweezer", 287e-6))
        w0_um = float(ga("w0_um", 1.0))
        U0_static = float(ga("U0_static", 0.01))
        U0_aux_max = float(ga("U0_aux_max", 0.03))
        starting_trap_fraction = float(ga("starting_trap_fraction", 0.5))
        trap_fraction = float(ga("trap_fraction", 0.5))
        final_trap_fraction = float(ga("final_trap_fraction", 0.5))
        g_dimless = float(ga("g_dimless", 0.0))

    w_aux = w * w_aux_factor

    params = dict(
        w=w,
        w_aux=w_aux,
        w_aux_factor=w_aux_factor,
        zR=zR,
        zR_aux=zR_aux,
        x_start=x_start,
        y_start=y_start,
        x_stop=x_stop,
        y_stop=y_stop,
        U0_static=U0_static,
        U0_aux_max=U0_aux_max,
        starting_trap_fraction=starting_trap_fraction,
        trap_fraction=trap_fraction,
        final_trap_fraction=final_trap_fraction,
        g_dimless=g_dimless,
    )

    scales = compute_scales(T_tweezer, w0_um, DEFAULTS)
    scales["T_atom_dimless"] = T_atom / T_tweezer

    print("Computing loss mask from energy criterion …")
    is_lost = compute_loss_mask(
        x,
        y,
        z,
        vx,
        vy,
        vz,
        t,
        ux,
        uy,
        ua,
        params,
        trap_fraction,
        g_dimless,
    )

    print("Computing energies …")
    KE, PE, E_tot = total_energy(x, y, z, vx, vy, vz, ux, uy, ua, params, g_dimless)

    n_shots = x.shape[1]
    n_survived = int(np.sum(~is_lost[-1]))
    print(f"Survival: {n_survived}/{n_shots}  ({100 * n_survived / n_shots:.1f}%)")

    return dict(
        t=t,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        ux=ux,
        uy=uy,
        ua=ua,
        is_lost=is_lost,
        KE=KE,
        PE=PE,
        E_tot=E_tot,
        params=params,
        scales=scales,
        n_shots=n_shots,
    )


def _output_dir(file_path: Path) -> Path:
    out = _THIS_DIR.parent / "images" / file_path.stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    results_dir = _THIS_DIR.parent / "ResultsForThesis"

    if len(sys.argv) >= 2:
        file_path = Path(sys.argv[1])
    else:
        candidates = sorted(
            results_dir.glob("forward3d_*.h5"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("ERROR: no forward3d_*.h5 found in results/")
            sys.exit(1)
        file_path = candidates[0]
        print(f"Auto-detected: {file_path.name}")

    data = load_forward3d(file_path)
    output_dir = _output_dir(file_path)
    print(f"Output → {output_dir}")

    if ENABLE_SURVIVAL:
        print("\n── Survival plot ──")
        plot_survival_3d(data, output_dir)

    if ENABLE_ENERGY:
        print("\n── Energy distributions ──")
        plot_energy_distributions_3d(data, output_dir)

    if ENABLE_PHASE_SPACE:
        print("\n── Phase-space animation ──")
        animate_phase_space_3d(
            data, output_dir, fps=FPS_PHASE_SPACE, dpi=DPI_PHASE_SPACE
        )

    if ENABLE_TRAJECTORIES:
        print("\n── Trajectory plots ──")
        plot_trajectories_3d(data, output_dir)

    if ENABLE_CONTROL_PROTOCOL:
        print("\n── Control protocol ──")
        plot_control_protocol_3d(data, output_dir)

    if ENABLE_TRANSPORT_ANIMATION:
        print("\n── 3D transport animation ──")
        create_transport_animation_3d(
            data,
            output_dir,
            fps=FPS_TRANSPORT,
            dpi=DPI_TRANSPORT,
            show_potentials=False,
            camera_mode="fixed",
        )

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
