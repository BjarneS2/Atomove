import sys
from pathlib import Path

# ── Make sure ./scripts/Visualization is on the import path ───────────────────
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / "Visualization"))

import h5py
import numpy as np

from constants3d import DEFAULTS, compute_scales    # type: ignore
from utils3d import compute_loss_mask, total_energy # type: ignore

# ── Visualization modules ──────────────────────────────────────────────────────
from plot_transport_3d        import create_transport_animation_3d  # type: ignore
from plot_survival_3d         import plot_survival_3d               # type: ignore
from plot_phase_space_3d      import animate_phase_space_3d         # type: ignore
from plot_energy_3d           import plot_energy_distributions_3d   # type: ignore
from plot_trajectories_3d     import plot_trajectories_3d           # type: ignore
from plot_control_protocol_3d import plot_control_protocol_3d       # type: ignore



def load_forward3d(file_path: Path) -> dict:

    with h5py.File(file_path, "r") as f:
        t    = f["t"][:]            # type: ignore
        ux   = f["ux"][:]           # type: ignore
        uy   = f["uy"][:]           # type: ignore
        ua   = f["ua"][:]           # type: ignore
        x    = np.array(f["x"]).T    # Julia writes (n_steps, n_shots); h5py reads transposed → fix to (n_steps, n_shots)
        y    = np.array(f["y"]).T
        z    = np.array(f["z"]).T
        vx   = np.array(f["vx"]).T
        vy   = np.array(f["vy"]).T
        vz   = np.array(f["vz"]).T

        atr = f.attrs
        def ga(k, default=None):
            return atr[k] if k in atr else default

        w            = float(ga("w", 1.2))                      # type: ignore
        w_aux_factor = float(ga("w_aux_factor", 1.1 / 1.2))     # type: ignore
        zR           = float(ga("zR",    15.0))                 # type: ignore
        zR_aux       = float(ga("zR_aux", 13.0))                # type: ignore
        x_start      = float(ga("x_start", 0.0))                # type: ignore
        y_start      = float(ga("y_start", 0.0))                # type: ignore
        x_stop       = float(ga("x_stop",  4.6))                # type: ignore
        y_stop       = float(ga("y_stop",  0.0))                # type: ignore
        T_atom       = float(ga("T_atom",   40e-6))             # type: ignore
        T_tweezer    = float(ga("T_tweezer", 287e-6))           # type: ignore
        w0_um        = float(ga("w0_um",  1.0))                 # type: ignore
        U0_static    = float(ga("U0_static",  0.01))            # type: ignore
        U0_aux_max   = float(ga("U0_aux_max", 0.03))            # type: ignore
        starting_trap_fraction = float(ga("starting_trap_fraction", 0.5))   # type: ignore
        trap_fraction          = float(ga("trap_fraction",           0.5))  # type: ignore
        final_trap_fraction    = float(ga("final_trap_fraction",     0.5))  # type: ignore

    w_aux = w * w_aux_factor

    params = dict(
        w          = w,
        w_aux      = w_aux,
        w_aux_factor = w_aux_factor,
        zR         = zR,
        zR_aux     = zR_aux,
        x_start    = x_start,
        y_start    = y_start,
        x_stop     = x_stop,
        y_stop     = y_stop,
        U0_static  = U0_static,
        U0_aux_max = U0_aux_max,
        starting_trap_fraction = starting_trap_fraction,
        trap_fraction          = trap_fraction,
        final_trap_fraction    = final_trap_fraction,
    )

    scales = compute_scales(T_tweezer, w0_um, DEFAULTS)
    # dimensionless temperature ratio used for ellipse sizing
    scales["T_atom_dimless"] = T_atom / T_tweezer

    # ── Energy arrays ──────────────────────────────────────────────────────────
    print("Computing energies …")
    KE, PE, E_tot = total_energy(x, y, z, vx, vy, vz, ux, uy, ua, params)

    n_shots    = x.shape[1]
    return dict(
        t       = t,
        x       = x,  y    = y,  z  = z,
        vx      = vx, vy   = vy, vz = vz,
        ux      = ux, uy   = uy, ua = ua,
        KE      = KE,
        PE      = PE,
        E_tot   = E_tot,
        params  = params,
        scales  = scales,
        n_shots = n_shots,
    )


def main():
    results_dir = _THIS_DIR.parent / "results"

    if len(sys.argv) >= 2:
        file_path = Path(sys.argv[1])
    else:
        candidates = sorted(results_dir.glob("forward3d_*.h5"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("ERROR: no forward3d_*.h5 found in results/")
            sys.exit(1)
        file_path = candidates[0]
        print(f"Auto-detected: {file_path.name}")
    

    # Now load in, and visualize the kinetic energies
    data = load_forward3d(file_path)
    
    speed = np.sqrt(data["vx"]**2 + data["vy"]**2 + data["vz"]**2)
    import matplotlib.pyplot as plt
    
    # Sanity check: print trap depths, speeds, and distance from maximum
    final_trap_fraction = data["params"]["final_trap_fraction"]
    U0_static = data["params"]["U0_static"]
    U0_aux_max = data["params"]["U0_aux_max"]
    max_speed = np.sqrt(2 * final_trap_fraction * U0_static)
    max_energy = final_trap_fraction * U0_static
    max_energy_beg = data["params"]["starting_trap_fraction"] * U0_static
    mean_final_speed = np.mean(speed[-1, :])
    print(f"Trap depths: U0_static = {U0_static}, U0_aux_max = {U0_aux_max}")
    print(f"Maximum expected energy: {max_energy}")
    print(f"Maximum expected energy in the beginning: {max_energy_beg}")
    print(f"Mean final speed: {mean_final_speed}")
    print(f"Maximum expected speed: {max_speed}")
    print(f"Difference from maximum: {max_speed - mean_final_speed}")
    
    t = data["t"]
    KE = data["KE"]
    E_tot = data["E_tot"]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Mean Kinetic Energy
    axs[0, 0].plot(t, np.mean(KE, axis=1), linewidth=1.5)
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Mean Kinetic Energy")
    axs[0, 0].set_title("Mean Kinetic Energy Over Time")
    axs[0, 0].grid(True, alpha=0.3)
    
    # Mean Potential Energy
    axs[0, 1].plot(t, np.mean(data["PE"], axis=1), linewidth=1.5)
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Mean Potential Energy")
    axs[0, 1].set_title("Mean Potential Energy Over Time")
    axs[0, 1].grid(True, alpha=0.3)
    
    # Mean Total Energy
    axs[1, 0].plot(t, np.mean(E_tot, axis=1), linewidth=1.5)
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Mean Total Energy")
    axs[1, 0].set_title("Mean Total Energy Over Time")
    axs[1, 0].grid(True, alpha=0.3)
    
    # mean of Speeds
    axs[1, 1].plot(t, np.mean(speed, axis=1), linewidth=1.5)
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Sum of Speeds")
    axs[1, 1].set_title("Sum of Speeds Over Time")
    axs[1, 1].grid(True, alpha=0.3)
    
    fig.tight_layout()

    out_path = file_path.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved energy plots to {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()
