"""
Trajectory plots for 3D forward dynamics.

Two panels: positions (x, y, z) and velocities (vx, vy, vz) over time,
one column per axis. Atoms coloured by final survived / lost status.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from utils3d import COLOR_ALIVE, COLOR_LOST


def plot_trajectories_3d(data: Dict, output_dir: Path, max_display: int = 400):
    t       = data["t"]
    x       = data["x"];   y  = data["y"];   z  = data["z"]
    vx      = data["vx"];  vy = data["vy"];  vz = data["vz"]
    ux      = data["ux"];  uy = data["uy"];  ua = data["ua"]
    is_lost = data["is_lost"]
    params  = data["params"]
    scales  = data["scales"]

    n_shots   = x.shape[1]
    lost_final = is_lost[-1]
    w0_um     = scales["w0_um"]

    rng = np.random.default_rng(0)
    idx = rng.permutation(n_shots)
    display_idx = idx[:max_display]

    pos_label = f"[w₀ = {w0_um:.1f} μm]"
    vel_label = "[m/s]"
    t_label   = "Time [μs]"

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(hspace=0.10, wspace=0.30)

    pos_axes = [axes[0, 0], axes[0, 1], axes[0, 2]]
    vel_axes = [axes[1, 0], axes[1, 1], axes[1, 2]]
    arrs_pos = [(x, "x"), (y, "y"), (z, "z")]
    arrs_vel = [(vx, "vx"), (vy, "vy"), (vz, "vz")]
    ctrl_pos = [(ux, params["x_start"], params["x_stop"]),
                (uy, params["y_start"], params["y_stop"]),
                (None, None, None)]

    v0          = scales.get("v0", 1.0)
    U0_static   = params["U0_static"]
    w           = params["w"]
    zR          = params["zR"]
    f_start     = params.get("starting_trap_fraction", 0.5)
    f_final     = params.get("final_trap_fraction",    0.5)
    v_ad        = np.sqrt(2.0 * U0_static) * v0
    v_ad_start  = np.sqrt(2.0 * f_start * U0_static) * v0
    v_ad_final  = np.sqrt(2.0 * f_final  * U0_static) * v0

    ua_max = np.max(np.abs(ua)) if np.size(ua) else 0.0
    aux_on = np.abs(ua) > 0.05 * ua_max if ua_max > 0 else np.ones_like(t, dtype=bool)
    aux_on[0] = 1
    for col, ((pos_arr, pos_name), (vel_arr, vel_name), (ctrl, p_start, p_stop)) in \
            enumerate(zip(arrs_pos, arrs_vel, ctrl_pos)):

        ax_p = pos_axes[col]
        ax_v = vel_axes[col]

        for s in display_idx:
            color = COLOR_LOST if lost_final[s] else COLOR_ALIVE
            alpha = 0.30 if not lost_final[s] else 0.20
            lw    = 0.7
            ax_p.plot(t, pos_arr[:, s], color=color, alpha=alpha, lw=lw)
            ax_v.plot(t, vel_arr[:, s], color=color, alpha=alpha, lw=lw)

        if ctrl is not None:
            ax_p.plot(t, ctrl, color="black", lw=2.0, ls="--",
                      alpha=0.8, label="Aux tweezer", zorder=10)
            ax_p.axhline(p_start, color="gray", ls=":", lw=1.2, alpha=0.6)
            ax_p.axhline(p_stop,  color="gray", ls="-.", lw=1.2, alpha=0.6)
            band_centre = np.where(aux_on, ctrl, p_stop)
            ax_p.plot(t, band_centre + w, color="red", lw=0.7, ls="-", alpha=0.6, zorder=9)
            ax_p.plot(t, band_centre - w, color="red", lw=0.7, ls="-", alpha=0.6, zorder=9)
        else:
            ax_p.axhline( zR, color="red", lw=0.7, ls="-", alpha=0.6)
            ax_p.axhline(-zR, color="red", lw=0.7, ls="-", alpha=0.6)

        ax_v.axhline( v_ad,       color="red", lw=0.9, ls="-",  alpha=0.7)
        ax_v.axhline(-v_ad,       color="red", lw=0.9, ls="-",  alpha=0.7)
        ax_v.axhline( v_ad_start, color="red", lw=0.7, ls="--", alpha=0.6)
        ax_v.axhline(-v_ad_start, color="red", lw=0.7, ls="--", alpha=0.6)
        ax_v.axhline( v_ad_final, color="red", lw=0.7, ls=":",  alpha=0.6)
        ax_v.axhline(-v_ad_final, color="red", lw=0.7, ls=":",  alpha=0.6)

        ax_v.axhline(0, color="black", lw=0.5, alpha=0.3)

        ax_p.set_ylabel(f"{pos_name}  {pos_label}", fontsize=10)
        ax_v.set_ylabel(f"{vel_name}  {vel_label}", fontsize=10)
        ax_v.set_xlabel(t_label, fontsize=10)

        for ax in (ax_p, ax_v):
            ax.grid(alpha=0.25, ls=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    n_surv = int(np.sum(~lost_final))
    fig.suptitle(
        f"3D trajectories  —  {n_shots} shots,  "
        f"survived: {n_surv}/{n_shots}  ({100*n_surv/n_shots:.1f}%)\n"
        f"(showing {min(max_display, n_shots)} trajectories)",
        fontsize=12, fontweight="bold",
    )

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=COLOR_ALIVE, lw=1.5, label="Survived"),
        Line2D([0], [0], color=COLOR_LOST,  lw=1.5, label="Lost"),
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="Aux tweezer"),
        Line2D([0], [0], color="red", lw=0.9, ls="-",  label=f"Trap width / $v_{{ad}}$={v_ad:.2f} m/s"),
        Line2D([0], [0], color="red", lw=0.7, ls="--", label=f"$v_{{ad}}$ @ start frac ({f_start:.2f}) = {v_ad_start:.2f} m/s"),
        Line2D([0], [0], color="red", lw=0.7, ls=":",  label=f"$v_{{ad}}$ @ final frac ({f_final:.2f}) = {v_ad_final:.2f} m/s"),
    ]
    axes[0, 2].legend(handles=legend_elems, loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    out = output_dir / "trajectories_3d.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")
