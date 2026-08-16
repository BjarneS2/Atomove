from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from style_thesis import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    CATEGORY_MARKERS,
    CATEGORY_ORDER,
    COLOR_AUX,
    COLOR_STATIC,
    savefig_thesis,
    style_axes,
)


def _nearest_index(t, target):
    return int(np.argmin(np.abs(t - target)))


def _plot_snapshot(
    ax, pos, vel, classification, t_idx, pos_label, vel_label, title, ellipses=None
):
    for cat in CATEGORY_ORDER:
        mask = classification[cat][t_idx]
        if np.any(mask):
            ax.scatter(
                pos[t_idx, mask],
                vel[t_idx, mask],
                s=35,
                color=CATEGORY_COLORS[cat],
                marker=CATEGORY_MARKERS[cat],
                alpha=0.65,
                label=CATEGORY_LABELS[cat],
            )
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)

    if ellipses:
        xlim_before, ylim_before = ax.get_xlim(), ax.get_ylim()
        theta = np.linspace(0, 2 * np.pi, 300)
        for spec in ellipses:
            pos_max, vel_max, centers, vel_center = (
                spec["pos_max"],
                spec["vel_max"],
                spec["centers"],
                spec.get("vel_center", 0.0),
            )
            for i, center in enumerate(centers):
                ax.plot(
                    center + pos_max * np.cos(theta),
                    vel_center + vel_max * np.sin(theta),
                    color=spec.get("color", "black"),
                    ls=spec.get("ls", "--"),
                    lw=1.2,
                    alpha=0.6,
                    label=spec.get("label") if i == 0 else None,
                )
        ax.set_xlim(xlim_before)
        ax.set_ylim(ylim_before)

    ax.set_xlabel(pos_label, fontsize=10)
    ax.set_ylabel(vel_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    style_axes(ax)


def _trap_boundary_params(coord_name, params):
    U0 = params["U0_static"]
    trap_fraction = params["trap_fraction"]
    vel_max = np.sqrt(max(2.0 * U0 * (1.0 - trap_fraction), 0.0))

    if coord_name in ("x", "y"):
        pos_max = params["w"] / np.sqrt(2.0)
        centers = sorted({params[f"{coord_name}_start"], params[f"{coord_name}_stop"]})
    else:
        pos_max = params["zR"]
        centers = [0.0]

    return pos_max, vel_max, centers


def _aux_trap_boundary_params(coord_name, data, t_idx):
    params = data["params"]
    U0_aux = params["U0_aux_max"] * data["ua"][t_idx]
    trap_fraction = params["trap_fraction"]
    vel_max = np.sqrt(max(2.0 * U0_aux * (1.0 - trap_fraction), 0.0))
    pos_max = params["w_aux"] / np.sqrt(2.0)
    u_coord = data[f"u{coord_name}"]
    center = u_coord[t_idx]
    u_vel = np.gradient(u_coord, data["t"])[t_idx]
    return pos_max, vel_max, [center], u_vel


def _plot_phase_space_thesis(
    data,
    classification,
    output_dir,
    show_ellipse,
    filename_suffix,
    show_aux_ellipse=False,
):
    t = data["t"]
    scales = data["scales"]
    params = data["params"]
    transport_time = data["transport_time"]
    w0_um = scales["w0_um"]

    t_indices = [
        0,
        _nearest_index(t, transport_time / 2.0),
        _nearest_index(t, transport_time),
        len(t) - 1,
    ]
    t_labels = ["Initial", "Midpoint", "Protocol end", "Final"]

    coord_specs = [
        ("x", data["x"], data["vx"], f"x [w₀ = {w0_um:.1f} μm]", True),
        ("y", data["y"], data["vy"], f"y [w₀ = {w0_um:.1f} μm]", True),
        ("z", data["z"], data["vz"], f"z [w₀ = {w0_um:.1f} μm]", False),
    ]
    if show_aux_ellipse:
        coord_specs = [spec for spec in coord_specs if spec[0] == "x"]

    for coord_name, pos, vel, pos_label, is_transport_axis in coord_specs:
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        axes = axes.flatten()

        static_ellipse = (
            _trap_boundary_params(coord_name, params) if show_ellipse else None
        )

        for ax, t_idx, t_label in zip(axes, t_indices, t_labels):
            ellipses = []
            if static_ellipse is not None:
                pos_max, vel_max, centers = static_ellipse
                ellipses.append(
                    dict(
                        pos_max=pos_max,
                        vel_max=vel_max,
                        centers=centers,
                        color="black",
                        ls="--",
                        label="Static trap boundary",
                    )
                )
            if show_aux_ellipse:
                pos_max, vel_max, centers, vel_center = _aux_trap_boundary_params(
                    coord_name, data, t_idx
                )
                ellipses.append(
                    dict(
                        pos_max=pos_max,
                        vel_max=vel_max,
                        centers=centers,
                        vel_center=vel_center,
                        color=COLOR_AUX,
                        ls="--",
                        label="Aux trap boundary",
                    )
                )

            _plot_snapshot(
                ax,
                pos,
                vel,
                classification,
                t_idx,
                pos_label,
                f"v{coord_name} [m/s]",
                f"{t_label} (t = {t[t_idx]:.1f} μs)",
                ellipses=ellipses,
            )
            if is_transport_axis:
                start_val = params[f"{coord_name}_start"]
                stop_val = params[f"{coord_name}_stop"]
                ax.axvline(start_val, color=COLOR_STATIC, ls=":", lw=1.0, alpha=0.5)
                ax.axvline(stop_val, color=COLOR_STATIC, ls="-.", lw=1.0, alpha=0.5)

        shared_idx = t_indices[:3]
        pos_shared = pos[shared_idx, :]
        vel_shared = vel[shared_idx, :]
        pos_pad = 0.05 * (np.nanmax(pos_shared) - np.nanmin(pos_shared) + 1e-12)
        vel_pad = 0.05 * (np.nanmax(vel_shared) - np.nanmin(vel_shared) + 1e-12)
        pos_lim = (np.nanmin(pos_shared) - pos_pad, np.nanmax(pos_shared) + pos_pad)
        vel_lim = (np.nanmin(vel_shared) - vel_pad, np.nanmax(vel_shared) + vel_pad)
        for ax in axes[:3]:
            ax.set_xlim(*pos_lim)
            ax.set_ylim(*vel_lim)

        axes[0].legend(loc="best", fontsize=8, framealpha=0.8)
        plt.tight_layout()
        savefig_thesis(
            fig,
            Path(output_dir) / f"phase_space_thesis_{coord_name}{filename_suffix}.png",
        )


def plot_phase_space_thesis(data, classification, output_dir):
    _plot_phase_space_thesis(
        data, classification, output_dir, show_ellipse=False, filename_suffix=""
    )


def plot_phase_space_thesis_with_ellipse(data, classification, output_dir):
    _plot_phase_space_thesis(
        data, classification, output_dir, show_ellipse=True, filename_suffix="_ellipse"
    )


def plot_phase_space_thesis_x_with_aux(data, classification, output_dir):
    _plot_phase_space_thesis(
        data,
        classification,
        output_dir,
        show_ellipse=True,
        filename_suffix="_aux",
        show_aux_ellipse=True,
    )
