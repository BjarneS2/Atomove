from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
from style_thesis import CATEGORY_COLORS, CATEGORY_LABELS, CATEGORY_ORDER, COLOR_AUX, savefig_thesis, style_axes


def _category_labels_final(classification, n_shots):
    labels = np.empty(n_shots, dtype=object)
    for cat in CATEGORY_ORDER:
        labels[classification[cat][-1]] = cat
    return labels


def plot_trajectories_thesis(data, classification, output_dir, style="individual", max_display=400):
    t = data["t"]
    x, y, z = data["x"], data["y"], data["z"]
    vx, vy, vz = data["vx"], data["vy"], data["vz"]
    ux, uy = data["ux"], data["uy"]
    params = data["params"]
    scales = data["scales"]
    transport_time = data["transport_time"]

    n_shots = x.shape[1]
    w0_um = scales["w0_um"]
    final_labels = _category_labels_final(classification, n_shots)
    survived_final = classification["survived"][-1]

    pos_label = f"[w₀ = {w0_um:.1f} μm]"
    vel_label = "[m/s]"

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.subplots_adjust(hspace=0.10, wspace=0.30)

    arrs_pos = [(x, "x"), (y, "y"), (z, "z")]
    arrs_vel = [(vx, "vx"), (vy, "vy"), (vz, "vz")]
    ctrl_pos = [(ux, params["x_start"], params["x_stop"]), (uy, params["y_start"], params["y_stop"]), (None, None, None)]

    rng = np.random.default_rng(0)
    display_idx = rng.permutation(n_shots)[:max_display]

    for col, ((pos_arr, pos_name), (vel_arr, vel_name), (ctrl, p_start, p_stop)) in enumerate(
        zip(arrs_pos, arrs_vel, ctrl_pos)
    ):
        ax_p = axes[0, col]
        ax_v = axes[1, col]

        if style == "individual":
            for s in display_idx:
                cat = final_labels[s]
                color = CATEGORY_COLORS[cat]
                ax_p.plot(t, pos_arr[:, s], color=color, alpha=0.25, lw=0.7)
                ax_v.plot(t, vel_arr[:, s], color=color, alpha=0.25, lw=0.7)
        else:
            pos_sub = pos_arr[:, survived_final]
            vel_sub = vel_arr[:, survived_final]
            pos_mean = pos_sub.mean(axis=1)
            pos_std = pos_sub.std(axis=1)
            vel_mean = vel_sub.mean(axis=1)
            vel_std = vel_sub.std(axis=1)
            ax_p.fill_between(t, pos_mean - pos_std, pos_mean + pos_std, color=CATEGORY_COLORS["survived"], alpha=0.25)
            ax_v.fill_between(t, vel_mean - vel_std, vel_mean + vel_std, color=CATEGORY_COLORS["survived"], alpha=0.25)

        pos_mean_line = pos_arr[:, survived_final].mean(axis=1)
        vel_mean_line = vel_arr[:, survived_final].mean(axis=1)
        ax_p.plot(t, pos_mean_line, color="black", lw=2.5, label="Mean (survived)")
        ax_v.plot(t, vel_mean_line, color="black", lw=2.5, label="Mean (survived)")

        if ctrl is not None:
            ax_p.plot(t, ctrl, color=COLOR_AUX, lw=1.5, ls=":", label="Aux tweezer", zorder=10)
            ax_p.axhline(p_start, color="gray", ls=":", lw=1.0, alpha=0.5)
            ax_p.axhline(p_stop, color="gray", ls="-.", lw=1.0, alpha=0.5)
        else:
            zR = params["zR"]
            ax_p.axhline(zR, color="red", lw=0.7, ls="-", alpha=0.5)
            ax_p.axhline(-zR, color="red", lw=0.7, ls="-", alpha=0.5)

        ax_v.axhline(0, color="black", lw=0.5, alpha=0.3)

        ax_p.set_ylabel(f"{pos_name}  {pos_label}", fontsize=10)
        ax_v.set_ylabel(f"{vel_name}  {vel_label}", fontsize=10)
        ax_v.set_xlabel("Time [μs]", fontsize=10)

        style_axes(ax_p)
        style_axes(ax_v)

    transport_mask = t <= transport_time
    dx = ux[transport_mask] - x[transport_mask][:, survived_final].mean(axis=1)
    dy = uy[transport_mask] - y[transport_mask][:, survived_final].mean(axis=1)
    d = np.sqrt(dx**2 + dy**2)
    metric = float(np.clip(np.sqrt(np.mean(d**2)) / params["w"], 0.0, 1.0))

    legend_elems = [
        Line2D([0], [0], color=CATEGORY_COLORS[c], lw=1.5, label=CATEGORY_LABELS[c]) for c in CATEGORY_ORDER
    ]
    legend_elems.append(Line2D([0], [0], color="black", lw=2.5, label="Mean (survived)"))
    axes[0, 2].legend(handles=legend_elems, loc="upper right", fontsize=8, framealpha=0.8)

    n_surv = int(np.sum(classification["survived"][-1]))
    fig.suptitle(
        f"Positions & velocities — {n_shots} shots, survived: {n_surv}/{n_shots} "
        f"({100*n_surv/n_shots:.1f}%)\n"
        f"Mean trajectory deviation from aux trap center (RMS/w): {metric:.2f}",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout()
    savefig_thesis(fig, Path(output_dir) / "trajectories_thesis.png")
