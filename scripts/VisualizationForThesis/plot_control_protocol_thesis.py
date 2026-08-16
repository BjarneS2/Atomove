from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from style_thesis import COLOR_AUX, COLOR_STATIC, savefig_thesis, style_axes

INSET_BBOX = [0.58, 0.08, 0.39, 0.42]


def _plot_active_split(ax, t, y, is_active, color, lw, label=None):
    n = len(t)
    idx = 0
    labelled = False
    while idx < n:
        run_active = is_active[idx]
        j = idx
        while j < n and is_active[j] == run_active:
            j += 1
        start = idx - 1 if idx > 0 else idx
        seg_t = t[start:j]
        seg_y = y[start:j]
        ls = "-" if run_active else ":"
        lbl = None
        if run_active and not labelled:
            lbl = label
            labelled = True
        ax.plot(seg_t, seg_y, color=color, lw=lw, ls=ls, label=lbl)
        idx = j


def plot_control_protocol_thesis(data, output_dir, show_insets=True):
    t = data["t"]
    ux = data["ux"]
    uy = data["uy"]
    ua = data["ua"]
    params = data["params"]
    scales = data["scales"]
    transport_time = data["transport_time"]
    w0_um = scales["w0_um"]

    ua_max = np.max(ua) if np.size(ua) else 0.0
    is_active = ua > 1e-3 * ua_max if ua_max > 0 else np.ones_like(t, dtype=bool)

    plot_uy = (not np.isclose(params["y_start"], params["y_stop"])) or (
        not np.allclose(uy, uy[0])
    )

    panels = [("ux", ux, params["x_start"], params["x_stop"])]
    if plot_uy:
        panels.append(("uy", uy, params["y_start"], params["y_stop"]))

    n_panels = len(panels) + 1
    pos_label = f"[w₀ = {w0_um:.1f} μm]"

    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3 * n_panels))
    if n_panels == 1:
        axes = [axes]

    transport_mask = t <= transport_time

    for ax, (name, arr, start_val, stop_val) in zip(axes, panels):
        ax.axvspan(t[0], transport_time, color="gray", alpha=0.06, zorder=0)
        _plot_active_split(ax, t, arr, is_active, COLOR_AUX, 2.0, label=f"{name} (aux tweezer)")
        ax.axhline(start_val, color=COLOR_STATIC, ls=":", lw=1.5, alpha=0.7, label=f"{name[-1]}_start")
        ax.axhline(stop_val, color=COLOR_STATIC, ls="-.", lw=1.5, alpha=0.7, label=f"{name[-1]}_stop")
        ax.set_ylabel(f"{name}  {pos_label}", fontsize=11)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
        style_axes(ax)

        if show_insets:
            axins = ax.inset_axes(INSET_BBOX)
            axins.set_facecolor("white")
            axins.patch.set_alpha(0.95)
            _plot_active_split(
                axins, t[transport_mask], arr[transport_mask], is_active[transport_mask],
                COLOR_AUX, 1.5,
            )
            axins.axhline(start_val, color=COLOR_STATIC, ls=":", lw=1.0, alpha=0.7)
            axins.axhline(stop_val, color=COLOR_STATIC, ls="-.", lw=1.0, alpha=0.7)
            axins.set_title("Transport window", fontsize=8, fontweight="bold")
            axins.tick_params(labelsize=7)
            style_axes(axins)

    ax = axes[-1]
    ax.axvspan(t[0], transport_time, color="gray", alpha=0.06, zorder=0)
    ax.plot(t, ua, color=COLOR_AUX, lw=2.0, label="ua (amplitude)")
    ax.fill_between(t, 0, ua, color=COLOR_AUX, alpha=0.20)
    ax.set_ylabel("ua  [dimless]", fontsize=11)
    ax.set_xlabel("Time [μs]", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    style_axes(ax)

    if show_insets:
        axins = ax.inset_axes(INSET_BBOX)
        axins.set_facecolor("white")
        axins.patch.set_alpha(0.95)
        axins.plot(t[transport_mask], ua[transport_mask], color=COLOR_AUX, lw=1.5)
        axins.fill_between(t[transport_mask], 0, ua[transport_mask], color=COLOR_AUX, alpha=0.20)
        axins.set_title("Transport window", fontsize=8, fontweight="bold")
        axins.tick_params(labelsize=7)
        style_axes(axins)

    for ax in axes[:-1]:
        ax.set_xlabel("")

    fig.suptitle("Control Protocol", fontsize=13, fontweight="bold")
    plt.tight_layout()
    suffix = "" if show_insets else "_no_inset"
    savefig_thesis(fig, Path(output_dir) / f"control_protocol_thesis{suffix}.png")
