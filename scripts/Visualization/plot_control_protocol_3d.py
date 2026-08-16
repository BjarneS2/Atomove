"""
Control protocol plot for 3D forward dynamics.

Shows ux(t), uy(t) (auxiliary tweezer position) and ua(t) (amplitude).
"""

from pathlib import Path
from typing import Dict  # noqa: UP035

import matplotlib.pyplot as plt
import numpy as np
from utils3d import COLOR_AUX, COLOR_STATIC


def plot_control_protocol_3d(data: Dict, output_dir: Path):
    t = data["t"]
    ux = data["ux"]
    uy = data["uy"]
    ua = data["ua"]
    params = data["params"]
    scales = data["scales"]
    w0_um = scales["w0_um"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    pos_label = f"[w₀ = {w0_um:.1f} μm]"

    ax = axes[0]
    ax.plot(t, ux, color=COLOR_AUX, lw=2.0, label="ux (aux tweezer)")
    ax.axhline(
        params["x_start"],
        color=COLOR_STATIC,
        ls=":",
        lw=1.5,
        alpha=0.7,
        label="x_start",
    )
    ax.axhline(
        params["x_stop"], color=COLOR_STATIC, ls="-.", lw=1.5, alpha=0.7, label="x_stop"
    )
    ax.set_ylabel(f"ux  {pos_label}", fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.25, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(t, uy, color=COLOR_AUX, lw=2.0, label="uy (aux tweezer)")
    ax.axhline(
        params["y_start"],
        color=COLOR_STATIC,
        ls=":",
        lw=1.5,
        alpha=0.7,
        label="y_start",
    )
    ax.axhline(
        params["y_stop"], color=COLOR_STATIC, ls="-.", lw=1.5, alpha=0.7, label="y_stop"
    )
    ax.set_ylabel(f"uy  {pos_label}", fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.25, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.plot(t, ua, color=COLOR_AUX, lw=2.0, label="ua (amplitude)")
    ax.fill_between(t, 0, ua, color=COLOR_AUX, alpha=0.20)
    ax.set_ylabel("ua  [dimless]", fontsize=11)
    ax.set_xlabel("Time [μs]", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.25, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("3D Control Protocol", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = output_dir / "control_protocol_3d.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")
