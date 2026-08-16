"""
Visualize the control protocol from 09.06.2026 and 05.07.2026
same plot, same time axis, top shows position over time, bottom
show the amplitude over time.
"""

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.linewidth"] = 1.0

COLOR_A = "#8B0000"
COLOR_B = "#001CBBBC"


def style_axes(ax):
    ax.grid(alpha=0.28, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load(h5_path):
    with h5py.File(h5_path, "r") as f:
        t_us = f["t_us"][:]
        x_um = f["ux_um"][:]
        ua = f["ua"][:]
    return t_us, x_um, ua


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_dir = repo_root / "ResultsForThesis"
    path_a = results_dir / "control3d_thermal_2026-06-09_22-34-02.h5"
    path_b = results_dir / "control3d_thermal_2026-07-05_21-03-39.h5"
    output_dir = repo_root / "images" / "Show_A_and_B"
    output_dir.mkdir(parents=True, exist_ok=True)

    t_a, x_a, ua_a = load(path_a)
    t_b, x_b, ua_b = load(path_b)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(t_a, x_a, color=COLOR_A, lw=2.0, label="Protocol optimized at 4μK")
    ax1.plot(
        t_b, x_b, color=COLOR_B, lw=2.0, ls="--", label="Protocol optimized at 16μK"
    )
    ax1.axhline(4.6, color="black", lw=2.0, ls=":")
    ax1.text(
        t_a[-1],
        4.5,
        "4.6μm",
        color="black",
        ha="right",
        va="top",
        fontsize=12,
    )
    ax1.text(
        t_a[-1],
        x_a[-1] + 0.1,
        f"{x_a[-1]:.3f}μm",
        color=COLOR_A,
        ha="right",
        va="bottom",
        fontsize=12,
    )
    ax1.text(
        t_b[-1] + 0.4,
        x_b[-1] - 0.5,
        f"{x_b[-1]:.3f}μm",
        color=COLOR_B,
        ha="right",
        va="top",
        fontsize=12,
    )
    ax1.text(
        t_a[0],
        x_a[0] + 0.5,
        f"{x_a[0]:.3f}μm",
        color=COLOR_A,
        ha="left",
        va="bottom",
        fontsize=12,
    )
    ax1.text(
        t_b[0] + 2,
        x_b[0] + 0.3,
        f"{x_b[0]:.3f}μm",
        color=COLOR_B,
        ha="left",
        va="top",
        fontsize=12,
    )
    ax1.set_ylabel("Position [μm]")
    ax1.legend(loc="best", framealpha=0.85)
    style_axes(ax1)

    ax2.plot(t_a, ua_a, color=COLOR_A, lw=1.8)
    ax2.plot(t_b, ua_b, color=COLOR_B, lw=1.8, ls="--")
    ax2.set_xlabel("Time [μs]")
    ax2.set_ylabel("Amplitude")
    style_axes(ax2)

    fig.tight_layout()

    for suffix in (".png", ".pdf"):
        out_path = output_dir / f"protocol_comparison_0609_0705{suffix}"
        fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"Saved: {out_path.name}")

    plt.close(fig)


if __name__ == "__main__":
    main()
