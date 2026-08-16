import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils3d import COLOR_ALIVE


def load_susceptibility_3d(h5_path: str):
    def _decode(v):
        if isinstance(v, bytes):
            return v.decode()
        if isinstance(v, np.ndarray) and v.dtype.kind in ("S", "O"):
            return v.item().decode() if v.ndim == 0 else v
        return v

    with h5py.File(h5_path, "r") as f:
        meta = {k: _decode(v) for k, v in f.attrs.items()}
        sweeps = {}
        for key in f.keys():
            grp = f[key]
            sweeps[key] = {
                "param_values":   grp["param_values"][:], # type: ignore
                "survival_rates": grp["survival_rates"][:], # type: ignore
                "n_lost":         grp["n_lost"][:], # type: ignore
                "param_units":    _decode(grp.attrs["param_units"]) if "param_units" in grp.attrs else "",
                "shots":          int(grp.attrs["shots"]) if "shots" in grp.attrs else int(meta.get("shots", 1000)), # type: ignore
            }
    return meta, sweeps


def plot_susceptibility_3d(h5_path: str, save_dir: str|None|Path = None) -> str:
    meta, sweeps = load_susceptibility_3d(h5_path)

    w0_um      = float(meta.get("w0_um", 1.0))
    shots_glob = int(meta.get("shots", 1000))
    T_atom_uK  = float(meta.get("T_atom",    40e-6)) * 1e6
    T_twz_uK   = float(meta.get("T_tweezer", 287e-6)) * 1e6

    panels = [
        {
            "key":    "temperature",
            "title":  "Atom temperature",
            "xlabel": "Temperature [μK]",
            "xscale": 1e6,
        },
        {
            "key":    "pos_offset_x",
            "title":  "Positional offset\n(transport dir, calibration)",
            "xlabel": "Offset along transport [μm]",
            "xscale": w0_um,
        },
        {
            "key":    "pos_offset_z",
            "title":  "Axial defocus\n(aux tweezer z-offset)",
            "xlabel": "Defocus Δz [μm]",
            "xscale": w0_um,
        },
        {
            "key":    "pos_noise",
            "title":  "Positional noise\n(beam pointing, white)",
            "xlabel": "σ per timestep [μm]",
            "xscale": w0_um,
        },
        {
            "key":    "pos_drift",
            "title":  "Positional drift\n(transport dir, linear)",
            "xlabel": "Total end-drift [μm]",
            "xscale": w0_um,
        },
        {
            "key":    "amp_offset",
            "title":  "Amplitude offset\n(calibration error)",
            "xlabel": "Constant offset [% of amplitude]",
            "xscale": 100.0,
        },
        {
            "key":    "amp_noise",
            "title":  "Amplitude noise\n(laser / chiller, white)",
            "xlabel": "σ per timestep [% of amplitude]",
            "xscale": 100.0,
        },
        {
            "key":    "amp_drift",
            "title":  "Amplitude drift\n(thermal stability, linear)",
            "xlabel": "Total end-drift [% of amplitude]",
            "xscale": 100.0,
        },
    ]

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    axes_active = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(8)]
    ax_hidden = fig.add_subplot(gs[2, 2])
    ax_hidden.set_visible(False)

    for ax, panel in zip(axes_active, panels):
        key = panel["key"]
        if key not in sweeps:
            ax.set_visible(False)
            continue

        sw     = sweeps[key]
        xs     = sw["param_values"] * panel["xscale"]
        sr     = sw["survival_rates"]
        shots  = sw["shots"]

        pct = 100.0 * sr
        err = 100.0 * np.sqrt(np.clip(sr * (1.0 - sr) / shots, 0, None))

        baseline = 100.0

        ax.errorbar(
            xs, pct, yerr=err,
            fmt="o-", color=COLOR_ALIVE,
            ecolor=COLOR_ALIVE, elinewidth=1.2, capsize=3,
            markersize=5, linewidth=1.8,
        )

        if baseline is not None:
            ax.axhline(
                baseline, color="gray", ls="--", lw=1.0, alpha=0.65,
                label=f"Baseline {baseline:.1f}%",
            )
            ax.legend(loc="lower left", fontsize=8, framealpha=0.7)

        ax.set_title(panel["title"], fontsize=9.5, fontweight="bold")
        ax.set_xlabel(panel["xlabel"], fontsize=9)
        ax.set_ylabel("Survival [%]", fontsize=9)
        ax.set_ylim(-2, 107)
        ax.grid(alpha=0.3, ls=":")
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "Protocol susceptibility / vulnerability\n"
        f"Shots per condition: {shots_glob}  |  "
        f"T_atom = {T_atom_uK:.0f} μK  |  "
        f"T_tweezer = {T_twz_uK:.0f} μK",
        fontsize=12, fontweight="bold", y=1.01,
    )

    if save_dir is None:
        h5_stem    = Path(h5_path).stem
        images_dir = Path(__file__).parent.parent / "images" / h5_stem
        images_dir.mkdir(parents=True, exist_ok=True)
        save_dir = images_dir

    out = Path(save_dir) / "susceptibility_3d.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return str(out)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        h5_path = sys.argv[1]
    else:
        results_dir = Path(__file__).parent.parent / "results"
        candidates  = sorted(
            results_dir.glob("susceptibility3d_*.h5"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("No susceptibility3d_*.h5 found in results/. Pass the file path as an argument.")
            sys.exit(1)
        h5_path = str(candidates[0])
        print(f"Using: {h5_path}")

    plot_susceptibility_3d(h5_path)
