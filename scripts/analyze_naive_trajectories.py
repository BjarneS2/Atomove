import sys
from pathlib import Path

import h5py

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "Visualization"))
sys.path.insert(0, str(_THIS_DIR / "VisualizationForThesis"))

from classify_thesis import compute_categories
from visualize_forward3D import load_forward3d

RESULTS_DIR = _THIS_DIR.parent / "ResultsForThesis"

TRAJ_ORDER = ["linear", "minjerk", "linear_offset", "minjerk_offset"]
BASE_ORDER = ["A", "B"]
TEMP_ORDER = [4, 16, 25, 40]

BASE_LABELS = {
    "A": "A (T_A, converged 4uK, 2026-06-09)",
    "B": "B (T_B, converged 16uK, 2026-07-05)",
}
TRAJ_LABELS = {
    "linear": "Linear",
    "minjerk": "Minimum-jerk",
    "linear_offset": "Linear (optimized offsets)",
    "minjerk_offset": "Minimum-jerk (optimized offsets)",
}


def _decode(v):
    return v.decode() if isinstance(v, bytes) else v


def find_runs():
    runs = {}
    for f in sorted(RESULTS_DIR.glob("forward3d_naive_*.h5")):
        with h5py.File(f, "r") as h:
            traj_type = _decode(h.attrs["traj_type"])
            base_run = _decode(h.attrs["base_run"])
            T_atom = float(h.attrs["T_atom"]) * 1e6
        temp_key = int(round(T_atom))
        key = (traj_type, base_run, temp_key)
        if key not in runs or f.stat().st_mtime > runs[key].stat().st_mtime:
            runs[key] = f
    return runs


def analyze_file(f):
    data = load_forward3d(f)
    with h5py.File(f, "r") as h:
        extension_factor = float(h.attrs.get("extension_factor", 0.0))
        T_transport = float(h.attrs.get("T_transport_us", float("nan")))
    classification = compute_categories(data)
    n_shots = data["x"].shape[1]

    n_survived = int(classification["survived"][-1].sum())
    n_left_behind = int(classification["left_behind"][-1].sum())
    n_recaptured = int(classification["recaptured"][-1].sum())
    n_lost = int(classification["lost"][-1].sum())

    return dict(
        n_shots=n_shots,
        survived=n_survived,
        left_behind=n_left_behind,
        recaptured=n_recaptured,
        lost=n_lost,
        survival_pct=100.0 * n_survived / n_shots,
        left_behind_pct=100.0 * n_left_behind / n_shots,
        recaptured_pct=100.0 * n_recaptured / n_shots,
        lost_pct=100.0 * n_lost / n_shots,
        surv_plus_recap_pct=100.0 * (n_survived + n_recaptured) / n_shots,
        T_transport=T_transport,
        extension_factor=extension_factor,
        file=f.name,
    )


def main():
    runs = find_runs()
    expected = [
        (traj, base, temp)
        for traj in TRAJ_ORDER
        for base in BASE_ORDER
        for temp in TEMP_ORDER
    ]
    missing = [k for k in expected if k not in runs]
    if missing:
        print("WARNING: missing runs:", missing)

    results = {}
    for key in expected:
        if key not in runs:
            continue
        results[key] = analyze_file(runs[key])

    lines = []
    lines.append(
        "| Trajectory | Base | T_atom | Survived | Recaptured | Lost | Left behind | Survival+Recaptured | Output file |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for traj in TRAJ_ORDER:
        for base in BASE_ORDER:
            for temp in TEMP_ORDER:
                key = (traj, base, temp)
                if key not in results:
                    continue
                r = results[key]
                lines.append(
                    f"| {TRAJ_LABELS[traj]} | {base} (T={r['T_transport']:.2f} us) | {temp} uK "
                    f"| {r['survival_pct']:.1f}% ({r['survived']}/{r['n_shots']}) "
                    f"| {r['recaptured_pct']:.1f}% ({r['recaptured']}/{r['n_shots']}) "
                    f"| {r['lost_pct']:.1f}% ({r['lost']}/{r['n_shots']}) "
                    f"| {r['left_behind_pct']:.1f}% ({r['left_behind']}/{r['n_shots']}) "
                    f"| {r['surv_plus_recap_pct']:.1f}% "
                    f"| `{r['file']}` |"
                )
    full_table = "\n".join(lines)

    lines2 = []
    header = "| Protocol | " + " | ".join(f"{t} uK" for t in TEMP_ORDER) + " |"
    sep = "|---|" + "---|" * len(TEMP_ORDER)
    lines2.append(header)
    lines2.append(sep)
    for traj in TRAJ_ORDER:
        for base in BASE_ORDER:
            if not any((traj, base, temp) in results for temp in TEMP_ORDER):
                continue
            row = [f"{TRAJ_LABELS[traj]} @ {base}"]
            for temp in TEMP_ORDER:
                key = (traj, base, temp)
                if key in results:
                    row.append(f"{results[key]['surv_plus_recap_pct']:.1f}%")
                else:
                    row.append("—")
            lines2.append("| " + " | ".join(row) + " |")
    matrix_table = "\n".join(lines2)

    print("\n## Full analysis — naive (linear / minimum-jerk) trajectories\n")
    print(full_table)
    print("\n## Survival + Recaptured [%] — 4 protocols x 4 temperatures\n")
    print(matrix_table)

    out_md = RESULTS_DIR / "forward_run_summary_naive_trajectories.md"
    with open(out_md, "w") as fh:
        fh.write("# Naive-trajectory forward-run campaign\n\n")
        fh.write(
            "Linear and minimum-jerk auxiliary-tweezer position trajectories, "
            "timed to the transport durations of the two converged optimized "
            "protocols (T_A = 21.60 us from the 2026-06-09 4uK-converged run, "
            "T_B = 24.92 us from the 2026-07-05 16uK-converged run). All other "
            "protocol parameters (aux amplitude ramp, geometry, trap depths) "
            "taken verbatim from the corresponding base protocol. "
            "5000 shots, SEED=101, extension_factor=2.0, final_trap_fraction "
            "forced to 0.7, as in the optimized-trajectory forward campaign.\n\n"
            "`Linear (optimized offsets)` / `Minimum-jerk (optimized offsets)` use "
            "the same linear/minimum-jerk shape but start/end at the optimized "
            "protocol's own ux/uy endpoints (e.g. for base A: ux 0.507 -> 3.668; "
            "for base B: ux 0.266 -> 3.907) instead of the nominal x_start/x_stop "
            "(0.0 -> 4.6) used by the plain linear/minimum-jerk rows.\n\n"
        )
        fh.write("## Full analysis\n\n")
        fh.write(full_table + "\n\n")
        fh.write("## Survival + Recaptured [%]\n\n")
        fh.write(matrix_table + "\n")
    print(f"\nSaved markdown summary to {out_md}")


if __name__ == "__main__":
    main()
