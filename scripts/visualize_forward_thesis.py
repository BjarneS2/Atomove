import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "Visualization"))
sys.path.insert(0, str(_THIS_DIR / "VisualizationForThesis"))

import h5py
from classify_thesis import compute_categories
from plot_classification_thesis import plot_classification_thesis
from plot_control_protocol_thesis import plot_control_protocol_thesis
from plot_phase_space_thesis import (
    plot_phase_space_thesis,
    plot_phase_space_thesis_with_ellipse,
    plot_phase_space_thesis_x_with_aux,
)
from plot_trajectories_thesis import plot_trajectories_thesis
from visualize_forward3D import load_forward3d

TRAJECTORY_STYLE = "individual"


def _output_dir(file_path):
    out = _THIS_DIR.parent / "images" / f"{file_path.stem}_thesis"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_transport_time(file_path, t):
    with h5py.File(file_path, "r") as f:
        extension_factor = float(f.attrs.get("extension_factor", 0.0))
    return (t[-1] - t[0]) / (1.0 + extension_factor)


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
            print("ERROR: no forward3d_*.h5 found in ResultsForThesis/")
            sys.exit(1)
        file_path = candidates[0]
        print(f"Auto-detected: {file_path.name}")

    data = load_forward3d(file_path)
    data["transport_time"] = _read_transport_time(file_path, data["t"])
    output_dir = _output_dir(file_path)
    print(f"Output → {output_dir}")

    classification = compute_categories(data)

    print("\n── Control protocol ──")
    plot_control_protocol_thesis(data, output_dir, show_insets=True)
    plot_control_protocol_thesis(data, output_dir, show_insets=False)

    print("\n── Trajectories ──")
    plot_trajectories_thesis(data, classification, output_dir, style=TRAJECTORY_STYLE)

    print("\n── Phase-space snapshots ──")
    plot_phase_space_thesis(data, classification, output_dir)
    plot_phase_space_thesis_with_ellipse(data, classification, output_dir)
    plot_phase_space_thesis_x_with_aux(data, classification, output_dir)

    print("\n── Survival classification ──")
    plot_classification_thesis(data, classification, output_dir)
    plot_classification_thesis(data, classification, output_dir, top_only=True)
    plot_classification_thesis(data, classification, output_dir, transport_only=True)
    plot_classification_thesis(data, classification, output_dir, control_row=True)
    plot_classification_thesis(data, classification, output_dir, summary_row=True)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
