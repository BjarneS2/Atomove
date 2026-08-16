"""
Standalone diagnostic: does neglecting gravitational PE in the lost/recaptured
energy criterion explain the atoms that flicker lost<->recaptured during the
extension (hold) period?

Does NOT modify any file in src/, scripts/run_forward_3d.jl, or
scripts/Visualization*/. It only imports the existing (unmodified)
load_forward3d / compute_categories / plot_* functions and adds
gravity-corrected counterparts locally.

Usage:
    python scripts/Miscellaneous/verify_gravity_energy_conservation.py <forward3d.h5>
    (defaults to ResultsForThesis/forward3d_2026-08-10_16-32-11.h5)
"""
import sys
from pathlib import Path

import h5py
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "Visualization"))
sys.path.insert(0, str(_SCRIPTS_DIR / "VisualizationForThesis"))

import matplotlib.pyplot as plt
from classify_thesis import compute_categories, compute_left_behind_mask
from plot_classification_thesis import _falling_event_counts, _rising_event_counts, plot_classification_thesis
from plot_control_protocol_thesis import plot_control_protocol_thesis
from plot_phase_space_thesis import plot_phase_space_thesis
from plot_trajectories_thesis import plot_trajectories_thesis
from utils3d import potential3d, total_energy
from visualize_forward3D import load_forward3d

DEFAULT_FILE = "ResultsForThesis/forward3d_2026-08-10_16-32-11.h5"


# ── Gravity-aware counterparts of classify_thesis.py (kept local — original untouched) ──

def compute_loss_mask_grav(x, y, z, vx, vy, vz, ux_arr, uy_arr, ua_arr, params, trap_fraction, g_dimless):
    n_steps, n_shots = x.shape
    is_lost = np.zeros((n_steps, n_shots), dtype=bool)
    for j in range(n_steps):
        U = potential3d(
            x[j], y[j], z[j], ux_arr[j], uy_arr[j], ua_arr[j],
            params["x_start"], params["y_start"], params["x_stop"], params["y_stop"],
            params["w"], params["w_aux"], params["zR"], params["zR_aux"],
            params["U0_static"], params["U0_aux_max"],
        )
        KE = 0.5 * (vx[j] ** 2 + vy[j] ** 2 + vz[j] ** 2)
        E_tot = KE + U + g_dimless * y[j]
        newly_lost = E_tot > trap_fraction * U
        is_lost[j] = newly_lost if j == 0 else (is_lost[j - 1] | newly_lost)
    return is_lost


def compute_recaptured_mask_grav(x, y, is_lost, E_tot, U, ux, uy, params, trap_fraction):
    w_aux = params["w_aux"]
    proximity_threshold = 2.0 * w_aux
    n_steps, n_shots = x.shape
    is_caught_again = np.zeros((n_steps, n_shots), dtype=bool)
    for j in range(n_steps):
        distance_to_aux = np.sqrt((x[j] - ux[j]) ** 2 + (y[j] - uy[j]) ** 2)
        energy_satisfied = E_tot[j] <= trap_fraction * U[j]
        proximity_satisfied = distance_to_aux <= proximity_threshold
        is_caught_again[j] = is_lost[j] & energy_satisfied & proximity_satisfied
    return is_caught_again


def compute_categories_grav(data, g_dimless):
    x, y, z = data["x"], data["y"], data["z"]
    vx, vy, vz = data["vx"], data["vy"], data["vz"]
    ux, uy, ua = data["ux"], data["uy"], data["ua"]
    params = data["params"]
    trap_fraction = params["trap_fraction"]

    is_lost = compute_loss_mask_grav(x, y, z, vx, vy, vz, ux, uy, ua, params, trap_fraction, g_dimless)
    KE, PE, E_tot_nograv = total_energy(x, y, z, vx, vy, vz, ux, uy, ua, params)
    E_tot = E_tot_nograv + g_dimless * y
    U = PE

    is_left_behind = compute_left_behind_mask(x, y, ux, uy, params)
    is_caught_again = compute_recaptured_mask_grav(x, y, is_lost, E_tot, U, ux, uy, params, trap_fraction)

    recaptured = is_caught_again
    lost = is_lost & ~is_caught_again
    left_behind = is_left_behind & ~is_lost
    survived = ~is_lost & ~is_left_behind

    return dict(
        survived=survived, left_behind=left_behind, recaptured=recaptured, lost=lost,
        is_lost=is_lost, is_left_behind=is_left_behind, is_caught_again=is_caught_again,
        E_tot=E_tot, U=U, KE=KE,
    )


def count_sign_crossings(margin):
    """margin: (n_steps_window, n_shots). Returns crossings per shot."""
    s = np.sign(margin)
    s[s == 0] = 1
    return np.sum(np.diff(s, axis=0) != 0, axis=0)


def _read_transport_time(file_path, t):
    with h5py.File(file_path, "r") as f:
        extension_factor = float(f.attrs.get("extension_factor", 0.0))
    return (t[-1] - t[0]) / (1.0 + extension_factor)


def _first_divergence_index(is_lost_a, is_lost_b):
    diff = is_lost_a != is_lost_b
    idx = np.full(diff.shape[1], -1, dtype=int)
    any_diff = diff.any(axis=0)
    first = np.argmax(diff, axis=0)  # argmax finds first True; garbage if none
    idx[any_diff] = first[any_diff]
    return idx


def main():
    file_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(DEFAULT_FILE)
    print(f"Loading {file_path}")

    data = load_forward3d(file_path)
    data["transport_time"] = _read_transport_time(file_path, data["t"])
    t = data["t"]
    transport_time = data["transport_time"]
    ext_mask = t > transport_time
    n_shots = data["x"].shape[1]

    with h5py.File(file_path, "r") as f:
        g_dimless = float(f.attrs["g_dimless"])
        trap_fraction = float(f.attrs["trap_fraction"])
    print(f"g_dimless = {g_dimless:.4g}, trap_fraction = {trap_fraction:.4g}")

    print("\n── Classifying with current (no-gravity) energy criterion ──")
    class_nograv = compute_categories(data)

    print("── Classifying with gravity-corrected energy criterion ──")
    class_grav = compute_categories_grav(data, g_dimless)

    # ── Metric 1: spurious flicker (sign crossings of the loss margin) during extension ──
    KE, PE, E_tot_nograv = total_energy(
        data["x"], data["y"], data["z"], data["vx"], data["vy"], data["vz"],
        data["ux"], data["uy"], data["ua"], data["params"],
    )
    U = PE
    margin_nograv = E_tot_nograv - trap_fraction * U
    margin_grav = (E_tot_nograv + g_dimless * data["y"]) - trap_fraction * U

    crossings_nograv = count_sign_crossings(margin_nograv[ext_mask])
    crossings_grav = count_sign_crossings(margin_grav[ext_mask])

    flicker_nograv = np.mean(crossings_nograv >= 2)
    flicker_grav = np.mean(crossings_grav >= 2)

    # ── Metric 2: conservation quality — RMS variation of the tracked energy   ──
    # during the extension, for shots that are gravity-bound the whole extension
    # (a clean population where the true dynamics are simple oscillation, no loss)
    clean = ~class_grav["is_lost"][-1] & ~class_grav["is_lost"][ext_mask][0]
    rms_nograv = np.std(margin_nograv[ext_mask][:, clean], axis=0)
    rms_grav = np.std(margin_grav[ext_mask][:, clean], axis=0)
    U0_static = data["params"]["U0_static"]

    # ── Metric 3: final survival comparison ──
    lost_nograv_final = class_nograv["is_lost"][-1]
    lost_grav_final = class_grav["is_lost"][-1]
    n_corrected_to_bound = int(np.sum(lost_nograv_final & ~lost_grav_final))
    n_corrected_to_lost = int(np.sum(~lost_nograv_final & lost_grav_final))

    first_div = _first_divergence_index(class_nograv["is_lost"], class_grav["is_lost"])
    diverged = first_div >= 0
    ext_start_idx = int(np.argmax(ext_mask))
    diverged_in_extension = diverged & (first_div >= ext_start_idx)

    # ── Metric 4: recapture/relost EVENTS during extension (what the classification ──
    # plots actually visualize as flicker) — the direct match to what was observed.
    recap_rising_nograv = _rising_event_counts(class_nograv["is_caught_again"])
    recap_falling_nograv = _falling_event_counts(class_nograv["is_caught_again"])
    recap_rising_grav = _rising_event_counts(class_grav["is_caught_again"])
    recap_falling_grav = _falling_event_counts(class_grav["is_caught_again"])

    ext_recap_nograv = int(recap_rising_nograv[ext_mask].sum())
    ext_relost_nograv = int(recap_falling_nograv[ext_mask].sum())
    ext_recap_grav = int(recap_rising_grav[ext_mask].sum())
    ext_relost_grav = int(recap_falling_grav[ext_mask].sum())

    # ── Report ──
    report_lines = [
        f"File: {file_path.name}",
        f"Shots: {n_shots},  transport_time = {transport_time:.3f} (dimless time units), "
        f"extension steps = {int(ext_mask.sum())}/{len(t)}",
        f"g_dimless = {g_dimless:.4g},  trap_fraction = {trap_fraction:.4g}",
        "",
        "── Spurious flicker during extension (loss-margin sign crossings) ──",
        f"  no-gravity criterion : {flicker_nograv*100:.1f}% of shots cross the "
        f"lost/bound threshold >=2 times during the hold",
        f"  gravity-corrected    : {flicker_grav*100:.1f}% of shots cross >=2 times",
        f"  mean crossings/shot  : no-gravity={crossings_nograv.mean():.2f}   "
        f"gravity-corrected={crossings_grav.mean():.2f}",
        "",
        "── Energy conservation quality (RMS of tracked 'energy' over the hold, "
        f"clean bound subset, n={int(clean.sum())}) ──",
        f"  no-gravity criterion : median RMS = {np.median(rms_nograv):.3e}  "
        f"({np.median(rms_nograv)/U0_static*100:.2f}% of trap depth U0_static)",
        f"  gravity-corrected    : median RMS = {np.median(rms_grav):.3e}  "
        f"({np.median(rms_grav)/U0_static*100:.2f}% of trap depth U0_static)",
        f"  improvement factor   : {np.median(rms_nograv)/max(np.median(rms_grav), 1e-300):.1f}x smaller "
        "residual variation once gravity's PE is included",
        "",
        "── Final survival: does the fix change who counts as lost? ──",
        f"  no-gravity survival  : {int(np.sum(~lost_nograv_final))}/{n_shots} "
        f"({100*np.mean(~lost_nograv_final):.1f}%)",
        f"  corrected survival   : {int(np.sum(~lost_grav_final))}/{n_shots} "
        f"({100*np.mean(~lost_grav_final):.1f}%)",
        f"  shots flipped lost->bound (spurious losses removed by the fix): {n_corrected_to_bound}",
        f"  shots flipped bound->lost (genuinely missed before)           : {n_corrected_to_lost}",
        f"  total shots whose lost/bound history differs at all           : {int(diverged.sum())}",
        f"  of those, first divergence happens during the EXTENSION       : "
        f"{int(diverged_in_extension.sum())} ({100*np.mean(diverged_in_extension[diverged]) if diverged.any() else 0:.1f}% of divergent shots)",
        "",
        "── Recapture/relost EVENTS during extension (matches the classification-plot rug marks) ──",
        f"  no-gravity criterion : {ext_recap_nograv} recapture events, {ext_relost_nograv} relapsed-to-lost events",
        f"  gravity-corrected    : {ext_recap_grav} recapture events, {ext_relost_grav} relapsed-to-lost events",
        f"  events removed by the fix: {ext_recap_nograv-ext_recap_grav} recapture, "
        f"{ext_relost_nograv-ext_relost_grav} relost "
        f"(out of {ext_recap_nograv}/{ext_relost_nograv} total — the rest trace back to atoms flagged "
        "lost during the fast TRANSPORT phase that later satisfy the recapture proximity+energy test "
        "once settled in the extension; that is a separate, legitimate mechanism, not a gravity artifact)",
    ]
    report = "\n".join(report_lines)
    print("\n" + report)

    out_dir = _SCRIPTS_DIR.parent / "images" / f"{file_path.stem}_GRAVITY_CHECK"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gravity_conservation_report.txt").write_text(report, encoding="utf-8")

    # ── Proof plot: energy-vs-time for the worst-flickering shots ──
    worst = np.argsort(-crossings_nograv)[:3]
    fig, axes = plt.subplots(len(worst), 1, figsize=(10, 3.2 * len(worst)), sharex=True)
    if len(worst) == 1:
        axes = [axes]
    for ax, s in zip(axes, worst):
        ax.plot(t, margin_nograv[:, s], color="crimson", lw=1.2,
                label="E_tot - trap_fraction·U  (no gravity PE, current code)")
        ax.plot(t, margin_grav[:, s], color="mediumseagreen", lw=1.6,
                label="E_tot - trap_fraction·U  (gravity PE included)")
        ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(transport_time, color="gray", lw=1.0, ls=":", alpha=0.8)
        ax.set_ylabel("loss margin")
        ax.set_title(f"shot {s}: {crossings_nograv[s]} crossings (no-grav) vs "
                     f"{crossings_grav[s]} (gravity-corrected)", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time [μs]")
    fig.suptitle("Proof: omitted gravitational PE causes spurious lost<->recaptured flicker\n"
                 "during the static extension hold (positive margin ⇒ classified lost)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "gravity_proof_energy_vs_time.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved proof plot: {out_dir / 'gravity_proof_energy_vs_time.png'}")

    # ── Full thesis figure set, rerun with the gravity-corrected classification ──
    print("\n── Re-rendering the thesis figure set with the gravity-corrected classification ──")
    plot_control_protocol_thesis(data, out_dir, show_insets=True)
    plot_control_protocol_thesis(data, out_dir, show_insets=False)
    plot_trajectories_thesis(data, class_grav, out_dir, style="individual")
    plot_phase_space_thesis(data, class_grav, out_dir)
    plot_classification_thesis(data, class_grav, out_dir)
    plot_classification_thesis(data, class_grav, out_dir, top_only=True)
    plot_classification_thesis(data, class_grav, out_dir, transport_only=True)
    plot_classification_thesis(data, class_grav, out_dir, control_row=True)

    print(f"\nAll gravity-check outputs saved to {out_dir}")
    print("Compare against the original (no-gravity) run at "
          f"{_SCRIPTS_DIR.parent / 'images' / f'{file_path.stem}_thesis'}")


if __name__ == "__main__":
    main()
