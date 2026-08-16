import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Visualization"))
from utils3d import compute_loss_mask, potential3d, total_energy  # type: ignore


def transport_unit_vector(params):
    dx = params["x_stop"] - params["x_start"]
    dy = params["y_stop"] - params["y_start"]
    L = np.hypot(dx, dy)
    if L == 0:
        return 1.0, 0.0, 0.0
    return dx / L, dy / L, L


def compute_left_behind_mask(x, y, ux, uy, params):
    ux_hat, uy_hat, _ = transport_unit_vector(params)
    w_aux = params["w_aux"]
    n_steps, n_shots = x.shape
    is_left_behind = np.zeros((n_steps, n_shots), dtype=bool)

    for j in range(n_steps):
        behind = (ux[j] - x[j]) * ux_hat + (uy[j] - y[j]) * uy_hat
        newly_left_behind = behind > 2.0 * w_aux
        if j == 0:
            is_left_behind[j] = newly_left_behind
        else:
            is_left_behind[j] = is_left_behind[j - 1] | newly_left_behind

    return is_left_behind


def compute_recaptured_mask(x, y, is_lost, E_tot, U, ux, uy, params, trap_fraction):
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


def compute_categories(data):
    x, y, z = data["x"], data["y"], data["z"]
    vx, vy, vz = data["vx"], data["vy"], data["vz"]
    ux, uy, ua = data["ux"], data["uy"], data["ua"]
    t = data["t"]
    params = data["params"]
    trap_fraction = params["trap_fraction"]
    g_dimless = params.get("g_dimless", 0.0)

    is_lost = compute_loss_mask(
        x, y, z, vx, vy, vz, t, ux, uy, ua, params, trap_fraction, g_dimless
    )
    KE, PE, E_tot = total_energy(x, y, z, vx, vy, vz, ux, uy, ua, params, g_dimless)

    n_steps, n_shots = x.shape
    U = np.zeros((n_steps, n_shots))
    for j in range(n_steps):
        U[j] = potential3d(
            x[j],
            y[j],
            z[j],
            ux[j],
            uy[j],
            ua[j],
            params["x_start"],
            params["y_start"],
            params["x_stop"],
            params["y_stop"],
            params["w"],
            params["w_aux"],
            params["zR"],
            params["zR_aux"],
            params["U0_static"],
            params["U0_aux_max"],
        )

    is_left_behind = compute_left_behind_mask(x, y, ux, uy, params)
    is_caught_again = compute_recaptured_mask(
        x, y, is_lost, E_tot, U, ux, uy, params, trap_fraction
    )

    recaptured = is_caught_again
    lost = is_lost & ~is_caught_again
    left_behind = is_left_behind & ~is_lost
    survived = ~is_lost & ~is_left_behind

    return dict(
        survived=survived,
        left_behind=left_behind,
        recaptured=recaptured,
        lost=lost,
        is_lost=is_lost,
        is_left_behind=is_left_behind,
        is_caught_again=is_caught_again,
        E_tot=E_tot,
        U=U,
        KE=KE,
    )


def category_of(categories, t_idx):
    n_shots = categories["survived"].shape[1]
    labels = np.empty(n_shots, dtype=object)
    labels[categories["survived"][t_idx]] = "survived"
    labels[categories["left_behind"][t_idx]] = "left_behind"
    labels[categories["recaptured"][t_idx]] = "recaptured"
    labels[categories["lost"][t_idx]] = "lost"
    return labels
