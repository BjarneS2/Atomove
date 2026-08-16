"""
Residual-energy and phase-space analysis for 3D single-atom transport.

Usage:
    python residual_energy_3d.py
    python residual_energy_3d.py --protocol path/to/control3d_*.h5
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

SEED = 101
kB, M_CS, G_SI = 1.380649e-23, 2.20695e-25, 9.81
W0_SI, T0_SI = 1e-6, 1e-6
V0 = W0_SI / T0_SI
E0 = M_CS * V0**2
G_DIMLESS = G_SI * T0_SI**2 / W0_SI
UK = E0 / kB * 1e6

_W1 = 1.0 / (2.0 - 2.0 ** (1 / 3))
_W0 = -(2.0 ** (1 / 3)) * _W1
YC = np.array([_W1 / 2, (_W0 + _W1) / 2, (_W0 + _W1) / 2, _W1 / 2])
YD = np.array([_W1, _W0, _W1, 0.0])


@dataclass
class Params3D:
    w: float = 1.2
    w_aux_factor: float = 1.0
    zR: float = 8.0
    zR_aux: float = 6.0
    x_start: float = 0.0
    y_start: float = 0.0
    x_stop: float = 8 * 4.6
    y_stop: float = 0.0
    U0_static: float = kB * 287e-6 / E0
    U0_aux_max: float = 2 * kB * 287e-6 / E0
    T_atom: float = 4e-6
    starting_trap_fraction: float = 0.8
    trap_fraction: float = 0.0
    z_aux_offset: float = 0.0

    @property
    def w_aux(self):
        return self.w * self.w_aux_factor


def _beam_U(x, y, z, cx, cy, cz, U0, w, zR):
    xi = z - cz
    wxi2 = w**2 * (1.0 + (xi / zR) ** 2)
    rho2 = (x - cx) ** 2 + (y - cy) ** 2
    return -U0 * (w**2 / wxi2) * np.exp(-2.0 * rho2 / wxi2)


def potential(x, y, z, ux, uy, ua, p: Params3D):
    return (
        _beam_U(x, y, z, p.x_start, p.y_start, 0.0, p.U0_static, p.w, p.zR)
        + _beam_U(x, y, z, p.x_stop, p.y_stop, 0.0, p.U0_static, p.w, p.zR)
        + _beam_U(x, y, z, ux, uy, p.z_aux_offset, ua * p.U0_aux_max, p.w_aux, p.zR_aux)
    )


def _beam_F(x, y, z, cx, cy, cz, U0, w, zR):
    xi, dx, dy = z - cz, x - cx, y - cy
    rho2 = dx**2 + dy**2
    wxi2 = w**2 * (1.0 + (xi / zR) ** 2)
    a = w**2 / wxi2
    f = a * np.exp(-2.0 * rho2 / wxi2)
    Fx = -4.0 * U0 * dx / wxi2 * f
    Fy = -4.0 * U0 * dy / wxi2 * f
    Fz = U0 * f * a * (xi / zR**2) * (4.0 * rho2 / wxi2 - 2.0)
    return Fx, Fy, Fz


def forces(x, y, z, ux, uy, ua, p: Params3D):
    F1 = _beam_F(x, y, z, p.x_start, p.y_start, 0.0, p.U0_static, p.w, p.zR)
    F2 = _beam_F(x, y, z, p.x_stop, p.y_stop, 0.0, p.U0_static, p.w, p.zR)
    Fa = _beam_F(x, y, z, ux, uy, p.z_aux_offset, ua * p.U0_aux_max, p.w_aux, p.zR_aux)
    return (
        F1[0] + F2[0] + Fa[0],
        F1[1] + F2[1] + Fa[1] - G_DIMLESS,
        F1[2] + F2[2] + Fa[2],
    )


def sample_thermal(p: Params3D, n_atoms: int, rng: np.random.Generator):
    Tdim = kB * p.T_atom / E0
    om_r2 = 4.0 * p.U0_static / p.w**2
    om_z2 = 2.0 * p.U0_static / p.zR**2
    sr, sz, sv = np.sqrt(Tdim / om_r2), np.sqrt(Tdim / om_z2), np.sqrt(Tdim)

    out = np.empty((6, 0))
    while out.shape[1] < n_atoms:
        m = 2 * (n_atoms - out.shape[1]) + 100
        x = p.x_start + rng.normal(0, sr, m)
        y = p.y_start + rng.normal(0, sr, m)
        z = rng.normal(0, sz, m)
        v = rng.normal(0, sv, (3, m))
        U = potential(x, y, z, p.x_start, p.y_start, 0.0, p)
        E = U + 0.5 * (v**2).sum(axis=0)
        keep = E < p.starting_trap_fraction * U
        out = np.hstack([out, np.vstack([x, y, z, v])[:, keep]])
    return out[:, :n_atoms]


def _ramp(s):
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(s, 0, 1)))


def _minjerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _minjerk_dd(s, T):
    s = np.clip(s, 0, 1)
    return (60 * s - 180 * s**2 + 120 * s**3) / T**2


def make_protocol(kind, p: Params3D, T_move, dt=0.05, t_ramp=20.0, hold=None):
    hold = 2 * T_move if hold is None else hold
    T_tot = 2 * t_ramp + T_move + hold
    t = np.arange(0.0, T_tot + dt, dt)
    s = np.clip((t - t_ramp) / T_move, 0, 1)
    d, dy = p.x_stop - p.x_start, p.y_stop - p.y_start

    if kind == "linear":
        shape = s
    elif kind == "minjerk":
        shape = _minjerk(s)
    elif kind == "sta":
        om2 = 4.0 * (p.U0_aux_max + p.U0_static) / p.w_aux**2
        shape = _minjerk(s) + _minjerk_dd(s, T_move) / om2
    else:
        raise ValueError(kind)

    ux = p.x_start + d * shape
    uy = p.y_start + dy * shape
    ua = _ramp(t / t_ramp) - _ramp((t - t_ramp - T_move) / t_ramp)
    return dict(
        t=t,
        ux=ux,
        uy=uy,
        ua=ua,
        dt=dt,
        t_hold_start=2 * t_ramp + T_move,
        label=f"{kind} (T={T_move:g} µs)",
    )


def load_protocol(path, p: Params3D, dt=0.05, hold=None, T_move_hint=100.0):
    path = Path(path)
    if path.suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as f:
            t, ux, uy, ua = (f[k][:] for k in ("t", "ux", "uy", "ua"))
    elif path.suffix == ".npz":
        d = np.load(path)
        t, ux, uy, ua = d["t"], d["ux"], d["uy"], d["ua"]
    else:
        t, ux, uy, ua = np.loadtxt(path, delimiter=",", unpack=True)

    hold = 2 * (t[-1] - t[0]) if hold is None else hold
    tn = np.arange(t[0], t[-1] + hold + dt, dt)
    f = lambda a, fill: np.interp(tn, t, a, right=fill)
    return dict(
        t=tn,
        ux=f(ux, ux[-1]),
        uy=f(uy, uy[-1]),
        ua=f(ua, 0.0),
        dt=dt,
        t_hold_start=t[-1],
        label=path.stem,
    )


def simulate(proto, p: Params3D, init, n_energy_samples=200):
    t, ux, uy, ua = proto["t"], proto["ux"], proto["uy"], proto["ua"]
    dt = proto["dt"]
    n = len(t)
    x, y, z, vx, vy, vz = (c.copy() for c in init)
    lost = np.zeros(x.shape, bool)

    e_stride = max(1, n // n_energy_samples)
    e_t, e_mean = [], []

    def u_at(tau):
        j = min(int(tau / dt), n - 1)
        f = tau / dt - j
        if j >= n - 1:
            return ux[-1], uy[-1], ua[-1]
        return (
            ux[j] + f * (ux[j + 1] - ux[j]),
            uy[j] + f * (uy[j + 1] - uy[j]),
            ua[j] + f * (ua[j + 1] - ua[j]),
        )

    for j in range(n - 1):
        tau = t[j]
        for c, d_ in zip(YC, YD):
            x += c * dt * vx
            y += c * dt * vy
            z += c * dt * vz
            tau += c * dt
            if d_ != 0.0:
                cx, cy, ca = u_at(tau)
                Fx, Fy, Fz = forces(x, y, z, cx, cy, ca, p)
                vx += d_ * dt * Fx
                vy += d_ * dt * Fy
                vz += d_ * dt * Fz

        U = potential(x, y, z, ux[j + 1], uy[j + 1], ua[j + 1], p)
        Etot = U + 0.5 * (vx**2 + vy**2 + vz**2)
        lost |= Etot > p.trap_fraction * U

        if j % e_stride == 0:
            e_t.append(t[j + 1])
            e_mean.append(Etot.mean())

    return (x, y, z, vx, vy, vz), lost, (np.array(e_t), np.array(e_mean))


def energy_drift_during_hold(proto, e_t, e_mean, p: Params3D):
    m = e_t >= proto["t_hold_start"] + 1.0
    if m.sum() < 2:
        return np.nan
    Eh = e_mean[m] * UK
    return np.max(np.abs(Eh - Eh[0]))


def energy_above_final_min(state, p: Params3D):
    x, y, z, vx, vy, vz = state
    U = potential(x, y, z, p.x_stop, p.y_stop, 0.0, p)
    Umin = potential(p.x_stop, p.y_stop, 0.0, p.x_stop, p.y_stop, 0.0, p)
    return 0.5 * (vx**2 + vy**2 + vz**2) + U - Umin, -Umin


def energy_above_start_min(state, p: Params3D):
    x, y, z, vx, vy, vz = state
    U = potential(x, y, z, p.x_start, p.y_start, 0.0, p)
    Umin = potential(p.x_start, p.y_start, 0.0, p.x_start, p.y_start, 0.0, p)
    return 0.5 * (vx**2 + vy**2 + vz**2) + U - Umin


def phase_space_volumes(state, cx, cy):
    x, y, z, vx, vy, vz = state
    dq = np.vstack([x - cx, y - cy, z, vx, vy, vz])
    C = np.cov(dq)
    ax = [np.sqrt(np.linalg.det(C[np.ix_([i, i + 3], [i, i + 3])])) for i in range(3)]
    return np.array(ax), np.sqrt(np.linalg.det(C))


def analyze(proto, p: Params3D, init):
    final, lost, (e_t, e_mean) = simulate(proto, p, init)
    delivered = (np.abs(final[0] - p.x_stop) < 1.5 * p.w) & (
        np.abs(final[1] - p.y_stop) < 1.5 * p.w
    )
    ok = ~lost & delivered
    if ok.sum() < 10:
        print(f"{proto['label']}: only {ok.sum()} delivered survivors")

    E_start = energy_above_start_min(init, p)
    E_final, depth_final = energy_above_final_min(final, p)
    dE = E_final - E_start

    V0_ax, V0_6d = phase_space_volumes(tuple(c[ok] for c in init), p.x_start, p.y_start)
    V1_ax, V1_6d = phase_space_volumes(tuple(c[ok] for c in final), p.x_stop, p.y_stop)

    Ti = p.T_atom * 1e6
    return dict(
        label=proto["label"],
        survival=(~lost).mean(),
        delivered=ok.mean(),
        dE_ok_uK=dE[ok] * UK,
        dE_lost_uK=dE[lost] * UK,
        E_final_ok_uK=E_final[ok] * UK,
        E_final_lost_uK=E_final[lost] * UK,
        E_thresh_uK=(1.0 - p.trap_fraction) * depth_final * UK,
        mean=dE[ok].mean() * UK,
        std=dE[ok].std() * UK,
        V_ratio_ax=V1_ax / V0_ax,
        V_ratio_6d=V1_6d / V0_6d,
        T_eff_ax=Ti * V1_ax / V0_ax,
        drift_uK=energy_drift_during_hold(proto, e_t, e_mean, p),
        e_t=e_t,
        e_mean_uK=e_mean * UK,
        final=tuple(c[ok] for c in final),
        init=tuple(c[ok] for c in init),
    )


def plot_results(protos, results, p: Params3D, outfile="residual_energy_3d.png"):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    colors = plt.cm.tab10(np.arange(len(results)))

    ax = axes[0, 0]
    for pr, c in zip(protos, colors):
        ax.plot(pr["t"], pr["ux"], color=c, label=pr["label"])
        ax.plot(pr["t"], pr["ua"] * p.x_stop, color=c, ls=":", alpha=0.5)
    ax.set(
        xlabel="t [µs]",
        ylabel="uₓ [µm]  (dotted: ua, scaled)",
        title="Control protocols",
    )
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for r, c in zip(results, colors):
        m = r["e_t"] >= 0
        ax.plot(r["e_t"][m], r["e_mean_uK"][m], color=c)
        ax.axvline(protos[0]["t_hold_start"], color="grey", ls="--", lw=0.8)
    ax.set(
        xlabel="t [µs]",
        ylabel="⟨E_tot⟩ all atoms [µK·k_B]",
        title="Energy conservation check\n(flat after dashed line ⇒ integrator OK)",
    )

    ax = axes[0, 2]
    for r, c in zip(results, colors):
        bins = np.linspace(0, 2.2 * r["E_thresh_uK"], 70)
        ax.hist(
            np.clip(r["E_final_ok_uK"], bins[0], bins[-1]),
            bins=bins,
            histtype="step",
            color=c,
            label=f"{r['label']} survived",
        )
        ax.hist(
            np.clip(r["E_final_lost_uK"], bins[0], bins[-1]),
            bins=bins,
            histtype="stepfilled",
            color=c,
            alpha=0.25,
            label=f"{r['label']} lost",
        )
    ax.axvline(
        results[0]["E_thresh_uK"],
        color="red",
        ls="--",
        label=f"(1−f)·depth = {results[0]['E_thresh_uK']:.0f} µK",
    )
    ax.set(
        xlabel="final E above trap minimum [µK·k_B]",
        ylabel="counts",
        title=f"Loss threshold (trap_fraction f = {p.trap_fraction})",
    )
    ax.legend(fontsize=7)

    ax = axes[0, 3]
    bp = ax.boxplot(
        [r["dE_ok_uK"] for r in results],
        patch_artist=True,
        flierprops=dict(
            marker="o",
            markerfacecolor="red",
            markeredgecolor="none",
            markersize=3,
            alpha=0.6,
        ),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for i, r in enumerate(results):
        ax.annotate(
            f"⟨ΔE⟩={r['mean']:.1f}±{r['std']:.1f}\ndeliv {100 * r['delivered']:.1f}%",
            xy=(i + 1, np.median(r["dE_ok_uK"])),
            xytext=(i + 1.28, r["mean"]),
            fontsize=7,
            va="center",
        )
    ax.set_xticks(
        np.arange(1, len(results) + 1),
        [r["label"] for r in results],
        rotation=15,
        fontsize=8,
    )
    ax.set(
        ylabel="ΔE per delivered atom [µK·k_B]",
        title="Residual energy (outliers in red)",
    )

    for k, (qi, vi, name) in enumerate([(0, 3, "x"), (1, 4, "y"), (2, 5, "z")]):
        ax = axes[1, k]
        for r, c in zip(results, colors):
            cq = [p.x_stop, p.y_stop, 0.0][k]
            ax.scatter(r["final"][qi] - cq, r["final"][vi], s=2, alpha=0.25, color=c)
        r0 = results[0]
        cq0 = [p.x_start, p.y_start, 0.0][k]
        ax.scatter(
            r0["init"][qi] - cq0,
            r0["init"][vi],
            s=2,
            alpha=0.25,
            color="k",
            label="initial (shared)",
        )
        ratios = ", ".join(f"{r['V_ratio_ax'][k]:.2f}" for r in results)
        ax.set(
            xlabel=f"{name} − {name}_trap [µm]",
            ylabel=f"v_{name} [m/s]",
            title=f"Phase space {name}  (V ratio: {ratios})",
        )
        ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 3]
    for r, c in zip(results, colors):
        both = np.concatenate([r["dE_ok_uK"], r["dE_lost_uK"]])
        lo, hi = np.percentile(both, [0.5, 99.5])
        bins = np.linspace(lo, hi, 70)
        ax.hist(
            np.clip(r["dE_ok_uK"], lo, hi),
            bins=bins,
            histtype="step",
            color=c,
            density=True,
        )
        ax.hist(
            np.clip(r["dE_lost_uK"], lo, hi),
            bins=bins,
            histtype="stepfilled",
            color=c,
            alpha=0.25,
            density=True,
        )
    ax.set(
        xlabel="ΔE [µK·k_B]  (filled: lost atoms)",
        ylabel="density",
        title="ΔE — survived vs lost",
    )

    fig.suptitle(
        f"3D forward Monte Carlo (Yoshida-4) — shared initial ensemble, "
        f"{len(results[0]['init'][0])} delivered survivors shown, "
        f"T_atom={p.T_atom * 1e6:g} µK, seed={SEED}"
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    print(f"saved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--atoms", type=int, default=10000)
    ap.add_argument("--T-move", type=float, default=370.0)
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args()

    p = Params3D()
    rng = np.random.default_rng(SEED)
    init = tuple(sample_thermal(p, args.atoms, rng))

    protos = [make_protocol(k, p, args.T_move, args.dt) for k in ("minjerk", "sta")]
    if args.protocol:
        protos.append(load_protocol(args.protocol, p, args.dt))

    results = []
    for pr in protos:
        r = analyze(pr, p, init)
        results.append(r)
        Tax = ", ".join(f"{v:.1f}" for v in r["T_eff_ax"])
        print(
            f"{r['label']:<24} surv={100 * r['survival']:5.1f}% (deliv {100 * r['delivered']:5.1f}%)  "
            f"ΔE = {r['mean']:7.2f} ± {r['std']:6.2f} µK·kB   "
            f"V/V0 (x,y,z) = {np.round(r['V_ratio_ax'], 2)}  6D = {r['V_ratio_6d']:.2f}   "
            f"T_eff = [{Tax}] µK   E-drift(hold) = {r['drift_uK']:.3f} µK"
        )

    plot_results(protos, results, p)


if __name__ == "__main__":
    main()
