"""
Cesium (Cs-133) dynamic polarizability plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

HARTREE_CM = 219474.6313705

D1_MATRIX_ELEMENT = 4.489
D2_MATRIX_ELEMENT = 6.324
E_6P1_CM = 11178.27
E_6P3_CM = 11732.31

ALPHA_CORE = 15.84 - 0.673


def alpha_6s(lam_nm):
    lam_nm = np.asarray(lam_nm, dtype=float)
    omega = (1e7 / lam_nm) / HARTREE_CM
    w1 = E_6P1_CM / HARTREE_CM
    w2 = E_6P3_CM / HARTREE_CM
    term = (
        D1_MATRIX_ELEMENT**2 * w1 / (w1**2 - omega**2)
        + D2_MATRIX_ELEMENT**2 * w2 / (w2**2 - omega**2)
    ) / 3.0
    return term + ALPHA_CORE


def _Ek_from_6p32_transition(lam_from_6p32_nm):
    return E_6P3_CM + 1e7 / lam_from_6p32_nm


TERMS_6P32 = [
    ("6s", 6.324, 0.0),
    ("7s", 6.48, 18535.53),
    ("5d3/2", 3.19, 14499.4),
    ("5d5/2", 9.7, 14596.84),
    ("6d3/2", 2.09, _Ek_from_6p32_transition(921.47)),
    ("6d5/2", 6.13, _Ek_from_6p32_transition(917.47)),
    ("7d3/2", 0.976, _Ek_from_6p32_transition(697.52)),
    ("7d5/2", 2.89, _Ek_from_6p32_transition(697.52)),
]
JV_6P32 = 1.5
PREFACTOR_6P32 = 2.0 / (3.0 * (2 * JV_6P32 + 1))


def alpha_6p32(lam_nm):
    lam_nm = np.asarray(lam_nm, dtype=float)
    omega = (1e7 / lam_nm) / HARTREE_CM
    total = np.zeros_like(omega)
    for _name, Dval, Ek in TERMS_6P32:
        wk = (Ek - E_6P3_CM) / HARTREE_CM
        total = total + Dval**2 * wk / (wk**2 - omega**2)
    return PREFACTOR_6P32 * total + ALPHA_CORE


TABLE_V_6S_7P1 = [
    (1172.40, 866, "14s"),
    (1189.3, 838, "12d3/2"),
    (1209.68, 807, "13s"),
    (1235.7, 774, "11d3/2"),
    (1266.4, 740, "12s"),
    (1313, 698, "10d3/2"),
    (1431, 623, "9d3/2"),
    (1535.0, 580, "10s"),
    (1727, 530, "8d3/2"),
]


def plot_magic_wavelength_figure(save_path="cs_magic_wavelength.pdf"):
    xv = np.array([p[0] for p in TABLE_V_6S_7P1])
    yv = np.array([p[1] for p in TABLE_V_6S_7P1])

    lam = np.linspace(1160, 1800, 2000)
    a6s = alpha_6s(lam)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax = axes[0]
    ax.plot(
        lam,
        a6s,
        color="#1f5fa8",
        lw=2.2,
        label=r"$\alpha(6s_{1/2})$  (computed, Eq. 2)",
    )
    ax.plot(xv, yv, color="#c0392b", lw=1.2, ls=":", alpha=0.6)
    ax.scatter(
        xv,
        yv,
        color="black",
        zorder=5,
        s=40,
        label=r"published magic $\lambda$ (Table V)",
    )
    for x, y, lbl in TABLE_V_6S_7P1:
        ax.annotate(
            f"{lbl}\n{x:.0f} nm",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            fontsize=6.8,
            ha="center",
        )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$\alpha(\omega)$ (a.u.)")
    ax.set_title(
        "Where $\\alpha(6s)$ meets $\\alpha(7p_{1/2})$:\npublished crossings (dots = exact values from paper)"
    )
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(400, 950)

    lamA, lamB = (
        1421.0,
        1538.1,
    )

    def w(lnm):
        return (1e7 / lnm) / HARTREE_CM

    wA, wB = w(lamA), w(lamB)

    def M(lnm):
        ww = w(lnm)
        return np.array([wA / (wA**2 - ww**2), wB / (wB**2 - ww**2)])

    Amat = np.array([M(1535.0), M(1480.0)])
    bvec = np.array([580, 0])
    A, B = np.linalg.solve(Amat, bvec)

    lam2 = np.linspace(lamA + 0.3, lamB - 0.3, 2000)
    w2 = w(lam2)
    branch = A * wA / (wA**2 - w2**2) + B * wB / (wB**2 - w2**2)

    ax = axes[1]
    ax.plot(
        lam2,
        branch,
        color="#c0392b",
        lw=2.2,
        label=r"$\alpha(7p_{1/2})$ — one branch (schematic)",
    )
    ax.plot(lam, a6s, color="#1f5fa8", lw=2.2, label=r"$\alpha(6s_{1/2})$")
    ax.axvline(lamA, color="gray", ls="--", lw=1)
    ax.axvline(lamB, color="gray", ls="--", lw=1)
    ax.annotate("resonance\n(pole)", (lamA, 900), fontsize=8, ha="center", color="gray")
    ax.annotate("resonance\n(pole)", (lamB, 900), fontsize=8, ha="center", color="gray")
    ax.scatter([1535.0], [580], color="black", zorder=5, s=50)
    ax.annotate(
        "magic wavelength\n1535 nm",
        (1535.0, 580),
        textcoords="offset points",
        xytext=(-70, -35),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", lw=1),
    )
    ax.set_xlim(lamA - 5, lamB + 5)
    ax.set_ylim(-1500, 1500)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$\alpha(\omega)$ (a.u.)")
    ax.set_title(
        "Schematic: true shape of ONE branch\n(pole positions illustrative, not exact)"
    )
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_ground_state_800_1400(save_path="cs_6s_polarizability_800_1400.pdf"):
    lam_D1 = 1e7 / E_6P1_CM
    lam_D2 = 1e7 / E_6P3_CM

    lam = np.linspace(800, 1400, 6000)
    eps = 0.05
    for pole in (lam_D1, lam_D2):
        lam = lam[np.abs(lam - pole) > eps]
    a6s = alpha_6s(lam)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(lam, a6s, color="#1f5fa8", lw=1.8)
    ax.axvline(lam_D1, color="gray", ls="--", lw=1)
    ax.axvline(lam_D2, color="gray", ls="--", lw=1)
    ax.annotate(
        "D2 line\n852.35 nm", (lam_D2, 3750), fontsize=8.5, ha="right", color="dimgray"
    )
    ax.annotate(
        "D1 line\n894.59 nm", (lam_D1, 3750), fontsize=8.5, ha="left", color="dimgray"
    )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-4000, 4000)
    ax.set_xlim(800, 1400)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$\alpha(6s_{1/2},\omega)$ (a.u.)")
    ax.set_title(
        "Cs ground-state ($6s_{1/2}$) dynamic polarizability, 800–1400 nm\n(Eq. 2, dominated by the D1/D2 resonances)"
    )
    ax.grid(alpha=0.3)

    ax.axvspan(800, lam_D2, color="tab:red", alpha=0.05)
    ax.axvspan(lam_D2, lam_D1, color="tab:orange", alpha=0.08)
    ax.axvspan(lam_D1, 1400, color="tab:green", alpha=0.05)
    ax.text(
        815,
        3200,
        "blue of both lines\n$\\alpha<0$ (repulsive)",
        fontsize=8,
        color="firebrick",
    )
    ax.text(
        1150,
        -2600,
        "red of both lines\n$\\alpha>0$ (attractive,\nstandard ODT regime)",
        fontsize=8,
        color="darkgreen",
        ha="center",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_933_vs_1064(save_path="cs133_933_vs_1064.pdf"):
    for lam0 in (933.0, 1064.0):
        ag, ae = alpha_6s(lam0), alpha_6p32(lam0)
        print(
            f"lambda = {lam0:7.1f} nm   alpha(6s_1/2) = {ag:8.2f} a.u.   "
            f"alpha(6p_3/2) = {ae:8.2f} a.u.   difference = {ae - ag:8.2f} a.u.  "
            f"({100 * (ae - ag) / ag:5.1f}% of alpha_6s)"
        )

    lam_magic = brentq(lambda l: alpha_6p32(l) - alpha_6s(l), 926, 934.9999)
    print("magic wavelength (D2 line, red-detuned branch):", lam_magic, "nm")

    lam = np.linspace(900, 1100, 3000)
    a6s = alpha_6s(lam)
    a6p32 = alpha_6p32(lam)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.plot(
        lam,
        a6s,
        color="#1f5fa8",
        lw=2.2,
        label=r"$\alpha(6s_{1/2})$ — Cs-133 ground state",
    )
    ax.plot(
        lam,
        a6p32,
        color="#c0392b",
        lw=2.2,
        label=r"$\alpha(6p_{3/2})$ — Cs-133 excited state",
    )

    for lam0, name, color in [
        (933.0, "933 nm\n", "darkgreen"),
        (1064.0, "1064 nm\n", "purple"),
    ]:
        ax.axvline(lam0, color=color, ls=":", lw=1.3)
        ag, ae = alpha_6s(lam0), alpha_6p32(lam0)
        ax.scatter([lam0, lam0], [ag, ae], color=color, zorder=5, s=45)
        ax.annotate(
            f"{name}\n$\\alpha_g$={ag:.0f}\n$\\alpha_e$={ae:.0f}",
            (lam0, max(ag, ae)),
            textcoords="offset points",
            xytext=(8, 10),
            fontsize=8.5,
            color=color,
            ha="left",
        )

    ax.scatter(
        [lam_magic], [alpha_6s(lam_magic)], color="black", zorder=6, s=55, marker="*"
    )
    ax.annotate(
        f"magic\n{lam_magic:.1f} nm",
        (lam_magic, alpha_6s(lam_magic)),
        textcoords="offset points",
        xytext=(-55, -38),
        fontsize=8.5,
        ha="center",
        arrowprops=dict(arrowstyle="->", lw=1),
    )

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlim(900, 1100)
    ax.set_ylim(-1500, 4500)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$\alpha(\omega)$ (a.u.)")
    ax.set_title(
        "Cs-133: ground- vs excited-state (D2 line) polarizability\n"
        "at 933 nm trap vs. non-magic 1064 nm trap"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_magic_wavelength_figure()
    plot_ground_state_800_1400()
    plot_933_vs_1064()
