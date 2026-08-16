"""
Analysis of the 2026-06-10 two-colour handoff ("stealing"/"delivering") runs.

MOPA (936 nm, AWG channel 0) carries the atom out over a min-jerk leg, hands it to
TiSaph (933 nm, channel 1) via a short amplitude crossfade at fixed position, and
TiSaph carries it back. Readout is the static MOPA pair.

@author: Bjarne Schuemann
"""

import collections
import glob
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "ExperimentalData", "20260610")
BIN, LOAD, SURV = 2, 1, 2
R_PIX = 4.0
UM_PER_PX = 32.25 / 67.274

SHOT_RE = re.compile(r"^(?P<prefix>.+?)_(?P<idx>\d+)_(?P<ts>\d{8}-\d{6})\.npy$")

MOVES = [
    ("First", 0.45, 0.45),
    ("Second", 0.65, 0.65),
    ("Third", 0.55, 0.55),
    ("Fourth", 0.35, 0.35),
    ("Fifth", 0.35, 0.45),
    ("Sixth", 0.42, 0.55),
    ("Seventh", 0.40, 0.65),
]

P_MOPA = {0.35: 7.3, 0.40: 9.5, 0.42: 10.5, 0.45: 11.8, 0.55: 17.2, 0.65: 23.0}
P_TISA = {0.35: 4.4, 0.45: 7.3, 0.55: 10.7, 0.65: 14.8}

C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def discover(folder):
    g = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(folder, "*.npy"))):
        m = SHOT_RE.match(os.path.basename(p))
        if m:
            g[m["prefix"]].append((int(m["idx"]), p))
    return {k: sorted(v) for k, v in g.items()}


G = discover(DATA)


def stack(prefix):
    return np.array(
        [np.load(p, allow_pickle=True)[()]["Images"] for _, p in G[prefix]]
    ).astype(np.int32)


def to_photons(im):
    return (im.astype(np.int32) - 200 * BIN**2) * 0.1


def peak_seeds(img, nmax, frac=0.25):
    im = ndimage.gaussian_filter(img.astype(float), 1.0)
    im = im - np.median(im)
    w, out = im.copy(), []
    yy, xx = np.mgrid[0 : im.shape[0], 0 : im.shape[1]]
    for _ in range(nmax):
        i = np.unravel_index(w.argmax(), w.shape)
        if w[i] < frac * im.max():
            break
        out.append((i[1], i[0]))
        w[((xx - i[1]) ** 2 + (yy - i[0]) ** 2) < 64] = -1e9
    return out


def centroid(img, cx, cy, r=3):
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    w = np.clip(img[m] - np.median(img), 0, None)
    return (xx[m] * w).sum() / w.sum(), (yy[m] * w).sum() / w.sum()


def locate(img, nmax):
    return sorted(centroid(img, x, y) for x, y in peak_seeds(img, nmax))


def site_masks(shape, loc, r):
    Y, X = np.mgrid[0 : shape[0], 0 : shape[1]]
    d = np.stack([(X - s[0]) ** 2 + (Y - s[1]) ** 2 for s in loc])
    own = d.argmin(0)
    return [(d[i] <= r * r) & (own == i) for i in range(len(loc))]


def roi_counts(images, masks):
    return np.array([[im[m].sum() for im in images] for m in masks])


def bimodal_pdf(k, A, m1, m2, s1, s2):
    return (1 - A) * norm.pdf(k, m1, s1) + A * norm.pdf(k, m2, s2)


def bimodal_threshold(pool):
    bins = np.arange(pool.min(), pool.max() + 2, 2) - 0.5
    ent, edg = np.histogram(pool, bins=bins, density=True)
    ctr = 0.5 * (edg[1:] + edg[:-1])
    p, _ = curve_fit(
        bimodal_pdf, ctr, ent, p0=[0.5, pool.max() / 6, pool.max() * 0.7, 5, 12],
        maxfev=40000,
    )
    A, m1, m2, s1, s2 = p
    g = np.linspace(min(m1, m2), max(m1, m2), 4000)
    d = (1 - A) * norm.pdf(g, m1, s1) - A * norm.pdf(g, m2, s2)
    return float(g[np.where(np.diff(np.sign(d)))[0][0]]), p


def wilson(k, n, z=1.0):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    dd = 1 + z * z / n
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return p, h


def minjerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


print("=" * 78)
print("GEOMETRY")
print("=" * 78)

raw = {n: stack(f"tweezerLoad2x2-{n}Move") for n, _, _ in MOVES}
ref_move = sum(raw[n][:, LOAD].std(0) + raw[n][:, SURV].std(0) for n, _, _ in MOVES)
sites = locate(ref_move, 4)
masks = site_masks(ref_move.shape, sites, R_PIX)
SRC, SPEC = 0, 1
sep_px = np.hypot(sites[1][0] - sites[0][0], sites[1][1] - sites[0][1])
print(f"scale                 {UM_PER_PX:.4f} um/px (32.25 um / 67.274 px, Mephisto ref)")
print(f"move-run sites        {[(round(a,2), round(b,2)) for a, b in sites]}")
print(f"site separation       {sep_px:.2f} px = {sep_px * UM_PER_PX:.2f} um")

nm_sites = {}
for lab in ("MOPA", "TiSaph"):
    s = stack(f"tweezerLoad2x2-NoMotion-{lab}")
    nm_sites[lab] = locate(s[:, LOAD].std(0) + s[:, SURV].std(0), 4)
    print(f"NoMotion-{lab:<7} 2x2 {[(round(a, 2), round(b, 2)) for a, b in nm_sites[lab]]}")

d_off = np.array(nm_sites["TiSaph"][0]) - np.array(nm_sites["MOPA"][0])
print(
    f"TiSaph - MOPA lattice offset  ({d_off[0]:+.2f}, {d_off[1]:+.2f}) px = "
    f"({d_off[0] * UM_PER_PX:+.2f}, {d_off[1] * UM_PER_PX:+.2f}) um"
)

gv = np.load(G["tweezerLoad2x2-FirstMove"][0][1], allow_pickle=True)[()]["globalvariables"]
v_peak = float(gv["param2"][0])
T_LEG = 500e-6
d_leg = v_peak * 1e-3 * T_LEG / 1.875 * 1e6
print(f"logged peak velocity  {v_peak:.2f} um/ms -> min-jerk leg d = {d_leg:.1f} um over {T_LEG*1e6:.0f} us")

print()
print("=" * 78)
print("DETECTION")
print("=" * 78)
cts = {
    n: (roi_counts(to_photons(raw[n][:, LOAD]), masks),
        roi_counts(to_photons(raw[n][:, SURV]), masks))
    for n, _, _ in MOVES
}
TH, fp = bimodal_threshold(np.concatenate([cts[n][0].ravel() for n, _, _ in MOVES]))
print(f"threshold {TH:.2f} photons   empty {min(fp[1], fp[2]):.1f}   filled {max(fp[1], fp[2]):.1f}")

fpos = fneg = 0
for n, _, _ in MOVES:
    L, S = cts[n]
    none = (L[SRC] <= TH) & (L[SPEC] <= TH)
    fpos += int((S[:, none] > TH).sum())
    fneg += int(none.sum()) * 2
print(f"false-positive control (SURV detections in shots with nothing loaded): {fpos}/{fneg} = {fpos/fneg:.4f}")

print()
print("=" * 78)
print("NO-MOTION BASELINE (imaging + hold only)")
print("=" * 78)
base = {}
for lab in ("MOPA", "TiSaph"):
    s = stack(f"tweezerLoad2x2-NoMotion-{lab}")
    mk = site_masks(s.shape[-2:], nm_sites[lab], R_PIX)
    L = roi_counts(to_photons(s[:, LOAD]), mk)
    S = roi_counts(to_photons(s[:, SURV]), mk)
    t, _ = bimodal_threshold(L.ravel())
    lo, su = L > t, S > t
    k, nn = int((lo & su).sum()), int(lo.sum())
    p, e = wilson(k, nn)
    base[lab] = (p, e, k, nn)
    print(f"{lab:<7} n_shots={s.shape[0]:3d}  thr={t:6.2f}  loaded={nn:3d}  survived={k:3d}  P={p:.3f}+-{e:.3f}")
b_all = (base["MOPA"][2] + base["TiSaph"][2], base["MOPA"][3] + base["TiSaph"][3])
BP, BE = wilson(*b_all)
print(f"pooled baseline P_hold = {BP:.3f} +- {BE:.3f}  ({b_all[0]}/{b_all[1]})")

print()
print("=" * 78)
print("PER-RUN RESULTS   (source = site L, spectator = site R)")
print("=" * 78)
print(f"{'run':<9}{'aM':>5}{'aT':>5}{'PM':>6}{'PT':>6}{'r':>6}{'Pmin':>6} | "
      f"{'n':>4}{'ldL':>5}{'ldR':>5} | {'onlyL':>9}{'both':>9} | {'P_surv':>14}{'->R':>6}")
rows = []
for n, aM, aT in MOVES:
    L, S = cts[n]
    lL, lR = L[SRC] > TH, L[SPEC] > TH
    sL, sR = S[SRC] > TH, S[SPEC] > TH
    onlyL, both = lL & ~lR, lL & lR
    k, nn = int((lL & sL).sum()), int(lL.sum())
    p, e = wilson(k, nn)
    PM, PT = P_MOPA[aM], P_TISA[aT]
    r = PT / PM
    pmin = PM * PT / (PM + PT)
    mis = int((lL & ~sL & sR).sum())
    rows.append(dict(name=n, aM=aM, aT=aT, PM=PM, PT=PT, r=r, pmin=pmin,
                     k=k, n=nn, p=p, e=e, nshots=L.shape[1],
                     ldrate=nn / L.shape[1], mis=mis))
    print(f"{n:<9}{aM:>5.2f}{aT:>5.2f}{PM:>6.1f}{PT:>6.1f}{r:>6.2f}{pmin:>6.2f} | "
          f"{L.shape[1]:>4}{int(lL.sum()):>5}{int(lR.sum()):>5} | "
          f"{int((onlyL & sL).sum()):>4}/{int(onlyL.sum()):<4}{int((both & sL).sum()):>4}/{int(both.sum()):<4} | "
          f"{k:>4}/{nn:<4} {p:.3f}+-{e:.3f}{mis:>6}")

print()
print("loading rate at source (data-quality check):")
for r_ in rows:
    flag = "   <-- ANOMALOUS" if r_["ldrate"] < 0.2 else ""
    print(f"  {r_['name']:<9} {r_['ldrate']:.3f} ({r_['n']}/{r_['nshots']}){flag}")

print()
print("=" * 78)
print("CONTROLLED COMPARISONS")
print("=" * 78)
by = {r_["name"]: r_ for r_ in rows}
print("A) fixed RECEIVER amplitude, vary DONOR:")
for aT, pair in [(0.45, ("Fifth", "First")), (0.55, ("Sixth", "Third")), (0.65, ("Seventh", "Second"))]:
    a, b = by[pair[0]], by[pair[1]]
    print(f"   aT={aT:.2f}:  aM={a['aM']:.2f} -> {a['p']:.3f}+-{a['e']:.3f} ({a['k']}/{a['n']})"
          f"   vs   aM={b['aM']:.2f} -> {b['p']:.3f}+-{b['e']:.3f} ({b['k']}/{b['n']})")
print("B) fixed DONOR amplitude, vary RECEIVER:")
a, b = by["Fourth"], by["Fifth"]
print(f"   aM=0.35:  aT={a['aT']:.2f} -> {a['p']:.3f}+-{a['e']:.3f} ({a['k']}/{a['n']})"
      f"   vs   aT={b['aT']:.2f} -> {b['p']:.3f}+-{b['e']:.3f} ({b['k']}/{b['n']})")

print()
print("C) pooled by depth-ratio family (SecondMove excluded, anomalous loading):")
fams = [("receiver shallower  r~0.62", ["First", "Third", "Fourth"]),
        ("depth matched       r~1.00", ["Fifth", "Sixth"]),
        ("receiver deeper     r~1.56", ["Seventh"])]
for lab, names in fams:
    k = sum(by[x]["k"] for x in names)
    nn = sum(by[x]["n"] for x in names)
    p, e = wilson(k, nn)
    print(f"   {lab}: {k:>3}/{nn:<3} = {p:.3f}+-{e:.3f}")

print()
print("D) pooled by DONOR amplitude:")
don = collections.defaultdict(lambda: [0, 0])
for r_ in rows:
    don[r_["aM"]][0] += r_["k"]
    don[r_["aM"]][1] += r_["n"]
for aM in sorted(don):
    k, nn = don[aM]
    p, e = wilson(k, nn)
    print(f"   aM={aM:.2f} (P_MOPA={P_MOPA[aM]:>4.1f} mW): {k:>3}/{nn:<3} = {p:.3f}+-{e:.3f}")

print()
print("=" * 78)
print("RF CHAIN LINEARITY  (optical power / amplitude^2)")
print("=" * 78)
for lab, tab in (("MOPA  ", P_MOPA), ("TiSaph", P_TISA)):
    ks = {a: p / a**2 for a, p in tab.items()}
    lo = ks[min(ks)]
    print(f"{lab}: " + "  ".join(f"a={a:.2f}:{ks[a]:6.2f}" for a in sorted(ks))
          + f"   compression at a_max = {(1 - ks[max(ks)] / lo) * 100:5.2f}%")

print()
print("=" * 78)
print("CROSSFADE DEPTH DIP  U_min/U_donor = r/(1+r)")
print("=" * 78)
for r_ in rows:
    print(f"  {r_['name']:<9} r={r_['r']:.2f}  U_min={r_['pmin']:5.2f} mW  "
          f"U_min/U_M={r_['pmin']/r_['PM']:.3f}  U_min/U_T={r_['pmin']/r_['PT']:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), facecolor="#fcfcfb")
for ax in axes.ravel():
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8.5)

ax = axes[0, 0]
ax.axhspan(BP - BE, BP + BE, color=C_AQUA, alpha=0.16, zorder=1)
ax.axhline(BP, color=C_AQUA, lw=2, zorder=2)
ax.text(0.655, BP + 0.022, "no-motion baseline", color=C_AQUA, fontsize=8.5, ha="right", weight="bold")
ok = [r_ for r_ in rows if r_["ldrate"] >= 0.2]
bad = [r_ for r_ in rows if r_["ldrate"] < 0.2]
ax.errorbar([r_["aM"] for r_ in ok], [r_["p"] for r_ in ok], yerr=[r_["e"] for r_ in ok],
            fmt="o", ms=8, color=C_BLUE, ecolor=C_BLUE, elinewidth=1.6, capsize=3, zorder=4,
            mec="#fcfcfb", mew=1.5)
ax.errorbar([r_["aM"] for r_ in bad], [r_["p"] for r_ in bad], yerr=[r_["e"] for r_ in bad],
            fmt="o", ms=8, mfc="#fcfcfb", color=C_BLUE, ecolor=C_BLUE, elinewidth=1.6,
            capsize=3, zorder=4, mew=1.5)
nudge = {"Fourth": (34, 6), "Fifth": (-4, -32), "Seventh": (-2, -32), "Sixth": (30, 4),
         "First": (30, 2), "Third": (0, -32), "Second": (0, 16)}
for r_ in rows:
    dx, dy = nudge[r_["name"]]
    ax.annotate(f"{r_['name']}\n$a_T$={r_['aT']:.2f}", (r_["aM"], r_["p"]),
                textcoords="offset points", xytext=(dx, dy), ha="center",
                fontsize=7.2, color=INK2)
ax.set_xlabel("donor (MOPA, ch0) amplitude $a_M$", color=INK2, fontsize=9.5)
ax.set_ylabel("P(atom recovered at source | loaded)", color=INK2, fontsize=9.5)
ax.set_title("(a)  survival is set by the donor amplitude alone", color=INK, fontsize=10.5, weight="bold", loc="left")
ax.set_ylim(-0.12, 1.08)
ax.set_xlim(0.30, 0.70)

ax = axes[0, 1]
pairs = [(0.45, "Fifth", "First"), (0.55, "Sixth", "Third"), (0.65, "Seventh", "Second")]
x = np.arange(len(pairs))
lowv = [by[p[1]]["p"] for p in pairs]
lowe = [by[p[1]]["e"] for p in pairs]
hiv = [by[p[2]]["p"] for p in pairs]
hie = [by[p[2]]["e"] for p in pairs]
ax.bar(x - 0.19, lowv, 0.36, yerr=lowe, color=C_BLUE, capsize=3, zorder=3,
       error_kw=dict(ecolor=INK2, elinewidth=1.2), label="low donor amplitude")
ax.bar(x + 0.19, hiv, 0.36, yerr=hie, color=C_ORANGE, capsize=3, zorder=3,
       error_kw=dict(ecolor=INK2, elinewidth=1.2), label="high donor amplitude")
for i, (aT, lo_, hi_) in enumerate(pairs):
    ax.text(i - 0.19, 0.03, f"$a_M$={by[lo_]['aM']:.2f}", ha="center", fontsize=7.5, color="#fcfcfb", weight="bold")
    lab_c = "#fcfcfb" if hiv[i] > 0.1 else INK2
    ax.text(i + 0.19, 0.03 if hiv[i] > 0.1 else 0.11, f"$a_M$={by[hi_]['aM']:.2f}",
            ha="center", fontsize=7.5, color=lab_c, weight="bold")
ax.axhline(BP, color=C_AQUA, lw=2, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels([f"receiver $a_T$={p[0]:.2f}\n({p[1]} vs {p[2]})" for p in pairs], fontsize=8)
ax.set_ylabel("P(recovered)", color=INK2, fontsize=9.5)
ax.set_title("(b)  at fixed receiver depth, raising the donor always hurts", color=INK, fontsize=10.5, weight="bold", loc="left")
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.set_ylim(0, 1.08)

ax = axes[1, 0]
s = np.linspace(0, 1, 400)
g = minjerk(s)
for r_, c, ls in ((by["First"], C_BLUE, "-"), (by["Fifth"], C_ORANGE, "-"), (by["Seventh"], C_AQUA, "-")):
    U = (r_["PM"] * (1 - g) ** 2 + r_["PT"] * g**2) / r_["PM"]
    ax.plot(s, U, ls, color=c, lw=2.2, zorder=3,
            label=f"{r_['name']}: $r$={r_['r']:.2f}, min={r_['r']/(1+r_['r']):.2f}")
sq = np.sqrt(1 - g) ** 2 + np.sqrt(g) ** 2
ax.plot(s, sq, "--", color=MUTED, lw=2, zorder=3, label="constant-depth (power) crossfade")
ax.set_xlabel("crossfade progress $s$   (4 $\\mu$s window)", color=INK2, fontsize=9.5)
ax.set_ylabel("total trap depth / donor depth", color=INK2, fontsize=9.5)
ax.set_title("(c)  amplitude crossfade always dips the depth to $r/(1{+}r)$", color=INK, fontsize=10.5, weight="bold", loc="left")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_ylim(0, 1.75)

ax = axes[1, 1]
for lab, tab, c in (("MOPA (ch0)", P_MOPA, C_BLUE), ("TiSaph (ch1)", P_TISA, C_ORANGE)):
    aa = sorted(tab)
    kk = [tab[a] / a**2 / (tab[aa[0]] / aa[0] ** 2) for a in aa]
    ax.plot(aa, kk, "o-", color=c, lw=2, ms=8, mec="#fcfcfb", mew=1.5, zorder=3, label=lab)
ax.axvspan(0.50, 0.70, color=C_ORANGE, alpha=0.10, zorder=1)
ax.text(0.60, 0.925, "survival collapses here", ha="center", fontsize=8, color=INK2)
ax.set_xlabel("channel amplitude $a$", color=INK2, fontsize=9.5)
ax.set_ylabel("normalised $P/a^2$  (RF-chain linearity)", color=INK2, fontsize=9.5)
ax.set_title("(d)  only the donor channel compresses", color=INK, fontsize=10.5, weight="bold", loc="left")
ax.legend(fontsize=8, frameon=False, loc="lower left")

fig.tight_layout()
out = os.path.join(HERE, "stealing_handoff.png")
fig.savefig(out, dpi=200, facecolor="#fcfcfb")
print(f"\nfigure -> {out}")
