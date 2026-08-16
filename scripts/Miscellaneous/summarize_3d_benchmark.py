import sys
from pathlib import Path

SCRIPTS = Path(r"c:\dev\GitHub\Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays\scripts")
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "Visualization"))
sys.path.insert(0, str(SCRIPTS / "VisualizationForThesis"))

from visualize_forward3D import load_forward3d
from classify_thesis import compute_categories

RESULTS_DIR = Path(r"c:\dev\GitHub\MasterThesisJulia\results\used_for_prep")

files = sorted(RESULTS_DIR.glob("forward3d_*_5000shots.h5"))
if not files:
    print(f"No forward3d files found in {RESULTS_DIR}")
    sys.exit(1)

print(f"Found {len(files)} 3D forward trajectory file(s)\n")

rows = []
for f in files:
    data = load_forward3d(f)
    data["params"]["trap_fraction"] = data["params"].get("trap_fraction", 0.5)
    cats = compute_categories(data)

    survived = cats["survived"][-1]
    lost = cats["lost"][-1]
    left_behind = cats["left_behind"][-1]
    recaptured = cats["recaptured"][-1]

    shots = survived.shape[0]
    n_survived = int(survived.sum())
    n_lost = int(lost.sum())
    n_left_behind = int(left_behind.sum())
    n_recaptured = int(recaptured.sum())
    total = n_survived + n_lost + n_left_behind + n_recaptured

    rows.append(dict(name=f.name, shots=shots, survived=n_survived, lost=n_lost,
                      left_behind=n_left_behind, recaptured=n_recaptured, total=total))

    print(f"=== {f.name} ===")
    print(f"  shots:       {shots}")
    print(f"  survived:    {n_survived} ({n_survived/shots*100:.2f}%)")
    print(f"  lost:        {n_lost} ({n_lost/shots*100:.2f}%)")
    print(f"  left behind: {n_left_behind} ({n_left_behind/shots*100:.2f}%)")
    print(f"  recaptured:  {n_recaptured} ({n_recaptured/shots*100:.2f}%)")
    print(f"  sum: {total} (should equal shots={shots})")
    print()

print("\n=== 3D-model benchmark summary (mutually exclusive categories) ===")
header = f"{'file':45s} {'shots':>6s} {'survived':>14s} {'lost':>14s} {'left_behind':>14s} {'recaptured':>14s} {'sum':>6s}"
print(header)
print("-" * len(header))
for r in rows:
    print(f"{r['name']:45s} {r['shots']:6d} "
          f"{r['survived']:5d} ({r['survived']/r['shots']*100:5.1f}%) "
          f"{r['lost']:5d} ({r['lost']/r['shots']*100:5.1f}%) "
          f"{r['left_behind']:5d} ({r['left_behind']/r['shots']*100:5.1f}%) "
          f"{r['recaptured']:5d} ({r['recaptured']/r['shots']*100:5.1f}%) "
          f"{r['total']:6d}")
