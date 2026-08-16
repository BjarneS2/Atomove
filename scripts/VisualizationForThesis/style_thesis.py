import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Visualization"))
from utils3d import COLOR_ALIVE, COLOR_AUX, COLOR_LOST, COLOR_STATIC  # noqa: E402

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.linewidth"] = 1.0

COLOR_LEFT_BEHIND = "orange"
COLOR_RECAPTURED = "dodgerblue"

CATEGORY_COLORS = {
    "survived": COLOR_ALIVE,
    "left_behind": COLOR_LEFT_BEHIND,
    "recaptured": COLOR_RECAPTURED,
    "lost": COLOR_LOST,
}

CATEGORY_MARKERS = {
    "survived": "o",
    "left_behind": "s",
    "recaptured": "^",
    "lost": "x",
}

CATEGORY_LABELS = {
    "survived": "Survived",
    "left_behind": "Left behind",
    "recaptured": "Recaptured",
    "lost": "Lost",
}

CATEGORY_ORDER = ["survived", "left_behind", "recaptured", "lost"]


def style_axes(ax):
    ax.grid(alpha=0.28, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig_thesis(fig, path):
    path = Path(path)
    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path.name} / {pdf_path.name}")
