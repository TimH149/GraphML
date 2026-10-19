"""Shared matplotlib style for the analysis figures.

Categorical colours are assigned in this fixed order (validated for colour-vision deficiency on
adjacent pairs in bars and lines; scatter plots use at most the first three).
Text uses the ink colours, never a series colour.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#898781"          # reference marks such as "random pairs"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": NEUTRAL, "axes.labelcolor": TEXT_SECONDARY, "text.color": TEXT,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "lines.linewidth": 2, "lines.markersize": 5,
})


def style_axis(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved " + path)
