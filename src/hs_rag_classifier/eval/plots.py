"""Reliability and coverage plots (PNG) for the MLflow artifacts.

Single-series line charts on the light reference palette: series blue, 2px line, 8px markers
with a surface ring, hairline solid grid, text in ink tokens. The title names the series, so
there is no legend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def _axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    return fig, ax


def _line(ax, x, y) -> None:
    ax.plot(x, y, color=SERIES, linewidth=2, solid_joinstyle="round", solid_capstyle="round",
            marker="o", markersize=8, markerfacecolor=SERIES, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=3)  # fmt: skip


def reliability_plot(bins: list[dict], ece: float, title: str, path: Path) -> Path:
    fig, ax = _axes(f"Reliability, top-1 at 8 digits ({title}); ECE {ece:.3f}",
                    "Mean confidence (15 equal-mass bins)", "Accuracy at 8 digits")  # fmt: skip
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, zorder=2)
    ax.annotate("perfect calibration", (0.62, 0.55), color=TEXT_SECONDARY, fontsize=8,
                rotation=37)  # fmt: skip
    _line(ax, [b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def coverage_plot(curve: list[dict], headline_rate: float, title: str, path: Path) -> Path:
    fig, ax = _axes(f"Accuracy of auto-accepted items at 8 digits ({title})",
                    "Share sent to review", "Accuracy at 8 digits")  # fmt: skip
    pts = [
        (c["actual_review_rate"], c["accuracy"]) for c in curve if c["accuracy"] == c["accuracy"]
    ]
    _line(ax, [x for x, _ in pts], [y for _, y in pts])
    head = min(curve, key=lambda c: abs(c["review_rate"] - headline_rate))
    if head["accuracy"] == head["accuracy"]:
        ax.annotate(f"{head['accuracy']:.3f} at {head['review_rate']:.0%} review",
                    (head["actual_review_rate"], head["accuracy"]), xytext=(8, -14),
                    textcoords="offset points", color=TEXT_PRIMARY, fontsize=8)  # fmt: skip
    ax.set_xlim(-0.01, max(0.51, max((x for x, _ in pts), default=0.5) + 0.01))
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
