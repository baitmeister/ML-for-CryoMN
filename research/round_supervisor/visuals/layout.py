"""Shared figure typography, plotting frame, and schematic marks."""

from dataclasses import dataclass
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D


INK, TEAL, BLUE, ORANGE, GREY = "#172f3c", "#08766b", "#3c62a2", "#ad4c25", "#526977"
PALE, MINT, SKY, PEACH, GRID, PAPER = "#edf1f2", "#d9eeea", "#e3ecf8", "#fae5d8", "#dbe2e4", "#fdfcf9"
PURPLE = "#72588c"
SECTIONS = ["How the supervisor works", "What the forecasts recovered", "What the lab and the replay established", "The Round 20 decision", "How the lab measures the endpoints"]
SECTION_COLORS = [TEAL, BLUE, ORANGE, PURPLE, "#38666e"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
                     "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.facecolor": PAPER, "figure.facecolor": PAPER, "pdf.fonttype": 42,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})


@dataclass(frozen=True)
class Plate:
    slug: str
    chapter: int
    title: str
    subtitle: str
    takeaway: str
    source: str
    draw: object
    kind: str = "Plot"


def page(plate, index, total):
    fig = plt.figure(figsize=(13.2, 7.5))
    color = SECTION_COLORS[plate.chapter]
    fig.text(.055, .943, SECTIONS[plate.chapter].upper(), fontsize=10, color=color, weight="bold", va="center")
    fig.text(.947, .943, f"{plate.kind.upper()}   {index:02d} / {total:02d}", ha="right", fontsize=9.5, color=GREY, va="center")
    def line(x, y, value, size, **style):
        item = fig.text(x, y, value, fontsize=size, **style)
        room = (.948-x)*fig.bbox.width
        width = item.get_window_extent(fig.canvas.get_renderer()).width
        if width > room:
            item.set_fontsize(size*room/width)
    line(.055, .852, plate.title, 25, weight="bold", va="baseline")
    line(.056, .791, plate.subtitle, 12.2, color=GREY, va="center")
    fig.add_artist(Line2D([.055, .946], [.15, .15], transform=fig.transFigure, color=GRID, lw=1))
    line(.056, .10, plate.takeaway, 12.2, color=INK, va="center")
    line(.056, .033, plate.source, 8.5, color=GREY, va="center")
    return fig


def compact_recipe(recipe):
    return re.sub(r"\d+(?:\.\d+)?", lambda m: f"{float(m.group()):.3g}", recipe)


def ax(fig, box=(.087, .257, .828, .445), grid="y"):
    a = fig.add_axes(box)
    a.set_axisbelow(True)
    if grid:
        a.grid(axis=grid, color=GRID, lw=.75)
    a.tick_params(length=0, pad=6)
    a.xaxis.labelpad = 8
    a.yaxis.labelpad = 6
    return a


def canvas(fig):
    a = fig.add_axes([.055, .193, .89, .545], frameon=False, xlim=(-.45, 100), ylim=(0, 100))
    a.set_axis_off()
    return a


def node(a, x, y, w, h, title, body="", *, color=TEAL, face=MINT, title_size=13, body_size=11.5):
    a.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.14,rounding_size=1.3",
                                       fc=face, ec=color, lw=1.1, zorder=2))
    if body:
        a.text(x+1.35, y+h-3.4, title, ha="left", va="top", fontsize=title_size, fontweight="bold", color=color, zorder=3)
        a.text(x+1.35, y+h-12.2, body, ha="left", va="top", fontsize=body_size, linespacing=1.28, zorder=3)
    else:
        a.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=color, zorder=3)


def arrow(a, points, *, color=GREY, dashed=False):
    for start, end in zip(points[:-2], points[1:-1]):
        a.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=1.45, ls="--" if dashed else "-", zorder=1)
    a.annotate("", xy=points[-1], xytext=points[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=1.45,
               mutation_scale=13, shrinkA=0, shrinkB=1, linestyle="--" if dashed else "-"), zorder=1)


def round_ticks(a):
    a.set_xticks(range(13, 20), [f"R{n}" for n in range(13, 20)])
    a.set_xlim(12.65, 19.35)


def key(color, label, marker="o", line="none", face=None):
    return Line2D([], [], label=label, marker=marker, ms=7, color=color, mfc=face or color,
                  mec=color, linestyle=line, lw=1.7)
