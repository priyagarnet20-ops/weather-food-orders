"""Shared matplotlib style for the project figures (light surface, thin marks, recessive grid).

Colours follow the dataviz reference palette: categorical slot 1 blue, slot 2 orange, slot 3 aqua,
neutral gray for baselines and context. Text always uses ink colours, never a series colour.
"""
import matplotlib as mpl

C = dict(
    bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0",
    blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", gray="#8a8983", light="#c9c8c2",
)


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": C["bg"], "axes.facecolor": C["bg"], "savefig.facecolor": C["bg"],
        "axes.edgecolor": C["light"], "axes.labelcolor": C["ink2"], "axes.titlecolor": C["ink"],
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": C["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "text.color": C["ink"],
        "font.size": 10, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "figure.dpi": 110,
    })
