"""Rendering helpers for tissue configurations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from .topology import Tissue, min_image


def draw_tissue(tissue: Tissue, ax=None, color_by="shape", cmap="viridis",
                vmin=None, vmax=None, lw=0.8, edgecolor="k", shear=0.0):
    """Draw all cells as (periodically unwrapped) polygons."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    L = tissue.L
    A, P = tissue.all_areas_perimeters()
    if color_by == "shape":
        vals = P / np.sqrt(A)
        vmin = 3.6 if vmin is None else vmin
        vmax = 4.6 if vmax is None else vmax
    elif color_by == "area":
        vals = A
    elif color_by == "z":
        vals = tissue.neighbor_number()
    else:
        vals = np.zeros(tissue.n_cells)

    patches, cvals = [], []
    # tile in 3x3 so cells crossing the boundary render fully
    for ci in range(tissue.n_cells):
        poly = tissue.cell_polygon(ci)
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                shift = np.array([sx * L + shear * sy * L, sy * L])
                pp = poly + shift
                # keep only tiles overlapping the primary box
                if (pp[:, 0].max() < -0.05 * L or pp[:, 0].min() > 1.05 * L or
                        pp[:, 1].max() < -0.05 * L or pp[:, 1].min() > 1.05 * L):
                    continue
                patches.append(Polygon(pp, closed=True))
                cvals.append(vals[ci])
    pc = PatchCollection(patches, cmap=cmap, edgecolor=edgecolor, linewidths=lw)
    pc.set_array(np.array(cvals))
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax, pc
