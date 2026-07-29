"""Movie of the faithful active-foam dynamics (curved-edge foam model).

Unlike make_movie.py (confluent surrogate), this renders the full
`FoamTissue`: curved cell contacts (intermediate vertices), extracellular
spaces (red background), and the complete transition machinery
(T1 / T2 / T2-reverse / T3-reverse / T4-adjacent / T4-non-adjacent) firing as
the tissue fluidizes.  Cells are coloured by shape factor s = P/sqrt(A)
(cf. Fig 4h).  Writes figures/active_foam_full_movie.gif.
"""
import os, sys, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.topology import build_periodic_voronoi
from activefoam.foam_full import FoamTissue

# ---- parameters -----------------------------------------------------------
W = 0.3              # W/T0 : partial adhesion -> extracellular spaces persist
DT = 1.0             # Delta T / T0  (active, fluidising regime)
RHO = 0.9            # cell density < 1 keeps films/spaces open (phi ~ 0.9)
SEED = 4
FRAMES = 70
STEPS_PER_FRAME = 14         # short total run: stay in the distributed-space
TRANS_EVERY = 14            # regime (spaces coarsen over long times, a model limit)
SPACE_COLOR = "#d1352b"     # extracellular space (red), as in Fig 2a
CMAP = plt.get_cmap("YlGnBu_r")
S_LO, S_HI = 3.7, 4.8       # shape-factor colour range


def draw(ft, ax):
    L = ft.bs
    ax.clear()
    ax.set_facecolor(SPACE_COLOR)                 # red = extracellular space
    s_all = np.full(ft.nFa, np.nan)
    for ci in range(ft.nFa):
        A = ft.f_area[ci]
        if A > 1e-9:
            s_all[ci] = ft._perimeter(ci) / np.sqrt(A)
    pats, cols = [], []
    for ci in range(ft.nFa):
        pts = ft._cell_polyline(ci)
        s = s_all[ci]
        c = CMAP((np.clip(s, S_LO, S_HI) - S_LO) / (S_HI - S_LO)) if np.isfinite(s) else "#c9d3e0"
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                pp = pts + np.array([sx * L, sy * L])
                if (pp[:, 0].max() < -0.05 * L or pp[:, 0].min() > 1.05 * L or
                        pp[:, 1].max() < -0.05 * L or pp[:, 1].min() > 1.05 * L):
                    continue
                pats.append(Polygon(pp, closed=True)); cols.append(c)
    ax.add_collection(PatchCollection(pats, facecolor=cols, edgecolor="k",
                                      lw=0.6, zorder=2))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    os.makedirs("figures", exist_ok=True)
    T = build_periodic_voronoi(6, np.random.default_rng(100 + SEED))
    ft = FoamTissue(T, w=W, rho=RHO, seed=SEED)
    for _ in range(150):                          # settle confluent
        ft.step(mu=0.0)
    for v in range(len(ft.vpos)):                 # open extracellular spaces
        ft.t2_reverse(v)
    ft._update_faces()
    for it in range(30):                          # brief settle (keep spaces open)
        ft.step(mu=DT)
        if it % TRANS_EVERY == 0:
            ft.do_transitions(mu=DT)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    frames = []
    for f in range(FRAMES):
        for it in range(STEPS_PER_FRAME):
            ft.step(mu=DT)
            if it % TRANS_EVERY == 0:
                ft.do_transitions(mu=DT)
        draw(ft, ax)
        ax.set_title(f"active foam  |  $W/T_0$={W}, $\\Delta T/T_0$={DT}   "
                     f"$\\phi$={ft.volume_fraction():.2f}", fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90); buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
        if f % 15 == 0:
            print(f"frame {f}/{FRAMES}  phi={ft.volume_fraction():.3f} "
                  f"V={len(ft.vpos)}", flush=True)

    out = "figures/active_foam_full_movie.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=70, loop=0, optimize=True)
    print("wrote", out, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
