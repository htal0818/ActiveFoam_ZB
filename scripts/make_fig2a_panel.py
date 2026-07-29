import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.topology import build_periodic_voronoi
from activefoam.foam_full import FoamTissue

def draw(ft, ax):
    L=ft.bs; pats=[]
    for ci in range(ft.nFa):
        pts=ft._cell_polyline(ci)
        for sx in(-1,0,1):
            for sy in(-1,0,1):
                pp=pts+np.array([sx*L,sy*L])
                if pp[:,0].max()<-0.08*L or pp[:,0].min()>1.08*L or pp[:,1].max()<-0.08*L or pp[:,1].min()>1.08*L: continue
                pats.append(Polygon(pp,closed=True))
    ax.add_collection(PatchCollection(pats,facecolor="#c9d3e0",edgecolor="k",lw=0.5))
    ax.set_xlim(0,L);ax.set_ylim(0,L);ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])

def cfg(w,rho):
    T=build_periodic_voronoi(6,np.random.default_rng(3))
    ft=FoamTissue(T,w=w,rho=rho,seed=1)
    for _ in range(150): ft.step(mu=0.0)
    for v in range(len(ft.vpos)): ft.t2_reverse(v)
    ft._update_faces()
    for it in range(350):
        ft.step(mu=0.5)
        if it % 15 == 0: ft.do_transitions()   # T1 + T3-reverse + T2 chain
    for it in range(700):
        ft.step(mu=0.0)
        if it % 15 == 0: ft.do_transitions()
    return ft

RHOS=[1.0,0.9,0.8]; WS=[0.0,0.5,1.5]
fig,axes=plt.subplots(len(RHOS),len(WS),figsize=(9,9))
for i,rho in enumerate(RHOS):
    for j,w in enumerate(WS):
        ft=cfg(w,rho); draw(ft,axes[i,j])
        axes[i,j].set_title(f"W/T0={w}, rho={rho}\nphi={ft.volume_fraction():.2f}",fontsize=9)
        print(f"rho={rho} W={w} phi={ft.volume_fraction():.3f}",flush=True)
fig.text(0.5,0.005,"Relative adhesion  W/T0  ->",ha="center")
fig.text(0.005,0.5,"Cell density  rho  ->",va="center",rotation=90)
plt.tight_layout(rect=[0.02,0.02,1,1])
plt.savefig("figures/foam_fig2a_panel.png",dpi=90)
print("saved foam_fig2a_panel.png")
