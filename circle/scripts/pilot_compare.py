"""Pilot comparison: disc against a matched periodic box (not a production ensemble).

    python circle/scripts/pilot_compare.py

Same preparation, same pressure convention (boundary normals, current-step
areas), same cell number, two seeds per point.  Writes
``circle/data/pilot_compare.json`` and ``figures/circle/disc_configs.png``.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                                     # noqa: E402

WS = (0.5, 1.0, 1.5)
RHOS = (1.0, 1.2, 1.4)
SEEDS = (1, 2)
N_SIDE = 10
N_CELLS = N_SIDE ** 2
BUDGET = 900


def run(task):
    from activefoam.budget import BudgetExceeded, call_with_budget
    from circle import protocol
    from circle.measure import touching_wall, wall_load, zone_statistics
    kind, w, rho, seed = task
    start = time.time()
    out = dict(kind=kind, w=w, rho=rho, seed=seed, valid=True)
    try:
        if kind == "disc":
            foam = call_with_budget(BUDGET, protocol.equilibrated, N_CELLS, w, rho, seed)
            report = protocol.validate(foam, strict=False)
            out.update(zone_statistics(foam))
            out.update(film_crossings=report["film_crossings"], nested=report["nested"],
                       microfold_checks=int(report["microfold_checks"]),
                       wall_load=wall_load(foam.wall)["total"],
                       accepted=foam.log.counts(), rejected=foam.log.counts("rejected"))
            if seed == SEEDS[0]:
                out["polygons"] = [foam._cell_polyline(c).tolist() for c in range(foam.nFa)]
                out["touching"] = touching_wall(foam).tolist()
                out["center"], out["radius"] = foam.center.tolist(), foam.radius
        else:
            foam = call_with_budget(BUDGET, protocol.periodic_equilibrated, N_SIDE, w, rho, seed)
            shape = foam.shape_factors()
            out.update(packing=foam.volume_fraction(), z=float(foam.neighbor_number().mean()),
                       shape=float(shape.mean()), area=float(foam.f_area.mean()))
    except (ValueError, BudgetExceeded) as exc:
        out.update(valid=False, error=str(exc)[:200])
    out["seconds"] = round(time.time() - start, 1)
    return out


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Patch, Polygon

    space, interior, wall_cell, ink = "#d1352b", "#c9d3e0", "#7f93ad", "#222222"
    fig, axes = plt.subplots(len(RHOS), len(WS), figsize=(9.2, 9.6))
    for i, rho in enumerate(RHOS[::-1]):
        for j, w in enumerate(WS):
            ax = axes[i, j]
            ax.set_aspect("equal"); ax.axis("off")
            hit = [r for r in results if r["kind"] == "disc" and r["w"] == w
                   and r["rho"] == rho and "polygons" in r]
            if not hit:
                ax.text(0.5, 0.5, "no valid\nconfiguration", ha="center", va="center",
                        transform=ax.transAxes, color=ink)
                continue
            r = hit[0]
            c, radius = np.array(r["center"]), r["radius"]
            ax.add_patch(Circle(c, radius, facecolor=space, edgecolor="none", zorder=1))
            for flag, color in ((False, interior), (True, wall_cell)):
                patches = [Polygon(np.array(p), closed=True)
                           for p, t in zip(r["polygons"], r["touching"]) if t == flag]
                ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor=ink,
                                                  linewidth=0.5, zorder=2))
            ax.add_patch(Circle(c, radius, facecolor="none", edgecolor=ink, linewidth=1.4,
                                zorder=3))
            ax.set_xlim(c[0] - 1.04 * radius, c[0] + 1.04 * radius)
            ax.set_ylim(c[1] - 1.04 * radius, c[1] + 1.04 * radius)
            ax.set_title(f"W/T0 = {w}, rho = {rho}\npacking {r['packing']:.3f}, "
                         f"{r['n_touching']} cells on the wall", fontsize=9, color=ink)
    fig.legend(handles=[Patch(facecolor=interior, edgecolor=ink, label="cell"),
                        Patch(facecolor=wall_cell, edgecolor=ink, label="cell touching the wall"),
                        Patch(facecolor=space, edgecolor="none", label="extracellular space")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Active foam in a rigid, frictionless, nonadhesive disc: {N_CELLS} cells, "
                 "equilibrated at zero activity", fontsize=11, color=ink)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    tasks = [(kind, w, rho, seed) for kind in ("disc", "periodic")
             for w in WS for rho in RHOS for seed in SEEDS]
    with Pool(int(os.environ.get("PILOT_WORKERS", "3"))) as pool:
        results = []
        for result in pool.imap_unordered(run, tasks):
            results.append(result)
            print({k: v for k, v in result.items()
                   if k not in ("polygons", "touching", "accepted", "rejected")}, flush=True)
    results.sort(key=lambda r: (r["kind"], r["w"], r["rho"], r["seed"]))
    figure(results, ROOT / "figures" / "circle" / "disc_configs.png")
    data = ROOT / "circle" / "data"
    data.mkdir(exist_ok=True)
    slim = [{k: v for k, v in r.items() if k not in ("polygons", "touching")} for r in results]
    (data / "pilot_compare.json").write_text(json.dumps(slim, indent=1))

    print("\n| W/T0 | rho | disc packing | disc cells on wall | disc interior z | periodic z "
          "| disc interior shape | periodic shape | disc interior area | periodic area "
          "| disc wall load | valid seeds (disc, periodic) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for w in WS:
        for rho in RHOS:
            d = [r for r in slim if r["kind"] == "disc" and r["w"] == w and r["rho"] == rho and r["valid"]]
            p = [r for r in slim if r["kind"] == "periodic" and r["w"] == w and r["rho"] == rho and r["valid"]]
            mean = lambda rows, key: (f"{np.mean([r[key] for r in rows]):.3f}" if rows else "n/a")
            print(f"| {w} | {rho} | {mean(d, 'packing')} | {mean(d, 'n_touching')} | "
                  f"{mean(d, 'z_interior')} | {mean(p, 'z')} | {mean(d, 'shape_interior')} | "
                  f"{mean(p, 'shape')} | {mean(d, 'area_interior')} | {mean(p, 'area')} | "
                  f"{mean(d, 'wall_load')} | {len(d)}, {len(p)} |")


if __name__ == "__main__":
    main()
