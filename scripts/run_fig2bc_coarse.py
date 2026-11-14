"""Coarse Fig 2b/2c sweep on the published (rho, W/T0) grid points.

Follows ts_simVolumeFraction.m: Voronoi start, a space opened at every vertex,
anneal at dT/T0 = 0.5, then quench at dT = 0, all on the reference transition
schedule (FoamTissue.advance: T1 every 3 steps, T4 / space reseeding every 5).
Run length matches the published template (anneal 2000 + quench 6000 steps).
Departures, for cost: 36 cells (paper: 100) and one seed (paper: reportedly 20).

z is the mean number of DISTINCT neighbouring cells; z_edges counts cell-cell
edges.  Each finished run is appended to data/fig2bc_coarse_runs.jsonl, so a
stuck run cannot lose the others.  Saves data/fig2bc_coarse.npz.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.topology import build_periodic_voronoi
from activefoam.foam_full import FoamTissue, SPACE

PSIS = [0.90, 0.96, 1.00, 1.06, 1.12, 1.20]
RHOS = [round(p * p, 4) for p in PSIS]
WS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SEEDS = [1]
ANNEAL, QUENCH, BLOCK = 2000, 6000, 100
MAX_SECONDS = 2400            # give up on a run that crawls (tangled / fragmented)
RUNS = "data/fig2bc_coarse_runs.jsonl"


def contacts(ft, ncell):
    pairs, nedge = set(), np.zeros(ncell)
    for f0, f1 in ft.e_f:
        f0, f1 = int(f0), int(f1)
        if f0 != SPACE and f1 != SPACE and f0 != f1:
            pairs.add((min(f0, f1), max(f0, f1)))
            nedge[f0] += 1; nedge[f1] += 1
    return 2.0 * len(pairs) / ncell, float(nedge.mean())


def equilibrate(w, rho, seed):
    t0 = time.time()
    T = build_periodic_voronoi(6, np.random.default_rng(100 + seed))
    ft = FoamTissue(T, w=w, rho=rho, seed=seed)
    ncell = ft.nFa
    for v in range(len(ft.vpos)):
        ft.t2_reverse(v)
    ft._update_faces()
    status = "ok"
    for blk in range((ANNEAL + QUENCH) // BLOCK):
        activity = 0.5 if blk * BLOCK < ANNEAL else 0.0
        for step in range(blk * BLOCK + 1, (blk + 1) * BLOCK + 1):
            ft.advance_scheduled(step, mu=activity)
        phi = ft.volume_fraction()
        if not np.isfinite(phi) or phi > 1.05:
            status = "invalid_geometry"; break
        if time.time() - t0 > MAX_SECONDS:
            status = "timeout"; break
    out = {"w": w, "rho": rho, "seed": seed, "status": status,
           "seconds": round(time.time() - t0), "steps": (blk + 1) * BLOCK}
    if status == "ok":
        tang = sum(ft._cell_self_intersects(c) for c in range(ncell))
        z, ze = contacts(ft, ncell)
        out.update(phi=float(ft.volume_fraction()), z=z, z_edges=ze, tangled=int(tang))
    return out


def _task(a):
    w, rho, s = a
    try:
        r = equilibrate(w, rho, s)
    except Exception as ex:
        r = {"w": w, "rho": rho, "seed": s, "status": f"exception: {ex!r}"}
    return r


def main():
    from multiprocessing import Pool
    os.makedirs("data", exist_ok=True)
    t0 = time.time()
    # slow, fragmented low-adhesion runs first so they do not all queue at the end
    tasks = sorted(((w, rho, s) for rho in RHOS for w in WS for s in SEEDS),
                   key=lambda a: (a[0], a[1]))
    res = []
    if "--resume" in sys.argv and os.path.exists(RUNS):      # keep finished runs
        res = [json.loads(l) for l in open(RUNS) if l.strip()]
        done = {(r["w"], r["rho"], r["seed"]) for r in res}
        tasks = [a for a in tasks if a not in done]
        print(f"resuming: {len(res)} done, {len(tasks)} to run", flush=True)
    else:
        open(RUNS, "w").close()
    with Pool(processes=4) as pool, open(RUNS, "a") as fh:
        for n, r in enumerate(pool.imap_unordered(_task, tasks), 1):
            res.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"[{n}/{len(tasks)}] {r}  [{time.time()-t0:.0f}s]", flush=True)
    grid = {q: np.full((len(RHOS), len(WS)), np.nan) for q in ("phi", "z", "z_edges", "tangled")}
    for r in res:
        if r["status"] == "ok":
            i, j = RHOS.index(r["rho"]), WS.index(r["w"])
            for q in grid:
                grid[q][i, j] = r[q]
    np.savez("data/fig2bc_coarse.npz", WS=np.array(WS), RHOS=np.array(RHOS), **grid)
    print("saved data/fig2bc_coarse.npz  total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
