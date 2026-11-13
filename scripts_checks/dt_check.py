"""Half-time-step check for the Fig 2b/2c equilibrium protocol.
Same confluent tiling per seed (prepared at dt=0.005), then anneal (tau_T at mu=0.5) and
quench (3 tau_T) at the given dt. Transition cadence stays in steps (3/5), as in the MATLAB."""
import json, sys, time, os
from copy import deepcopy
import numpy as np
sys.path.insert(0, os.getcwd())
from activefoam import foam_measure as fm

def run(a):
    w, rho, seed, dt = a
    t0 = time.time()
    try:
        ft = fm.initial_voronoi(10, seed, True, w=w, rho=rho)
        ft._restore(deepcopy(fm.relaxed_tiling(10, seed, True)))
        ft.dt = dt
        ft.e_t = np.array([ft.fixed_tension(e) for e in range(len(ft.e_v))])
        for v in range(len(ft.vpos)): ft.t2_reverse(v)
        fm.evolve(ft, ft.tauT, 0.5)
        ft.e_t = np.array([ft.fixed_tension(e) for e in range(len(ft.e_v))])
        fm.evolve(ft, 3 * ft.tauT, 0., start=round(ft.tauT / ft.dt))
        q = fm.quality(ft)
        r = dict(w=w, rho=rho, seed=seed, dt=dt, status="ok", phi=float(fm.validate(ft)),
                 z=float(ft.neighbor_number().mean()), checks=int(q["quality_checks"]),
                 microfold_checks=int(q["microfold_checks"]), max_depth=float(q["max_crossing_depth"]))
    except Exception as ex:
        r = dict(w=w, rho=rho, seed=seed, dt=dt, status=f"failed: {ex!r}"[:200])
    r["seconds"] = round(time.time() - t0)
    return r

if __name__ == "__main__":
    from multiprocessing import Pool
    out = sys.argv[1]
    done = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    have = {(d["w"], d["rho"], d["seed"], d["dt"]) for d in done}
    tasks = [(w, rho, s, dt) for dt in (0.0025, 0.005) for (w, rho) in ((0.0, 0.81), (0.4, 1.0), (0.5, 0.81)) for s in range(1, 6)]
    tasks = [t for t in tasks if t not in have]
    with Pool(4) as p, open(out, "a") as fh:
        for r in p.imap_unordered(run, tasks):
            fh.write(json.dumps(r) + "\n"); fh.flush(); print(r, flush=True)
    print("DONE")
