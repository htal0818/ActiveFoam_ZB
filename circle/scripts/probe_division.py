"""Pilot: forced neighbour exchange with and without a synchronous division.

    python circle/scripts/probe_division.py

One founder cell divides at t = 2, 6 and 10 tau_R to 16 cells, exactly as in
``division_lineage.py`` (no growth, no feedback, no tension noise, W/T0 = 1,
dt = 0.001, film spacing 0.025, wall radius 0.7).  At t = 14 the run branches:

  divided   all 16 cells divide (16 -> 32), then 4 tau_R of relaxation;
  skipped   no division, the same 4 tau_R of relaxation.

Both branches are then probed at t = 18 with ``probe.force_exchange``.
Daughters are half the size of their mothers while the model's T1 length is
fixed, so contacts in the divided branch start closer to it.  Work is therefore
reported per unit shortening (a mean resisting force) next to the raw work.

Writes ``circle/data/probe_division_20261114/records.json``.
"""
import copy
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                                     # noqa: E402

TAG = "probe_division_20261114"
DT, SPACING, RADIUS = 0.001, 0.025, 0.7
PER_BRANCH = 10
_cache = {}


def branch(name):
    """The foam of one branch at t = 18 tau_R (cached per process)."""
    if name in _cache:
        return _cache[name]
    from circle import protocol
    from circle.benchmarks import free_area
    from circle.biology import BiologyFoam
    from circle.disc import DiscFoam, circle_polyline
    center = np.full(2, 4 * RADIUS)
    ring = circle_polyline(center, np.sqrt(free_area(1.0) / np.pi), SPACING)
    base = DiscFoam.from_closed_cells(RADIUS, [ring], center=center, rho=1.0, w=1.0, dt=DT,
                                      edpc=SPACING, seed=1)
    foam = BiologyFoam.from_disc(base, gain=0.0, growth_rate=0.0)
    divisions = {round(t / DT) for t in (2.0, 6.0, 10.0)}
    if name == "divided":
        divisions.add(round(14.0 / DT))
    for step in range(1, round(18.0 / DT) + 1):
        foam.advance_scheduled(step, mu=0.0)
        if step in divisions:
            assert foam.divide_synchronously(), foam.division_log[-1]
    status = protocol.validate(foam, strict=False)["geometry_status"]
    _cache[name] = (foam, status)
    return _cache[name]


def chosen(foam):
    from circle import probe
    eligible = probe.contacts(foam, min_length=2 * probe.T1_LENGTH)
    rng = np.random.default_rng(7)
    return [int(e) for e in rng.choice(eligible, size=min(PER_BRANCH, len(eligible)),
                                       replace=False)], len(eligible)


def run(task):
    from circle import probe
    name, index = task
    start = time.time()
    foam, status = branch(name)
    edges, eligible = chosen(foam)
    if index >= len(edges):
        return None
    e = edges[index]
    lengths = foam.edge_lengths()
    row = dict(branch=name, n_cells=int(foam.nFa), edge=e, eligible=eligible,
               geometry_status=status, length=float(lengths[e]),
               mean_contact_length=float(np.mean(lengths[probe.contacts(foam, 0.0)])),
               mean_area=float(np.mean(foam.f_area)), packing=float(foam.volume_fraction()))
    try:
        result = probe.force_exchange(foam, e, release=4.0)
        row.update({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in result.items() if k not in ("converged", "hold_time")})
        row["all_converged"] = bool(result["all_converged"])
    except ValueError as exc:
        row.update(outcome="failed", error=str(exc)[:160])
    row["seconds"] = round(time.time() - start, 1)
    return row


def main():
    tasks = [(name, i) for i in range(PER_BRANCH) for name in ("divided", "skipped")]
    with Pool(int(os.environ.get("PILOT_WORKERS", "4")), maxtasksperchild=None) as pool:
        rows = []
        for row in pool.imap_unordered(run, tasks, chunksize=1):
            if row is None:
                continue
            rows.append(row)
            print({k: v for k, v in row.items() if k not in ("separation", "force")}, flush=True)
    rows.sort(key=lambda r: (r["branch"], r["edge"]))
    out = ROOT / "circle" / "data" / TAG
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(rows, indent=1))
    print("\n| branch | cells | probed | mean contact length | work | work / shortening | "
          "stiffness | peak force | outcomes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in ("skipped", "divided"):
        hit = [r for r in rows if r["branch"] == name and r.get("outcome") not in (None, "failed")]
        done = [r for r in hit if r["outcome"] != "preempted"]
        outcomes = {}
        for r in hit:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        if not done:
            print(f"| {name} | n/a | 0 |  |  |  |  |  | {outcomes} |")
            continue
        mean = lambda f: np.mean([f(r) for r in done])
        sem = lambda f: np.std([f(r) for r in done], ddof=1) / np.sqrt(len(done))
        per = lambda r: r["work"] / (r["d0"] - r["final_separation"])
        print(f"| {name} | {done[0]['n_cells']} | {len(done)} | {done[0]['mean_contact_length']:.3f} | "
              f"{mean(lambda r: r['work']):.4f} ± {sem(lambda r: r['work']):.4f} | "
              f"{mean(per):.3f} ± {sem(per):.3f} | "
              f"{mean(lambda r: r['stiffness']):.2f} ± {sem(lambda r: r['stiffness']):.2f} | "
              f"{mean(lambda r: r['peak_force']):.3f} | {outcomes} |")


if __name__ == "__main__":
    main()
