"""Pilot: forced neighbour exchange in the disc and in a matched periodic box.

    python circle/scripts/probe_pilot.py

Follows the comparison of Thomas & Hopyan (Biophys. J. 2023) between a periodic
sheet and a bounded flat disc, with their energy barrier replaced by the
mechanical work to shorten a contact to the T1 length (``circle.probe``).
Each prepared state is probed on a random sample of contacts; the state is
restored between probes.  A pilot with two seeds per point, not an ensemble.

Writes ``circle/data/probe_20261114/records.json`` and
``figures/circle/probe_20261114/forced_exchange.png``.
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

TAG = "probe_20261114"
WS = (0.5, 1.0, 1.5)
RHO = 1.3
SIDES = (6, 10)                    # periodic n_side; the disc has n_side**2 cells
SEEDS = (1, 2)
PER_STATE = 12
BUDGET = 3000


def run(task):
    from activefoam.budget import BudgetExceeded, call_with_budget
    from circle import probe, protocol
    domain, side, w, seed = task
    start = time.time()
    head = dict(domain=domain, n_cells=side * side, w=w, rho=RHO, seed=seed)
    rows = []
    try:
        if domain == "disc":
            foam = call_with_budget(BUDGET, protocol.equilibrated, side * side, w, RHO, seed)
            status = protocol.validate(foam, strict=False)["geometry_status"]
        else:
            foam = call_with_budget(BUDGET, protocol.periodic_equilibrated, side, w, RHO, seed)
            status = "periodic"
        rng = np.random.default_rng(1000 + seed)
        eligible = probe.contacts(foam)
        zones = {e: probe.zone(foam, e) for e in eligible}
        chosen = []
        groups = sorted(set(zones.values()))
        for name in groups:
            members = [e for e in eligible if zones[e] == name]
            take = min(len(members), PER_STATE // len(groups))
            chosen += list(rng.choice(members, size=take, replace=False))
        lengths = foam.edge_lengths()
        for e in chosen:
            e = int(e)
            row = dict(head, edge=e, zone=zones[e], length=float(lengths[e]),
                       sides=[len(foam.f_v[int(c)]) for c in foam.e_f[e]])
            try:
                result = call_with_budget(600, probe.force_exchange, foam, e)
                row.update({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in result.items() if k not in ("converged", "hold_time")})
                row["all_converged"] = bool(result["all_converged"])
            except (ValueError, BudgetExceeded) as exc:
                row.update(outcome="failed", error=str(exc)[:160])
            rows.append(row)
        info = dict(head, valid=True, geometry_status=status, contacts=len(eligible),
                    packing=float(foam.volume_fraction()),
                    mean_area=float(np.mean(foam.f_area)))
    except (ValueError, BudgetExceeded) as exc:
        info = dict(head, valid=False, error=str(exc)[:160])
    info["seconds"] = round(time.time() - start, 1)
    return info, rows


def summarise(rows, states):
    """Per-state means first, then mean and range over seeds (no pooling of contacts)."""
    table = []
    for side in SIDES:
        for w in WS:
            for domain, zone in (("periodic", "periodic"), ("disc", "interior"), ("disc", "wall")):
                per_seed, outcomes, stiff, n = [], {}, [], 0
                for seed in SEEDS:
                    hit = [r for r in rows if r["domain"] == domain and r["zone"] == zone
                           and r["n_cells"] == side * side and r["w"] == w and r["seed"] == seed
                           and r.get("outcome") not in (None, "failed")]
                    done = [r for r in hit if r["outcome"] != "preempted"]
                    for r in hit:
                        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
                    if done:
                        per_seed.append(np.mean([r["work"] / (r["d0"] - r["final_separation"])
                                                 for r in done]))
                        stiff.append(np.mean([r["stiffness"] for r in done]))
                        n += len(done)
                table.append(dict(n_cells=side * side, w=w, domain=domain, zone=zone,
                                  seeds=len(per_seed), contacts=n,
                                  mean_force=float(np.mean(per_seed)) if per_seed else None,
                                  seed_values=[float(x) for x in per_seed],
                                  stiffness=float(np.mean(stiff)) if stiff else None,
                                  outcomes=outcomes))
    return table


def figure(rows, table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#0b0b0b", "#52514e"
    colors = {"periodic": "#2a78d6", "interior": "#eb6834", "wall": "#1baf7a"}
    labels = {"periodic": "periodic box", "interior": "disc, interior", "wall": "disc, wall layer"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)

    ax = axes[0]
    for zone in ("periodic", "interior", "wall"):
        shown = [r for r in rows if r["zone"] == zone and r["w"] == 1.0 and r["n_cells"] == 100
                 and r.get("outcome") not in (None, "failed", "preempted")][:6]
        for i, r in enumerate(shown):
            shortening = (r["d0"] - np.array(r["separation"])) / r["d0"]
            ax.plot(shortening, r["force"], color=colors[zone], linewidth=1.6, alpha=0.75,
                    label=labels[zone] if i == 0 else None)
    ax.set_xlabel("shortening of the contact / initial separation", color=ink, fontsize=10)
    ax.set_ylabel("holding force  (units of T0)", color=ink, fontsize=10)
    ax.set_title("Force needed to shorten one contact\n100 cells, W/T0 = 1", color=ink, fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    for ax, n_cells in zip(axes[1:], (36, 100)):
        for k, zone in enumerate(("periodic", "interior", "wall")):
            x, y = [], []
            for w in WS:
                entry = next(t for t in table if t["n_cells"] == n_cells and t["w"] == w
                             and t["zone"] == zone)
                for value in entry["seed_values"]:
                    ax.plot(w + 0.04 * (k - 1), value, marker="o", markersize=5,
                            color=colors[zone], alpha=0.55, linestyle="none")
                if entry["mean_force"] is not None:
                    x.append(w + 0.04 * (k - 1))
                    y.append(entry["mean_force"])
            ax.plot(x, y, color=colors[zone], linewidth=2, label=labels[zone])
        ax.set_xticks(WS)
        ax.set_xlabel("adhesion  W/T0", color=ink, fontsize=10)
        ax.set_ylabel("work to the T1 length / shortening  (T0)", color=ink, fontsize=10)
        ax.set_title(f"Mean resisting force, {n_cells} cells\nline: mean of seeds, dots: each seed",
                     color=ink, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Forced neighbour exchange at rho = {RHO}, zero activity (pilot, "
                 f"{len(SEEDS)} seeds, {PER_STATE} contacts per state)", color=ink, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    tasks = [(domain, side, w, seed) for side in SIDES for domain in ("disc", "periodic")
             for w in WS for seed in SEEDS]
    tasks.sort(key=lambda t: -t[1])                 # start the large systems first
    states, rows = [], []
    with Pool(int(os.environ.get("PILOT_WORKERS", "4"))) as pool:
        for info, state_rows in pool.imap_unordered(run, tasks):
            states.append(info)
            rows += state_rows
            print(info, {o: sum(r.get("outcome") == o for r in state_rows)
                         for o in sorted({r.get("outcome") for r in state_rows})}, flush=True)
    key = lambda r: (r["domain"], r["n_cells"], r["w"], r["seed"])
    states.sort(key=key)
    rows.sort(key=lambda r: key(r) + (r["edge"],))
    table = summarise(rows, states)
    out = ROOT / "circle" / "data" / TAG
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(dict(states=states, contacts=rows,
                                                      summary=table), indent=1))
    figure(rows, table, ROOT / "figures" / "circle" / TAG / "forced_exchange.png")
    print("\n| cells | W/T0 | domain, zone | contacts | mean resisting force (seeds) | stiffness "
          "| outcomes |")
    print("|---|---|---|---|---|---|---|")
    for t in table:
        seeds = ", ".join(f"{v:.3f}" for v in t["seed_values"])
        mean = "n/a" if t["mean_force"] is None else f"{t['mean_force']:.3f}"
        stiff = "n/a" if t["stiffness"] is None else f"{t['stiffness']:.2f}"
        print(f"| {t['n_cells']} | {t['w']} | {t['domain']}, {t['zone']} | {t['contacts']} | "
              f"{mean} ({seeds}) | {stiff} | {t['outcomes']} |")


if __name__ == "__main__":
    main()
