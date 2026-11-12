"""Compare full-foam 3a, 4a, and 4g checkpoints with the publisher workbooks."""
import argparse
from contextlib import contextmanager
import io
import json
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook

SOURCE = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41567-021-01215-1/"
    "MediaObjects/41567_2021_1215_MOESM{}_ESM.xlsx"
)
ACTIVITIES = [.25, .5, .75, 1., 1.25, 1.5]
COLORS = dict(zip(ACTIVITIES, ["#547db5", "#dc9625", "#87ad39", "#dc4b34", "#8776b0", "#c87225"]))
COLORS[0.] = "#41bcb3"


def workbook(cache, figure):
    path = cache / f"publisher_fig{figure}.xlsx"
    if not path.exists():
        with urlopen(SOURCE.format(figure + 9), timeout=60) as response:
            raw = response.read()
        with load_source(raw) as book:
            rows = list(book.worksheets[0].values)
        path.write_bytes(raw)
        return rows
    with load_source(path.read_bytes()) as book:
        return list(book.worksheets[0].values)


@contextmanager
def load_source(raw):
    book = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    try:
        yield book
    finally:
        book.close()


def columns(rows, *indices):
    return np.array([[r[i] for i in indices] for r in rows[5:]
                     if all(isinstance(r[i], (int, float)) for i in indices)], dtype=float).T


def compare(ax, reference_x, reference_y, x, values, mu, log_y=False, label=None):
    color = COLORS[mu]
    ax.plot(reference_x, reference_y, "--", color=color, alpha=.55, lw=1)
    if not len(values):
        return {"mu": mu, "model_seeds": 0}
    mean = np.mean(values, axis=0)
    ax.plot(x, mean, "o-" if ax.get_xscale() == "linear" else "-",
            color=color, lw=1.35, ms=3, label=label or f"{mu:g} (n={len(values)})")
    mask = (x >= reference_x[0]) & (x <= reference_x[-1]) & np.isfinite(mean)
    if log_y:
        mask &= mean > 0
        target = np.interp(np.log(x[mask]), np.log(reference_x), np.log10(reference_y))
        error = np.log10(mean[mask]) - target
    else:
        target = np.interp(x[mask], reference_x, reference_y)
        error = mean[mask] - target
    return {
        "mu": mu, "model_seeds": len(values), "points_compared": int(mask.sum()),
        "bias": float(error.mean()) if len(error) else None,
        "rms": float(np.sqrt(np.mean(error**2))) if len(error) else None,
        "error_unit": "log10 MSD" if log_y else "panel units",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.reference_cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    fig3, fig4 = (workbook(args.reference_cache, i) for i in (3, 4))
    records = []
    for job in manifest["jobs"]:
        path = args.data / f'{job["id"]}.npz'
        if path.exists() and job["kind"] in {"dynamics", "stress"}:
            with np.load(path) as raw:
                data = {key: raw[key] for key in
                        ("tlag", "msd", "shape_mean", "time", "stress") if key in raw}
            records.append((job["kind"], job["parameters"], data))

    fig, (ax_msd, ax_stress, ax_shape) = plt.subplots(1, 3, figsize=(14, 4.7))
    ax_msd.set(xscale="log", yscale="log", xlabel=r"Lag / $\tau_T$", ylabel=r"MSD / $L_0^2$", title="3a")
    ax_stress.set(xscale="log", xlabel=r"Time / $\tau_R$", ylabel=r"$\sigma_{xy}/\sigma_0$", title="4a")
    ax_shape.set(xlabel=r"$W/T_0$", ylabel=r"$\langle P/\sqrt{A}\rangle$", title="4g")
    result = {"3a": [], "4a": [], "4g": []}

    for i, mu in enumerate(ACTIVITIES):
        ref_t, ref_y = columns(fig3, 3*i, 3*i+1)
        selected = [d for kind, p, d in records if kind == "dynamics" and p["mu"] == mu and p["w"] == 1.]
        t = selected[0]["tlag"] if selected else np.empty(0)
        # Publisher workbook time is in tau_R. Panel 3a is in tau_T=10 tau_R.
        result["3a"].append(compare(ax_msd, ref_t/10, ref_y, t,
                                   [d["msd"] for d in selected], mu, log_y=True))

        ref_w, ref_s = columns(fig4, 46, 47+2*i)
        by_w = {}
        for kind, p, d in records:
            if kind == "dynamics" and p["mu"] == mu:
                by_w.setdefault(p["w"], []).append(float(d["shape_mean"]))
        model_w = sorted({j["parameters"]["w"] for j in manifest["jobs"]
                          if j["kind"] == "dynamics" and j["parameters"]["mu"] == mu})
        shape = [np.mean(by_w[w]) if w in by_w else np.nan for w in model_w]
        score = compare(ax_shape, ref_w, ref_s, np.asarray(model_w),
                        [shape] if by_w else [], mu, label=f"{mu:g}")
        score.pop("model_seeds")
        score["seeds_by_adhesion"] = {str(w): len(by_w.get(w, ())) for w in model_w}
        result["4g"].append(score)

    for i, mu in enumerate([0., .5, 1., 1.5]):
        ref_t, ref_y = columns(fig4, 4*i, 4*i+1)
        positive = ref_t > 0
        selected = [d for kind, p, d in records if kind == "stress" and p["mu"] == mu]
        t = selected[0]["time"] if selected else np.empty(0)
        result["4a"].append(compare(ax_stress, ref_t[positive], ref_y[positive], t[t > 0],
                                   [d["stress"][t > 0] for d in selected], mu))

    for ax in (ax_msd, ax_stress, ax_shape):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, title=r"$\Delta T/T_0$")
    fig.suptitle("Original source data (dashed) and local simulation (solid)"
                 + (" — short diagnostic pilot" if manifest["pilot"] else ""))
    fig.tight_layout()
    fig.savefig(args.out / "compare_active_panels.png", dpi=180)
    fig.savefig(args.out / "compare_active_panels.pdf")
    (args.out / "compare_active_panels.json").write_text(
        json.dumps({"source_hash": manifest["source_hash"], "pilot": manifest["pilot"],
                    "comparison": result}, indent=2))


if __name__ == "__main__":
    main()
