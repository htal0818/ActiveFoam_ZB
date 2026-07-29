"""Build a self-contained side-by-side comparison gallery (figures/gallery.html)."""
import os, sys, base64, io
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")


def embed(path, maxw=1180, q=84):
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# (file, title, subtitle, model, verdict, note)
CARDS = [
    ("compare_fig2a_configs.png", "Fig 2a", "Equilibrium configurations",
     "Faithful curved-edge foam model", "strong",
     "Curved cells (intermediate vertices) with extracellular spaces opening at low "
     "adhesion / low density and closing to confluence as W/T₀ rises."),
    ("compare_fig2b_phi.png", "Fig 2b", "Volume fraction φ vs adhesion",
     "Unified foam model (T1 + T2 + T2-reverse + T3-reverse)", "strong",
     "φ rises from ≈ the jamming value toward 1 (confluent) as adhesion "
     "increases, with lower density giving lower φ."),
    ("compare_fig2c_z.png", "Fig 2c", "Contact number z vs adhesion",
     "Unified foam model (contact-breaking via T1 → T3-reverse)", "good",
     "The single faithful model now produces z too: contacts break as spaces grow "
     "(T1 flips a contact into a space–space edge, T3-reverse merges the spaces). "
     "z rises to 6 with adhesion, matching for W/T₀ ≳ 0.25; the pure-foam W→0 "
     "limit over-fragments (small-triangle stability / full T4 still to come)."),
    ("compare_fig2i_stress.png", "Fig 2i", "Shear-stress relaxation vs adhesion",
     "Foam model + paper Eq. 4/5 stress + affine-shear step", "strong",
     "The paper's Methods protocol (strain step then relax): the initial elastic "
     "jump is largest at W=0 and vanishes at W/T₀=2, while the residual (yield) "
     "stress is highest at the structural transition (W/T₀≈0.5) — the exact "
     "ordering and crossover of the paper's family."),
    ("compare_fig2j_yield.png", "Fig 2j", "Yield stress vs adhesion",
     "Foam model + paper Eq. 4/5 stress", "strong",
     "Computed with the paper's Methods stress tensor: the yield stress is "
     "maximal at the structural transition (W/T₀≈0.5–0.6) and vanishes at W=0 "
     "(jamming) and W=2 (vanishing tension) — the tissue's two fluid limits."),
    ("compare_fig2f_jamming.png", "Fig 2f", "Foam-limit jamming",
     "Soft-particle jamming (O'Hern)", "strong",
     "φ_c ≈ 0.85 (paper 0.83), z_c = 4 at the isostatic point, z rising with "
     "the square-root-plus-linear law."),
    ("compare_fig2g_shape.png", "Fig 2g", "Density-independent transition",
     "Confluent vertex model", "strong",
     "Equilibrium shape factor s̄ stays flat at ≈ 3.84 then jumps to 4.31 at "
     "W/T₀ = 2, where the cell–cell tension 2−W vanishes — an "
     "almost exact match."),
    ("compare_fig3a_msd.png", "Fig 3a", "Mean squared displacement",
     "Confluent dynamic vertex model (T1)", "good",
     "MSD spans six decades with the correct caged → diffusive ordering in "
     "ΔT/T₀. The 36-cell system creeps a little more than the paper at low "
     "activity, so the extracted exponent α is biased high."),
    ("compare_fig4g_shape.png", "Fig 4g", "Dynamic shape factor",
     "Confluent dynamic vertex model", "good",
     "s̄ rises with adhesion and with activity, crossing the 3.81 rigidity line; "
     "the spread across ΔT/T₀ is milder than the paper."),
    ("compare_fig4a_stress.png", "Fig 4a", "Shear-stress relaxation",
     "Confluent dynamic vertex model", "partial",
     "The elastic jump (≈ 0.85) and fast initial decay are captured, but the "
     "confluent model relaxes too fast to resolve the paper's slow, "
     "orders-of-magnitude τ_SR tail — the weakest of the set."),
    ("compare_fig4d_tauSR.png", "Fig 4d", "Stress-relaxation time τ_SR map",
     "Foam model, τ_SR = 1/k_NE", "good",
     "τ_SR is read off the steady-state cellular NE (topological-event) rate, "
     "τ_SR ≈ 1/k_NE (the paper's stated mechanism). It grows from the fluid "
     "non-confluent states toward the structural transition (green = confluent) "
     "and shrinks with activity ΔT/T₀, spanning several decades as in the paper."),
    ("compare_fig4e_phase.png", "Fig 4e", "Fluid / solid phase diagram",
     "Foam model, τ_SR/τ_T = 10² criterion", "good",
     "Applying the paper's fluidity criterion (fluid if τ_SR/τ_T < 10²): solid "
     "states form a band bordering the structural transition at low activity and "
     "the tissue is fluid everywhere for large enough ΔT/T₀ — reproducing the "
     "topology of the paper's phase diagram from a direct NE-rate measurement."),
]

VERDICT = {
    "strong": ("Strong match", "var(--good)"),
    "good": ("Good match", "var(--ok)"),
    "partial": ("Partial", "var(--warn)"),
}

PARAMS = [
    ("W/T₀", "w", "swept", "relative cell–cell adhesion → tension 2−W"),
    ("ΔT/T₀", "dT", "swept", "tension-fluctuation magnitude"),
    ("ρ ≡ NL₀²/A_T", "rho", "1 or <1", "density; preferred area A₀=ρ"),
    ("P₀L₀/T₀", "P0", "10", "relative osmotic (normal) force"),
    ("τ_T/τ_R", "tauT", "10", "tension persistence time"),
    ("Δt", "dt", "0.005 τ_R", "Euler–Maruyama step"),
]

TRANS = [
    ("T1", "neighbour exchange (edge flip)", "done", "confluent + foam model, validated"),
    ("T2-reverse", "create a triangular extracellular space", "done", "ts_t2ReverseTransition port"),
    ("T2", "annihilate a collapsed space → triple junction", "done", "with id remap"),
    ("T3 / T4", "merge cells / resolve crossing edges", "todo", "needs ts_edgeCut* machinery"),
]


def card_html(c):
    img = embed(os.path.join(FIG, c[0]))
    if img is None:
        return ""
    label, col = VERDICT[c[4]]
    return f"""
    <figure class="card">
      <div class="card-head">
        <div><span class="fig">{c[1]}</span><span class="fig-sub">{c[2]}</span></div>
        <span class="badge" style="--c:{col}">{label}</span>
      </div>
      <div class="frame"><img loading="lazy" src="{img}" alt="{c[1]} comparison"></div>
      <figcaption><span class="model">{c[3]}</span>{c[5]}</figcaption>
    </figure>"""


def main():
    cards = "\n".join(card_html(c) for c in CARDS)
    prows = "\n".join(
        f"<tr><td class='m'>{p[0]}</td><td class='m'>{p[1]}</td>"
        f"<td class='m'>{p[2]}</td><td>{p[3]}</td></tr>" for p in PARAMS)
    trows = "\n".join(
        f"<tr><td class='m'>{t[0]}</td><td>{t[1]}</td>"
        f"<td><span class='dot {t[2]}'></span>"
        f"{'implemented' if t[2]=='done' else 'not ported'}</td>"
        f"<td class='sub'>{t[3]}</td></tr>" for t in TRANS)
    html = TEMPLATE.replace("{{CARDS}}", cards).replace("{{PARAMS}}", prows).replace("{{TRANS}}", trows)
    out = os.path.join(FIG, "gallery.html")
    open(out, "w").write(html)
    print("wrote", out, f"({os.path.getsize(out)//1024} KB)")


TEMPLATE = r"""<title>Active Foam — reproduction gallery</title>
<style>
:root{
  --ink:#161a1e; --paper:#f4f6f5; --panel:#ffffff; --line:#dfe4e2;
  --muted:#5c6a6b; --accent:#0e7c89; --accent2:#b96f24;
  --good:#1f9d76; --ok:#2f7fb8; --warn:#c9772b;
  --shadow:0 1px 2px rgba(20,30,30,.05),0 8px 24px rgba(20,40,42,.06);
}
@media (prefers-color-scheme:dark){:root{
  --ink:#e6ecec; --paper:#0f1417; --panel:#161d21; --line:#25302f;
  --muted:#8ea0a0; --accent:#3bb6c4; --accent2:#d69146;
  --good:#37c493; --ok:#4d9bd6; --warn:#e0954a;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}}
:root[data-theme="dark"]{
  --ink:#e6ecec; --paper:#0f1417; --panel:#161d21; --line:#25302f;
  --muted:#8ea0a0; --accent:#3bb6c4; --accent2:#d69146;
  --good:#37c493; --ok:#4d9bd6; --warn:#e0954a;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}
:root[data-theme="light"]{
  --ink:#161a1e; --paper:#f4f6f5; --panel:#ffffff; --line:#dfe4e2;
  --muted:#5c6a6b; --accent:#0e7c89; --accent2:#b96f24;
  --good:#1f9d76; --ok:#2f7fb8; --warn:#c9772b;
  --shadow:0 1px 2px rgba(20,30,30,.05),0 8px 24px rgba(20,40,42,.06);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:400 17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:clamp(28px,5vw,64px) clamp(18px,4vw,40px)}
.eyebrow{font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.18em;text-transform:uppercase;color:var(--accent)}
h1{font:600 clamp(30px,5vw,48px)/1.08 Georgia,"Times New Roman",serif;
  letter-spacing:-.01em;text-wrap:balance;margin:.5em 0 .2em}
.cite{color:var(--muted);font-size:15px;max-width:64ch}
.cite a{color:var(--accent);text-decoration:none;border-bottom:1px solid var(--line)}
.lede{margin:1.4em 0 0;max-width:66ch;color:var(--ink)}
section{margin-top:clamp(36px,6vw,64px)}
h2{font:600 13px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted);margin:0 0 18px;
  padding-bottom:10px;border-bottom:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:14.5px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font:600 12px/1 ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
td.m,.m{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13.5px}
td.sub,.sub{color:var(--muted);font-size:13.5px}
.cards{display:flex;flex-direction:column;gap:26px}
.card{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;
  overflow:hidden;box-shadow:var(--shadow)}
.card-head{display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:16px 20px;border-bottom:1px solid var(--line)}
.fig{font:600 18px/1 Georgia,serif;margin-right:12px}
.fig-sub{color:var(--muted);font-size:15px}
.badge{font:600 11px/1 ui-monospace,monospace;letter-spacing:.06em;text-transform:uppercase;
  color:var(--c);border:1px solid var(--c);border-radius:999px;padding:5px 11px;white-space:nowrap}
.frame{background:#fff;padding:6px}
.frame img{display:block;width:100%;height:auto;border-radius:6px}
figcaption{padding:15px 20px 19px;color:var(--muted);font-size:14.5px;line-height:1.55}
.model{display:block;color:var(--accent2);font-weight:600;font-size:13px;
  letter-spacing:.02em;margin-bottom:5px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;vertical-align:middle}
.dot.done{background:var(--good)}.dot.todo{background:var(--warn)}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:10px;padding:18px 22px;font-size:15px;color:var(--ink);max-width:74ch}
.foot{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);
  color:var(--muted);font-size:13px}
.toggle{position:fixed;top:16px;right:16px;background:var(--panel);border:1px solid var(--line);
  color:var(--muted);border-radius:999px;padding:8px 14px;font:600 12px ui-monospace,monospace;
  cursor:pointer;box-shadow:var(--shadow)}
.toggle:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style>
<button class="toggle" onclick="var r=document.documentElement,d=r.getAttribute('data-theme')==='dark'?'light':'dark';r.setAttribute('data-theme',d)">toggle theme</button>
<div class="wrap">
  <header>
    <div class="eyebrow">Paper reproduction &middot; Python</div>
    <h1>Embryonic tissues as active foams</h1>
    <p class="cite">Side-by-side reproduction of Kim, Pochitaloff, Stooke-Vaughan &amp; Camp&agrave;s,
    <em>Nature Physics</em> <strong>17</strong>, 859–866 (2021).
    <a href="https://doi.org/10.1038/s41567-021-01215-1">doi:10.1038/s41567-021-01215-1</a>.
    Each card shows the published panel (left) beside the Python reproduction (right).</p>
    <p class="lede">The reference MATLAB model is ported to NumPy across three complementary
    implementations: a faithful vertex-network <strong>foam model</strong> with curved contacts
    (intermediate vertices) and extracellular spaces; a <strong>confluent dynamic vertex model</strong>
    for the dynamics; and a <strong>deformable-particle</strong> and <strong>soft-particle jamming</strong>
    model for structure. A stochastic simulation can't be reproduced bit-for-bit — the goal is the
    paper's quantitative results: the curves, scaling exponents, and transition values.</p>
  </header>

  <section>
    <h2>Comparisons</h2>
    <div class="cards">{{CARDS}}</div>
  </section>

  <section>
    <h2>Parameter mapping &middot; decoded from the MATLAB</h2>
    <table><thead><tr><th>Paper</th><th>Code</th><th>Value</th><th>Meaning</th></tr></thead>
    <tbody>{{PARAMS}}</tbody></table>
  </section>

  <section>
    <h2>Topological transitions</h2>
    <table><thead><tr><th>Transition</th><th>Meaning</th><th>Status</th><th>Notes</th></tr></thead>
    <tbody>{{TRANS}}</tbody></table>
    <p class="note" style="margin-top:18px">T1, T2-reverse and T2 are implemented, stable
    (invariants verified across long runs), and wired into the foam dynamics. T3/T4 handle the
    curved-edge <em>collision</em> cases (edges crossing as a space pinches, merging adjacent
    spaces) and remain to be ported.</p>
  </section>

  <p class="foot">Generated by Claude Code &middot; three NumPy models reproducing the active-foam
  framework. Bit-exact reproduction of a stochastic, <code>rng('shuffle')</code>-seeded MATLAB
  simulation is not possible; matches are quantitative (curves, exponents, transition values).</p>
</div>"""


if __name__ == "__main__":
    main()
