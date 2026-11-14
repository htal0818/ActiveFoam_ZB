"""Write the finished experiment report; refuse incomplete batches."""
import hashlib,json,platform,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import scipy,numba
from circle.scripts.plot_collective import load,main as plot
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'circle/data/collective_dynamics_20261114'
OUT=ROOT/'figures/circle/collective_dynamics_20261114'


def main():
    for family,count in [('division',3),('mechanics',12),('activity',40),('resolution',4)]:
        rows=load(family)
        if len(rows)!=count:raise RuntimeError(f'{family}: {len(rows)}/{count} cases finished')
    plot();summary=json.loads((OUT/'summary.json').read_text())
    lines=['# Collective-dynamics pilot results','',
        'These are controlled finite-time experiments with the unchanged foam equations and rigid one-sided wall. '
        'The disc surface is flat. No energy or target shape index is introduced. '
        'All runs, including preemptions and geometry failures, are retained.','',
        '## Synchronous division without growth','',
        'One identical eight-cell founder state; five synchronous rounds to 256. '
        'The table covers the same four relaxation times after the final division. '
        'The skipped-final-division control has 128 cells. Pair loss and gain count separately; '
        'changes must persist for 0.2 relaxation times.','',
        '| Interval / τR | Divide: pair changes | Skip: pair changes | Divide rate / cell / τR | Skip rate | Rate ratio | Final shape index | Contact survival |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['division']:
        ratio=r['divide_rate']/r['skip_rate'] if r['skip_rate'] else np.inf
        lines.append(f"| {r['interval']:g} | {r['divide_changes']} | {r['skip_changes']} | {r['divide_rate']:.4f} | {r['skip_rate']:.4f} | {ratio:.2f} | {r['shape_final']:.4f} | {r['survival']:.3f} |")
    lines+=['','Division creates substantial transient remodeling in these runs. '
        'These windows do not establish sustained fluidity after driving stops. '
        'There is only one founder realization, and dividing versus skipping changes cell number. '
        'All runs conserve total preferred area at 8 and hold the wall fixed; actual cell area may relax.','',
        '![Division timing](division_timing.png)','![Final snapshots](division_snapshots.png)','',
        'Dwell sensitivity (divide / skip counts):','',
        '| Interval | Dwell 0.1 | Dwell 0.2 | Dwell 0.5 | Maximum tolerated microfold count |',
        '|---:|---:|---:|---:|---:|']
    for r in summary['division']:
        vals=[f"{r['dwell_sensitivity'][str(d)]['divide']} / {r['dwell_sensitivity'][str(d)]['skip']}" for d in [.1,.2,.5]]
        lines.append(f"| {r['interval']:g} | {' | '.join(vals)} | {r['max_tolerated_microfolds']} |")
    lines+=['','No resolved crossing is accepted by the disc checks. Tolerated local microfolds remain; '
        'the 256-cell results have not been independently converged under mesh/timestep refinement. '
        'The interval-four rerun matches the historical saved polygons exactly. The old count of '
        '184 final-generation persistent changes becomes 381 when zero-duration within-step reversals '
        'are grouped before applying physical dwell. The historical grouped log and the new observer '
        'agree at all three dwell thresholds; this is a measurement change, not changed dynamics.','',
        '## Local exchange, propagation and facilitation','',
        'N=36, W=1, rho=1.3; two seeds; quenches 1 and 5 τT. Selected contacts are the longest eligible '
        'ones, with at most two candidates. This sample cannot establish a population barrier distribution.','',
        '| Domain/layer | Seed | Quench / τT | Contact pair | Excess work / shortening | Initial shape index | Outcome | Secondary excess changes |',
        '|---|---:|---:|---|---:|---:|---|---:|']
    for r in summary['mechanics']:
        lines.append(f"| {r['zone']} | {r['seed']} | {r['quench_tauT']:g} | {r['old_pair']} | {r['excess_work_per_shortening']:.4f} | {r['shape']:.4f} | {r['outcome']} | {r['secondary_excess']} |")
    mechanical=load('mechanics')
    opening=sum(r.get('status')=='complete' and len(r.get('direct_pairs',[]))==1 and
                all(v<0 for v in r['curve']['opposite']) for r in mechanical)
    lines+=['',f'In {opening} of {len(mechanical)} cases both opposite faces are extracellular space. '
            'The imposed foam T1 opens the selected contact; it does not immediately create a four-cell neighbor swap. '
            'All reported reversals are contact reclosures. The work therefore measures resistance to contact '
            'opening along this protocol, not a demonstrated barrier to a lasting four-cell exchange.']
    local=[z for r in mechanical for z in r.get('facilitation',[])]
    valid=sum(all(v['complete'] for v in z['curves'].values()) for z in local)
    lines+=['',f'Only {valid} of {len(local)} local before/after/sham triples have three complete curves. '
            'The others preempt or fail to settle and cannot supply a quantitative facilitation difference.']
    lines+=['','Secondary changes exclude pairs directly modified by the imposed T1 and subtract the '
        'matched unforced sham. A release-without-T1 control is also saved. '
        'Complete force contractions can still lead to reversal rather than a lasting exchange.','',
        '![Local response](local_rearrangement.png)','![Preparation and boundary](boundary_preparation.png)','',
        'Local facilitation measures resistance to 10% shortening of a common nearby or distant '
        'contact. Its comparison with the loaded-before-T1 state and its comparison with the unforced '
        'sham answer different questions. Negative after-minus-loaded values alone do not show that '
        'the tissue has become softer than its unforced state. All three local curves must converge '
        'for a difference to enter the plot. Missing/preempted curves are not zeros.','',
        '## Tension persistence at fixed variance','',
        'Reference ΔT=0.25 at τT=10, W=1.5, rho=1.3. All 12 weak-noise cases complete. '
        'The exact Euler raw contact variance is fixed at 0.00312578, with the original half-amplitude '
        'free-boundary noise. Each run burns for 5 τT and is observed for 100 τR. '
        'Table entries show the two seeds separately. MSD removes translation and unwraps periodic trajectories; '
        'these endpoint displacements are not diffusion coefficients.','',
        '| Domain | τT | Persistent pair changes, seeds 1 / 2 | Final MSD / mean initial cell area, seeds 1 / 2 |',
        '|---|---:|---|---|']
    for domain in ['disc','periodic']:
        for tau in [2.,10.,20.]:
            rows=sorted([r for r in summary['activity'] if r['domain']==domain and r['tau']==tau],key=lambda r:r['seed'])
            lines.append(f"| {domain} | {tau:g} | {' / '.join(str(r['changes']) for r in rows)} | {' / '.join(format(r['msd'],'.5f') for r in rows)} |")
    lines+=['','Longer persistence produces larger displacements in this small pilot, but contact rearrangement '
        'is rare. All four zero-noise controls have zero persistent pair changes. '
        'Two seeds and a finite observation window do not establish a fluid/solid phase boundary.','',
        '![Persistence](tension_persistence.png)','',
        '## Failed regimes and numerical checks','',
        '| Reference ΔT at τT=10 | Completed / attempted activity cases |',
        '|---:|---:|']
    act=load('activity')
    for mu in [0.,.25,.5,3.]:
        rows=[r for r in act if r['parameters']['mu_reference']==mu]
        lines.append(f"| {mu:g} | {sum(r['status']=='complete' for r in rows)} / {len(rows)} |")
    lines+=['','Failed geometry is a numerical limitation, not evidence of melting. '
        'The selected refinement check keeps the initial prepared state, halves dt to 0.0025 and '
        'spacing to 0.05, and preserves the same raw OU variance. It does not share a Brownian path '
        'with the coarser run.','', '| Domain | Reference ΔT | Refined result |','|---|---:|---|']
    for r in load('resolution'):
        p=r['parameters'];lines.append(f"| {p['domain']} | {p['reference_mu']:g} | {r['status']}: {r.get('error','completed observation')} |")
    lines+=['','The mechanical reference gates pass in both domains: maximum work change 1.12% across '
        'step-count, hold-window and timestep refinements, and maximum weak-pulse discrepancy 0.36%. '
        'Input state, RNG and cache identity preservation are covered by tests. '
        'The existing full suite passed 196 tests; the expanded measurement file separately passed four tests.','',
        'Periodic trajectories use the existing planar geometry validation, which lacks the disc’s '
        'independent global crossing/nesting audit. Together with selected contacts, few seeds and '
        'unconverged high-activity geometry, this limits quantitative claims about boundary stiffening.','',
        'Protocol and commands: [COLLECTIVE_EXPERIMENTS.md](../../../circle/COLLECTIVE_EXPERIMENTS.md). '
        'Detailed failures and per-case statistics: [summary.json](summary.json).']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    titles=[('division_timing','Synchronous division timing'),('division_snapshots','256-cell snapshots'),
            ('local_rearrangement','Local exchange and facilitation'),('boundary_preparation','Boundary and preparation'),
            ('tension_persistence','Tension persistence at fixed variance')]
    html=['<!doctype html><meta charset="utf-8"><title>Collective dynamics experiments</title>',
          '<style>body{font:17px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#183238}img{width:100%;height:auto}a{color:#006f7a}</style>',
          '<h1>Collective dynamics experiments</h1><p>Same foam equations; fixed circular wall. Controlled pilots, not a phase diagram.</p>',
          '<p><a href="FINDINGS.md">Findings and limitations</a> · <a href="summary.json">Numerical summary and failures</a></p>']
    for name,title in titles:
        html.append(f'<h2>{title}</h2><p><a href="{name}.png">PNG</a> · <a href="{name}.pdf">PDF</a></p><img src="{name}.png" alt="{title}">')
    (OUT/'index.html').write_text('\n'.join(html)+'\n')
    paths=sorted((ROOT/'activefoam').glob('*.py'))+sorted((ROOT/'circle').rglob('*.py'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),base_commit='efb0f8f1bf6f0dc9b73f6cdeb4d8fab7f5ab3524',
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,source_sha256=hashes,
        commands=['python -m circle.scripts.collective_gates','python -m circle.scripts.collective_mechanics',
            'python -m circle.scripts.collective_division','python -m circle.scripts.collective_activity --mu 3 .5 .25 0',
            'python -m circle.scripts.collective_resolution','python -m circle.scripts.audit_collective_division',
            'python -m circle.scripts.report_collective'])
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
