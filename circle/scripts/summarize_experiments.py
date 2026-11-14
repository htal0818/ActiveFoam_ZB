"""Produce numerical findings and seed-level group summaries for the experiment gallery."""
import csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle.scripts.run_experiments import load_records
from circle.scripts.plot_experiments import scalar


def main():
    _,records=load_records(ROOT/'circle/data/experiments_20261114');out=ROOT/'figures/circle/experiments_20261114'
    keys=('family','n','w','rho','mu','tau','aspect','pattern','dt','spacing');groups=defaultdict(list)
    for r in records:groups[tuple(r['task'][k] for k in keys)].append(r)
    rows=[]
    for key,group in groups.items():
        accepted=[r for r in group if r['accepted']];row=dict(zip(keys,key));row.update(attempted=len(group),completed=len(accepted))
        for metric in ('shape_mean','contacts','event_rate','packing_mean','final_neighbor_overlap','final_continuous_survival'):
            values=[scalar(r,metric) for r in accepted];row[metric+'_mean']=float(np.mean(values)) if values else None
            row[metric+'_seed_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        rows.append(row)
    with (out/'group_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    pulse=[]
    for n in (36,100):
        shams={r['task']['seed']:r for r in records if r['task']['family']=='pulse' and r['task']['pattern']=='uniform' and r['task']['n']==n and r['accepted']}
        for pattern in ('pulse','wall','sector'):
            group=[r for r in records if r['task']['family']=='pulse' and r['task']['pattern']==pattern and r['task']['n']==n and r['accepted'] and r['task']['seed'] in shams]
            delta=[]
            for r in group:
                sham=shams[r['task']['seed']]
                delta.append(np.array([np.mean(s['shape']) for s in r['samples']])-np.array([np.mean(s['shape']) for s in sham['samples']]))
            curves=np.array(delta);mean=curves.mean(axis=0);peak=int(mean.argmax());times=[s['time'] for s in group[0]['samples']]
            pulse.append(dict(n=n,pattern=pattern,paired_seeds=len(group),peak_mean_shape_change=float(mean[peak]),
                time_of_peak=times[peak],seed_sd_at_peak=float(curves[:,peak].std(ddof=1)),
                final_mean_shape_change=float(mean[-1]),final_seed_sd=float(curves[:,-1].std(ddof=1))))
    (out/'pulse_effects.json').write_text(json.dumps(pulse,indent=2)+'\n')
    report=['# Findings from the first dynamics experiments','',
        '**143 simulations were executed: 137 experiment runs plus six validation-cadence controls.** The main batch completed 96 runs; 41 stopped on geometry errors. All six additional controls passed every-timestep checks after preparation and matched the corresponding ordinary trajectories and event records exactly. The regression suite passes 158 tests.','',
        'These are finite-time exploratory results with three seeds per scientific point. They do not establish phase boundaries. Passing the checks retains the documented microfold tolerance and does not certify unmonitored preparation intervals.','',
        '## Frequent operations mostly reverse local contacts','',
        'For N=36, ρ=1.2, μ=0.5, the two completed W=1.5 seeds produced 80 and 222 contact-changing transactions over 10τT, involving only 11 and 12 distinct pairs. Their final initial-neighbor overlap was 98.85% and 96.55%, while unbroken contact survival was 89.66% and 90.80%. Many initially lost contacts therefore re-formed. The third seed failed geometry checks. Event counts alone would overstate lasting neighbor exchange.','',
        'The event catalog separates committed T1/T3/T4 operations and provides waiting-time and clustering diagnostics. Repeated reversals can form large event clusters without a collective rearrangement cascade; no avalanche claim is made.','',
        '## Persistence is different from noise amplitude','',
        '| Protocol | τT/τR | Mean transactions / cell / τR | Mean shape index | Seeds |','|---|---:|---:|---:|---:|']
    for row in rows:
        if row['family'].startswith('persistence_'):
            report.append(f"| {row['family'].replace('persistence_','').replace('_',' ')} | {row['tau']:g} | {row['event_rate_mean']:.5g} | {row['shape_mean_mean']:.5f} | {row['completed']}/{row['attempted']} |")
    report+=['',
        'At fixed μ, shorter persistence also increases stationary tension variance. Matching the raw Euler–Maruyama variance largely removes the shape trend and changes the event-rate trend. The τT=2 fixed-variance runs recorded no events within their 20τR observation windows; that is a finite detection limit, not proof of a zero rate. Observation times scale with τT, and the τT=10 controls are duplicated across protocols rather than independent replicas.','',
        '## Activity pulses give a transient response and recovery','',
        'All 24 pulse/sham runs completed. Every intervention matches its same-seed sham before activation. μ rises from 0.25 to 1 during t∈[10,20)τR. The table describes the peak of the **seed-mean paired shape-change curve**, not the mean of individual peak amplitudes.','',
        '| N | Activated region | Peak mean Δq | Seed SD at that time | Peak time / τR | Final mean Δq |','|---:|---|---:|---:|---:|---:|']
    for row in pulse:report.append(f"| {row['n']} | {row['pattern']} | {row['peak_mean_shape_change']:.4f} | {row['seed_sd_at_peak']:.4f} | {row['time_of_peak']:g} | {row['final_mean_shape_change']:.4f} |")
    report+=['',
        'The uniform pulse raises the whole-tissue shape index by roughly 0.05 before relaxation. Wall and sector pulses produce smaller whole-tissue responses. They activate different numbers of films and are not matched for total noise input, so this is not a per-film efficiency comparison. Small residual changes and variation among seeds remain visible in the paired curves.','',
        '## Long observations remain mostly caged at W=1','',
        'After 20τT warmup and 50τT observation, all three W=1 runs retained 98.85–100% of their initial neighbors. At lag 12.5τT, MSD/A₀ ranges from 0.00558 to 0.00642 (RMS displacement about 0.075–0.080L₀). Shape block means show some slow drift. These runs do not establish long-time diffusion or a solid/fluid transition. None of the three W=1.5 trajectories completed the long protocol.','',
        '## Equal-area ellipses vary wall curvature at fixed N','',
        '| Aspect a/b | Mean q ± seed SD | Mean contacts ± seed SD | Completed |','|---:|---:|---:|---:|']
    for row in rows:
        if row['family']=='geometry':report.append(f"| {row['aspect']:g} | {row['shape_mean_mean']:.4f} ± {row['shape_mean_seed_sd']:.4f} | {row['contacts_mean']:.4f} ± {row['contacts_seed_sd']:.4f} | {row['completed']}/{row['attempted']} |")
    report+=['',
        'The whole-tissue shape differences are comparable to seed variation in this small sample. Aspect 2 loses one seed during preparation. Local curvature/contact/shape profiles and final foam snapshots are provided; global anisotropy and local curvature still covary, so there is no isolated causal curvature conclusion.','',
        '## Numerical limits determine the next step','',
        '- All eight μ=1.5 refinement tests failed, including dt=0.0025 and spacing=0.075. No high-activity physical phase is assigned. All 18 constant-μ=1 active-adhesion runs also failed their longer protocols, although the shorter μ=1 pulses completed.',
        '- Of 12 low-adhesion/density runs, only four W=0.35 trajectories completed; all W=0.25 attempts failed. The completed W=0.35 runs have measured packing about 0.765 at ρ=0.9 and 0.846 at ρ=1. These are conditional observations, not proof of a fragmentation transition.',
        '- Selected baseline controls pass every-step checks, but transient crossings were detected when checking preparation more finely. Preparation and transition/integration geometry need further work before increasing activity or making quantitative phase-boundary claims.',
        '- A useful next experiment is a refined-mesh, multi-seed confirmation of the paired pulse response and fixed-variance persistence trend, while diagnosing the first crossing events in the saved failed configurations. Expanding the high-activity grid before fixing this would chiefly map numerical failure.',
        '',
        'See [the figure gallery](index.html), [methods and acceptance counts](REPORT.md), [per-run summaries](summary.csv), [group means and seed SD](group_summary.csv), and [contact reversals](contact_reversals.csv). Raw compressed trajectories and event catalogs are in `circle/data/experiments_20261114/runs/`.']
    (out/'FINDINGS.md').write_text('\n'.join(report).rstrip()+'\n')
    print(json.dumps(pulse,indent=2))


if __name__=='__main__':main()
