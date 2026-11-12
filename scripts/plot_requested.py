"""Render only measured full-foam runs from a reproduction manifest.

Missing/failed configurations remain masked. No interpolation/fallback model.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
ROOT=Path(__file__).resolve().parents[1]
COLORS=['#547db5','#dc9625','#87ad39','#dc4b34','#8776b0','#c87225']
ACTIVITY_COLORS = dict(zip([.25, .5, .75, 1., 1.25, 1.5], COLORS))
STRESS_COLORS = {0.: '#41bcb3', .5: COLORS[1], 1.: COLORS[3], 1.5: COLORS[5]}


def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('--out',type=Path,default=ROOT/'figures'/'full_foam')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.data/'manifest.json').read_text());records=[]
    for j in manifest['jobs']:
        f=a.data/(j['id']+'.npz')
        if f.exists():
            with np.load(f,allow_pickle=False) as d:
                records.append((j,{k:d[k].copy() for k in d.files}))
    prefix='PILOT — short validation only' if manifest['pilot'] else 'Full curved-edge foam'
    notes=[]
    produced=[]
    quality=[dict(id=j['id'],checks=int(d.get('quality_checks',0)),microfold_checks=int(d.get('microfold_checks',0)),max_crossing_depth=float(d.get('max_crossing_depth',0.))) for j,d in records]
    (a.out/'geometry_quality.json').write_text(json.dumps(quality,indent=2))
    def save(fig,name):
        produced.append(name)
        if any(q['microfold_checks'] for q in quality):
            fig.text(.5,.005,'Sub-threshold junction folds detected; see geometry_quality.json',ha='center',fontsize=7)
        fig.tight_layout(rect=[0,.025,1,1]);fig.savefig(a.out/(name+'.png'),dpi=180)
        fig.savefig(a.out/(name+'.pdf'));plt.close(fig)
    eq=[(j,d) for j,d in records if j['kind']=='equilibrium']
    all_eq=[j for j in manifest['jobs'] if j['kind']=='equilibrium']
    if all_eq:
        ws=sorted({j['parameters']['w'] for j in all_eq});rhos=sorted({j['parameters']['rho'] for j in all_eq})
        comparison = {'WS': ws, 'RHOS': rhos}
        for quantity,name,label,limits in [('phi','fig2b',r'$\phi$',(0.6,1.)),('z','fig2c',r'$z$',(0.,6.))]:
            means=np.full((len(rhos),len(ws)),np.nan);sd=means.copy();counts=np.zeros_like(means,int)
            for i,rho in enumerate(rhos):
                for k,w in enumerate(ws):
                    vals=[float(d[quantity]) for j,d in eq if j['parameters']['rho']==rho and j['parameters']['w']==w]
                    if vals:means[i,k]=np.mean(vals);sd[i,k]=np.std(vals);counts[i,k]=len(vals)
            fig,ax=plt.subplots(figsize=(6.3,4.8));ax.set_facecolor('0.85')
            im=ax.pcolormesh(ws,rhos,np.ma.masked_invalid(means),shading='nearest',cmap='RdYlBu_r',vmin=limits[0],vmax=limits[1])
            finite = means[np.isfinite(means)]
            levels = (.8, .9) if quantity == 'phi' else (3., 4., 5.)
            if finite.size and min(means.shape) >= 2:
                levels = [level for level in levels if finite.min() < level < finite.max()]
                if levels:
                    contours = ax.contour(ws, rhos, np.ma.masked_invalid(means),
                                          levels=levels, colors='k', linestyles='--',
                                          linewidths=.8, corner_mask=False)
                    ax.clabel(contours, fmt='%g', fontsize=8)
            fig.colorbar(im,ax=ax,label=label)
            ax.set(xlabel=r'$W/T_0$',ylabel=r'$\rho$',title=f'{name[3:]} | {prefix}\n{manifest["n_side"]**2} cells; up to {manifest["seeds"]} seeds per point')
            ax.set_xlim(ws[0],ws[-1]);ax.set_ylim(rhos[0],rhos[-1])
            ax.text(.02,.02,'Gray = missing/invalid run',transform=ax.transAxes,fontsize=8)
            save(fig,name)
            np.savez(a.out/(name+'_summary.npz'),w=ws,rho=rhos,mean=means,sd=sd,count=counts)
            comparison[quantity] = means
            comparison[f'{quantity}_count'] = counts
        np.savez(a.out/'fig2bc_model.npz', **comparison)
    dyn=[(j,d) for j,d in records if j['kind']=='dynamics']
    ms=[(j,d) for j,d in dyn if j['parameters']['w']==1.]
    if ms:
        fig,ax=plt.subplots(figsize=(6.8,5.));inset=ax.inset_axes([.13,.64,.31,.30])
        for mu in sorted({j['parameters']['mu'] for j,d in ms}):
            color = ACTIVITY_COLORS[mu]
            ds=[d for j,d in ms if j['parameters']['mu']==mu];t=ds[0]['tlag']
            values=np.array([d['msd'] for d in ds]);mean=values.mean(0);sd=values.std(0)
            ax.loglog(t,mean,color=color,label=f'{mu:g} (n={len(ds)})')
            if len(ds)>1:ax.fill_between(t,np.maximum(mean-sd,1e-20),mean+sd,color=color,alpha=.2)
            mask=(t>=10)&(t<=100)&(mean>0)
            if mask.sum()>=3:
                alpha=np.polyfit(np.log(t[mask]),np.log(mean[mask]),1)[0]
                inset.plot(mu,alpha,'o',color=color);notes.append(dict(quantity='alpha',mu=mu,value=float(alpha),window_tauT=[10,100]))
        inset.axhline(1,color='0.5',lw=.6);inset.set(xlabel=r'$\Delta T/T_0$',ylabel=r'$\alpha$');inset.tick_params(labelsize=7)
        ax.set(xlabel=r'$t/\tau_T$',ylabel=r'$\mathrm{MSD}/L_0^2$',title=f'3a | {prefix}; W/T₀=1, ρ=1')
        ax.legend(title=r'$\Delta T/T_0$',fontsize=8);save(fig,'fig3a')
    if dyn:
        fig,ax=plt.subplots(figsize=(6.8,4.8))
        jobs = [j['parameters'] for j in manifest['jobs'] if j['kind'] == 'dynamics']
        for mu in sorted({j['parameters']['mu'] for j,d in dyn}):
            ws = sorted({p['w'] for p in jobs if p['mu'] == mu})
            means, errors = [], []
            for w in ws:
                vals=[float(d['shape_mean']) for j,d in dyn if j['parameters']['w']==w and j['parameters']['mu']==mu]
                means.append(np.mean(vals) if vals else np.nan)
                errors.append(np.std(vals) if len(vals) > 1 else np.nan)
            means, errors = np.asarray(means), np.asarray(errors)
            color = ACTIVITY_COLORS[mu]
            ax.plot(ws, means, 'o-', ms=3, color=color, label=f'{mu:g}')
            ax.fill_between(ws, means-errors, means+errors, color=color, alpha=.18)
        ax.axhline(3.81,color='0.5',ls='--',lw=.7,label='3.81 reference')
        ax.axvline(1., color='0.5', ls='--', lw=.7)
        ax.set(xlabel=r'$W/T_0$',ylabel=r'$\langle P/\sqrt{A}\rangle$',title=f'4g | {prefix}; ρ=1')
        ax.legend(title=r'$\Delta T/T_0$',fontsize=8,ncol=2);save(fig,'fig4g')
    stress=[(j,d) for j,d in records if j['kind']=='stress']
    if stress:
        fig,ax=plt.subplots(figsize=(6.8,4.8))
        inset = ax.inset_axes([.65, .57, .31, .36])
        def stretched(t,offset,amp,tau,beta):return offset+amp*np.exp(-(t/tau)**beta)
        for mu in sorted({j['parameters']['mu'] for j,d in stress}):
            color = STRESS_COLORS[mu]
            ds=[d for j,d in stress if j['parameters']['mu']==mu];t=ds[0]['time'];vals=np.array([d['stress'] for d in ds]);mean=vals.mean(0);sd=vals.std(0)
            ok=t>0;ax.semilogx(t[ok],mean[ok],color=color,label=f'{mu:g} (n={len(ds)})')
            if len(ds)>1:ax.fill_between(t[ok],mean[ok]-sd[ok],mean[ok]+sd[ok],color=color,alpha=.2)
            mask=t>=1.
            positive = mask & (mean > 0)
            inset.loglog(t[positive], mean[positive], color=color, lw=.8)
            sigma_a = float(np.mean([d['sigma_A'] for d in ds]))
            if sigma_a > 0 and mu > 0:
                inset.axhline(sigma_a, color=color, lw=.65, ls=':')
            if not manifest['pilot'] and mu > 0:
                try:
                    fit,cov=curve_fit(stretched,t[mask],mean[mask],p0=[mean[-1],max(mean[mask][0]-mean[-1],.01),30.,.5],bounds=([-np.inf,0,.005,.1],[np.inf,np.inf,1e6,1.0]),maxfev=20000)
                    ax.semilogx(t[mask],stretched(t[mask],*fit),'k--',lw=.8)
                    fit_curve = stretched(t[mask], *fit)
                    inset.loglog(t[mask], np.where(fit_curve > 0, fit_curve, np.nan), 'k--', lw=.65)
                    notes.append(dict(quantity='stress_fit',mu=mu,parameters=fit.tolist(),parameter_sd=np.sqrt(np.diag(cov)).tolist(),fit_start_tauR=1.,sigma_A=sigma_a))
                except (ValueError,RuntimeError) as ex:notes.append(dict(quantity='stress_fit',mu=mu,error=str(ex)))
        ax.set(xlabel=r'$t/\tau_R$',ylabel=r'$\sigma_{xy}/\sigma_0$',title=f'4a | {prefix}; W/T₀=0.5, ρ=1, strain=1.5')
        inset.set(xlabel=r'$t/\tau_R$', ylabel=r'$\sigma_{xy}/\sigma_0$')
        inset.tick_params(labelsize=7)
        inset.xaxis.label.set_size(8)
        inset.yaxis.label.set_size(8)
        ax.legend(title=r'$\Delta T/T_0$',fontsize=8,loc='lower left');save(fig,'fig4a')
    (a.out/'analysis.json').write_text(json.dumps(dict(source_hash=manifest['source_hash'],completed=len(records),requested=len(manifest['jobs']),notes=notes),indent=2))
    cards=''.join(f'<section><h2>{name.replace("fig","Figure ")}</h2><a href="{name}.png"><img src="{name}.png" alt="{name}"></a><p><a href="{name}.pdf">PDF</a> · <a href="{name}.png">PNG</a></p></section>' for name in produced)
    html=f"""<!doctype html><meta charset="utf-8"><title>Active foam reproduction</title>
    <style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172331;background:#f7f8fa}}h1{{font-size:30px}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:20px}}section{{background:white;padding:16px;border:1px solid #dde2e7;border-radius:8px}}img{{width:100%;height:auto}}p{{line-height:1.5}}code{{overflow-wrap:anywhere}}</style>
    <h1>Active foam: requested figure reproductions</h1><p>{prefix}. {len(records)}/{len(manifest['jobs'])} simulations completed. {manifest['n_side']**2} cells; {manifest['seeds']} seed(s) per parameter point. Full model throughout; dt=0.005 τ_R.</p>
    <p>This finite-size, coarse-grid run is not the published 10–20-run ensemble. Sub-threshold junction folds are recorded; resolved tangles are rejected. Missing data are not imputed.</p>
    <p><a href="../../REPRODUCTION.md">Protocols and limitations</a> · <a href="geometry_quality.json">Geometry diagnostics</a> · <a href="analysis.json">Fit results</a></p><main>{cards}</main><p>Source hash: <code>{manifest['source_hash']}</code></p>"""
    (a.out/'index.html').write_text(html)
    print(f'Rendered {len(records)}/{len(manifest["jobs"])} recorded simulations into {a.out}')
if __name__=='__main__':main()
