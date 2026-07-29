import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.deformable import DeformableTissue
os.makedirs("data", exist_ok=True)
WS=np.round(np.arange(0.0,1.51,0.25),3); RHOS=[1.0,0.9,0.8]; SEEDS=[1,2,3]
def run(w,rho,s):
    dt=DeformableTissue(n_cells=36,m=24,rho=rho,w=w,seed=s)
    dt.minimize(n_steps=1800,dt=0.005)
    return dt.volume_fraction(), dt.contact_number()
def main():
    t0=time.time(); phi=np.zeros((len(RHOS),len(WS))); zz=np.zeros((len(RHOS),len(WS)))
    for i,rho in enumerate(RHOS):
        for j,w in enumerate(WS):
            pv=[];zv=[]
            for s in SEEDS:
                p,z=run(w,rho,s); pv.append(p); zv.append(z)
            phi[i,j]=np.mean(pv); zz[i,j]=np.mean(zv)
        print(f"rho={rho}: phi={np.round(phi[i],3)} z={np.round(zz[i],2)} [{time.time()-t0:.0f}s]",flush=True)
    np.savez("data/fig2bc_def.npz",WS=WS,RHOS=np.array(RHOS),phi=phi,z=zz)
    print("saved")
main()
