"""Benchmark identical seeded full-foam states; exclude JIT compilation."""
import sys,time,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from activefoam.foam_full import FoamTissue
from activefoam.topology import build_periodic_voronoi

def main():
    base=FoamTissue(build_periodic_voronoi(6,np.random.default_rng(1)),w=.3,rho=.9,seed=1)
    for v in range(len(base.vpos)):base.t2_reverse(v)
    warm=copy.deepcopy(base);warm.accelerated=True;warm.step(.5);warm.do_transitions()
    result={}
    for fast in (False,True):
        f=copy.deepcopy(base);f.accelerated=fast
        t=time.perf_counter()
        for i in range(100):f.step(.5)
        step=(time.perf_counter()-t)/100
        t=time.perf_counter();f.do_transitions();transition=time.perf_counter()-t
        result['numba' if fast else 'python']={'step_seconds':step,'transition_seconds':transition}
    result['step_speedup']=result['python']['step_seconds']/result['numba']['step_seconds']
    print(json.dumps(result,indent=2),flush=True)
    Path(__file__).resolve().parents[1].joinpath('data/benchmark.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
