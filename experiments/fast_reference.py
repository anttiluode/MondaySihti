"""Optional Numba acceleration of G3's independent reference implementation.

The authoritative source comparison remains upstream Operaattori's original solver.
This is the same backward-Euler + implicit NMDA system, compiled for throughput.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit
from experiments.nonlinear import Result

@njit(cache=True)
def fast_run(P,X,sites,ga,gn):
    nsite=len(sites);nt=ga.shape[1]
    state=np.zeros(P.shape[0]);prev=np.zeros(nsite);soma=np.zeros(nt)
    R=X[sites,:];maxiter=0;maxerr=0.
    for ti in range(nt):
        passive=P@state;ps=passive[sites];z=prev.copy()
        for itr in range(31):
            v=-70.+z;b=1./(1.+np.exp(-.078*v)/3.57)
            j=(ga[:,ti]+gn[:,ti]*b)*(-v)
            resid=z-ps-R@j
            if np.max(np.abs(resid))<=1e-10:break
            bp=.078*b*(1.-b)
            dj=gn[:,ti]*bp*(-v)-(ga[:,ti]+gn[:,ti]*b)
            jac=np.eye(nsite)-R@np.diag(dj)
            delta=np.linalg.solve(jac,resid)
            old=float(np.max(np.abs(resid)));alpha=1.
            accepted=False
            for _ in range(9):
                cand=z-alpha*delta
                vv=-70.+cand;bb=1./(1.+np.exp(-.078*vv)/3.57)
                jj=(ga[:,ti]+gn[:,ti]*bb)*(-vv)
                rr=cand-ps-R@jj
                if float(np.max(np.abs(rr)))<old:
                    z=cand;accepted=True;break
                alpha*=.5
                if alpha<1./256-1e-15:break
            if not accepted:raise ValueError('Numba Newton failed')
        if itr>=30:raise ValueError('Numba Newton iteration limit')
        maxiter=max(maxiter,itr)
        v=-70.+z;b=1./(1.+np.exp(-.078*v)/3.57)
        j=(ga[:,ti]+gn[:,ti]*b)*(-v)
        state=passive+X@j
        maxerr=max(maxerr,float(np.max(np.abs(state[sites]-z))))
        soma[ti]=state[0];prev=z
    return soma,maxiter,maxerr

class FastCable:
    def __init__(self):
        from experiments.nonlinear import ReferenceCable,SITES
        ref=ReferenceCable();self.P=ref.P;self.X=ref.X;self.sites=np.asarray(SITES,dtype=np.int64)
    def run(self,ga,gn):
        y,it,err=fast_run(self.P,self.X,self.sites,ga,gn)
        return Result(y,True,it,err)

def main():
    import json
    from pathlib import Path
    from experiments.nonlinear import ReferenceCable,report,program,SEEDS
    a=ReferenceCable();b=FastCable()
    for seed in [1,2,17,26,200]:
        rng=np.random.default_rng(seed)
        ga,gn,pga,pgn=program(rng,(seed%4,(seed//4)%4))
        for u,v in [(ga,gn),(pga,pgn)]:
            x=a.run(u,v);y=b.run(u,v)
            d=np.max(np.abs(x.soma_depolarization_mV-y.soma_depolarization_mV))
            print('pilot',seed,'max',d)
            if d>1e-8:raise ValueError(f'Numba disagreement {d}')
    out=report(runner=b)
    path=Path('data/nonlinear_results.json');path.write_text(json.dumps(out,indent=2)+'\n')
    print('Saved',path)
    for k,v in out['methods'].items():print(k,v['mean_rmse'])
    print('GATES',out['gates'])

if __name__=='__main__':main()
