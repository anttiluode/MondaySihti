#!/usr/bin/env python3
"""G3: hidden input history -> noisy soma observations -> later probe response.

An explicitly synthetic, supervised inverse-prediction challenge, not QKV recovery.
When --operaattori is supplied, uses pinned original CausalGraphCircuit and
cross-checks the local reference implementation on every evaluated program.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from experiments.bridge import cable_graph

DT = .15
NSTEPS = 200
SITES = [2, 3, 4, 5, 6]
PRE_TIMES = [4.05, 6.45, 8.55, 10.5, 12.6, 14.55, 16.5, 18.0]
POST_TIMES = [20.7, 23.25, 26.1, 28.8]
PRE_IDX = [int(round(t/DT)) for t in PRE_TIMES]
POST_IDX = [int(round(t/DT)) for t in POST_TIMES]
ALPHAS = [1e-4, 1e-2, 1e-1, 1., 10.]
SEEDS = [101, 211, 307, 419, 523]
TRAIN, VAL, TEST = 96, 48, 64
TRAIN_PAIRS = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,3),(2,0),(2,2),(3,0),(3,1)]
VAL_PAIRS = [(0,3),(1,2),(2,1)]
TEST_PAIRS = [(2,3),(3,2),(3,3)]

@dataclass
class Result:
    soma_depolarization_mV: np.ndarray
    all_steps_converged: bool
    max_newton_iterations: int
    max_site_consistency_mV: float

class ReferenceCable:
    """Independent, restricted-equation reference for upstream numerical comparison.

    Implements Operaattori's implicit local AMPA/NMDA cable equations (not a
    substitute for the original; CI invokes and checks the actual upstream).
    """
    def __init__(self):
        G,C=cable_graph(); n=len(C)
        A=G+np.diag(C/DT)
        B=np.zeros((n,len(SITES))); B[SITES,np.arange(len(SITES))]=1
        RHS=np.column_stack([np.diag(C/DT),B])
        Z=np.linalg.solve(A,RHS)
        self.P=Z[:,:n]; self.X=Z[:,n:]; self.R=self.X[SITES]

    def run(self,ga,gn):
        state=np.zeros(7); prev=np.zeros(len(SITES)); soma=np.zeros(ga.shape[1]); maxiter=0; maxerr=0.
        for i in range(ga.shape[1]):
            passive=self.P@state; ps=passive[SITES]; z=prev.copy()
            for itr in range(31):
                v=-70.+z; b=1./(1.+np.exp(-.078*v)/3.57)
                j=(ga[:,i]+gn[:,i]*b)*(-v)
                resid=z-ps-self.R@j
                if np.max(np.abs(resid))<=1e-10: break
                bp=.078*b*(1.-b)
                dj=gn[:,i]*bp*(-v)-(ga[:,i]+gn[:,i]*b)
                jac=np.eye(len(SITES))-self.R@np.diag(dj)
                delta=np.linalg.solve(jac,resid)
                original=float(np.max(np.abs(resid))); alpha=1.
                accepted=False
                for _ in range(9):
                    candidate=z-alpha*delta
                    vv=-70.+candidate; bb=1./(1.+np.exp(-.078*vv)/3.57)
                    jj=(ga[:,i]+gn[:,i]*bb)*(-vv)
                    rr=candidate-ps-self.R@jj
                    if float(np.max(np.abs(rr)))<original:
                        z=candidate;accepted=True;break
                    alpha*=.5
                    if alpha<1./256-1e-15:break
                if not accepted:raise ArithmeticError('Independent Newton solve failed')
            else:raise ArithmeticError('Independent Newton did not converge')
            maxiter=max(maxiter,itr)
            v=-70.+z; b=1./(1.+np.exp(-.078*v)/3.57)
            j=(ga[:,i]+gn[:,i]*b)*(-v)
            state=passive+self.X@j
            maxerr=max(maxerr,float(np.max(np.abs(state[SITES]-z))))
            soma[i]=state[0];prev=z
        return Result(soma,True,maxiter,maxerr)


def pulse(t,onset,rise,decay):
    lag=t-onset
    a=np.where(lag>0,np.exp(-np.maximum(lag,0)/decay)-np.exp(-np.maximum(lag,0)/rise),0.)
    m=float(np.max(a));return a/m if m>0 else a


def program(rng,pair):
    # Inputs are never passed to the learned observer, only to the hidden plant.
    tt=np.arange(NSTEPS)*DT
    ga=np.zeros((len(SITES),NSTEPS));gn=np.zeros_like(ga)
    conditioning=pulse(tt,3.0+rng.uniform(-.45,.45),.65,14.)
    a=rng.uniform(.35,1.55)
    ga[-1]+=.125*a*conditioning;gn[-1]+=.34*a*conditioning
    for which,(site,at) in enumerate(zip(pair,[5.0,13.2])):
        wave=pulse(tt,at+rng.uniform(-1.0,1.0),.45,8.0)
        scale=rng.uniform(.48,1.70)
        ga[site]+=.085*scale*wave;gn[site]+=.38*scale*wave
    probega=ga.copy();probegn=gn.copy()
    probe=pulse(tt,18.7,.38,5.5)
    # Standardized controlled probe to port #3 (SITES index 1).
    probega[1]+=.06*probe;probegn[1]+=.28*probe
    return ga,gn,probega,probegn


def make_split(runner,seed,n,pairs,noise=.06,reference=None,comparison=None):
    rng=np.random.default_rng(seed)
    readings=[];targets=[];gaps=[]
    for k in range(n):
        pair=pairs[int(rng.integers(len(pairs)))]
        ga,gn,pga,pgn=program(rng,pair)
        baseline=runner.run(ga,gn);intervened=runner.run(pga,pgn)
        if not baseline.all_steps_converged or not intervened.all_steps_converged:raise ArithmeticError('upstream solver nonconvergence')
        if comparison is not None:
            # Fresh real upstream response versus independent implementation,
            # on both the baseline and probe programs.
            for xa,xb in [(baseline,comparison.run(ga,gn)),(intervened,comparison.run(pga,pgn))]:
                err=float(np.max(np.abs(xa.soma_depolarization_mV-xb.soma_depolarization_mV)))
                if err>1e-8:raise AssertionError(f'Original-vs-reference mismatch: {err:.12g} mV')
                gaps.append(err)
        observed=baseline.soma_depolarization_mV[PRE_IDX]+rng.normal(0,noise,len(PRE_IDX))
        target=(intervened.soma_depolarization_mV-baseline.soma_depolarization_mV)[POST_IDX]
        readings.append(observed);targets.append(target)
    return np.asarray(readings),np.asarray(targets),max(gaps,default=0.)


def residue8(X):
    return np.column_stack((X[:,:-1]-X[:,1:],X[:,-1]))


def restore8(R):
    return np.flip(np.cumsum(np.flip(R,axis=1),axis=1),axis=1)


def features(X,degree=2):
    if degree==1:return X
    n,d=X.shape
    return np.column_stack([X]+[(X[:,i]*X[:,j]) for i in range(d) for j in range(i,d)])


def fit_ridge(X,Y,W,alpha):
    mean=X.mean(0); scale=X.std(0);scale[scale<1e-8]=1.
    Z=(X-mean)/scale
    mY=Y.mean(0)
    ZZ=features(Z);ww=features((W-mean)/scale)
    mx=ZZ.mean(0); ZZ-=mx; ww-=mx
    beta=np.linalg.solve(ZZ.T@ZZ+alpha*np.eye(ZZ.shape[1]),ZZ.T@(Y-mY))
    return ww@beta+mY


def err(a,b):return float(np.sqrt(np.mean((a-b)**2)))


def select(X,YY,split,n=3):
    a,b,c=split;best=None;selected=None
    for idx in itertools.combinations(range(X.shape[1]),n):
        Xs=X[:,list(idx)]
        for alpha in ALPHAS:
            pred=fit_ridge(Xs[:a],YY[:a],Xs[a:a+b],alpha)
            v=err(pred,YY[a:a+b]);key=(v,idx,alpha)
            if best is None or key<best:
                best=key;selected=pred
    validation,idx,alpha=best
    pred=fit_ridge(X[:a,list(idx)],YY[:a],X[a+b:a+b+c,list(idx)],alpha)
    return dict(val_rmse=validation,test_rmse=err(pred,YY[a+b:a+b+c]),choice=list(idx),alpha=alpha),pred


def select_pca(X,YY,split,k):
    a,b,c=split
    mu=X[:a].mean(0);_,_,vt=np.linalg.svd(X[:a]-mu,full_matrices=False)
    Z=(X-mu)@vt[:k].T
    alpha=min(ALPHAS,key=lambda al:err(fit_ridge(Z[:a],YY[:a],Z[a:a+b],al),YY[a:a+b]))
    pred=fit_ridge(Z[:a],YY[:a],Z[a+b:a+b+c],alpha)
    return dict(test_rmse=err(pred,YY[a+b:a+b+c]),alpha=alpha),pred


def select_snapshot(X,YY,split):
    a,b,c=split;X=X[:,-1,None]
    alpha=min(ALPHAS,key=lambda al:err(fit_ridge(X[:a],YY[:a],X[a:a+b],al),YY[a:a+b]))
    pred=fit_ridge(X[:a],YY[:a],X[a+b:a+b+c],alpha)
    return dict(test_rmse=err(pred,YY[a+b:a+b+c]),alpha=alpha),pred


def run_seed(seed,runner,comparison=None,ntrain=TRAIN,nval=VAL,ntest=TEST):
    # Disjoint sequences/port-pair families; disjoint PRNG streams.
    X0,Y0,d0=make_split(runner,seed*10+1,ntrain,TRAIN_PAIRS,comparison=comparison)
    X1,Y1,d1=make_split(runner,seed*10+2,nval,VAL_PAIRS,comparison=comparison)
    X2,Y2,d2=make_split(runner,seed*10+3,ntest,TEST_PAIRS,comparison=comparison)
    X=np.vstack([X0,X1,X2]); Y=np.vstack([Y0,Y1,Y2]); N=(ntrain,nval,ntest)
    R=residue8(X)
    if float(np.max(np.abs(X-restore8(R))))>1e-12:raise AssertionError('Residue failed exact reconstruction')
    results={};preds={}
    for name,arr in [('raw_three',X),('residue_three',R)]:results[name],preds[name]=select(arr,Y,N)
    results['pca_three'],preds['pca_three']=select_pca(X,Y,N,3)
    results['snapshot'],preds['snapshot']=select_snapshot(X,Y,N)
    results['full_eight'],preds['full_eight']=select_pca(X,Y,N,8) # higher observation budget; not a privileged true-state oracle
    return dict(seed=seed,methods=results,example=dict(pre_soma_observed=X2[0].round(6).tolist(),
                 response_actual=Y2[0].round(6).tolist(),
                 predictions={k:v[0].round(6).tolist() for k,v in preds.items()}),
                diagnostics=dict(reference_error_mV=max(d0,d1,d2),
        residue_roundtrip_max=float(np.max(np.abs(X-restore8(R)))),
        target_rms=float(np.sqrt(np.mean(Y**2))),target_variation=float(np.sqrt(np.mean((Y-Y.mean(0))**2))),
        first_example_readings=X2[0].round(6).tolist(),first_example_probe=Y2[0].round(6).tolist()))


def report(seeds=SEEDS,runner=None,comparison=None,ntrain=TRAIN,nval=VAL,ntest=TEST):
    if runner is None:runner=ReferenceCable()
    rows=[run_seed(s,runner,comparison,ntrain,nval,ntest) for s in seeds]
    methods={name:dict(mean_rmse=float(np.mean([r['methods'][name]['test_rmse'] for r in rows])),
          per_seed_rmse=[r['methods'][name]['test_rmse'] for r in rows])
         for name in ['snapshot','raw_three','residue_three','pca_three','full_eight']}
    raw,res,pca,snap=[methods[x]['mean_rmse'] for x in ['raw_three','residue_three','pca_three','snapshot']]
    gates=dict(G3_0_original_source_verified=bool(comparison is not None and max(r['diagnostics']['reference_error_mV'] for r in rows)<=1e-8),
               G3_1_exact_residue_inverse=all(r['diagnostics']['residue_roundtrip_max']<1e-12 for r in rows),
               G3_2_residue_5pct_better_than_matched=bool(res<=.95*min(raw,pca)),
               G3_3_both_histories_5pct_better_than_snapshot=bool(raw<=.95*snap and res<=.95*snap))
    return dict(schema='mondaysihti.nonlinear.v1',date='2026-10-12',source='7-compartment illustrative passive cable with implicit voltage-dependent AMPA/NMDA',
                context='Inputs fully hidden from learner; eight noisy soma pre-probe samples; four post-probe response targets',
                data_sizes=[ntrain,nval,ntest],seeds=seeds,observed_times_ms=PRE_TIMES,probe_times_ms=POST_TIMES,
                method='Equal three-scalar observation budgets, quadratic ridge regression, train-validation selected regularization; disjoint port-pair families',
                assumptions=dict(known_family='Only learned polynomial feature mapping, not hidden circuit/source input schedule',
                                 simulation_only=True,independent_upstream_verification=comparison is not None),
                methods=methods,by_seed=rows,gates=gates)


def main():
    p=argparse.ArgumentParser();p.add_argument('--operaattori');p.add_argument('--output',default='data/nonlinear_results.json');p.add_argument('--check',action='store_true');p.add_argument('--pilot',action='store_true')
    args=p.parse_args();compare=None;runner=ReferenceCable()
    if args.operaattori:
        repo=Path(args.operaattori).resolve()
        if not (repo/'reduced'/'causal_graph_circuit.py').is_file():p.error('original Operaattori module missing')
        sys.path.insert(0,str(repo));from reduced.causal_graph_circuit import CausalGraphCircuit
        G,C=cable_graph();runner=CausalGraphCircuit.from_dense_passive_graph(G,C,SITES,soma_node=0,dt_ms=DT)
        compare=ReferenceCable()
    if args.pilot:
        result=report([1],runner,comparison=compare,ntrain=18,nval=10,ntest=12)
    else:
        result=report(runner=runner,comparison=compare)
    print('Results')
    for name,m in result['methods'].items():print(name,round(m['mean_rmse'],7),'mV')
    print('gates',result['gates'])
    print('target rms',result['by_seed'][0]['diagnostics']['target_rms'])
    if args.check:
        prior=json.loads(Path(args.output).read_text())
        for name in prior['methods']:
            gap=abs(result['methods'][name]['mean_rmse']-prior['methods'][name]['mean_rmse'])
            if gap>1e-6:raise SystemExit(f'FAILED {name} difference {gap} mV')
        if args.operaattori and not result['gates']['G3_0_original_source_verified']:
            raise SystemExit('FAILED original source verification')
        print('NONLINEAR UPSTREAM VALIDATION PASSED')
    elif not args.pilot:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print('WROTE',args.output)

if __name__=='__main__':main()
