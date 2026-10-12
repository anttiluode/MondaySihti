#!/usr/bin/env python3
"""A matched-budget Sihti residue/neuronal-observability bridge.

No claim of biological identification: both plants are known synthetic linear systems.
Decoder is supervised using simulated x0 labels, with all tuning on validation.
The pinned upstream Operaattori compiler is checked with --operaattori.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np

TIMES = (0, 1, 2, 4, 8, 16, 32)
SEEDS = (11, 19, 23, 31, 47)
ALPHAS = (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)
N_TRAIN, N_VAL, N_TEST = 400, 140, 180


def image_plant():
    n = 8
    rgb = np.empty((n,n,3))
    rgb[:] = [0.18, 0.26, 0.45]
    rgb[1:5, 1:4] = [0.82, 0.32, 0.16]
    rgb[3:7, 5:7] = [0.15, 0.75, 0.39]
    # Synthetic appearance graph with local color affinities (not a natural image).
    W = np.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            a = i*n+j
            for di,dj in ((1,0),(0,1)):
                h,k=i+di,j+dj
                if h<n and k<n:
                    b=h*n+k
                    dist2=float(np.sum((rgb[i,j]-rgb[h,k])**2))
                    weight=0.015+np.exp(-dist2/(2*0.18**2))
                    W[a,b]=W[b,a]=weight
    D=W.sum(axis=1)
    P=0.5*np.eye(n*n)+0.5*(W/D[:,None])
    C=np.eye(n*n)[[0,7,18,27,36,45,56,63]]
    return P,C,rgb


def cable_graph():
    # Same reference seven-compartment graph as MondayRepo's forward fixture.
    edges=[(0,1,.21),(1,2,.18),(2,3,.15),(3,4,.13),(4,5,.12),(1,6,.18)]
    leak=np.array([.035,.028,.025,.023,.023,.021,.025])
    G=np.diag(leak)
    for i,j,g in edges:
        G[i,i]+=g;G[j,j]+=g;G[i,j]-=g;G[j,i]-=g
    C=np.array([.10,.08,.07,.06,.06,.06,.06])
    return G,C


def cable_plant(operaattori=None):
    G,C=cable_graph()
    dt=.1
    D=np.diag(C/dt)
    reference=np.linalg.solve(G+D,D)
    if operaattori is not None:
        repo=Path(operaattori).resolve()
        if not (repo/'reduced'/'causal_graph_circuit.py').exists():
            raise ValueError(f'Cannot find Operaattori module in {repo}')
        sys.path.insert(0,str(repo))
        from reduced.causal_graph_circuit import compile_dense_passive_graph
        original,_=compile_dense_passive_graph(G,C,[2,3,4,5,6],dt_ms=dt)
        err=float(np.max(np.abs(original-reference)))
        if err>1e-12:
            raise AssertionError(f'UPSTREAM MISMATCH: {err:.3g}')
        print(f'ORIGINAL OPERAATTORI PASSIVE COMPILER CHECK PASSED: {err:.3g} max |ΔP|')
        P=original
    else:
        P=reference
    # Exactly ONE measured soma channel.
    H=np.eye(len(C))[[0]]
    return P,H


def observation(P,C,times=TIMES):
    return np.vstack([C@np.linalg.matrix_power(P,t) for t in times])


def octave_difference(n=7):
    # One residue between each adjacent DYADIC checkpoint, plus terminal core.
    D=np.zeros((n,n))
    for i in range(n-1): D[i,i]=1;D[i,i+1]=-1
    D[-1,-1]=1
    return D


def synthetic_states(domain,P,rng,n):
    d=len(P)
    raw=rng.normal(0,1,(n,d))
    if domain=='image':
        # Mixed fast/slow spatial content; no source separation oracle.
        smooth=(raw@P.T+raw@(P@P).T)/2
        return 0.65*raw+0.90*smooth
    states=1.7*raw
    # Random dendritic initial perturbation, soma not preferentially stimulated.
    positions=rng.integers(2,6,size=n)
    states[np.arange(n),positions]+=rng.normal(0,2,n)
    return states


def ridge(train_x,train_y,eval_x,alpha):
    # Center from train only. The same formulation is used by all encoders.
    mx=train_x.mean(axis=0);my=train_y.mean(axis=0)
    x=train_x-mx
    w=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@(train_y-my))
    return (eval_x-mx)@w+my


def rmse(a,b):
    return float(np.sqrt(np.mean((a-b)**2)))


def train_decode(F, X, splits, alpha, val_only=False):
    a,b,c=splits
    pred=ridge(F[:a],X[:a],F[a:a+b] if val_only else F[a+b:a+b+c],alpha)
    truth=X[a:a+b] if val_only else X[a+b:a+b+c]
    return rmse(pred,truth),pred


def choose(F,X,splits,k,kind,nchan):
    a,b,c=splits
    # Train/validation-only grid-search over which TIME BANDS to store.
    best=None
    for idx in itertools.combinations(range(len(TIMES)),k):
        cols=np.concatenate([np.arange(i*nchan,(i+1)*nchan) for i in idx])
        feat=F[:,cols]
        for alpha in ALPHAS:
            score,_=train_decode(feat,X,splits,alpha,True)
            key=(score,idx,alpha)
            if best is None or key<best[0]:best=(key,cols)
    (v,idx,alpha),cols=best
    err,pred=train_decode(F[:,cols],X,splits,alpha)
    return dict(test=err,val=v,choice=list(idx),alpha=alpha),pred


def fit_pca(F,X,splits,k):
    a,b,c=splits
    mu=F[:a].mean(0)
    # PCA basis is fitted on train only; k stored channels, same as the others.
    _,_,vt=np.linalg.svd(F[:a]-mu,full_matrices=False)
    projected=(F-mu)@vt[:k].T
    scores=[(train_decode(projected,X,splits,alpha,True)[0],alpha) for alpha in ALPHAS]
    _,alpha=min(scores)
    err,pred=train_decode(projected,X,splits,alpha)
    return dict(test=err,alpha=alpha),pred


def benchmark_one(domain,P,C,seed,k=3):
    rng=np.random.default_rng(seed)
    total=N_TRAIN+N_VAL+N_TEST
    X=synthetic_states(domain,P,rng,total)
    O=observation(P,C)
    nchan=C.shape[0]
    noise=.045 if domain=='image' else .015
    F=X@O.T+rng.normal(0,noise,(total,len(O)))
    R=np.kron(octave_difference(),np.eye(nchan))
    F_res=F@R.T
    splits=(N_TRAIN,N_VAL,N_TEST)
    out={}; preds={}
    for method,data in [('raw',F),('residue',F_res)]:
        out[method],preds[method]=choose(data,X,splits,k,method,nchan)
    out['pca'],preds['pca']=fit_pca(F,X,splits,k*nchan)
    for method,data in [('full_raw',F),('full_residue',F_res)]:
        # Full dimensionality; model class and #columns identical. No tuning on test.
        al=min(ALPHAS,key=lambda z:train_decode(data,X,splits,z,True)[0])
        out[method]={'alpha':al}
        out[method]['test'],preds[method]=train_decode(data,X,splits,al)
    # An exact, unregularized invariant diagnostic on the LINEAR observation operator.
    rank_a=int(np.linalg.matrix_rank(O,tol=1e-10))
    rank_b=int(np.linalg.matrix_rank(R@O,tol=1e-10))
    obs_error=float(np.max(np.abs(np.linalg.solve(R,R@O)-O)))
    assert rank_a==rank_b
    # Predict a withheld future soma/pixel measurement from the recovered latent state.
    future=C@np.linalg.matrix_power(P,48)
    Yfuture=X[N_TRAIN+N_VAL:]@future.T
    for key,pred in preds.items():
        out[key]['future_rmse']=rmse(pred@future.T,Yfuture)
    return dict(seed=seed, methods=out, diagnostics=dict(rank_raw=rank_a,rank_residue=rank_b,invertible_diff=obs_error,observation_rows=len(O),state_dims=len(P)))


def create_report(operaattori=None):
    cable,cobs=cable_plant(operaattori)
    image,iobs,rgb=image_plant()
    runs={}
    for domain,P,C in [('image',image,iobs),('cable',cable,cobs)]:
        rows=[benchmark_one(domain,P,C,s) for s in SEEDS]
        methods={}
        for name in ['raw','residue','pca','full_raw','full_residue']:
            scores=[row['methods'][name]['test'] for row in rows]
            future=[row['methods'][name]['future_rmse'] for row in rows]
            methods[name]={'mean_state_rmse':float(np.mean(scores)),
                           'mean_future_rmse':float(np.mean(future)),
                           'per_seed_state_rmse':scores,'per_seed_future_rmse':future,
                           'mean_state_relative_to_pca':float(np.mean(scores)/np.mean([r['methods']['pca']['test'] for r in rows]))}
        runs[domain]=dict(methods=methods,by_seed=rows,static=dict(observability_rank=rows[0]['diagnostics']['rank_raw'],state_dims=len(P),channels=len(C)))
    # Visualize one reproducible plant-state residue sequence, no selection based on results.
    vis_rng=np.random.default_rng(1234)
    demo={}
    for dom,P,C in [('image',image,iobs),('cable',cable,cobs)]:
        x=synthetic_states(dom,P,vis_rng,1)[0]
        states=np.stack([np.linalg.matrix_power(P,t)@x for t in TIMES])
        residues=np.vstack([states[i]-states[i+1] for i in range(len(TIMES)-1)]+[states[-1]])
        demo[dom]={'initial':x.round(5).tolist(),
                   'states':states.round(5).tolist(),
                   'bands':residues.round(5).tolist(),
                   'observation':(states@C.T).round(5).tolist(),
                   'reconstruction_error':float(np.max(np.abs(residues.sum(axis=0)-x)))}
    report=dict(schema='mondaysihti.bridge.v1',date='2026-10-12',code_scope='illustrative synthetic linear operators',
                times=list(TIMES),seeds=list(SEEDS),ntrain=N_TRAIN,nval=N_VAL,ntest=N_TEST,
                budget_time_bands=3,noise_sd={'image':.045,'cable_mV':.015},
                note='All 7 measurement times sampled before compression; equal stored scalar budget; feature selection & ridge alpha tuned on validation only. Not a measurement-budget experiment.',
                cases=runs,illustration=demo,
                gates={
                    'G0_exact_telescope':max(demo[z]['reconstruction_error'] for z in demo)<1e-12,
                    'G1_observability_rank_preserved':all(runs[d]['static']['observability_rank']==runs[d]['by_seed'][0]['diagnostics']['rank_residue'] for d in runs),
                    'G2_residue_beats_both_matched_baselines_both_domains':all(runs[d]['methods']['residue']['mean_state_rmse'] < min(runs[d]['methods']['raw']['mean_state_rmse'],runs[d]['methods']['pca']['mean_state_rmse']) for d in runs)
                })
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--operaattori',default=None)
    p.add_argument('--output',default='data/results.json')
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    report=create_report(a.operaattori)
    for d in report['cases']:
        print('\n'+d, 'rank',report['cases'][d]['static']['observability_rank'],'/',report['cases'][d]['static']['state_dims'])
        for k,v in report['cases'][d]['methods'].items():print('%14s state %9.5f future %9.5f'%(k,v['mean_state_rmse'],v['mean_future_rmse']))
    print('GATES',report['gates'])
    if a.check:
        target=json.loads(Path(a.output).read_text())
        for domain in ['image','cable']:
            for k in report['cases'][domain]['methods']:
                d=abs(report['cases'][domain]['methods'][k]['mean_state_rmse']-target['cases'][domain]['methods'][k]['mean_state_rmse'])
                if d>1e-9:raise SystemExit(f'FAIL {domain}/{k}: {d}')
        print('UPSTREAM REPRO CHECK PASSED')
    else:
        out=Path(a.output);out.parent.mkdir(exist_ok=True,parents=True)
        out.write_text(json.dumps(report,indent=2)+'\n')
        print('WROTE',out)

if __name__=='__main__':main()
