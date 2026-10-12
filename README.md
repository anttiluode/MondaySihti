# MondaySihti — The Residue Bridge

[**Interactive instrument (GitHub Pages)**](https://anttiluode.github.io/MondaySihti/) · [Data](data/results.json) · [Generator](experiments/bridge.py)

**Does Sihti's multiscale residue representation recover hidden neuronal state better than equally compact raw-history representations?** This project asks this question in two *controlled linear simulators*, without claiming a new biological model.

## The precise shared mechanism

For a fixed passive linear operator `P`, state relaxes as `x[t+1]=P x[t]`. Image diffusion constructs `P` from local color-affinity weights. Operaattori's passive cable compiler produces `P=(G+diag(C/dt))^-1 diag(C/dt)` using a seven-compartment illustrative morphology. There are **no active AMPA/NMDA nonlinearities** in this particular experiment (they would make the operator state-dependent). The site replays stored, reproducibly generated results.

At checkpoints `t=(0,1,2,4,8,16,32)` define adjacent residuals `r[j]=x[t[j]]-x[t[j+1]]` and terminal core `r[6]=x[32]`. Their sum is identically the initial state, even for nonlinear state trajectories. This bookkeeping says **nothing by itself about source identification**.

The observer instead sees **only** `y[t]=H x[t]+noise`: 8 of 64 pixels, or a single soma voltage among 7 compartments. If `O=stack(H P^t)` and `D` is the invertible 7×7 adjacent-difference-plus-core transform, the full residue observation operator is `(D⊗I)O`. **The rank is preserved exactly.** If the image observation operator has deficient rank, no invertible residue transform can fix it; even full rank does not imply stable inversion under noise.

## Experiment and fairness

- Five deterministic training/test seeds: `11, 19, 23, 31, 47`.
- Per seed: 400 training examples, 140 validation examples, and 180 independent held-out test examples. Target is the *true synthetic initial hidden state*. Secondary metric is prediction of `H P^48 x0`.
- All seven noisy observations are collected first; 3 stored time bands are selected for each compressed method. This is a **storage** budget, not an observation-count budget. With `m` sensors each method stores exactly `3m` scalars.
- Three-band **raw** and **residue** methods each search the same 35 candidate time-triplets on validation. PCA is trained only from training observations and stores `3m` coefficients. All are decoded with a linear ridge decoder whose regularization is chosen on validation. Full-history raw and full-history residue ridge models are reference upper-capacity comparisons (seven bands).
- Residues transform measured *noise too*: adjacent differences have correlated errors, whereas raw sensor noise was independent. Ordinary coordinate-sensitive ridge regression need not give identical full-history performance even though the representations are invertibly related.
- The inverse decoder is supervised with **oracle synthetic hidden-state labels during training**; it is *not* a blind, unsupervised biological source decoder. Exact physical matrices are used to generate data, but **not supplied to the trained linear decoder**.

## Recorded gates

| Gate | Criterion | Outcome |
| --- | --- | --- |
| G0 | Adding all six difference bands plus terminal core reconstructs the original simulated state to float precision | **PASS** |
| G1 | Residue-transformed and raw observation matrices have identical rank | **PASS** |
| G2 | Three-band residues beat *both* equally sized raw-history selection and trained PCA in hidden-state RMSE *for both domains* | **FAIL** |

### Result summary — mean held-out hidden-state RMSE, five seeds

| Method | Image (synthetic units) | Cable (mV) |
| --- | ---: | ---: |
| Best 3 raw checkpoints | 0.80056 | 1.45125 |
| Best 3 residue bands | 0.83569 | 1.45399 |
| Train-fitted PCA | **0.79853** | 1.45315 |
| Full 7 raw checkpoints | **0.75320** | **1.26457** |
| Full 7 residue bands | 0.76036 | 1.26564 |

These are means, **not population-confidence claims**. Per-seed values and future-response RMSE are saved in [data/results.json](data/results.json); no post hoc gate changes were made to reclassify G2.

Interpretation: this particular compressed Sihti representation does **not** deliver a general advantage. The full representation is algebraically invertible but the *measured channels* may still be insufficient or ill-conditioned. The 64-cell image has observation matrix rank **56/64**; the passive 7-compartment cable has rank **7/7** at the seven chosen times, but its singular values span ~82,000×. The latter has full formal rank, not robust noisy state identification.

The image color graph is constructed from three flat synthetic color regions and positive neighbor affinities, not a semantic picture. Cable initial conditions are artificial distributed voltage perturbations, **not realistic ongoing neuronal stimulation**. Neither represents real brain EEG, learned sensory coding, or literal QKV. Sihti's other, more interesting possibility—**state-dependent operator changes**—is outside this passive experiment.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python experiments/bridge.py --check
python -m http.server 8000
# then browse http://localhost:8000
```

The upstream test uses **the exact original** [Operaattori `reduced/causal_graph_circuit.py`](https://github.com/anttiluode/Operaattori/blob/f2398939dc2ef2ec829093953c20637c74e82cd0/reduced/causal_graph_circuit.py) at commit `f2398939dc2ef2ec829093953c20637c74e82cd0`:

```bash
python experiments/bridge.py --operaattori /path/to/Operaattori --check
```

The GitHub Actions `ci.yml` checks out that pinned repository and verifies the **exact passive cable compiler** before recomputing every test metric. We do **not** claim validation of Operaattori's full nonlinear solver here. Pages is deployed by a new workflow `pages.yml`; if GitHub Pages hasn't been enabled, go to repository **Settings → Pages → Build and deployment → Source: GitHub Actions**. The GitHub Actions run must succeed before claiming a live Pages site.

## What would justify a second gate?

Use a controlled *state-dependent* neuronal operator (including local NMDA), hide some input history, train predictors of new interventions, and compare raw history, invertibly transformed residues, and a learned observer with matched capacity, sensor count and computational budget. A strong positive result would be reusable causal prediction **beyond matched baselines**, not merely exact telescoping.

## Project relations

[Sihti](https://github.com/anttiluode/Sihti) · [Sihti2](https://github.com/anttiluode/Sihti2) · [SighImageFactorization](https://github.com/anttiluode/SighImageFactorization) · [Operaattori](https://github.com/anttiluode/Operaattori) · [MondayRepo](https://github.com/anttiluode/MondayRepo) · [SihtiSort](https://github.com/anttiluode/SihtiSort)

MIT. The repository does not import code from the above projects into its Git tree; CI imports the original pinned Operaattori source from a separate checkout for validation.
