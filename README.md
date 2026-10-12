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

---

## G3 · Nonlinear history: the residue advantage does not materialize

[Open the interactive nonlinear gate](https://anttiluode.github.io/MondaySihti/#nonlinear) · [Protocol frozen before evaluation](PROTOCOL_G3.md) · [All five seeds and held-out traces](data/nonlinear_results.json) · [Run against pinned original Operaattori](.github/workflows/g3-upstream.yml)

G3 moves beyond the passive operator. A **seven-compartment illustrative neuron** is simulated with local AMPA/NMDA conductances and voltage-dependent magnesium block in the same compiled causal-cable equations as Operaattori. Each synthetic history contains two prior dendritic input pulses and a conditioning pulse; the observer is **not told** their port locations, timings or amplitudes. It receives only eight noisy soma readings before a fixed probe, and its supervised target consists of four values of the future soma response caused by that probe (probe run minus matched no-probe run).

Train, validation and test **withhold whole ordered port-pair combinations**: 96/48/64 histories per seed. Five frozen seeds: 101, 211, 307, 419, 523. The same degree-two polynomial ridge model receives exactly **three scalar observations** from raw checkpoint selection, difference-band selection or trained PCA. Choices/regularization are trained and tuned without test targets. The full-eight method has a larger information/parameter budget and is not a fair control for G3-2. All results are synthetic with known labels; neither the input history nor the dendritic state is recovered.

| Representation | Mean held-out future probe-response RMSE (mV) | Observation budget |
| --- | ---: | ---: |
| Last soma reading alone | 0.44994 | 1 |
| **Three raw soma readings** | **0.40357** | 3 |
| **Three Sihti residues** | 0.48023 | 3 |
| Train-fitted PCA | 0.45765 | 3 |
| Eight full-history PCA coordinates | 0.61366 | 8 (unmatched, overfit possible) |

Results are mean held-out RMSE over all five independent splits; all per-seed values are preserved in the JSON. The full-eight model can do *worse* because its degree-two expansion has many more coefficients relative to training examples and the test contains new stimulus combinations; it is not an oracle and **not** a lower bound on error.

**Gate verdicts:**

- **G3-1 PASS:** full temporal difference representation restores the full eight noisy soma readings to machine precision; no new observability is created.
- **G3-2 FAIL (primary):** three residues do not beat matched raw and PCA by at least 5%. Residue RMSE is about **19% higher than raw**.
- **G3-3 FAIL:** raw-three improves on a single soma snapshot, but residue-three does not. Both were required.
- **G3-0 pending upstream execution in CI:** the local high-throughput Numba solver was compared to the independent Python solver on ten pilot forward runs, all exactly matched in that environment. The exact pinned original Operaattori runtime also passed the separate small pilot in [the branch CI run](https://github.com/anttiluode/MondaySihti/actions/runs/38225860712). The [dedicated original-upstream G3 workflow](.github/workflows/g3-upstream.yml) verifies **all** evaluation programs and their held-out metrics; claim G3-0 PASS only after that workflow succeeds.

**Reproduce the benchmark:**

```bash
pip install numpy==2.1.3 numba==0.61.2
OPENBLAS_NUM_THREADS=1 python -m experiments.fast_reference
python -m unittest discover -s tests -v

# Requires an independent checkout of pinned original Operaattori:
OPENBLAS_NUM_THREADS=1 python -m experiments.nonlinear --operaattori /path/to/Operaattori --check --output data/nonlinear_results.json
```

The frozen result contradicts the claim that a compressed Sihti residue representation inherently extracts more useful hidden-state information than ordinary time samples. It is a **negative gate**, not evidence against all possible state-dependent or nonlinear residue algorithms. The next justified question would change the mechanism—not merely the representation—such as adapting a measurement or an operator based on residual prediction error, while testing against capacity-matched recurrent or direct-history alternatives.
