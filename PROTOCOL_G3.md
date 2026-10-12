# G3 — nonlinear dendrite, hidden stimulus histories

## Claim to try to falsify

Can three Sihti-style temporal residue channels predict responses of a nonlinear dendritic cable to a new fixed probe **better than equally sized raw-history selection and train-fitted PCA**? The observer never receives the dendritic state or the historical stimulus ports/amplitudes. It receives **only eight noisy pre-probe soma readings**. All output labels come from a separate, synthetic known-answer simulator.

## Forward model and true intervention

The seven-compartment graph is the same illustrative reference as MondayRepo. Use Operaattori's pinned `CausalGraphCircuit` passive backward-Euler propagation and implicit voltage-dependent AMPA/NMDA current. Two prior conductance pulses at hidden ports (among sites 2–5) plus a hidden conditioning pulse at site 6 produce a dendritic state. At a fixed later time, compare the actual probe response to an otherwise identical counterfactual without that probe. The target consists of four values from **one soma trace**, i.e. `y_probe(t) − y_no_probe(t)` at fixed post-probe times.

Histories are split by *ordered early/late port combination* before sampling their amplitudes, so held-out combinations are not present in the training/validation combinations. The learner does not get the port identities or hidden amplitudes. Noise is added only to its eight pre-probe soma observations; the target is from the noiseless simulator. No repeated seeds between train, validation, and test.

## Primary representations and controls

- Three raw soma readings, selected from all 56 combinations using training + validation only.
- Three Sihti residue channels, selected from 56 combinations, where `r_i = y_i − y_{i+1}` for seven bands and the last channel is the terminal reading. All eight channels together reconstruct the eight sampled readings exactly.
- Three PCA coordinates fitted only on the training observations.
- A single last-soma-reading snapshot baseline (control, smaller capacity).
- All eight soma readings (higher-budget reference, explicitly **not** a matched-capacity control).

All *three-scalar* methods use the identical degree-two polynomial feature map and standardized ridge regression. The same regularization grid is selected using validation data only. Target dimension and test histories are identical. Model fit is fully supervised using synthetic probe-response labels; this is NOT a blind unsupervised source decoder.

## Prespecified gates

- **G3-0, source fidelity:** when `--operaattori` is supplied, import the **actual** pinned Operaattori `CausalGraphCircuit` and verify every generated observation/target against the local reference runner at max error ≤ 1e-8 mV. This is a software equivalence check, not biological validation.
- **G3-1, information identity:** the complete eight-channel Sihti temporal difference representation reconstructs the eight noisy soma readings at machine precision, and retains the same linear observation span.
- **G3-2, residue advantage (primary):** residue-three's mean held-out response RMSE is at least 5% smaller than **both** raw-three and PCA-three, over all fresh seeds. Otherwise **FAIL**. Retain all comparisons.
- **G3-3, historical observability:** raw-three and residue-three individually achieve at least 5% lower held-out RMSE than the latest-soma-only snapshot; this is a secondary gate.

No retuning on test seeds. Pilot data may be used to establish a functioning simulation and meaningful nonzero outputs only; quantitative claims and gate results must come from the frozen evaluation splits. No positive causal-recovery or QKV claim can follow from these synthetic tests.

## Limitations

Temporal residues are a change of coordinates in measured soma history, not access to dendritic compartments. The original forward model and conductance laws are assumed known to the experiment designer, not to the trained ridge predictor. A model could recover *predictive effects* while remaining unable to reconstruct the true hidden synapses or past input schedule. If the fixed probe does not expose useful nonlinear history dependence, record that and report it rather than redefining the task after seeing evaluation results.