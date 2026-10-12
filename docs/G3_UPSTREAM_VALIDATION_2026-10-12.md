# MondaySihti G3 original nonlinear-cable validation — 12 October 2026

**Result: PASS.** [GitHub Actions complete logs](https://github.com/anttiluode/MondaySihti/actions/runs/38226873814)

- Verified source: [anttiluode/Operaattori](https://github.com/anttiluode/Operaattori), commit `f2398939dc2ef2ec829093953c20637c74e82cd0`.
- Original `reduced/causal_graph_circuit.py` Git blob SHA-1: `433621a1a5a39a6ea3c2dcf47082c8e2a9085a47`.
- Original numerical entrypoint: `CausalGraphCircuit.from_dense_passive_graph` and `.run`, with implicit AMPA/NMDA current and passive propagation. No copied source was imported for the upstream run.
- Simulation: illustrative seven-compartment morphology; two hidden prior dendritic pulses, one conditioning pulse, fixed later probe; noisy eight-sample **soma** observations; four differential response target samples.
- Data: five evaluation seeds `101,211,307,419,523`; per seed 96 train + 48 validation + 64 held-out histories with disjoint early/late port-pair sets.
- Verification: **every baseline and probe program** recomputed through original upstream runtime; each **entire soma trace** checked numerically against an independent local implementation with tolerance `1e-8` mV. Then predictions for the entire frozen benchmark recomputed using upstream response labels and compared with stored mean RMSEs within `1e-6` mV.
- Outcome in run logs: `G3_0_original_source_verified: True`; `NONLINEAR UPSTREAM VALIDATION PASSED`; 17 Python unit tests passed.

## Frozen mean held-out results (mV RMSE)

| Method | Error |
| --- | ---: |
| Latest soma reading (snapshot) | 0.4499376 |
| Three raw readings | **0.4035691** |
| Three residue channels | 0.4802332 |
| Three PCA components | 0.4576528 |
| Eight-dimensional full history (unmatched capacity) | 0.6136621 |

**G3-1 PASS:** full residue transform invertible on soma observations.

**G3-2 FAIL:** compressed residue features do not beat both equal-budget controls by the prespecified 5% margin.

**G3-3 FAIL:** raw three samples improve on snapshot but residue-three does not; both were required.

## Claim boundary

This validates that the held-out results are reproducible from the **original pinned Operaattori mathematical cable implementation** for this synthetic stimulus family, not that Operaattori's biological assumptions are correct or that actual neuronal hidden states or Q/K/V have been identified. The observer's network is a supervised degree-two polynomial ridge predictor and does not receive hidden stimulus identities or compartment states. The result is about prediction under specified compression, not source separation or impossibility of all residue-based recovery methods.

See [frozen G3 protocol](../PROTOCOL_G3.md), [complete JSON](../data/nonlinear_results.json), and [upstream verification workflow](../.github/workflows/g3-upstream.yml).
