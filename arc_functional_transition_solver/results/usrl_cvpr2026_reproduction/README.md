# USRL reproduction evidence index

All scores are deterministic, seed-20260726, compute-bounded diagnostics. No
artifact in this directory substantiates the paper's 47.2% pass@2 claim.

## Audits and smoke tests

- `paper_architecture_audit.json`: 7,103,505-parameter public tensor contract.
- `protocol_audit.json`: content-addressed strict and paper-transductive splits.
- `synthetic_overfit_smoke.json`: bounded optimization sanity check.
- `paper_shape_cost_probe.json`: A800 inference timing and memory probe.

## Selected ARC pilots

- `final_strict_shallow_3000.json`: strict direct-CE reference.
- `final_strict_shallow_contrastive_3000.json`: matched contrastive ablation.
- `final_transductive_shallow_3000.json`: paper-transductive direct-CE reference.
- `final_transductive_shallow_mask_3000.json`: cosine-mask training corruption,
  without the official DRM iterative sampler.
- `final_transductive_deep_nomask_3000.json`: matched 4x3 direct-CE recurrence.
- `final_transductive_deep_cosine_mask_no_sampler_3000.json`: matched 4x3
  cosine-mask training corruption, again without the DRM sampler.

The `config` object is authoritative for recurrence depth. Early generated deep
artifacts used a generic claim-boundary sentence that said `1x1`; the selected
deep cosine-mask filename and the result report correct that metadata ambiguity.
The model outputs, metrics, configuration, and trace were not changed.

`summary.json` contains the compact comparison used by the result report. The
full interpretation and continuation gates are in
`notes/results/usrl-cvpr2026-reproduction-20260726.md`.
