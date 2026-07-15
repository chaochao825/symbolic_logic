# Provenance

## User-provided reference documents

The supplied documents were read as UTF-8. They were not copied into this repository; their external locations and SHA-256 hashes are recorded so the interpretation can be audited.

| Document | SHA-256 | Role in this project |
|---|---|---|
| `D:\xwechat_files\wxid_zqxxbf2n217z22_a684\msg\file\2026-07\txt1.txt` | `89C7DF27C692D7263E4D156F966309C3C8E0A3D4C08A5BB330C1B1D47EA204A9` | Positions LGN as a trainable, hardenable predicate compiler rather than a complete symbolic reasoner. |
| `D:\xwechat_files\wxid_zqxxbf2n217z22_a684\msg\file\2026-07\txt2.txt` | `8AE3E23F4E15251D81008D7F02E99B18546092856D59EEC52670368656A183DF` | Identifies soft/dense computation, relation enumeration, rule selection, and constraint verification as candidate gate-compilation targets. |
| `C:\Users\chaochao\.codex\attachments\43faaf28-6b6a-4e22-b7a9-8ce3b478d1fb\pasted-text.txt` | `D65338FED98B4046B887F7B102E7A99457B308FB6898781437C97172111C82B2` | Independent review that identifies the missing raw-pixel end-to-end model, learned compilation loop, sparsity, and evidence-chain issues. |
| `C:\Users\chaochao\.codex\attachments\e12f7bbf-fa93-43f5-b58a-989c57897c40\pasted-text-1.txt` | `4EEA2C8BEC3EE380BF4B65288226DC0CCE935B7202F055DF6CE341774F5677A0` | Requests an operational discrete theory based on Boolean representation codes, circuit/netlist descriptions, residual codes, exact small synthesis, and honest incompressibility controls. |
| `C:\Users\chaochao\.codex\attachments\6b499159-ff3c-48cb-b541-178603224ce7\pasted-text-1.txt` | `88234A4DA6A6C09C41131648788B1717AD4E059BFAB5750E32C0B288F1F14C12` | Sharpens the target around paid Boolean description length, basis-aware circuit complexity, exact small-function synthesis, and the separation of representation hardness from search hardness. |
| `C:\Users\chaochao\.codex\attachments\6b499159-ff3c-48cb-b541-178603224ce7\pasted-text-2.txt` | `1831682DE804858958F4BF8B81366A3F7B9AAC464908B1C21308AE0BCEC5A86E` | Emphasizes formula-versus-DAG accounting, decision-tree/BDD short-circuit structure, treewidth, exact-oracle validation, synthesis, and formal equivalence boundaries. |

## Result provenance

- Raw results: `results/*.csv`.
- Aggregate bootstrap summaries: `results/summary_metrics.csv` (2,500 rows after adding CA local, collective, pathfinding, official-replay, and complexity sections).
- Runtime metadata: `results/run_metadata.json`.
- New hardening artifacts: `results/learned_gate_results.csv` and `results/learned_gate_bfs_results.csv` (full run: 40 local rows and 360 graph rows).
- New state/complexity artifacts: `results/state_transition_results.csv` (26 rows), `results/planning_frontier_results.csv` (22 rows), `results/noncompressible_scaling_results.csv` (60 rows), and `results/probability_marginalization_results.csv` (100 rows).
- End-to-end visual artifacts: `results/end_to_end_gridworld_results.csv` (36 rows), `results/end_to_end_gridworld_metadata.json`, and three encoder/gate model pairs under `results/gridworld_models/`.
- Logic-discovery artifacts: `results/logic_discovery_results.csv` (125 rows), covering five seeds, four primitive rules, distractors, composition transfer, task-only topology search, and hardening negative controls.
- Rate-reduction artifacts: `results/rate_reduction_results.csv` (60 rows) and `results/rate_logic_metadata.json`; the 46.09-second full run started from clean commit `5cddc7f4bd4df878359e1e60a765cd0f4db22aa2` on host `hi-X640-G40`, Python 3.10.20/NumPy 1.25.2, with one BLAS/OpenMP thread through SSH profile 210.
- Discrete-theory artifacts: `results/discrete_function_mdl_results.csv` (147 rows), `results/discrete_task_mdl_results.csv` (307 rows), `results/exact_formula_balanced4_results.csv` (12,870 rows), `results/discrete_representation_code_results.csv` (8 rows), and `results/discrete_theory_checks.json`; the 34.90-second full run started from the same clean commit and locked environment. Source and artifact normalized SHA-256 values are recorded in `results/discrete_theory_metadata.json`.
- Basis-aware artifacts: `results/runs/circuit_bias_v0_full_210_5cddc7f/` contains 1,024 exact formula-basis rows, 256 Boolean-diagnostic rows, aggregate/family/named-task tables, a parity-vs-random search bridge, and metadata. The 1.09-second full run started clean from `5cddc7f4bd4df878359e1e60a765cd0f4db22aa2`; exactness is limited to formula size, while DAG/code fields are retained-witness upper bounds.
- External Hard-LGN audit: `results/external/hard_lgn_v23/` records normalized hashes and selected rows from separate-project commit `39c73d099bdcd7068f99bdea7667ae54578c193c`. It is audited external evidence, not a `symbolic_logic` rerun; the selected source artifacts do not record their Python/package environment.
- Cellular-automata artifacts: `results/cellular_automata_local_rule_results.csv` (269 rows), `results/cellular_automata_task_results.csv` (402 rows), `results/cellular_automata_complexity_results.csv` (307 rows), `results/cellular_automata_official_results.csv` (19 rows), and `results/cellular_automata_trajectories.npz`. The final 573.80-second full run used Python 3.10.4, NumPy 1.25.2, and pandas 2.3.3 and started clean from merge commit `a604e8745f3b19ade0c5a785c131d10c37d79a70`; source and artifact hashes are recorded in `results/cellular_automata_metadata.json`.
- Official DiffLogic-CA inference artifacts are downloaded unchanged from `pages@4c0246d9f7a2912cb7201f6bfe5fcda0fe373904`: GoL SHA-256 `82d59f04...ecba` and checkerboard SHA-256 `8a22e6e1...3458`. The committed manifest contains full digests, byte counts, Git blobs, source URLs, and the mixed Apache-2.0/CC-BY license notice.
- The official JAX training attempt is separately recorded in `results/difflogic_ca_training_attempt.json`: no optimization epoch ran because SSH profiles 210/236 timed out and both local PyPI/mirror dependency fetches were blocked. No frozen-circuit result is relabeled as training reproduction.
- CA figure source: `figures/cellular_automata_plot.py`; it reads only the four CA CSV tables and generates `figures/cellular_automata_results.pdf` and `.png`.
- Figure source: `figures/results_plot.py`; it reads only the raw CSV files, including `figures/learned_gate_integration.pdf`.
- Core exactness tests: `tests/test_core.py`.
- Discrete-code and artifact-integrity tests: `tests/test_boolean_mdl.py`.
- CA semantic and artifact-integrity tests: `tests/test_cellular_automata.py`, `tests/test_difflogic_ca.py`, and `tests/test_cellular_automata_artifacts.py`.
- Previous QA: a fresh bundle clone of artifact commit `892ec4a1d1a3348bec14c2e5da93054e9546f729` on SSH profile 210/Python 3.8.10 passed the then-current 39 repository tests and reproduced the 1,974-row summary; it is historical and superseded by the current QA below.
- Pre-CA inductive-bias QA: a fresh detached worktree of artifact/report commit `5cb67b436d7b4053b98624273277246506e6d66f` on SSH profile 210, Python 3.10.20/NumPy 1.25.2, passed all 53 repository tests in 4.51 seconds; regenerating the 1,974-row summary was byte-identical and left the worktree clean. Two independent read-only final audits reported no P0 or P1 finding; their non-blocking report-label findings were corrected before the final documentation commit.
- Current combined in-place QA: after the clean-start CA run, all 88 repository tests passed in 13.42 seconds and `src/summarize_results.py` regenerated the 2,500-row aggregate table. A fresh-clone verification remains the next post-push check.
- Terminology debt retained for artifact fidelity: the module-level prose in `src/logic_core.py` still calls the legacy GateBeam a “gate-DAG synthesizer”, while its recursive `Expr` and every current contract/report correctly count formula references. Changing that hashed source after the clean run would make the committed runtime manifest false; a future rerun should rename the legacy docstring together with new metadata.
- The previous mojibake-encoded report is preserved at `trash/20260712_evidence_report_revision/evidence_report_zh_legacy.md`; the current report is UTF-8.
