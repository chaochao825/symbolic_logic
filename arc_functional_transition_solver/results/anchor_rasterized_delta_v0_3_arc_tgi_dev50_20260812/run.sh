#!/usr/bin/env bash
set -u

repo=/home/spco/sow_linear/codex_worktrees/arc_failure_matrix_20260810/arc_functional_transition_solver
out="$repo/results/anchor_rasterized_delta_v0_3_arc_tgi_dev50_20260812"
challenges="$repo/results/arc_tgi_arcmini_cohort_v2_20260810/development_challenges.json"
solutions="$repo/results/arc_tgi_arcmini_cohort_v2_20260810/development_solutions.json"
parent_freeze="$repo/results/counterfactual_transition_v4_arc_tgi_dev_20260812/candidate_freeze_a.json"
protocol="$repo/notes/research/anchor-rasterized-delta-v0.3-gate-20260812.md"

mkdir -p "$out"
cd "$repo" || exit 90
git rev-parse HEAD > "$out/source_commit.txt"
git status --short -- \
  src/afts_arc/anchor_mask.py \
  src/afts_arc/anchor_mask_reachability.py \
  src/afts_arc/anchor_mask_topology_gate.py \
  src/afts_arc/executable_workspace.py \
  scripts/afts_anchor_mask_topology_gate.py \
  tests/test_anchor_mask.py \
  tests/test_anchor_mask_reachability.py \
  tests/test_anchor_mask_topology_gate.py \
  notes/literature/arga-arcana-minimal-semantic-recruitment-20260812.md \
  notes/research/anchor-rasterized-delta-v0.3-gate-20260812.md \
  > "$out/scoped_source_status.txt"
python --version > "$out/python_version.txt" 2>&1
{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  uname -a
  nproc
} > "$out/environment.txt"

ruff check \
  src/afts_arc/anchor_mask.py \
  src/afts_arc/anchor_mask_reachability.py \
  src/afts_arc/anchor_mask_topology_gate.py \
  src/afts_arc/executable_workspace.py \
  scripts/afts_anchor_mask_topology_gate.py \
  tests/test_anchor_mask.py \
  tests/test_anchor_mask_reachability.py \
  tests/test_anchor_mask_topology_gate.py \
  > "$out/ruff.stdout.log" \
  2> "$out/ruff.stderr.log"
status=$?
printf '%s\n' "$status" > "$out/ruff.status"

if [ "$status" -eq 0 ]; then
  env PYTHONPATH=src python -m pytest \
    tests/test_anchor_mask.py \
    tests/test_anchor_mask_reachability.py \
    tests/test_anchor_mask_topology_gate.py \
    tests/test_counterfactual_transition.py \
    tests/test_counterfactual_transition_gate.py \
    tests/test_executable_workspace.py \
    tests/test_object_graph_rewrite.py \
    tests/test_object_program_workspace.py \
    tests/test_recolor_topology_gate.py \
    tests/test_scene_graph.py \
    tests/test_stateful_object_graph_rewrite.py \
    tests/test_stateful_rewrite_failure_audit.py \
    tests/test_stateful_scene.py \
    tests/test_visual_trace_repair.py -q \
    > "$out/dependency_pytest.stdout.log" \
    2> "$out/dependency_pytest.stderr.log"
  status=$?
fi
printf '%s\n' "$status" > "$out/dependency_pytest.status"

freeze_one() {
  local label="$1"
  /usr/bin/time -v env PYTHONPATH=src \
    python scripts/afts_anchor_mask_topology_gate.py freeze \
    --challenges "$challenges" \
    --parent-freeze "$parent_freeze" \
    --cohort-id b237836bbaf9340497292f632efb7c2208e140a877b75a5bd3783ff11b4e2a27 \
    --scientific-lane outcome_exposed_development \
    --protocol "$protocol" \
    --max-first-stage-trials 2000 \
    --max-parents 4 \
    --reserved-native-trials 2048 \
    --workers 8 \
    --output "$out/candidate_freeze_${label}.json" \
    > "$out/freeze_${label}.stdout.log" \
    2> "$out/freeze_${label}.stderr.log"
}

score_one() {
  local label="$1"
  env PYTHONPATH=src python scripts/afts_anchor_mask_topology_gate.py score \
    --freeze "$out/candidate_freeze_${label}.json" \
    --solutions "$solutions" \
    --output "$out/result_${label}.json" \
    > "$out/score_${label}.stdout.log" \
    2> "$out/score_${label}.stderr.log"
}

if [ "$status" -eq 0 ]; then
  freeze_one a
  status=$?
fi
printf '%s\n' "$status" > "$out/freeze_a.status"
if [ "$status" -eq 0 ]; then
  freeze_one b
  status=$?
fi
printf '%s\n' "$status" > "$out/freeze_b.status"
if [ "$status" -eq 0 ]; then
  cmp -s "$out/candidate_freeze_a.json" "$out/candidate_freeze_b.json"
  status=$?
fi
printf '%s\n' "$status" > "$out/freeze_replay.status"
if [ "$status" -eq 0 ]; then
  score_one a
  status=$?
fi
printf '%s\n' "$status" > "$out/score_a.status"
if [ "$status" -eq 0 ]; then
  score_one b
  status=$?
fi
printf '%s\n' "$status" > "$out/score_b.status"
if [ "$status" -eq 0 ]; then
  cmp -s "$out/result_a.json" "$out/result_b.json"
  status=$?
fi
printf '%s\n' "$status" > "$out/score_replay.status"

if [ "$status" -eq 0 ]; then
  sha256sum \
    "$out/candidate_freeze_a.json" \
    "$out/candidate_freeze_b.json" \
    "$out/result_a.json" \
    "$out/result_b.json" \
    > "$out/replay_sha256.txt"
  sha256sum \
    src/afts_arc/anchor_mask.py \
    src/afts_arc/anchor_mask_reachability.py \
    src/afts_arc/anchor_mask_topology_gate.py \
    src/afts_arc/executable_workspace.py \
    scripts/afts_anchor_mask_topology_gate.py \
    tests/test_anchor_mask.py \
    tests/test_anchor_mask_reachability.py \
    tests/test_anchor_mask_topology_gate.py \
    notes/literature/arga-arcana-minimal-semantic-recruitment-20260812.md \
    notes/research/anchor-rasterized-delta-v0.3-gate-20260812.md \
    > "$out/source_sha256.txt"
fi
printf '%s\n' "$status" > "$out/status"
exit "$status"
