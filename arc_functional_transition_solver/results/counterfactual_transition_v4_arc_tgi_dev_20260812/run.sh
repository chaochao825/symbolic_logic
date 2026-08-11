#!/usr/bin/env bash
set -u

repo=/home/spco/sow_linear/codex_worktrees/arc_failure_matrix_20260810/arc_functional_transition_solver
out="$repo/results/counterfactual_transition_v4_arc_tgi_dev_20260812"
challenges="$repo/results/arc_tgi_arcmini_cohort_v2_20260810/development_challenges.json"
solutions="$repo/results/arc_tgi_arcmini_cohort_v2_20260810/development_solutions.json"
v3_freeze="$repo/results/stateful_object_graph_rewrite_v3_arc_tgi_dev_20260811/candidate_freeze_a.json"
protocol="$repo/notes/design/counterfactual-transition-v4.md"

mkdir -p "$out"
cd "$repo" || exit 90
git rev-parse HEAD > "$out/source_commit.txt"
python --version > "$out/python_version.txt" 2>&1
{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  uname -a
  nproc
} > "$out/environment.txt"

PYTHONPATH=src python -m pytest \
  tests/test_scene_graph.py \
  tests/test_object_code.py \
  tests/test_counterfactual_transition.py \
  tests/test_counterfactual_transition_gate.py \
  tests/test_stateful_scene.py \
  tests/test_stateful_object_graph_rewrite.py \
  tests/test_stateful_rewrite_failure_audit.py -q \
  > "$out/semantic_tests.stdout.log" \
  2> "$out/semantic_tests.stderr.log"
status=$?

freeze_one() {
  local label="$1"
  /usr/bin/time -v env PYTHONPATH=src \
    python scripts/afts_counterfactual_transition_gate.py freeze \
    --challenges "$challenges" \
    --v3-freeze "$v3_freeze" \
    --cohort-id b237836bbaf9340497292f632efb7c2208e140a877b75a5bd3783ff11b4e2a27 \
    --scientific-lane outcome_exposed_development \
    --protocol "$protocol" \
    --workers 8 \
    --output "$out/candidate_freeze_${label}.json" \
    > "$out/freeze_${label}.stdout.log" \
    2> "$out/freeze_${label}.stderr.log"
}

score_one() {
  local label="$1"
  PYTHONPATH=src python scripts/afts_counterfactual_transition_gate.py score \
    --freeze "$out/candidate_freeze_${label}.json" \
    --solutions "$solutions" \
    --output "$out/result_${label}.json" \
    > "$out/score_${label}.stdout.log" \
    2> "$out/score_${label}.stderr.log"
}

if [ "$status" -eq 0 ]; then
  freeze_one a &
  pid_a=$!
  freeze_one b &
  pid_b=$!
  wait "$pid_a" || status=$?
  wait "$pid_b" || status=$?
fi
if [ "$status" -eq 0 ] && ! cmp -s "$out/candidate_freeze_a.json" "$out/candidate_freeze_b.json"; then
  status=42
fi
if [ "$status" -eq 0 ]; then
  score_one a || status=$?
fi
if [ "$status" -eq 0 ]; then
  score_one b || status=$?
fi
if [ "$status" -eq 0 ] && ! cmp -s "$out/result_a.json" "$out/result_b.json"; then
  status=43
fi
if [ "$status" -eq 0 ]; then
  sha256sum \
    "$out/candidate_freeze_a.json" \
    "$out/candidate_freeze_b.json" \
    "$out/result_a.json" \
    "$out/result_b.json" \
    > "$out/sha256.txt"
fi
printf '%s\n' "$status" > "$out/status"
exit "$status"
