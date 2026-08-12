#!/usr/bin/env bash

set -euo pipefail

EXP=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp004_prospective_v1
RUNNER="${EXP}/run_varc_shard_exp004.sh"

test -s "${EXP}/visual_launch_commitment.json"
for session in afts_exp004_varc_0 afts_exp004_varc_1 afts_exp004_varc_2; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    printf 'refusing to reuse tmux session: %s\n' "${session}" >&2
    exit 2
  fi
done
for shard_index in 0 1 2; do
  test ! -e "${EXP}/varc_shard_${shard_index}_launcher_exit_code.txt"
done

gpu_ids=(0 1 3)
for shard_index in 0 1 2; do
  gpu_id=${gpu_ids[shard_index]}
  session="afts_exp004_varc_${shard_index}"
  tmux new-session -d -s "${session}" \
    "bash '${RUNNER}' '${shard_index}' '${gpu_id}' > '${EXP}/varc_shard_${shard_index}_launcher.stdout.log' 2> '${EXP}/varc_shard_${shard_index}_launcher.stderr.log'"
  tmux list-panes -t "${session}" -F '#{pane_pid}' \
    > "${EXP}/varc_shard_${shard_index}_launcher.pid"
done

date --iso-8601=seconds > "${EXP}/visual_provider_started_at.txt"
printf 'started shard sessions on GPUs 0, 1, and 3\n'
