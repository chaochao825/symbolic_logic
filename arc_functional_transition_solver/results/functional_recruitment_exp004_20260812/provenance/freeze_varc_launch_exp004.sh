#!/usr/bin/env bash

set -euo pipefail

EXP=/home/spco/sow_linear/codex_artifacts/hypothesis_population_20260812/exp004_prospective_v1
PROJ=/home/spco/sow_linear/codex_worktrees/arc_failure_matrix_20260810/arc_functional_transition_solver
VARC=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/external/VARC
VARC_PY=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/.venv-varc/bin/python
CHECKPOINT="${VARC}/saves/offline_train_ViT/checkpoint_best.pt"

test -s "${EXP}/pre_visual_commitment.json"
test -s "${EXP}/varc_data_preparation_manifest.json"
test ! -e "${EXP}/visual_launch_commitment.json"
[[ "$(git -C "${VARC}" rev-parse HEAD)" == \
  bd478ecf362e6499a988b05f33223e5c5fc6a6be ]]
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == \
  c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3 ]]

for shard_index in 0 1 2; do
  test ! -e "${EXP}/varc_shard_${shard_index}_predictions_v1"
  test ! -e "${EXP}/varc_shard_${shard_index}_run_v1"
done
for gpu_id in 0 1 3; do
  if ! flock -n "/tmp/afts-varc-gpu${gpu_id}.lock" -c true; then
    printf 'VARC GPU lock is held for GPU %s\n' "${gpu_id}" >&2
    exit 3
  fi
done
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "${EXP}/varc_gpu_inventory_prelaunch.csv"

cd "${PROJ}"
PYTHONPATH=src "${VARC_PY}" - "${EXP}" "${PROJ}" "${VARC}" "${CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

from afts_arc.experiment_safety import atomic_write_json, canonical_sha256, file_sha256

exp = Path(sys.argv[1])
project = Path(sys.argv[2])
varc = Path(sys.argv[3])
checkpoint = Path(sys.argv[4])
gpu_rows = (
    exp / "varc_gpu_inventory_prelaunch.csv"
).read_text(encoding="utf-8").splitlines()
gpu_inventory = {}
for row in gpu_rows:
    fields = [field.strip() for field in row.split(",")]
    gpu_inventory[int(fields[0])] = {
        "memory_total": fields[4],
        "memory_used": fields[3],
        "name": fields[2],
        "utilization": fields[5],
        "uuid": fields[1],
    }
mapping = [
    {"gpu_id": 0, "shard_index": 0},
    {"gpu_id": 1, "shard_index": 1},
    {"gpu_id": 3, "shard_index": 2},
]
for record in mapping:
    gpu_id = record["gpu_id"]
    memory_used = int(str(gpu_inventory[gpu_id]["memory_used"]).split()[0])
    utilization = int(str(gpu_inventory[gpu_id]["utilization"]).split()[0])
    if memory_used > 1000 or utilization > 10:
        raise ValueError(f"GPU {gpu_id} is not idle at launch commitment")
body = {
    "checkpoint_sha256": file_sha256(checkpoint),
    "close_and_score_script_sha256": file_sha256(
        exp / "close_varc_and_score_exp004.sh"
    ),
    "data_preparation_sha256": file_sha256(
        exp / "varc_data_preparation_manifest.json"
    ),
    "freeze_launch_script_sha256": file_sha256(
        exp / "freeze_varc_launch_exp004.sh"
    ),
    "gpu_assignment": mapping,
    "gpu_inventory": {str(key): gpu_inventory[key] for key in sorted(gpu_inventory)},
    "launcher_sha256": file_sha256(
        project / "scripts" / "launch_afts_arc_varc_cohort.sh"
    ),
    "pre_visual_commitment_sha256": file_sha256(
        exp / "pre_visual_commitment.json"
    ),
    "query_gold_opened": False,
    "run_shard_script_sha256": file_sha256(exp / "run_varc_shard_exp004.sh"),
    "score_equal_native_cost_script_sha256": file_sha256(
        exp / "score_equal_native_cost.py"
    ),
    "schema": "afts.exp004-visual-launch-commitment/v1",
    "shard_assignment_sha256": file_sha256(
        exp / "varc_shard_assignment_v1" / "assignment.json"
    ),
    "source_commit": "bd478ecf362e6499a988b05f33223e5c5fc6a6be",
    "source_tracked_diff_sha256": file_sha256(
        exp / "varc_source_tracked.diff"
    ),
    "start_shards_script_sha256": file_sha256(
        exp / "start_varc_shards_exp004.sh"
    ),
    "task_runner_sha256": file_sha256(
        project / "scripts" / "run_afts_arc_varc_task.sh"
    ),
    "task_count": 98,
    "visual_provider_started": False,
}
atomic_write_json(
    exp / "visual_launch_commitment.json",
    {"launch_commitment_id": canonical_sha256(body), **body},
)
print(
    json.dumps(
        {
            "gpu_assignment": mapping,
            "launch_commitment_id": canonical_sha256(body),
            "task_count": 98,
        },
        sort_keys=True,
    )
)
PY
