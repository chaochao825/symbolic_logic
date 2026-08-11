#!/usr/bin/env bash

set -euo pipefail

: "${AFTS_VARC_ROOT:?Set AFTS_VARC_ROOT to the pinned VARC source directory.}"
: "${AFTS_VARC_PYTHON:?Set AFTS_VARC_PYTHON to the frozen VARC interpreter.}"
: "${AFTS_VARC_CHECKPOINT:?Set AFTS_VARC_CHECKPOINT to checkpoint_best.pt.}"
: "${AFTS_VARC_CHECKPOINT_SHA256:?Set the expected checkpoint SHA-256.}"
: "${AFTS_VARC_SOURCE_COMMIT:?Set the expected VARC source commit.}"
: "${AFTS_VARC_DATA_ROOT_NAME:?Set the query-blind data-root name.}"
: "${AFTS_VARC_SAVE_NAME:?Set the immutable provider save name.}"

if [[ $# -ne 1 ]]; then
  printf 'usage: %s TASK_ID\n' "$0" >&2
  exit 2
fi

task_id=$1
if [[ ! "${task_id}" =~ ^[A-Za-z0-9_-]{1,200}$ ]]; then
  printf 'unsafe VARC task ID: %s\n' "${task_id}" >&2
  exit 2
fi

for required_path in \
  "${AFTS_VARC_ROOT}/test_time_train_ARC.py" \
  "${AFTS_VARC_PYTHON}" \
  "${AFTS_VARC_CHECKPOINT}" \
  "${AFTS_VARC_ROOT}/raw_data/${AFTS_VARC_DATA_ROOT_NAME}/data/eval_color_permute_ttt_9/${task_id}"; do
  if [[ ! -e "${required_path}" ]]; then
    printf 'Required path does not exist: %s\n' "${required_path}" >&2
    exit 2
  fi
done

actual_checkpoint_sha256=$(sha256sum "${AFTS_VARC_CHECKPOINT}" | awk '{print $1}')
if [[ "${actual_checkpoint_sha256}" != "${AFTS_VARC_CHECKPOINT_SHA256}" ]]; then
  printf 'VARC checkpoint SHA-256 mismatch\n' >&2
  exit 3
fi
actual_source_commit=$(git -C "${AFTS_VARC_ROOT}" rev-parse HEAD)
if [[ "${actual_source_commit}" != "${AFTS_VARC_SOURCE_COMMIT}" ]]; then
  printf 'VARC source commit mismatch\n' >&2
  exit 3
fi

runtime_task_name=${task_id%%_*}
runtime_prediction_path="${AFTS_VARC_ROOT}/outputs/${AFTS_VARC_SAVE_NAME}_attempt_0/${runtime_task_name}_predictions.json"
prediction_path="${AFTS_VARC_ROOT}/outputs/${AFTS_VARC_SAVE_NAME}_attempt_0/${task_id}_predictions.json"
if [[ "${runtime_prediction_path}" != "${prediction_path}" && -e "${runtime_prediction_path}" ]] || [[ -e "${prediction_path}" ]]; then
  printf 'Refusing to replace a runtime or canonical prediction artifact\n' >&2
  exit 2
fi

cd "${AFTS_VARC_ROOT}"
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

/usr/bin/time -v "${AFTS_VARC_PYTHON}" test_time_train_ARC.py \
  --epochs 100 \
  --depth 10 \
  --batch-size 8 \
  --image-size 64 \
  --patch-size 2 \
  --learning-rate 3e-4 \
  --weight-decay 0 \
  --embed-dim 512 \
  --num-heads 8 \
  --num-colors 12 \
  --resume-checkpoint "${AFTS_VARC_CHECKPOINT}" \
  --lr-scheduler cosine \
  --train-split "eval_color_permute_ttt_9/${task_id}" \
  --data-root "raw_data/${AFTS_VARC_DATA_ROOT_NAME}" \
  --eval-split "eval_color_permute_ttt_9/${task_id}" \
  --resume-skip-task-token \
  --architecture vit \
  --eval-save-name "${AFTS_VARC_SAVE_NAME}" \
  --num-attempts 10 \
  --ttt-num-each 1 \
  --seed 42

if [[ ! -s "${runtime_prediction_path}" ]]; then
  printf 'VARC did not produce a non-empty prediction artifact: %s\n' \
    "${runtime_prediction_path}" >&2
  exit 4
fi
if [[ "${runtime_prediction_path}" != "${prediction_path}" ]]; then
  mv -- "${runtime_prediction_path}" "${prediction_path}"
fi
