#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s TASK_ID\n' "$0" >&2
  exit 2
fi

ROOT=/home/wangmeiqi/codex_runs/arc_visual_posterior_bridge_20260809
VARC=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/external/VARC
PYTHON=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807/.venv-varc/bin/python
TASK_ID=$1
CHECKPOINT="$VARC/saves/offline_train_ViT/checkpoint_best.pt"
EXPECTED_CHECKPOINT_SHA=c52d719fe9a41066aa8e96548ae6f83a0abf3568facd5103e176d7005e8421b3

if [[ ! "$TASK_ID" =~ ^[0-9a-f]{8}$ ]]; then
  printf 'invalid ARC task ID: %s\n' "$TASK_ID" >&2
  exit 2
fi
actual_checkpoint_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
if [[ "$actual_checkpoint_sha" != "$EXPECTED_CHECKPOINT_SHA" ]]; then
  printf 'checkpoint SHA-256 mismatch\n' >&2
  exit 3
fi

cd "$VARC"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

/usr/bin/time -v "$PYTHON" test_time_train_ARC.py \
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
  --resume-checkpoint "$CHECKPOINT" \
  --lr-scheduler cosine \
  --train-split "eval_color_permute_ttt_9/$TASK_ID" \
  --data-root raw_data/ARC2_QUERY_BLIND_STRUCTURE_V1 \
  --eval-split "eval_color_permute_ttt_9/$TASK_ID" \
  --resume-skip-task-token \
  --architecture vit \
  --eval-save-name query_blind_structure_v1 \
  --num-attempts 10 \
  --ttt-num-each 1 \
  --seed 42
