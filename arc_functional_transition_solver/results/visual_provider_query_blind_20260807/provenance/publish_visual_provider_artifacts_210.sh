#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

ROOT=/home/wangmeiqi/codex_runs/arc_visual_provider_probe_20260807
REPO=/home/wangmeiqi/codex_runs/symbolic_logic_arc_online_control_20260724/repo/arc_functional_transition_solver
DEST="$REPO/results/visual_provider_query_blind_20260807"

copy_new() {
    local source=$1
    local target=$2
    if [[ -e "$target" ]]; then
        if cmp --silent "$source" "$target"; then
            return
        fi
        echo "refusing to replace different existing artifact: $target" >&2
        exit 1
    fi
    install -m 0644 "$source" "$target"
}

copy_new "$ROOT/aggregate_v2_1_v3_confirm/summary.json" \
    "$DEST/aggregate/summary.json"

copy_new "$ROOT/cohort_v2/manifest.json" "$DEST/v2_1/cohort_manifest.json"
copy_new "$ROOT/exposure_registry_v2.json" "$DEST/v2_1/exposure_registry.json"
copy_new "$ROOT/provider_contract_v2_1.json" "$DEST/v2_1/provider_contract.json"
copy_new "$ROOT/from236_v2_1/provider_raw_validation_v2_1.json" \
    "$DEST/v2_1/raw_validation.json"
copy_new "$ROOT/pre_gold_freeze_receipt_v2_1.json" \
    "$DEST/v2_1/pre_gold_freeze_receipt.json"
copy_new "$ROOT/comparison_freeze_receipt_v2_1.json" \
    "$DEST/v2_1/comparison_freeze_receipt.json"
copy_new "$ROOT/baseline_v2/summary.json" "$DEST/v2_1/baseline_summary.json"
copy_new "$ROOT/object_code_v2/summary.json" "$DEST/v2_1/object_code_summary.json"
copy_new "$ROOT/score_v2_1/frozen_predictions.json" \
    "$DEST/v2_1/frozen_predictions.json"
copy_new "$ROOT/score_v2_1/summary.json" "$DEST/v2_1/score_summary.json"
copy_new "$ROOT/posterior_audit_v2_1/replay_receipt.json" \
    "$DEST/v2_1/posterior_replay_receipt.json"
copy_new "$ROOT/posterior_audit_v2_1/summary.json" \
    "$DEST/v2_1/posterior_summary.json"
copy_new "$ROOT/posterior_bridge_v2_1/bridge_candidates.json" \
    "$DEST/v2_1/bridge_candidates.json"
copy_new "$ROOT/posterior_bridge_v2_1/summary.json" \
    "$DEST/v2_1/bridge_summary.json"

copy_new "$ROOT/cohort_v3_confirm/manifest.json" \
    "$DEST/v3_confirm/cohort_manifest.json"
copy_new "$ROOT/exposure_registry_v3_confirm.json" \
    "$DEST/v3_confirm/exposure_registry.json"
copy_new "$ROOT/provider_contract_v3_confirm.json" \
    "$DEST/v3_confirm/provider_contract.json"
copy_new "$ROOT/from236_v3_confirm/provider_raw_validation_v3_confirm.json" \
    "$DEST/v3_confirm/raw_validation.json"
copy_new "$ROOT/pre_gold_freeze_receipt_v3_confirm.json" \
    "$DEST/v3_confirm/pre_gold_freeze_receipt.json"
copy_new "$ROOT/comparison_freeze_receipt_v3_confirm.json" \
    "$DEST/v3_confirm/comparison_freeze_receipt.json"
copy_new "$ROOT/baseline_v3_confirm/summary.json" \
    "$DEST/v3_confirm/baseline_summary.json"
copy_new "$ROOT/object_code_v3_confirm/summary.json" \
    "$DEST/v3_confirm/object_code_summary.json"
copy_new "$ROOT/score_v3_confirm/frozen_predictions.json" \
    "$DEST/v3_confirm/frozen_predictions.json"
copy_new "$ROOT/score_v3_confirm/summary.json" \
    "$DEST/v3_confirm/score_summary.json"
copy_new "$ROOT/posterior_audit_v3_confirm/replay_receipt.json" \
    "$DEST/v3_confirm/posterior_replay_receipt.json"
copy_new "$ROOT/posterior_audit_v3_confirm/summary.json" \
    "$DEST/v3_confirm/posterior_summary.json"

copy_new "$ROOT/build_visual_v3_pre_gold.py" \
    "$DEST/provenance/build_visual_v3_pre_gold.py"
copy_new "$ROOT/build_visual_v3_comparison_freeze.py" \
    "$DEST/provenance/build_visual_v3_comparison_freeze.py"
copy_new "$ROOT/publish_visual_provider_artifacts_210.sh" \
    "$DEST/provenance/publish_visual_provider_artifacts_210.sh"

for cohort in v2_1 v3_confirm; do
    archive="$DEST/raw/${cohort}_predictions.tar.gz"
    if [[ -e "$archive" ]]; then
        echo "refusing to replace existing artifact: $archive" >&2
        exit 1
    fi
done

(
    cd "$ROOT/from236_v2_1/external/VARC/outputs/query_blind_pilot_v2_attempt_0"
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --format=ustar \
        -cf - ./*_predictions.json
) | gzip -n > "$DEST/raw/v2_1_predictions.tar.gz"
(
    cd "$ROOT/from236_v3_confirm/external/VARC/outputs/query_blind_confirm_v3_attempt_0"
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --format=ustar \
        -cf - ./*_predictions.json
) | gzip -n > "$DEST/raw/v3_confirm_predictions.tar.gz"

if [[ -e "$DEST/SHA256SUMS" ]]; then
    echo "refusing to replace existing artifact: $DEST/SHA256SUMS" >&2
    exit 1
fi
(
    cd "$DEST"
    find . -type f ! -name SHA256SUMS -print0 \
        | sort -z \
        | xargs -0 sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS
)
