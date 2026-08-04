#!/bin/bash
#
# ReTabSyn 全流程批量实验脚本
#

set -e

CSV="csv/example/wilt.csv"
TARGET="class"
N_SEEDS=10
BASE_SEED=42
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "  ReTabSyn Pipeline - Batch Experiments"
echo "  CSV: $CSV"
echo "  Target: $TARGET"
echo "  Seeds: $N_SEEDS x (base=$BASE_SEED)"
echo "  Output: $RESULTS_DIR/"
echo "=========================================="

# ── Small Data 场景（多组 n_samples）────────────────────
SMALL_N_SAMPLES=(32 64 128 256)

for N in "${SMALL_N_SAMPLES[@]}"; do
    echo ""
    echo ">>> Small Data Scenario (n_samples=$N)"
    python run_pipeline.py \
        --csv "$CSV" \
        --scenario small \
        --target "$TARGET" \
        --scenario-kwargs "{\"target_col\": \"$TARGET\", \"n_samples\": $N}" \
        --n-seeds "$N_SEEDS" \
        --base-seed "$BASE_SEED" \
        --output "$RESULTS_DIR/small_n${N}.csv"
done

# ── Imbalanced 场景 ──────────────────────────────────────
echo ""
echo ">>> Imbalanced Scenario (minority_prev=0.05)"
python run_pipeline.py \
    --csv "$CSV" \
    --scenario imbalanced \
    --target "$TARGET" \
    --scenario-kwargs "{\"target_col\": \"$TARGET\", \"minority_prev\": 0.05}" \
    --n-seeds "$N_SEEDS" \
    --base-seed "$BASE_SEED" \
    --output "$RESULTS_DIR/imbalanced_prev005.csv"

# ── Distribution Shift 场景（需要含 split_col 的数据集）────
# echo ""
# echo ">>> Distribution Shift Scenario"
# python run_pipeline.py \
#     --csv "$CSV" \
#     --scenario shift \
#     --target "$TARGET" \
#     --scenario-kwargs "{\"split_col\": \"your_split_column\"}" \
#     --n-seeds "$N_SEEDS" \
#     --base-seed "$BASE_SEED" \
#     --output "$RESULTS_DIR/shift.csv"

echo ""
echo "Done."
