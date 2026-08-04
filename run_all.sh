#!/bin/bash
#
# ReTabSyn 全流程批量实验脚本
#

set -e

CSV="csv/example/wilt.csv"
TARGET="class"
N_SEEDS=10
BASE_SEED=42

echo "=========================================="
echo "  ReTabSyn Pipeline - Batch Experiments"
echo "  CSV: $CSV"
echo "  Target: $TARGET"
echo "  Seeds: $N_SEEDS x (base=$BASE_SEED)"
echo "=========================================="

# ── Small Data 场景 ──────────────────────────────────────
echo ""
echo ">>> [1/2] Small Data Scenario (n_samples=64)"
python run_pipeline.py \
    --csv "$CSV" \
    --scenario small \
    --target "$TARGET" \
    --scenario-kwargs "{\"target_col\": \"$TARGET\", \"n_samples\": 64}" \
    --n-seeds "$N_SEEDS" \
    --base-seed "$BASE_SEED"

# ── Imbalanced 场景 ──────────────────────────────────────
echo ""
echo ">>> [2/2] Imbalanced Scenario (minority_prev=0.05)"
python run_pipeline.py \
    --csv "$CSV" \
    --scenario imbalanced \
    --target "$TARGET" \
    --scenario-kwargs "{\"target_col\": \"$TARGET\", \"minority_prev\": 0.05}" \
    --n-seeds "$N_SEEDS" \
    --base-seed "$BASE_SEED"

# ── Distribution Shift 场景（需要含 split_col 的数据集）────
# echo ""
# echo ">>> [3/3] Distribution Shift Scenario"
# python run_pipeline.py \
#     --csv "$CSV" \
#     --scenario shift \
#     --target "$TARGET" \
#     --scenario-kwargs "{\"split_col\": \"your_split_column\"}" \
#     --n-seeds "$N_SEEDS" \
#     --base-seed "$BASE_SEED"

echo ""
echo "Done."
