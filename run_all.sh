#!/bin/bash
#
# ReTabSyn 全流程批量实验脚本
# 通过配置文件 (configs/*.json) 指定合成算法及其参数
#

set -e

CSV="csv/example/wilt.csv"
TARGET="class"
N_SEEDS=10
BASE_SEED=42
RESULTS_DIR="results"

# 合成算法列表（自动加载 configs/synth_<name>.json）
SYNTH_ALGOS=(
    "smote"
    "tvae"
    # "great"
    # "pta"
    # "retabsyn"
)

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "  ReTabSyn Pipeline - Batch Experiments"
echo "  CSV: $CSV"
echo "  Target: $TARGET"
echo "  Seeds: $N_SEEDS x (base=$BASE_SEED)"
echo "  Output: $RESULTS_DIR/"
echo "  Algorithms: ${SYNTH_ALGOS[*]}"
echo "=========================================="

# ── Small Data 场景（多组 n_samples）────────────────────
SMALL_N_SAMPLES=(32 64 128 256)

for ALGO in "${SYNTH_ALGOS[@]}"; do
    for N in "${SMALL_N_SAMPLES[@]}"; do
        echo ""
        echo ">>> Small Data | $ALGO | n_samples=$N"
        python run_pipeline.py \
            --csv "$CSV" \
            --scenario small \
            --target "$TARGET" \
            --synthesizer "$ALGO" \
            --scenario-kwargs "{\"target_col\": \"$TARGET\", \"n_samples\": $N}" \
            --n-seeds "$N_SEEDS" \
            --base-seed "$BASE_SEED" \
            --output "$RESULTS_DIR/small_${ALGO}_n${N}.csv"
    done
done

# ── Imbalanced 场景 ──────────────────────────────────────
for ALGO in "${SYNTH_ALGOS[@]}"; do
    echo ""
    echo ">>> Imbalanced | $ALGO | minority_prev=0.05"
    python run_pipeline.py \
        --csv "$CSV" \
        --scenario imbalanced \
        --target "$TARGET" \
        --synthesizer "$ALGO" \
        --scenario-kwargs "{\"target_col\": \"$TARGET\", \"minority_prev\": 0.05}" \
        --n-seeds "$N_SEEDS" \
        --base-seed "$BASE_SEED" \
        --output "$RESULTS_DIR/imbalanced_${ALGO}_prev005.csv"
done

# ── Distribution Shift 场景（需要含 split_col 的数据集）────
# for ALGO in "${SYNTH_ALGOS[@]}"; do
#     echo ""
#     echo ">>> Distribution Shift | $ALGO"
#     python run_pipeline.py \
#         --csv "$CSV" \
#         --scenario shift \
#         --target "$TARGET" \
#         --synthesizer "$ALGO" \
#         --scenario-kwargs "{\"split_col\": \"your_split_column\"}" \
#         --n-seeds "$N_SEEDS" \
#         --base-seed "$BASE_SEED" \
#         --output "$RESULTS_DIR/shift_${ALGO}.csv"
# done

echo ""
echo "Done."
