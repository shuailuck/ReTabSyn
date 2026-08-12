#!/bin/bash
#
# ReTabSyn 全流程批量实验脚本
#
# 每个场景: 先生成数据，再遍历算法验证。
#

set -e

CSV="csv/example/wilt.csv"
TARGET="class"
N_SEEDS=10
BASE_SEED=42
RESULTS_DIR="results"
SCENARIO_DIR="scenario_data"
SYNTH_DIR="synth_data"

SYNTH_ALGOS=(
    "smote"
    "tvae"
    # "great"      # 启用 retabsyn 前先启用 great（GReaT 训练→存模型，retabsyn 加载→DPO）
    # "retabsyn"
    # "pta"
    # "cartgenir"
)

SMALL_NS=(32 64 128 256)
IMBALANCED_PREVS=(0.01 0.05 0.10)

mkdir -p "$RESULTS_DIR" "$SCENARIO_DIR" "$SYNTH_DIR"

echo "=========================================="
echo "  ReTabSyn Pipeline - Batch Experiments"
echo "  CSV: $CSV"
echo "  Target: $TARGET"
echo "  Seeds: $N_SEEDS x (base=$BASE_SEED)"
echo "  Algorithms: ${SYNTH_ALGOS[*]}"
echo "=========================================="

# ====================================================================
# Small Data 场景
# ====================================================================
echo ""
echo "========== Small Data =========="

for N in "${SMALL_NS[@]}"; do
    LABEL="n${N}"
    KWARGS="{\"target_col\": \"$TARGET\", \"n_samples\": $N}"

    # 生成场景数据
    python generate_scenario_data.py \
        --csv "$CSV" --target "$TARGET" \
        --scenarios "{\"small\": [[\"$LABEL\", $KWARGS]]}" \
        --output-dir "$SCENARIO_DIR" --base-seed "$BASE_SEED" --n-seeds "$N_SEEDS"

    # 各算法
    for ALGO in "${SYNTH_ALGOS[@]}"; do
        echo ""
        echo ">>> small/$LABEL | $ALGO"
        python run_pipeline.py \
            --scenario-data-dir "$SCENARIO_DIR/small" --scenario-label "$LABEL" \
            --target "$TARGET" --synthesizer "$ALGO" \
            --n-seeds "$N_SEEDS" --base-seed "$BASE_SEED" \
            --output "$RESULTS_DIR/small_${ALGO}_${LABEL}.csv" \
            --synth-output-dir "$SYNTH_DIR"
    done
done

# ====================================================================
# Imbalanced 场景
# ====================================================================
echo ""
echo "========== Imbalanced =========="

for PREV in "${IMBALANCED_PREVS[@]}"; do
    LABEL="prev$(echo "$PREV" | sed 's/0\.//')"
    KWARGS="{\"target_col\": \"$TARGET\", \"minority_prev\": $PREV}"

    # 生成场景数据
    python generate_scenario_data.py \
        --csv "$CSV" --target "$TARGET" \
        --scenarios "{\"imbalanced\": [[\"$LABEL\", $KWARGS]]}" \
        --output-dir "$SCENARIO_DIR" --base-seed "$BASE_SEED" --n-seeds "$N_SEEDS"

    # 各算法
    for ALGO in "${SYNTH_ALGOS[@]}"; do
        echo ""
        echo ">>> imbalanced/$LABEL | $ALGO"
        python run_pipeline.py \
            --scenario-data-dir "$SCENARIO_DIR/imbalanced" --scenario-label "$LABEL" \
            --target "$TARGET" --synthesizer "$ALGO" \
            --n-seeds "$N_SEEDS" --base-seed "$BASE_SEED" \
            --output "$RESULTS_DIR/imbalanced_${ALGO}_${LABEL}.csv" \
            --synth-output-dir "$SYNTH_DIR"
    done
done

# ====================================================================
# Distribution Shift 场景 (需要含 split_col 的数据集，例如 wilt 的 "class")
# ====================================================================
# echo ""
# echo "========== Distribution Shift =========="
#
# SPLIT_COL="class"
# SHIFT_NS=(32 64 128 256 512)
#
# for N in "${SHIFT_NS[@]}"; do
#     LABEL="n${N}"
#     KWARGS="{\"split_col\": \"$SPLIT_COL\", \"target_col\": \"$TARGET\", \"n_train\": $N, \"n_test\": $N}"
#
#     python generate_scenario_data.py \
#         --csv "$CSV" --target "$TARGET" \
#         --scenarios "{\"shift\": [[\"$LABEL\", $KWARGS]]}" \
#         --output-dir "$SCENARIO_DIR" --base-seed "$BASE_SEED" --n-seeds "$N_SEEDS"
#
#     for ALGO in "${SYNTH_ALGOS[@]}"; do
#         echo ""
#         echo ">>> shift/$LABEL | $ALGO"
#         python run_pipeline.py \
#             --scenario-data-dir "$SCENARIO_DIR/shift" --scenario-label "$LABEL" \
#             --target "$TARGET" --synthesizer "$ALGO" \
#             --n-seeds "$N_SEEDS" --base-seed "$BASE_SEED" \
#             --output "$RESULTS_DIR/shift_${ALGO}_${LABEL}.csv" \
#             --synth-output-dir "$SYNTH_DIR"
#     done
# done

echo ""
echo "Done."
