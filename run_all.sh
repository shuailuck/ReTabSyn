#!/bin/bash
#
# ReTabSyn 全流程批量实验脚本
#
# 三阶段: 1=场景生成, 2=合成, 3=评估
# 通过 RUN_STAGE_* 控制运行哪些阶段
#

set -e

# ── 阶段开关 ──────────────────────────────────────────
RUN_STAGE1=1   # 1=生成场景数据, 0=跳过（使用已有 scenario_data）
RUN_STAGE23=1  # 1=合成+评估, 0=跳过

# ── 全局配置 ──────────────────────────────────────────
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
    # "great"
    # "retabsyn"
    # "pta"
    # "cart"
    # "cartgenir"
)

SMALL_NS=(32 64 128 256)
IMBALANCED_PREVS=(0.01 0.05 0.10)
NOISY_RATIOS=(0.1 0.2 0.3 0.4)

mkdir -p "$RESULTS_DIR" "$SCENARIO_DIR" "$SYNTH_DIR"

echo "=========================================="
echo "  ReTabSyn Pipeline"
echo "  CSV: $CSV  |  Target: $TARGET"
echo "  Seeds: $N_SEEDS x (base=$BASE_SEED)"
echo "  Stages: 1=$RUN_STAGE1, 2+3=$RUN_STAGE23"
echo "  Algorithms: ${SYNTH_ALGOS[*]}"
echo "=========================================="

# ====================================================================
# Stage 1: 场景数据生成
# ====================================================================
if [ "$RUN_STAGE1" -eq 1 ]; then
    echo ""
    echo "========== Stage 1: Scenario Generation =========="

    # Small Data
    for N in "${SMALL_NS[@]}"; do
        python pipeline.py --stages 1 \
            --csv "$CSV" --scenario small --scenario-label "n${N}" \
            --scenario-kwargs "{\"target_col\": \"$TARGET\", \"n_samples\": $N}" \
            --target "$TARGET" --output-dir "$SCENARIO_DIR" \
            --base-seed "$BASE_SEED" --n-seeds "$N_SEEDS"
    done

    # Imbalanced
    for PREV in "${IMBALANCED_PREVS[@]}"; do
        LABEL="prev$(echo "$PREV" | sed 's/0\.//')"
        python pipeline.py --stages 1 \
            --csv "$CSV" --scenario imbalanced --scenario-label "$LABEL" \
            --scenario-kwargs "{\"target_col\": \"$TARGET\", \"minority_prev\": $PREV}" \
            --target "$TARGET" --output-dir "$SCENARIO_DIR" \
            --base-seed "$BASE_SEED" --n-seeds "$N_SEEDS"
    done

    # Noisy Label
    for RATIO in "${NOISY_RATIOS[@]}"; do
        LABEL="nr$(awk "BEGIN{printf \"%.0f\", $RATIO*100}")"
        python pipeline.py --stages 1 \
            --csv "$CSV" --scenario noisy_label --scenario-label "$LABEL" \
            --scenario-kwargs "{\"target_col\": \"$TARGET\", \"noise_ratio\": $RATIO}" \
            --target "$TARGET" --output-dir "$SCENARIO_DIR" \
            --base-seed "$BASE_SEED" --n-seeds "$N_SEEDS"
    done
fi

# ====================================================================
# Stage 2+3: 合成 + 评估
# ====================================================================
if [ "$RUN_STAGE23" -eq 1 ]; then
    echo ""
    echo "========== Stage 2+3: Synthesis + Evaluation =========="

    # 遍历所有已生成的场景目录
    for SCENARIO_DIR_ENTRY in "$SCENARIO_DIR"/*/; do
        SCENARIO_NAME=$(basename "$SCENARIO_DIR_ENTRY")
        # 从目录中提取所有 label
        LABELS=$(ls "$SCENARIO_DIR_ENTRY" | grep '^train_' | sed -E 's/^train_([^_]+)_seed[0-9]+\.csv$/\1/' | sort -u)

        for LABEL in $LABELS; do
            for ALGO in "${SYNTH_ALGOS[@]}"; do
                echo ""
                echo ">>> $SCENARIO_NAME/$LABEL | $ALGO"
                python pipeline.py --stages 2,3 \
                    --scenario-data-dir "$SCENARIO_DIR/$SCENARIO_NAME" \
                    --scenario-label "$LABEL" --scenario "$SCENARIO_NAME" \
                    --target "$TARGET" --synthesizer "$ALGO" \
                    --n-seeds "$N_SEEDS" --base-seed "$BASE_SEED" \
                    --output "$RESULTS_DIR/${SCENARIO_NAME}_${ALGO}_${LABEL}.csv" \
                    --synth-output-dir "$SYNTH_DIR"
            done
        done
    done
fi

echo ""
echo "Done."
