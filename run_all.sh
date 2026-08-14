#!/bin/bash
#
# ReTabSyn 全流程批量实验脚本
#
# 每个 (场景, 算法) 组合执行完整流程：生成场景数据 → 合成 → 评估。
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
echo "  Algorithms: ${SYNTH_ALGOS[*]}"
echo "=========================================="

# ── Small Data 场景 ────────────────────────────────────
echo ""
echo "========== Small Data =========="
for N in "${SMALL_NS[@]}"; do
    LABEL="n${N}"
    for ALGO in "${SYNTH_ALGOS[@]}"; do
        echo ""
        echo ">>> small/$LABEL | $ALGO"
        python pipeline.py \
            --csv "$CSV" --scenario small --scenario-label "$LABEL" \
            --scenario-kwargs "{\"n_samples\": $N}" \
            --target "$TARGET" --synthesizer "$ALGO" \
            --n-seeds "$N_SEEDS" --base-seed "$BASE_SEED" \
            --scenario-output-dir "$SCENARIO_DIR" \
            --synth-output-dir "$SYNTH_DIR" \
            --output "$RESULTS_DIR/small_${ALGO}_${LABEL}.csv"
    done
done

# ── Imbalanced 场景 ─────────────────────────────────────
echo ""
echo "========== Imbalanced =========="
for PREV in "${IMBALANCED_PREVS[@]}"; do
    LABEL="prev$(echo "$PREV" | sed 's/0\.//')"
    for ALGO in "${SYNTH_ALGOS[@]}"; do
        echo ""
        echo ">>> imbalanced/$LABEL | $ALGO"
        python pipeline.py \
            --csv "$CSV" --scenario imbalanced --scenario-label "$LABEL" \
            --scenario-kwargs "{\"minority_prev\": $PREV}" \
            --target "$TARGET" --synthesizer "$ALGO" \
            --n-seeds "$N_SEEDS" --base-seed "$BASE_SEED" \
            --scenario-output-dir "$SCENARIO_DIR" \
            --synth-output-dir "$SYNTH_DIR" \
            --output "$RESULTS_DIR/imbalanced_${ALGO}_${LABEL}.csv"
    done
done

# ── Noisy Label 场景 ────────────────────────────────────
echo ""
echo "========== Noisy Label =========="
for RATIO in "${NOISY_RATIOS[@]}"; do
    LABEL="nr$(awk "BEGIN{printf \"%.0f\", $RATIO*100}")"
    for ALGO in "${SYNTH_ALGOS[@]}"; do
        echo ""
        echo ">>> noisy_label/$LABEL | $ALGO"
        python pipeline.py \
            --csv "$CSV" --scenario noisy_label --scenario-label "$LABEL" \
            --scenario-kwargs "{\"noise_ratio\": $RATIO}" \
            --target "$TARGET" --synthesizer "$ALGO" \
            --n-seeds "$N_SEEDS" --base-seed "$BASE_SEED" \
            --scenario-output-dir "$SCENARIO_DIR" \
            --synth-output-dir "$SYNTH_DIR" \
            --output "$RESULTS_DIR/noisy_label_${ALGO}_${LABEL}.csv"
    done
done

echo ""
echo "Done."
