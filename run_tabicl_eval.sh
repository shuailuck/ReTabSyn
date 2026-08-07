#!/bin/bash
#
# TabICL In-Context Learning 评测脚本
#
# 直接读取预生成的场景数据和合成数据，用 TabICL 做 ICL 推理。
# 场景数据和合成数据由 run_all.sh 产出。
#

set -e

SCENARIO_DIR="scenario_data"
SYNTH_DIR="synth_data"
RESULTS_DIR="results/tabicl"
TARGET="class"
N_SEEDS=10
BASE_SEED=42

SYNTH_ALGOS=(
    "smote"
    "tvae"
    # "cartgenir"
)

SMALL_NS=(32 64 128 256)
IMBALANCED_PREVS=(0.01 0.05 0.10)

ALGO_LIST=$(IFS=,; echo "${SYNTH_ALGOS[*]}")
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "  TabICL ICL Evaluation"
echo "  Scenario: $SCENARIO_DIR"
echo "  Synth: $SYNTH_DIR"
echo "  Target: $TARGET"
echo "  Seeds: $N_SEEDS x (base=$BASE_SEED)"
echo "  Algorithms: ${SYNTH_ALGOS[*]}"
echo "=========================================="

# ── Small Data ──────────────────────────────────────────
echo ""
echo "========== Small Data =========="

for N in "${SMALL_NS[@]}"; do
    LABEL="n${N}"
    LABEL_LIST="$LABEL"
    echo ""
    echo "--- small/$LABEL ---"
    python run_tabicl_eval.py \
        --scenario-dir "$SCENARIO_DIR/small" \
        --synth-dir "$SYNTH_DIR" \
        --synth-algos "$ALGO_LIST" \
        --labels "$LABEL_LIST" \
        --target "$TARGET" \
        --base-seed "$BASE_SEED" --n-seeds "$N_SEEDS" \
        --output "$RESULTS_DIR/small_${LABEL}.csv"
done

# ── Imbalanced ──────────────────────────────────────────
echo ""
echo "========== Imbalanced =========="

for PREV in "${IMBALANCED_PREVS[@]}"; do
    LABEL="prev$(echo "$PREV" | sed 's/0\.//')"
    LABEL_LIST="$LABEL"
    echo ""
    echo "--- imbalanced/$LABEL ---"
    python run_tabicl_eval.py \
        --scenario-dir "$SCENARIO_DIR/imbalanced" \
        --synth-dir "$SYNTH_DIR" \
        --synth-algos "$ALGO_LIST" \
        --labels "$LABEL_LIST" \
        --target "$TARGET" \
        --base-seed "$BASE_SEED" --n-seeds "$N_SEEDS" \
        --output "$RESULTS_DIR/imbalanced_${LABEL}.csv"
done

echo ""
echo "Done."
