"""
run_quality_eval.py: 合成数据质量评估 CLI。

评估两个维度:
  1. 统计相似性 (Chi-squared, Total Variation, KL Divergence)
  2. 标签正确性 (LR, DT, RF, MLP)
"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from evaluator.quality import statistical_similarity, label_accuracy


def run(
    *,
    scenario_dir: str,
    synth_dir: str,
    synth_algos: list[str],
    scenario_labels: list[str],
    scenario: str = "",
    target_col: str,
    base_seed: int = 42,
    n_seeds: int = 10,
    output_csv: str | None = None,
):
    rows = []

    for label in scenario_labels:
        for algo in synth_algos:
            stat_accum = defaultdict(list)
            label_accum = defaultdict(list)

            for i in range(n_seeds):
                seed = base_seed + i
                train_path = os.path.join(scenario_dir,
                                          f"train_{label}_seed{seed}.csv")
                test_path = os.path.join(scenario_dir,
                                         f"test_{label}_seed{seed}.csv")
                if not os.path.exists(train_path):
                    continue
                real_train = pd.read_csv(train_path)
                real_test = pd.read_csv(test_path)

                synth_path = os.path.join(
                    synth_dir, f"{scenario}_{algo}_{label}_seed{seed}.csv"
                )
                if not os.path.exists(synth_path):
                    continue
                synth_df = pd.read_csv(synth_path)

                # 1. 统计相似性 (逐列，取均值)
                stat_df = statistical_similarity(real_train, synth_df)
                stat_accum["chi2_pvalue"].append(stat_df["chi2_pvalue"].mean())
                stat_accum["total_variation"].append(stat_df["total_variation"].mean())
                stat_accum["kl_divergence"].append(stat_df["kl_divergence"].mean())

                # 2. 标签正确性
                acc = label_accuracy(real_train, synth_df, target_col, seed=seed)
                for model, score in acc.items():
                    if not np.isnan(score):
                        label_accum[model].append(score)

            print(f"\n{'=' * 60}")
            print(f"  {label} / {algo}")
            print(f"  {'Metric':<24} {'Mean':>8} {'SE':>8}")
            print(f"  {'-' * 42}")

            # 输出统计相似性
            for metric in ["chi2_pvalue", "total_variation", "kl_divergence"]:
                vals = stat_accum[metric]
                if vals:
                    arr = np.array(vals)
                    mean, se = float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))
                    print(f"  {metric:<24} {mean:>8.4f} {se:>8.4f}")
                    rows.append({
                        "label": label, "algo": algo, "dimension": "statistical",
                        "metric": metric, "mean": round(mean, 4), "se": round(se, 4),
                    })

            # 输出标签正确性
            for model, vals in label_accum.items():
                if vals:
                    arr = np.array(vals)
                    mean, se = float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))
                    print(f"  acc_{model:<19} {mean:>8.4f} {se:>8.4f}")
                    rows.append({
                        "label": label, "algo": algo, "dimension": "label_accuracy",
                        "metric": f"acc_{model}", "mean": round(mean, 4),
                        "se": round(se, 4),
                    })

    df = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved to: {output_csv}")
    return df


def _parse_args():
    p = argparse.ArgumentParser(description="合成数据质量评估")
    p.add_argument("--scenario-dir", required=True, help="场景数据目录")
    p.add_argument("--synth-dir", required=True, help="合成数据目录")
    p.add_argument("--synth-algos", type=str, required=True,
                   help="合成算法列表，逗号分隔")
    p.add_argument("--labels", type=str, required=True,
                   help="场景标签列表，逗号分隔")
    p.add_argument("--scenario", default="", help="场景名")
    p.add_argument("--target", required=True, help="目标列名")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--output", default=None, help="结果 CSV 输出路径")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        scenario_dir=args.scenario_dir,
        synth_dir=args.synth_dir,
        synth_algos=args.synth_algos.split(","),
        scenario_labels=args.labels.split(","),
        scenario=args.scenario,
        target_col=args.target,
        base_seed=args.base_seed,
        n_seeds=args.n_seeds,
        output_csv=args.output,
    )
