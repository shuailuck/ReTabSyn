"""
run_pipeline.py: ReTabSyn 全流程实验脚本

流程: 场景数据生成 → 数据合成 → 下游评估
通过 seed 控制单次流程中的所有随机性，支持 multi-seed 重复实验并汇总均值 ± 标准误。

用法示例:
  python run_pipeline.py \
      --csv csv/example/wilt.csv \
      --scenario small \
      --target class \
      --scenario-kwargs '{"target_col": "class", "n_samples": 64}' \
      --n-seeds 10
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from collections import defaultdict

from scenario import ScenarioFactory
from synthesis.smote_synthesizer import SmoteSynthesizer
from evaluator.utility import evaluate_all_modes


# ---------------------------------------------------------------------------
# 单 Seed 流程
# ---------------------------------------------------------------------------

def run_single_seed(
    *,
    seed: int,
    csv_path: str,
    scenario_name: str,
    scenario_kwargs: dict,
    synthesizer_name: str,
    synthesizer_kwargs: dict,
    target_col: str,
    n_synth_samples: int | None = None,
    metric: str = "auroc",
) -> dict[str, dict[str, float]]:
    """
    单个 seed 的完整流程。

    Parameters
    ----------
    seed : 控制本轮的场景划分与合成随机性。
    csv_path : 原始 CSV 数据路径。
    scenario_name : "small" | "imbalanced" | "shift"
    scenario_kwargs : 场景构造参数（不含 seed，由本函数注入）。
    synthesizer_name : 合成算法名称，当前支持 "smote"。
    synthesizer_kwargs : 合成器构造参数（不含 random_state）。
    target_col : 目标列名，用于下游评估。
    n_synth_samples : 合成样本数，默认与 real_train 等量。
    metric : "auroc" | "prauc"

    Returns
    -------
    {"real": {clf: score}, "synthetic": {clf: score}, "augment": {clf: score}}
    """
    # 1. 加载原始数据
    df = pd.read_csv(csv_path)

    # 2. 构建场景
    scenario = ScenarioFactory.create(scenario_name, seed=seed, **scenario_kwargs)
    real_train, real_test = scenario.build(df)

    # 3. 数据合成
    synth_df = None
    if synthesizer_name == "smote":
        synth = SmoteSynthesizer(random_state=seed, **synthesizer_kwargs)
        synth.fit(real_train)
        n_samples = n_synth_samples if n_synth_samples is not None else len(real_train)
        synth_df = synth.sample(n_samples=n_samples)
    else:
        raise ValueError(f"未知合成算法: {synthesizer_name}")

    # 4. 下游评估
    return evaluate_all_modes(
        real_train=real_train,
        real_test=real_test,
        synth_df=synth_df,
        target_col=target_col,
        seed=seed,
        metric=metric,
    )


# ---------------------------------------------------------------------------
# 多 Seed 聚合
# ---------------------------------------------------------------------------

def aggregate_results(
    all_results: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    聚合多个 seed 的结果，计算每个 (mode, classifier) 的均值与标准误。

    Returns
    -------
    {mode: {classifier: (mean, standard_error)}}
    """
    accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for res in all_results:
        for mode, clf_scores in res.items():
            for clf, score in clf_scores.items():
                if not np.isnan(score):
                    accum[mode][clf].append(score)

    aggregated: dict[str, dict[str, tuple[float, float]]] = {}
    for mode, clf_dict in accum.items():
        aggregated[mode] = {}
        for clf, vals in clf_dict.items():
            arr = np.array(vals)
            aggregated[mode][clf] = (float(arr.mean()), float(arr.std() / np.sqrt(len(arr))))
    return aggregated


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    csv_path: str,
    scenario_name: str,
    scenario_kwargs: dict,
    synthesizer_name: str = "smote",
    synthesizer_kwargs: dict | None = None,
    target_col: str,
    n_synth_samples: int | None = None,
    n_seeds: int = 10,
    base_seed: int = 42,
    metric: str = "auroc",
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    完整 multi-seed 实验入口。

    Returns
    -------
    {mode: {classifier: (mean, standard_error)}}
    """
    if synthesizer_kwargs is None:
        synthesizer_kwargs = {}

    all_results: list[dict[str, dict[str, float]]] = []
    for i in range(n_seeds):
        seed = base_seed + i
        res = run_single_seed(
            seed=seed,
            csv_path=csv_path,
            scenario_name=scenario_name,
            scenario_kwargs=scenario_kwargs,
            synthesizer_name=synthesizer_name,
            synthesizer_kwargs=synthesizer_kwargs,
            target_col=target_col,
            n_synth_samples=n_synth_samples,
            metric=metric,
        )
        all_results.append(res)

    return aggregate_results(all_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReTabSyn 全流程实验脚本")

    p.add_argument("--csv", required=True, help="原始 CSV 数据路径")
    p.add_argument("--scenario", required=True, choices=["small", "imbalanced", "shift"],
                   help="测试场景")
    p.add_argument("--target", required=True, help="目标列名（用于下游评估）")
    p.add_argument("--scenario-kwargs", type=str, default="{}",
                   help="场景构造参数，JSON 字典字符串。"
                        " small: {\"target_col\":..., \"n_samples\":64, \"balance_mode\":\"raw\"}"
                        " imbalanced: {\"target_col\":..., \"minority_prev\":0.01}"
                        " shift: {\"split_col\":...}")
    p.add_argument("--synthesizer", default="smote", help="合成算法 (默认 smote)")
    p.add_argument("--synth-kwargs", type=str, default="{}",
                   help="合成器参数，JSON 字典字符串，如 '{\"k_neighbors\": 5}'")
    p.add_argument("--n-synth", type=int, default=None, help="合成样本数（默认与训练集等量）")
    p.add_argument("--n-seeds", type=int, default=10, help="重复实验次数 (默认 10)")
    p.add_argument("--base-seed", type=int, default=42, help="起始随机种子 (默认 42)")
    p.add_argument("--metric", default="auroc", choices=["auroc", "prauc"],
                   help="评估指标 (默认 auroc)")

    return p.parse_args()


def _print_results(result: dict[str, dict[str, tuple[float, float]]]) -> None:
    """格式化打印结果表格。"""
    modes = ["real", "synthetic", "augment"]
    for mode in modes:
        if mode not in result:
            continue
        print(f"\n{'=' * 50}")
        print(f"  Mode: {mode}")
        print(f"  {'Classifier':<12} {'Mean':>8} {'SE':>8}")
        print(f"  {'-' * 30}")
        for clf, (mean, se) in result[mode].items():
            print(f"  {clf:<12} {mean:>8.4f} {se:>8.4f}")


if __name__ == "__main__":
    args = _parse_args()

    result = run_pipeline(
        csv_path=args.csv,
        scenario_name=args.scenario,
        scenario_kwargs=json.loads(args.scenario_kwargs),
        synthesizer_name=args.synthesizer,
        synthesizer_kwargs=json.loads(args.synth_kwargs),
        target_col=args.target,
        n_synth_samples=args.n_synth,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        metric=args.metric,
    )

    _print_results(result)
