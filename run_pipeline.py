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
      --n-seeds 10 \
      --output results/small_n64.csv
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from scenario import ScenarioFactory
from synthesis.smote_synthesizer import SmoteSynthesizer
from synthesis.great_synthesizer import GreatSynthesizer
from synthesis.pta_synthesizer import PTASynthesizer
from synthesis.tvae_synthesizer import TVAESynthesizer
from synthesis.retabsyn_synthesizer import ReTabSynSynthesizer
from synthesis.cartgenir_synthesizer import CARTGenIRSynthesizer
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
    单个 seed 的完整流程，返回 {mode: {classifier: score}}。
    """
    df = pd.read_csv(csv_path)

    scenario = ScenarioFactory.create(scenario_name, seed=seed, **scenario_kwargs)
    real_train, real_test = scenario.build(df)

    synth_df = None
    _SYNTHESIZERS = {
        "smote": SmoteSynthesizer,
        "great": GreatSynthesizer,
        "pta": PTASynthesizer,
        "tvae": TVAESynthesizer,
        "retabsyn": ReTabSynSynthesizer,
        "cartgenir": CARTGenIRSynthesizer,
    }
    synth_cls = _SYNTHESIZERS.get(synthesizer_name)
    if synth_cls is None:
        raise ValueError(f"未知合成算法: {synthesizer_name}")

    synth = synth_cls(random_state=seed, **synthesizer_kwargs)
    synth.fit(real_train)
    n_samples = n_synth_samples if n_synth_samples is not None else len(real_train)
    synth_df = synth.sample(n_samples=n_samples)

    return evaluate_all_modes(
        real_train=real_train,
        real_test=real_test,
        synth_df=synth_df,
        target_col=target_col,
        seed=seed,
        metric=metric,
    )


# ---------------------------------------------------------------------------
# 原始结果展平
# ---------------------------------------------------------------------------

def _flatten_raw_results(
    all_results: list[dict[str, dict[str, float]]],
    base_seed: int,
    scenario_name: str,
    scenario_kwargs: dict,
    metric: str,
) -> pd.DataFrame:
    """将多 seed 的嵌套结果展平为长格式 DataFrame。"""
    rows = []
    for i, res in enumerate(all_results):
        seed = base_seed + i
        for mode, clf_scores in res.items():
            for clf, score in clf_scores.items():
                rows.append({
                    "seed": seed,
                    "scenario": scenario_name,
                    **{f"param_{k}": v for k, v in scenario_kwargs.items()},
                    "mode": mode,
                    "classifier": clf,
                    "metric": metric,
                    "score": score,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def aggregate_results(
    all_results: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """聚合多个 seed 的结果，返回 {mode: {classifier: (mean, se)}}。"""
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
    output_csv: str | None = None,
) -> tuple[dict[str, dict[str, tuple[float, float]]], pd.DataFrame]:
    """
    完整 multi-seed 实验入口。

    Returns
    -------
    (aggregated, raw_df)
        aggregated : {mode: {classifier: (mean, se)}}
        raw_df : 每行一个 (seed, mode, classifier) 的长格式 DataFrame
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

    aggregated = aggregate_results(all_results)
    raw_df = _flatten_raw_results(all_results, base_seed, scenario_name, scenario_kwargs, metric)

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        raw_df.to_csv(output_csv, index=False)
        print(f"\nRaw results saved to: {output_csv}")

    return aggregated, raw_df


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
                   help="场景构造参数，JSON 字典。"
                        " small: {\"target_col\":..., \"n_samples\":64}"
                        " imbalanced: {\"target_col\":..., \"minority_prev\":0.01}"
                        " shift: {\"split_col\":...}")
    p.add_argument("--synthesizer", default="smote",
                   choices=["smote", "great", "pta", "tvae", "retabsyn", "cartgenir"],
                   help="合成算法，自动从 configs/synth_<name>.json 加载参数")
    p.add_argument("--synth-kwargs", type=str, default="{}",
                   help="合成器参数覆盖，JSON 字典，如 '{\"k_neighbors\": 5}'（会覆盖配置文件中的同名字段）")
    p.add_argument("--n-synth", type=int, default=None, help="合成样本数（默认与训练集等量）")
    p.add_argument("--n-seeds", type=int, default=10, help="重复实验次数 (默认 10)")
    p.add_argument("--base-seed", type=int, default=42, help="起始随机种子 (默认 42)")
    p.add_argument("--metric", default="auroc", choices=["auroc", "prauc"],
                   help="评估指标 (默认 auroc)")
    p.add_argument("--output", default=None, help="原始结果 CSV 输出路径")

    return p.parse_args()


def _print_results(result: dict[str, dict[str, tuple[float, float]]]) -> None:
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

    # 自动加载 configs/synth_<name>.json
    config_path = f"configs/synth_{args.synthesizer}.json"
    synth_kwargs = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        synth_kwargs.update(cfg.get("synth_kwargs", {}))
        print(f"Loaded config: {config_path}")

    # CLI --synth-kwargs 覆盖配置文件中的值
    synth_kwargs.update(json.loads(args.synth_kwargs))

    aggregated, raw_df = run_pipeline(
        csv_path=args.csv,
        scenario_name=args.scenario,
        scenario_kwargs=json.loads(args.scenario_kwargs),
        synthesizer_name=args.synthesizer,
        synthesizer_kwargs=synth_kwargs,
        target_col=args.target,
        n_synth_samples=args.n_synth,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        metric=args.metric,
        output_csv=args.output,
    )

    _print_results(aggregated)
