"""
pipeline.py: ReTabSyn 全流程流水线。

三阶段:
  1. 场景数据生成 (scenario data generation)
  2. 数据合成 (synthesis)
  3. 下游评估 (evaluation)

通过 --stages 控制运行哪些阶段。
"""

from __future__ import annotations
import argparse
import json
import os
import sys
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
from synthesis.cart_synthesizer import CARTSynthesizer

SYNTHESIZERS = {
    "smote": SmoteSynthesizer,
    "great": GreatSynthesizer,
    "pta": PTASynthesizer,
    "tvae": TVAESynthesizer,
    "retabsyn": ReTabSynSynthesizer,
    "cart": CARTSynthesizer,
    "cartgenir": CARTGenIRSynthesizer,
}


# ===========================================================================
# Stage 1: 场景数据生成
# ===========================================================================

def generate_scenario_data(
    csv_path: str,
    target_col: str,
    scenarios: dict,
    output_dir: str = "scenario_data",
    base_seed: int = 42,
    n_seeds: int = 10,
) -> None:
    """生成所有场景/配置/seed 的 train/test 切分并保存。"""
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

    for scenario_name, configs in scenarios.items():
        scenario_dir = os.path.join(os.path.normpath(output_dir), scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        for label, kwargs in configs:
            kwargs = dict(kwargs)
            if "target_col" in kwargs:
                kwargs["target_col"] = target_col

            for i in range(n_seeds):
                seed = base_seed + i
                scenario = ScenarioFactory.create(scenario_name, seed=seed, **kwargs)
                train_df, test_df = scenario.build(df)

                train_df.to_csv(os.path.join(scenario_dir, f"train_{label}_seed{seed}.csv"), index=False)
                test_df.to_csv(os.path.join(scenario_dir, f"test_{label}_seed{seed}.csv"), index=False)
                print(f"[generate] {scenario_name}/{label}/seed{seed} train={train_df.shape} test={test_df.shape}")


# ===========================================================================
# Stage 2+3: 合成 + 评估
# ===========================================================================

def _resolve_seed_kwargs(kwargs: dict, seed: int, label: str = "") -> dict:
    result = {}
    for k, v in kwargs.items():
        if isinstance(v, str):
            v = v.replace("{seed}", str(seed)).replace("{label}", label)
        result[k] = v
    return result


def run_single_seed(
    *,
    seed: int,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    scenario: object,           # BaseScenario 实例（用于获取 evaluator）
    synthesizer_name: str,
    synthesizer_kwargs: dict,
    target_col: str,
    n_synth_samples: int | None,
    scenario_label: str = "",
    synth_output_dir: str | None = None,
    scenario_name: str = "",
) -> dict:
    """单 seed: 合成 + 评估。"""
    synth_cls = SYNTHESIZERS.get(synthesizer_name)
    if synth_cls is None:
        raise ValueError(f"未知合成算法: {synthesizer_name}")

    synth = synth_cls(random_state=seed, **_resolve_seed_kwargs(synthesizer_kwargs, seed, scenario_label))
    synth.fit(real_train)
    n_samples = n_synth_samples if n_synth_samples is not None else len(real_train)
    synth_df = synth.sample(n_samples=n_samples)

    if synth_output_dir and scenario_label:
        os.makedirs(synth_output_dir, exist_ok=True)
        fname = f"{scenario_name}_{synthesizer_name}_{scenario_label}_seed{seed}.csv"
        synth_df.to_csv(os.path.join(synth_output_dir, fname), index=False)

    # 用场景对应的 evaluator 评估
    evaluator = scenario.evaluator if scenario is not None else None
    if evaluator is None:
        from evaluator.downstream import DownstreamEvaluator
        evaluator = DownstreamEvaluator()

    return evaluator.evaluate(
        real_train=real_train,
        real_test=real_test,
        synth_df=synth_df,
        target_col=target_col,
        seed=seed,
    )


def _load_or_build_scenario(
    *,
    seed: int,
    csv_path: str,
    scenario_name: str,
    scenario_kwargs: dict,
    scenario_data_dir: str | None,
    scenario_label: str,
    df_cache: pd.DataFrame | None,
):
    """加载预生成数据，或现场构建场景。返回 (scenario, real_train, real_test)。"""
    if scenario_data_dir:
        if not scenario_label:
            raise ValueError("--scenario-label 不能为空（使用预生成场景数据时必填）")
        base = os.path.normpath(scenario_data_dir)
        real_train = pd.read_csv(os.path.join(base, f"train_{scenario_label}_seed{seed}.csv"))
        real_test = pd.read_csv(os.path.join(base, f"test_{scenario_label}_seed{seed}.csv"))
        return None, real_train, real_test

    if df_cache is None:
        df_cache = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    scenario = ScenarioFactory.create(scenario_name, seed=seed, **scenario_kwargs)
    real_train, real_test = scenario.build(df_cache)
    return scenario, real_train, real_test


def run_synthesis_eval(
    *,
    csv_path: str = "",
    scenario_name: str = "",
    scenario_kwargs: dict | None = None,
    synthesizer_name: str = "smote",
    synthesizer_kwargs: dict | None = None,
    target_col: str,
    n_synth_samples: int | None = None,
    n_seeds: int = 10,
    base_seed: int = 42,
    output_csv: str | None = None,
    scenario_data_dir: str | None = None,
    scenario_label: str = "",
    synth_output_dir: str | None = None,
) -> dict:
    """合成 + 评估（multi-seed）。"""
    if synthesizer_kwargs is None:
        synthesizer_kwargs = {}
    if scenario_kwargs is None:
        scenario_kwargs = {}

    df_cache = None
    all_results = []
    for i in range(n_seeds):
        seed = base_seed + i
        scenario, real_train, real_test = _load_or_build_scenario(
            seed=seed, csv_path=csv_path, scenario_name=scenario_name,
            scenario_kwargs=scenario_kwargs, scenario_data_dir=scenario_data_dir,
            scenario_label=scenario_label, df_cache=df_cache,
        )
        res = run_single_seed(
            seed=seed, real_train=real_train, real_test=real_test,
            scenario=scenario, synthesizer_name=synthesizer_name,
            synthesizer_kwargs=synthesizer_kwargs, target_col=target_col,
            n_synth_samples=n_synth_samples, scenario_label=scenario_label,
            synth_output_dir=synth_output_dir, scenario_name=scenario_name,
        )
        all_results.append(res)

    aggregated = aggregate_results(all_results)
    if output_csv:
        _save_results(aggregated, all_results, scenario_name, scenario_kwargs,
                      synthesizer_name, output_csv, base_seed)
    return aggregated


# ===========================================================================
# 聚合与保存
# ===========================================================================

def aggregate_results(all_results: list[dict]) -> dict:
    accum = defaultdict(lambda: defaultdict(list))
    for res in all_results:
        for mode, clf_scores in res.items():
            for clf, score in clf_scores.items():
                if not np.isnan(score):
                    accum[mode][clf].append(score)
            valid = [s for s in clf_scores.values() if not np.isnan(s)]
            if valid:
                accum[mode]["Average"].append(float(np.mean(valid)))

    aggregated = {}
    for mode, clf_dict in accum.items():
        aggregated[mode] = {}
        for clf, vals in clf_dict.items():
            arr = np.array(vals)
            aggregated[mode][clf] = (float(arr.mean()), float(arr.std() / np.sqrt(len(arr))))
    return aggregated


def _save_results(aggregated, all_results, scenario_name, scenario_kwargs,
                  synthesizer_name, output_csv, base_seed):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    agg_rows = []
    for mode, clf_dict in aggregated.items():
        if "Average" in clf_dict:
            mean, se = clf_dict["Average"]
            agg_rows.append({
                "scenario": scenario_name,
                **{f"param_{k}": v for k, v in scenario_kwargs.items()},
                "synthesizer": synthesizer_name,
                "mode": mode, "mean": round(mean, 4), "se": round(se, 4),
            })

    raw_rows = []
    for i, res in enumerate(all_results):
        for mode, clf_scores in res.items():
            for clf, score in clf_scores.items():
                raw_rows.append({
                    "seed": base_seed + i, "mode": mode,
                    "classifier": clf, "score": round(score, 4),
                })

    out_df = pd.concat([pd.DataFrame(agg_rows), pd.DataFrame(raw_rows)],
                       axis=0, ignore_index=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")


def _print_results(result: dict) -> None:
    print(f"\n{'=' * 40}")
    print(f"  {'Mode':<14} {'Mean':>8} {'SE':>8}")
    print(f"  {'-' * 32}")
    for mode in ["real", "synthetic", "augment"]:
        if mode in result and "Average" in result[mode]:
            mean, se = result[mode]["Average"]
            print(f"  {mode:<14} {mean:>8.4f} {se:>8.4f}")


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="ReTabSyn 全流程流水线")

    p.add_argument("--stages", default="1,2,3",
                   help="运行的阶段，逗号分隔: 1=场景生成, 2=合成, 3=评估。默认全部")

    # Stage 1 参数
    p.add_argument("--csv", help="原始 CSV 路径（阶段1需要）")
    p.add_argument("--scenario", help="场景名（阶段1需要）")
    p.add_argument("--scenario-kwargs", type=str, default="{}",
                   help="场景构造参数 JSON")
    p.add_argument("--scenarios", type=str, default=None,
                   help="多场景配置 JSON（阶段1批量模式）")

    # Stage 2 参数
    p.add_argument("--synthesizer", default="smote",
                   choices=list(SYNTHESIZERS.keys()), help="合成算法")
    p.add_argument("--synth-kwargs", type=str, default="{}",
                   help="合成器参数 JSON（覆盖 configs 中的值）")
    p.add_argument("--n-synth", type=int, default=None, help="合成样本数")

    # Stage 2+3 共用
    p.add_argument("--target", required=True, help="目标列名")
    p.add_argument("--scenario-data-dir", default=None, help="预生成场景数据目录")
    p.add_argument("--scenario-label", default="", help="场景配置标签")

    # Stage 3 参数
    p.add_argument("--output", default=None, help="评估结果 CSV 输出路径")
    p.add_argument("--synth-output-dir", default=None, help="合成数据 CSV 输出目录")

    # 通用
    p.add_argument("--output-dir", default="scenario_data", help="场景数据输出目录")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--metric", default="auroc", choices=["auroc", "prauc"])
    return p.parse_args()


def _load_synth_config(synthesizer: str, synth_kwargs_str: str) -> dict:
    config_path = f"configs/synth_{synthesizer}.json"
    kwargs = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            kwargs.update(json.load(f).get("synth_kwargs", {}))
        print(f"Loaded config: {config_path}")
    kwargs.update(json.loads(synth_kwargs_str))
    return kwargs


def main():
    args = _parse_args()
    stages = set(int(s) for s in args.stages.split(","))

    synth_kwargs = _load_synth_config(args.synthesizer, args.synth_kwargs)

    # Stage 1: 场景数据生成
    if 1 in stages:
        if not args.csv or not args.scenario:
            print("错误: 阶段1需要 --csv 和 --scenario")
            sys.exit(1)
        if args.scenarios:
            scenarios = json.loads(args.scenarios)
        else:
            label = args.scenario_label or "default"
            scenarios = {args.scenario: [[label, json.loads(args.scenario_kwargs)]]}
        generate_scenario_data(
            csv_path=args.csv, target_col=args.target, scenarios=scenarios,
            output_dir=args.output_dir, base_seed=args.base_seed, n_seeds=args.n_seeds,
        )

    # Stage 2+3: 合成 + 评估
    if 2 in stages or 3 in stages:
        aggregated = run_synthesis_eval(
            csv_path=args.csv or "",
            scenario_name=args.scenario or "",
            scenario_kwargs=json.loads(args.scenario_kwargs),
            synthesizer_name=args.synthesizer,
            synthesizer_kwargs=synth_kwargs,
            target_col=args.target,
            n_synth_samples=args.n_synth,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
            output_csv=args.output,
            scenario_data_dir=args.scenario_data_dir,
            scenario_label=args.scenario_label,
            synth_output_dir=args.synth_output_dir,
        )
        _print_results(aggregated)


if __name__ == "__main__":
    main()
