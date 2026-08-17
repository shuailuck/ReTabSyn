"""
pipeline.py: ReTabSyn 全流程流水线。

完整流程：场景数据生成 → 合成 → 评估（默认全部执行）。
对每个 seed 由 scenario 内部编排三阶段。
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from scenario import ScenarioFactory
from scenario.scenario import SaveConfig
from synthesis.smote_synthesizer import SmoteSynthesizer
from synthesis.great_synthesizer import GreatSynthesizer
from synthesis.pta_synthesizer import PTASynthesizer
from synthesis.tvae_synthesizer import TVAESynthesizer
from synthesis.retabsyn_synthesizer import ReTabSynSynthesizer
from synthesis.cartgenir_synthesizer import CARTGenIRSynthesizer
from synthesis.cart_synthesizer import CARTSynthesizer
from synthesis.evolve_synthesizer import EvolveSynthesizer

SYNTHESIZERS = {
    "smote": SmoteSynthesizer,
    "great": GreatSynthesizer,
    "pta": PTASynthesizer,
    "tvae": TVAESynthesizer,
    "retabsyn": ReTabSynSynthesizer,
    "cart": CARTSynthesizer,
    "cartgenir": CARTGenIRSynthesizer,
    "evolve": EvolveSynthesizer,
}


def _resolve_seed_kwargs(kwargs: dict, seed: int, label: str = "") -> dict:
    result = {}
    for k, v in kwargs.items():
        if isinstance(v, str):
            v = v.replace("{seed}", str(seed)).replace("{label}", label)
        result[k] = v
    return result


def run_pipeline(
    *,
    csv_path: str,
    scenario_name: str,
    scenario_label: str,
    scenario_kwargs: dict,
    synthesizer_name: str,
    synthesizer_kwargs: dict | None,
    target_col: str,
    n_synth_samples: int | None = None,
    n_seeds: int = 10,
    base_seed: int = 42,
    scenario_output_dir: str = "scenario_data",
    synth_output_dir: str | None = None,
    output_csv: str | None = None,
) -> dict:
    """完整流程：生成场景数据 → 合成 → 评估（multi-seed）。

    对每个 seed: 构建场景 → 保存场景数据 → 合成 → 保存合成数据 → 评估。
    """
    if synthesizer_kwargs is None:
        synthesizer_kwargs = {}
    if scenario_kwargs is None:
        scenario_kwargs = {}

    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    synth_cls = SYNTHESIZERS.get(synthesizer_name)
    if synth_cls is None:
        raise ValueError(f"未知合成算法: {synthesizer_name}")

    scenario_dir = os.path.join(scenario_output_dir, scenario_name)
    save_config = SaveConfig(
        scenario_name=scenario_name,
        scenario_output_dir=scenario_dir,
        scenario_label=scenario_label,
        synth_output_dir=synth_output_dir or "",
        synthesizer_name=synthesizer_name,
    )

    all_results = []
    for i in range(n_seeds):
        seed = base_seed + i
        scenario = ScenarioFactory.create(
            scenario_name, seed=seed,
            **{**scenario_kwargs, "target_col": target_col, "save_config": save_config},
        )

        # 1. 场景数据生成（场景内部处理存在检测 + 保存）
        scenario.build(df)

        # 2. 合成（场景内部处理存在检测 + 保存）
        kwargs = _resolve_seed_kwargs(synthesizer_kwargs, seed, scenario_label)
        scenario.synthesize(synth_cls, kwargs, n_synth_samples)

        # 3. 评估
        results = scenario.evaluate(target_col)
        all_results.append(results)

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
    for mode in result:
        if "Average" in result[mode]:
            mean, se = result[mode]["Average"]
            print(f"  {mode:<14} {mean:>8.4f} {se:>8.4f}")


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="ReTabSyn 全流程流水线（生成-合成-评估）")

    p.add_argument("--csv", required=True, help="原始 CSV 路径")
    p.add_argument("--scenario", required=True, help="场景名")
    p.add_argument("--scenario-label", default="", help="场景配置标签（用于文件名）")
    p.add_argument("--scenario-kwargs", type=str, default="{}", help="场景构造参数 JSON")

    p.add_argument("--synthesizer", default="smote", choices=list(SYNTHESIZERS.keys()))
    p.add_argument("--synth-kwargs", type=str, default="{}", help="合成器参数 JSON")
    p.add_argument("--n-synth", type=int, default=None, help="合成样本数")

    p.add_argument("--target", required=True, help="目标列名")

    p.add_argument("--scenario-output-dir", default="scenario_data", help="场景数据输出目录")
    p.add_argument("--synth-output-dir", default=None, help="合成数据输出目录")
    p.add_argument("--output", default=None, help="评估结果 CSV 输出路径")

    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=42)
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
    synth_kwargs = _load_synth_config(args.synthesizer, args.synth_kwargs)

    aggregated = run_pipeline(
        csv_path=args.csv,
        scenario_name=args.scenario,
        scenario_label=args.scenario_label,
        scenario_kwargs=json.loads(args.scenario_kwargs),
        synthesizer_name=args.synthesizer,
        synthesizer_kwargs=synth_kwargs,
        target_col=args.target,
        n_synth_samples=args.n_synth,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        scenario_output_dir=args.scenario_output_dir,
        synth_output_dir=args.synth_output_dir,
        output_csv=args.output,
    )
    _print_results(aggregated)


if __name__ == "__main__":
    main()
