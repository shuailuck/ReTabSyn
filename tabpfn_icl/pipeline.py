"""
tabpfn/pipeline.py: 批量评估流水线。

按场景、算法、seed 遍历，聚合结果并保存。
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from tabpfn_icl.core import TabPFNEngine
from tabpfn_icl.context import (
    ContextBuilder, RealOnlyBuilder, SynthOnlyBuilder,
    MixedBuilder, FilteredMixedBuilder, AllInBuilder,
)


def get_default_strategies() -> list[ContextBuilder]:
    """默认策略集合。"""
    return [
        RealOnlyBuilder(),
        SynthOnlyBuilder(),
        AllInBuilder(),
        MixedBuilder(),
        FilteredMixedBuilder(),
    ]


def get_imbalanced_strategies() -> list[ContextBuilder]:
    """不平衡场景策略集合：只用合成数据作 context。"""
    return [
        RealOnlyBuilder(),
        SynthOnlyBuilder(),
    ]


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
    metric: str = "auroc",
    output_csv: str | None = None,
    strategies: list[ContextBuilder] | None = None,
):
    """批量评估入口。

    Parameters
    ----------
    strategies : 自定义策略列表，None 则根据 scenario 自动选择
    """
    if strategies is None:
        if scenario == "imbalanced":
            strategies = get_imbalanced_strategies()
        else:
            strategies = get_default_strategies()

    engine = TabPFNEngine(metric=metric)
    rows = []

    for label in scenario_labels:
        for algo in synth_algos:
            all_scores = defaultdict(list)

            for i in range(n_seeds):
                seed = base_seed + i
                train_path = os.path.join(scenario_dir,
                                          f"train_{label}_seed{seed}.csv")
                test_path = os.path.join(scenario_dir,
                                         f"test_{label}_seed{seed}.csv")
                if not os.path.exists(train_path):
                    print(f"  SKIP {label}/seed{seed}: train file not found")
                    continue
                real_train = pd.read_csv(train_path)
                real_test = pd.read_csv(test_path)

                synth_path = os.path.join(
                    synth_dir,
                    f"{scenario}_{algo}_{label}_seed{seed}.csv"
                )
                synth_df = (pd.read_csv(synth_path)
                            if os.path.exists(synth_path) else None)

                data = engine.preprocess(
                    real_train, real_test, synth_df, target_col
                )

                for builder in strategies:
                    builder.seed = seed
                    score = engine.evaluate(data, builder, target_col)
                    if not np.isnan(score):
                        all_scores[builder.name].append(score)

            _print_and_record(all_scores, label, algo, metric, rows)

    df = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved to: {output_csv}")
    return df


def _print_and_record(
    all_scores: dict[str, list[float]],
    label: str,
    algo: str,
    metric: str,
    rows: list,
):
    print(f"\n{'=' * 50}")
    print(f"  {label} / {algo}")
    print(f"  {'Strategy':<28} {'Mean':>8} {'SE':>8}")
    print(f"  {'-' * 46}")
    for mode, vals in all_scores.items():
        if vals:
            arr = np.array(vals)
            mean, se = float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))
            print(f"  {mode:<28} {mean:>8.4f} {se:>8.4f}")
            rows.append({
                "label": label, "algo": algo, "strategy": mode,
                "metric": metric, "mean": round(mean, 4), "se": round(se, 4),
            })
