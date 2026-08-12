"""
run_tabpfn_eval.py: TabPFN In-Context Learning 评估 CLI。

薄封装，核心逻辑在 tabpfn/ 包中。
"""

import argparse
from tabpfn_icl.pipeline import run, get_default_strategies, get_imbalanced_strategies


def _parse_args():
    p = argparse.ArgumentParser(description="TabPFN ICL 评估")
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
    p.add_argument("--metric", default="auroc", choices=["auroc", "prauc"])
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
        metric=args.metric,
        output_csv=args.output,
    )
