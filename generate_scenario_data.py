"""
generate_scenario_data.py: 预先生成场景数据切分，保存到本地。

所有算法共享同一份场景数据，保证输入一致且避免重复计算。

用法示例:
  python generate_scenario_data.py \
      --csv csv/example/wilt.csv \
      --target class \
      --scenarios '{
          "small": [["n32", {"n_samples": 32}], ["n64", {"n_samples": 64}]],
          "imbalanced": [["prev005", {"minority_prev": 0.05}]]
      }' \
      --n-seeds 10
"""

import os
import argparse
import json
import pandas as pd

from scenario import ScenarioFactory


def generate_scenario_data(
    csv_path: str,
    target_col: str,
    scenarios: dict,
    output_dir: str = "scenario_data",
    base_seed: int = 42,
    n_seeds: int = 10,
):
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

    for scenario_name, configs in scenarios.items():
        scenario_dir = os.path.join(os.path.normpath(output_dir), scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        for label, kwargs in configs:
            if "target_col" in kwargs:
                kwargs["target_col"] = target_col

            for i in range(n_seeds):
                seed = base_seed + i
                scenario = ScenarioFactory.create(scenario_name, seed=seed, **kwargs)
                train_df, test_df = scenario.build(df)

                train_path = os.path.join(scenario_dir, f"train_{label}_seed{seed}.csv")
                test_path = os.path.join(scenario_dir, f"test_{label}_seed{seed}.csv")
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)

                print(f"[{scenario_name}/{label}/seed{seed}] train={train_df.shape}, test={test_df.shape}")


def _parse_args():
    p = argparse.ArgumentParser(description="预生成场景数据切分")
    p.add_argument("--csv", required=True, help="原始 CSV 路径")
    p.add_argument("--target", required=True, help="目标列名")
    p.add_argument("--scenarios", type=str, required=True,
                   help="场景配置，JSON 格式。"
                        ' {"small": [["n32", {"n_samples": 32}]], '
                        '  "imbalanced": [["prev005", {"minority_prev": 0.05}]]}')
    p.add_argument("--output-dir", default="scenario_data", help="输出根目录")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_scenario_data(
        csv_path=args.csv,
        target_col=args.target,
        scenarios=json.loads(args.scenarios),
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        n_seeds=args.n_seeds,
    )
    print("\nDone.")
