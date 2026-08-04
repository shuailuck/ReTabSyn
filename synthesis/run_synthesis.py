"""
统一的数据合成命令行入口脚本
Usage:
    python run_synthesis.py --algo great --data_path data/real.csv --out_dir synth_data/ --n_samples 1000
"""
import os
import sys
import argparse
import importlib
import pandas as pd


def get_synthesizer_class(algo_name: str):
    """
    根据算法名称动态加载模块与类
    例如: algo_name="great" -> 导入 great_synthesizer 模块中的 GreatSynthesizer 类
    """
    module_name = f"{algo_name.lower()}_synthesizer"
    
    # 转换为驼峰命名法：great -> GreatSynthesizer, tab_syn -> TabSynSynthesizer
    class_name = "".join([part.capitalize() for part in algo_name.split("_")]) + "Synthesizer"
    
    try:
        module = importlib.import_module(module_name)
        synthesizer_cls = getattr(module, class_name)
        return synthesizer_cls
    except (ImportError, AttributeError) as e:
        print(f"[Error] 无法加载算法 '{algo_name}' (尝试加载模块 '{module_name}', 类 '{class_name}'): {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Tabular Data Synthesis Runner")
    parser.add_argument("--algo", type=str, required=True, 
                        help="合成算法名称，例如: great, tvae, tabsyn, pta, synrl, retabsyn")
    parser.add_argument("--data_path", type=str, required=True, help="真实数据路径 (.csv)")
    parser.add_argument("--out_dir", type=str, default="synth_outputs", help="生成数据的保存目录")
    parser.add_argument("--n_samples", type=int, default=None, help="合成样本数量（默认与真实训练集等量）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--integer_cols", type=str, default=None, help="逗号分割的整数列名（如 Age,Count）")

    args = parser.parse_args()

    # 1. 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 2. 读取数据
    print(f"\n[1/3] 读取真实数据: {args.data_path}")
    df_real = pd.read_csv(args.data_path)

    # 解析 integer_cols 参数
    integer_columns = args.integer_cols.split(",") if args.integer_cols else None

    # 3. 动态实例化Synthesizer
    print(f"[2/3] 初始化并拟合算法: {args.algo.upper()} (Seed: {args.seed})")
    SynthesizerCls = get_synthesizer_class(args.algo)
    synthesizer = SynthesizerCls(integer_columns=integer_columns, random_state=args.seed)

    # 训练模型
    synthesizer.fit(df_real)

    # 4. 采样与后处理
    n_samples = args.n_samples if args.n_samples is not None else len(df_real)
    print(f"[3/3] 生成 {n_samples} 条合成数据...")
    synth_df = synthesizer.sample(n_samples=n_samples)

    # 5. 保存结果
    out_filename = f"{args.algo}_synth_seed{args.seed}.csv"
    out_path = os.path.join(args.out_dir, out_filename)
    synth_df.to_csv(out_path, index=False)
    print(f"✅ 生成完毕！合成数据已保存至: {out_path}\n")


if __name__ == "__main__":
    main()