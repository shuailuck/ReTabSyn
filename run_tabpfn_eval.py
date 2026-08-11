"""
run_tabpfn_eval.py: TabPFN In-Context Learning 评估。

利用合成数据作为 TabPFN 的额外 context examples，增强对 test 数据的推理能力。
TabPFN 无需训练 —— 前向传播时直接使用 context 样本进行 in-context 推理。
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score


# ===========================================================================
# TabPFN ICL 核心评估
# ===========================================================================

def _evaluate_tabpfn_icl(
    *,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame | None,
    target_col: str,
    seed: int = 42,
    metric: str = "auroc",
    max_context: int = 2000,
    weight_synth: float = 0.5,
    filter_mode: str = "none",
    max_synth_ratio: float = 3.0,
) -> float:
    """
    TabPFN In-Context Learning 评估。

    Parameters
    ----------
    weight_synth : 合成样本在 context 中的比例 [0, 1]，filter_mode="none" 时使用
    filter_mode : "none" 全部混入 | "density+ratio" 密度过滤 + 配比控制
    max_synth_ratio : filter_mode="density+ratio" 时 synth:real 最大比 (default 3.0)
    """
    from tabpfn import TabPFNClassifier

    np.random.seed(seed)

    feat_cols = [c for c in real_test.columns if c != target_col]
    if not feat_cols:
        raise ValueError("没有可用的特征列")

    X_train, X_test, X_synth = _preprocess(real_train, real_test, synth_df, feat_cols)

    y_train = real_train[target_col]
    y_test = real_test[target_col]
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test], axis=0).astype(str))
    y_train_enc = le.transform(y_train.astype(str))
    y_test_enc = le.transform(y_test.astype(str))

    # ── #1: 密度筛选 — Isolation Forest 过滤合成数据中的离群点 ──
    if filter_mode == "density+ratio" and X_synth is not None and len(X_synth) > 0:
        X_synth, y_synth_raw = _filter_by_isolation_forest(
            X_train, X_synth, synth_df, target_col, le, seed
        )
    else:
        y_synth_raw = synth_df[target_col] if synth_df is not None and target_col in (synth_df.columns if synth_df is not None else []) else None

    # ── 构建 context ──
    if filter_mode == "density+ratio":
        # #3: 配比控制 — synth:real 不超过 max_synth_ratio
        n_real_use = min(len(X_train), max_context // 2)
        n_synth_max = min(int(n_real_use * max_synth_ratio), max_context - n_real_use)
        n_synth_use = min(len(X_synth), n_synth_max) if X_synth is not None else 0
    else:
        # 原有逻辑：一股脑混入
        n_context = min(max_context, len(X_train) + (len(X_synth) if X_synth is not None else 0))
        n_synth_use = int(n_context * weight_synth) if X_synth is not None else 0
        n_synth_use = min(n_synth_use, len(X_synth)) if X_synth is not None else 0
        n_real_use = n_context - n_synth_use

    context_X, context_y = [], []
    if n_real_use > 0:
        idx = np.random.choice(len(X_train), size=min(n_real_use, len(X_train)), replace=False)
        context_X.append(X_train[idx])
        context_y.append(y_train_enc[idx])
    if n_synth_use > 0 and X_synth is not None:
        idx = np.random.choice(len(X_synth), size=n_synth_use, replace=False)
        context_X.append(X_synth[idx])
        if y_synth_raw is not None:
            context_y.append(le.transform(y_synth_raw[idx].astype(str)))
        else:
            # fallback: use train labels
            context_y.append(y_train_enc[:n_synth_use])

    X_context = np.vstack(context_X) if context_X else X_train
    y_context = np.concatenate(context_y) if context_y else y_train_enc
    print(f"  TabPFN context: {len(y_context)} samples (real={n_real_use}, synth={n_synth_use}, mode={filter_mode})")

    clf = TabPFNClassifier(random_state=seed, n_estimators=4)
    clf.fit(X_context, y_context)
    y_prob = clf.predict_proba(X_test)

    n_classes = len(np.unique(y_test_enc))
    if n_classes == 2:
        return float(
            average_precision_score(y_test_enc, y_prob[:, 1])
            if metric == "prauc"
            else roc_auc_score(y_test_enc, y_prob[:, 1])
        )
    return float(roc_auc_score(y_test_enc, y_prob, multi_class="ovr", average="macro"))


# ---------------------------------------------------------------------------
# Isolation Forest 密度筛选
# ---------------------------------------------------------------------------

def _filter_by_isolation_forest(
    X_train: np.ndarray,
    X_synth: np.ndarray,
    synth_df: pd.DataFrame,
    target_col: str,
    label_encoder: LabelEncoder,
    seed: int,
) -> tuple[np.ndarray, pd.Series]:
    """用 Isolation Forest 在真实数据上拟合，剔除合成数据中的离群点。

    Returns
    -------
    (filtered_X_synth, filtered_y_synth)
    """
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(random_state=seed, contamination=0.1)
    iso.fit(X_train)
    # 预测合成样本：1=inlier, -1=outlier
    preds = iso.predict(X_synth)
    mask = preds == 1
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"  [IF] 过滤掉 {n_removed}/{len(X_synth)} 条合成样本")
    y_synth = synth_df[target_col]
    return X_synth[mask], y_synth[mask].reset_index(drop=True)


def _evaluate_all_modes(
    *,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame | None,
    target_col: str,
    seed: int = 42,
    metric: str = "auroc",
    max_context: int = 2000,
) -> dict[str, float]:
    """对比多种 context 策略。

    real_only : 仅真实数据
    synth_only : 全部合成数据（一股脑）
    real_plus_synth : 真实+合成混入（一股脑）
    synth_filtered : 合成数据经 IF 过滤 + 配比控制
    real_plus_synth_filtered : 真实 + IF 过滤合成 + 配比控制
    """
    results = {}
    results["real_only"] = _evaluate_tabpfn_icl(
        real_train=real_train, real_test=real_test, synth_df=None,
        target_col=target_col, seed=seed, metric=metric, max_context=max_context,
    )
    if synth_df is not None and len(synth_df) >= 10:
        results["synth_only"] = _evaluate_tabpfn_icl(
            real_train=real_train, real_test=real_test, synth_df=synth_df,
            target_col=target_col, seed=seed, metric=metric, max_context=max_context,
            weight_synth=1.0, filter_mode="none",
        )
        results["real_plus_synth"] = _evaluate_tabpfn_icl(
            real_train=real_train, real_test=real_test, synth_df=synth_df,
            target_col=target_col, seed=seed, metric=metric, max_context=max_context,
            weight_synth=0.5, filter_mode="none",
        )
        # ── 新策略: IF 密度筛选 + 配比控制 ──
        results["synth_filtered"] = _evaluate_tabpfn_icl(
            real_train=real_train, real_test=real_test, synth_df=synth_df,
            target_col=target_col, seed=seed, metric=metric, max_context=max_context,
            filter_mode="density+ratio",
        )
        results["real_plus_synth_filtered"] = _evaluate_tabpfn_icl(
            real_train=real_train, real_test=real_test, synth_df=synth_df,
            target_col=target_col, seed=seed, metric=metric, max_context=max_context,
            weight_synth=0.5, filter_mode="density+ratio",
        )
    return results


# ===========================================================================
# 预处理
# ===========================================================================

def _preprocess(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame | None,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    X_train = real_train[feat_cols].copy()
    X_test = real_test[feat_cols].copy()
    X_synth = synth_df[feat_cols].copy() if synth_df is not None else None

    for col in feat_cols:
        if X_train[col].dtype == object or X_train[col].dtype.name == "category":
            le = LabelEncoder()
            all_vals = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
            if X_synth is not None:
                all_vals = pd.concat([all_vals, X_synth[col].astype(str)], axis=0)
            le.fit(all_vals)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            if X_synth is not None:
                X_synth[col] = le.transform(X_synth[col].astype(str))
        else:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
            if X_synth is not None:
                X_synth[col] = X_synth[col].fillna(median_val)

    return X_train.values, X_test.values, X_synth.values if X_synth is not None else None


# ===========================================================================
# 批量评估入口
# ===========================================================================

def run(
    *,
    scenario_dir: str,
    synth_dir: str,
    synth_algos: list[str],
    scenario_labels: list[str],
    target_col: str,
    base_seed: int = 42,
    n_seeds: int = 10,
    metric: str = "auroc",
    output_csv: str | None = None,
):
    rows = []
    for label in scenario_labels:
        for algo in synth_algos:
            all_scores = defaultdict(list)

            for i in range(n_seeds):
                seed = base_seed + i
                train_path = os.path.join(scenario_dir, f"train_{label}_seed{seed}.csv")
                test_path = os.path.join(scenario_dir, f"test_{label}_seed{seed}.csv")
                if not os.path.exists(train_path):
                    print(f"  SKIP {label}/seed{seed}: train file not found")
                    continue
                real_train = pd.read_csv(train_path)
                real_test = pd.read_csv(test_path)

                synth_path = os.path.join(synth_dir, f"{algo}_{label}_seed{seed}.csv")
                synth_df = pd.read_csv(synth_path) if os.path.exists(synth_path) else None

                scores = _evaluate_all_modes(
                    real_train=real_train, real_test=real_test, synth_df=synth_df,
                    target_col=target_col, seed=seed, metric=metric,
                )
                for mode, score in scores.items():
                    if not np.isnan(score):
                        all_scores[mode].append(score)

            print(f"\n{'=' * 50}")
            print(f"  {label} / {algo}")
            print(f"  {'Mode':<18} {'Mean':>8} {'SE':>8}")
            print(f"  {'-' * 36}")
            for mode in ["real_only", "synth_only", "real_plus_synth",
                          "synth_filtered", "real_plus_synth_filtered"]:
                vals = all_scores.get(mode, [])
                if vals:
                    arr = np.array(vals)
                    mean, se = float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))
                    print(f"  {mode:<18} {mean:>8.4f} {se:>8.4f}")
                    rows.append({
                        "label": label, "algo": algo, "mode": mode,
                        "metric": metric, "mean": round(mean, 4), "se": round(se, 4),
                    })

    df = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved to: {output_csv}")
    return df


def _parse_args():
    p = argparse.ArgumentParser(description="TabPFN ICL 评估")
    p.add_argument("--scenario-dir", required=True, help="场景数据目录 (如 scenario_data/small)")
    p.add_argument("--synth-dir", required=True, help="合成数据目录 (如 synth_data)")
    p.add_argument("--synth-algos", type=str, required=True,
                   help="合成算法列表，逗号分隔 (如 smote,tvae)")
    p.add_argument("--labels", type=str, required=True,
                   help="场景标签列表，逗号分隔 (如 n32,n64)")
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
        target_col=args.target,
        base_seed=args.base_seed,
        n_seeds=args.n_seeds,
        metric=args.metric,
        output_csv=args.output,
    )
