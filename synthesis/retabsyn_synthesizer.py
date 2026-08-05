"""
ReTabSyn 合成器：基于预训练 GReaT checkpoint 的 DPO 微调。

流程:
  1. 加载预训练 GReaT checkpoint（由 GreatSynthesizer 产出）
  2. 构建偏好对: 对真实数据施加扰动，构造 chosen(原始) vs rejected(扰动)
  3. DPO 微调: 用偏好对优化模型，使其倾向生成正确值
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig
from be_great import GReaT

from DPOUtils import DPOPositiveTrainer, calculate_max_seq_len
from DPOFeatureAndTarget import create_perturbed_dataset

from synthesis.synthesizer import BaseTabularSynthesizer


class ReTabSynSynthesizer(BaseTabularSynthesizer):
    """ReTabSyn: 加载 GReaT checkpoint → DPO 偏好优化。"""

    def __init__(
        self,
        integer_columns: list = None,
        random_state: int = 42,
        # ── GReaT checkpoint 路径 ──
        great_checkpoint_path: str = "",
        llm: str = "gpt2",
        guided_sampling: bool = False,
        # ── DPO / 偏好对构建参数 ──
        dpo_epochs: int = 3,
        beta: float = 0.1,
        lambda_reg: float = 0.1,
        perturb_target_prob: float = 0.5,
        split_ratio: float = 1.0,
        shuffle_columns: bool = False,
        tau: float = 1.0,
        min_corr_threshold: float = 0.1,
        quantile_Q: int = 10,
        test_train_ratio: float = 0.2,
        minor_to_major_ratio: float = -1,
        # ── 其他 ──
        device: str | None = None,
        checkpoint_dir: str = "./retabsyn_checkpoints",
    ):
        """
        Parameters
        ----------
        great_checkpoint_path : GReaT checkpoint 目录路径（必填），
                               base model 自动从 {path}/base_model 加载
        llm : 基座 LLM 名称（需与 GReaT checkpoint 一致）
        guided_sampling : 采样时是否使用引导采样
        dpo_epochs : DPO 微调轮数
        beta : DPO beta 参数
        lambda_reg : DPO 正则化系数 (防止 chosen 概率下降)
        perturb_target_prob : 仅扰动 target 列的概率
        split_ratio : 额外列对采样概率 p
        shuffle_columns : 偏好对构建时打乱列顺序
        tau : 相关性采样 softmax 温度
        min_corr_threshold : 列对相关性最低阈值
        quantile_Q : 数值列分位数数量
        test_train_ratio : DPO 验证集比例
        minor_to_major_ratio : 少数/多数类比例 (-1 = 不均衡)
        checkpoint_dir : DPO 模型输出目录
        """
        super().__init__(integer_columns=integer_columns, random_state=random_state)
        self.great_checkpoint_path = great_checkpoint_path
        self.llm = llm
        self.guided_sampling = guided_sampling
        self.dpo_epochs = dpo_epochs
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.perturb_target_prob = perturb_target_prob
        self.split_ratio = split_ratio
        self.shuffle_columns = shuffle_columns
        self.tau = tau
        self.min_corr_threshold = min_corr_threshold
        self.quantile_Q = quantile_Q
        self.test_train_ratio = test_train_ratio
        self.minor_to_major_ratio = minor_to_major_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self._columns = []

    # -------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame):
        self._columns = [c for c in df.columns]
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        target_column = df.columns[-1]
        cond_col = df.columns[0]

        if not self.great_checkpoint_path:
            raise ValueError("great_checkpoint_path 是必填参数，请指定预训练的 GReaT checkpoint 路径")

        base_model_path = os.path.join(self.great_checkpoint_path, "base_model")

        # 1. 加载预训练 GReaT checkpoint
        tokenizer = AutoTokenizer.from_pretrained(self.llm)
        max_seq_len = calculate_max_seq_len(df, tokenizer)

        self._great_model = GReaT(llm=self.llm, max_length=max_seq_len)
        print(f"[ReTabSyn] 加载 GReaT checkpoint: {self.great_checkpoint_path}")
        self._great_model.load_from_dir(self.great_checkpoint_path)
        self._great_model.model = AutoModelForCausalLM.from_pretrained(base_model_path)
        self._great_model._update_column_information(df)
        self._great_model._update_conditional_information(df, cond_col)

        # 2. 构建 DPO 偏好对
        dataset = create_perturbed_dataset(
            df, target_column,
            p=self.split_ratio,
            shuffle=self.shuffle_columns,
            tau=self.tau,
            min_corr_threshold=self.min_corr_threshold,
            Q=self.quantile_Q,
            perturb_target_prob=self.perturb_target_prob,
        )

        if self.minor_to_major_ratio != -1:
            dataset = self._balance_dataset(dataset, df, target_column)

        train_dataset, eval_dataset = dataset.train_test_split(
            test_size=self.test_train_ratio
        ).values()

        # 3. DPO 微调
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model_ref = AutoModelForCausalLM.from_pretrained(base_model_path)

        dpo_output_dir = os.path.join(self.checkpoint_dir, "dpo_model")
        training_args = DPOConfig(
            beta=self.beta,
            output_dir=dpo_output_dir,
            num_train_epochs=self.dpo_epochs,
            logging_steps=10,
            save_steps=100,
        )

        dpo_trainer = DPOPositiveTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            lambda_reg=self.lambda_reg,
        )
        dpo_trainer.train()
        dpo_trainer.save_model()

        # 将 DPO 微调后的模型注入 GReaT
        self._great_model.model = model
        self._great_model._update_column_information(df)
        self._great_model._update_conditional_information(df, cond_col)
        self._max_seq_len = max_seq_len

    # -------------------------------------------------------------------
    # DPO 数据集类别均衡
    # -------------------------------------------------------------------

    def _balance_dataset(self, dataset, train_df, target_column):
        import random as _random
        _random.seed(self.random_state)

        target_counts = train_df[target_column].value_counts()
        majority_class = target_counts.index[0]
        target_mapping = train_df[target_column].tolist()

        majority_indices = []
        minority_indices = []
        for idx, val in enumerate(target_mapping):
            if val == majority_class:
                majority_indices.append(idx)
            else:
                minority_indices.append(idx)

        total_minority = len(minority_indices)
        desired_majority = int(total_minority / self.minor_to_major_ratio)

        if desired_majority < len(majority_indices):
            sampled_majority = _random.sample(majority_indices, desired_majority)
            balanced_indices = minority_indices + sampled_majority
            _random.shuffle(balanced_indices)
            balanced_data = [dataset[i] for i in balanced_indices]
            from datasets import Dataset as HFDataset2
            return HFDataset2.from_list(balanced_data)
        return dataset

    # -------------------------------------------------------------------
    # Sample
    # -------------------------------------------------------------------

    def _sample(self, n_samples: int) -> pd.DataFrame:
        torch.manual_seed(self.random_state + 1)
        np.random.seed(self.random_state + 1)

        synth = self._great_model.sample(
            n_samples=n_samples,
            max_length=self._max_seq_len,
            guided_sampling=self.guided_sampling,
        )
        return synth[self._columns] if set(self._columns).issubset(synth.columns) else synth
