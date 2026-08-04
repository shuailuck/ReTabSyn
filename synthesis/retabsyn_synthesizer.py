"""
ReTabSyn 合成器：GReaT 预训练 + DPO (Direct Preference Optimization) 微调。

两阶段流程:
  1. GReaT 预训练: 将表格行序列化为文本，微调 LLM（可选 SMOTE 数据增强）
  2. DPO 微调: 构建偏好对（扰动后 vs 真实值），用 DPO 优化模型生成质量
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig
from be_great import GReaT

from DPOUtils import DPOPositiveTrainer, calculate_max_seq_len, suggest_synthetic_count
from DPOFeatureAndTarget import create_perturbed_dataset
from smote import generate_smote_synthetic

from synthesis.synthesizer import BaseTabularSynthesizer


class ReTabSynSynthesizer(BaseTabularSynthesizer):
    """ReTabSyn: GReaT 基座 + DPO 偏好优化。

    训练流程:
      1. SMOTE 数据增强 (可选，支持固定数量/比例/动态建议三种模式)
      2. GReaT 预训练: LLM 学习表格数据的文本表示
      3. 构建偏好对: 对真实数据施加扰动，构造 chosen(原始) vs rejected(扰动)
      4. DPO 微调: 用偏好对优化模型，使其倾向生成正确值
    """

    def __init__(
        self,
        integer_columns: list = None,
        random_state: int = 42,
        # ── GReaT 预训练参数 ──
        llm: str = "gpt2",
        great_epochs: int = 50,
        batch_size: int = 32,
        fp16: bool = True,
        efficient_finetuning: str = "",
        guided_sampling: bool = False,
        shuffle_data: bool = False,
        resume_from_checkpoint: bool = False,
        save_steps: int = 30000,
        # ── SMOTE 增强参数 ──
        n_aug: int = 0,
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
        llm : 基座 LLM，如 "gpt2", "distilgpt2"
        great_epochs : GReaT 预训练轮数
        batch_size : 训练批量大小
        fp16 : 是否启用混合精度
        efficient_finetuning : 高效微调方法，如 "lora"
        guided_sampling : 采样时是否使用引导采样 (逐特征生成)
        shuffle_data : 训练时是否打乱数据
        resume_from_checkpoint : 是否从 checkpoint 恢复训练
        save_steps : 每隔多少步保存一次 checkpoint
        n_aug : SMOTE 增强样本数。>0=固定数量, 0=不增强,
                -1~-100=训练集的 abs(n_aug) 倍,
                -114514=动态建议策略
        dpo_epochs : DPO 微调轮数
        beta : DPO beta 参数 (控制偏离参考模型的惩罚)
        lambda_reg : DPO 正则化系数 (防止 chosen 概率下降)
        perturb_target_prob : 仅扰动 target 列的概率 (vs 基于相关性扰动多列)
        split_ratio : 偏好对构建时额外列对的采样概率 p
        shuffle_columns : 偏好对构建时是否随机打乱列顺序
        tau : 相关性采样 softmax 温度参数
        min_corr_threshold : 列对相关性的最低阈值
        quantile_Q : 数值列分位数数量
        test_train_ratio : DPO 数据集中验证集比例
        minor_to_major_ratio : DPO 数据集少数类/多数类比例 (-1 = 不均衡)
        checkpoint_dir : 模型保存目录
        """
        super().__init__(integer_columns=integer_columns, random_state=random_state)
        # GReaT
        self.llm = llm
        self.great_epochs = great_epochs
        self.batch_size = batch_size
        self.fp16 = fp16
        self.efficient_finetuning = efficient_finetuning
        self.guided_sampling = guided_sampling
        self.shuffle_data = shuffle_data
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_steps = save_steps
        # SMOTE
        self.n_aug = n_aug
        # DPO
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
        # Other
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
        train = df.copy().reset_index(drop=True)

        # 1. SMOTE 数据增强
        n_aug = self._resolve_n_aug(train)
        if n_aug > 0:
            train = self._augment_with_smote(train, n_aug)

        # 2. GReaT 预训练
        tokenizer = AutoTokenizer.from_pretrained(self.llm)
        max_seq_len = calculate_max_seq_len(train, tokenizer)

        great_kwargs = {
            "llm": self.llm,
            "batch_size": self.batch_size,
            "epochs": self.great_epochs,
            "fp16": self.fp16,
        }
        if self.efficient_finetuning:
            great_kwargs["efficient_finetuning"] = self.efficient_finetuning

        self._great_model = GReaT(**great_kwargs)
        self._great_model.fit(
            train,
            resume_from_checkpoint=self.resume_from_checkpoint,
            conditional_col=cond_col,
            shuffle_data=self.shuffle_data,
        )

        # 保存 GReaT checkpoint
        ckpt_path = os.path.join(self.checkpoint_dir, "great_checkpoint")
        base_model_path = os.path.join(self.checkpoint_dir, "base_model")
        self._great_model.save(ckpt_path)
        base_model = self._great_model.model
        base_model.save_pretrained(base_model_path)

        # 3. 构建 DPO 偏好对
        dataset = create_perturbed_dataset(
            train, target_column,
            p=self.split_ratio,
            shuffle=self.shuffle_columns,
            tau=self.tau,
            min_corr_threshold=self.min_corr_threshold,
            Q=self.quantile_Q,
            perturb_target_prob=self.perturb_target_prob,
        )

        # 类别均衡
        if self.minor_to_major_ratio != -1:
            dataset = self._balance_dataset(dataset, train, target_column)

        train_dataset, eval_dataset = dataset.train_test_split(
            test_size=self.test_train_ratio
        ).values()

        # 4. DPO 微调
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

        # 将 DPO 微调后的模型注入 GReaT 用于采样
        self._great_model.model = model
        self._great_model._update_column_information(train)
        self._great_model._update_conditional_information(train, cond_col)
        self._max_seq_len = max_seq_len

    # -------------------------------------------------------------------
    # SMOTE n_aug 解析 (三种模式: 固定数量 / 比例 / 动态建议)
    # -------------------------------------------------------------------

    def _resolve_n_aug(self, df: pd.DataFrame) -> int:
        if self.n_aug == -114514:
            n = suggest_synthetic_count(df)
            print(f"[ReTabSyn] 动态建议 SMOTE 增强数: {n}")
            return n
        elif self.n_aug < 0:
            ratio = abs(self.n_aug)
            n = int(len(df) * ratio)
            print(f"[ReTabSyn] SMOTE 增强: {n} ({ratio}x 训练集)")
            return n
        return self.n_aug

    # -------------------------------------------------------------------
    # SMOTE 增强
    # -------------------------------------------------------------------

    def _augment_with_smote(self, df: pd.DataFrame, n_aug: int) -> pd.DataFrame:
        integer_cols = [
            c for c in df.columns
            if df[c].dtype in ("int64", "int32", "int16", "int8")
        ]
        synth = generate_smote_synthetic(
            df,
            n_samples=n_aug,
            integer_columns=integer_cols,
            random_state=self.random_state,
        )
        return pd.concat([df, synth], ignore_index=True)

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
