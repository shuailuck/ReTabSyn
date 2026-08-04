"""
P-TA 合成器：基于 PPO 增强 LLM 的表格数据合成。

Paper: P-TA: Using Proximal Policy Optimization to Enhance Tabular Data
       Augmentation via Large Language Models (ACL 2024 Findings)

流程: GReaT 微调 → 分类器(判别器)训练 → GAN(PPO式)优化 → 采样生成
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AdamW,
)
from datasets import Dataset as HFDataset, DatasetDict
from sklearn.model_selection import train_test_split

from synthesis.synthesizer import BaseTabularSynthesizer


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _format_row(row: pd.Series) -> str:
    """将表格行序列化为文本: 'col1 is val1, col2 is val2, ...'"""
    return ", ".join([f"{col} is {row[col]}" for col in row.index])


def _parse_text(text: str) -> dict:
    """从序列化文本解析回 dict: {'col1': 'val1', ...}"""
    result = {}
    for part in text.split(", "):
        if " is " in part:
            key, value = part.split(" is ", 1)
            result[key] = value
    return result


def _remove_random_values(input_text: str, num_remove: int = 2) -> tuple[str, dict[int, str]]:
    """随机遮盖 N 个特征值，返回 (corrupted_text, {position: column_name})"""
    tokens = input_text.split(", ")
    candidates = [i for i in range(len(tokens)) if " is " in tokens[i]]

    if len(candidates) < num_remove:
        num_remove = len(candidates)

    remove_indices = random.sample(candidates, num_remove)

    new_tokens = []
    missing_slots = {}

    for i, token in enumerate(tokens):
        if i in remove_indices:
            col_name = token.split(" is ")[0]
            new_tokens.append(f"{col_name} is")
            missing_slots[i] = col_name
        else:
            new_tokens.append(token)

    return ", ".join(new_tokens), missing_slots


# ---------------------------------------------------------------------------
# 分类器 (判别器)
# ---------------------------------------------------------------------------

class _TextClassifier(nn.Module):
    """二分类判别器：区分真实 vs 合成样本。"""

    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits


class _TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label),
        }


# ---------------------------------------------------------------------------
# GReaT Trainer
# ---------------------------------------------------------------------------

class _GreatTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        outputs = model(**inputs, labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


# ---------------------------------------------------------------------------
# PTA 合成器
# ---------------------------------------------------------------------------

class PTASynthesizer(BaseTabularSynthesizer):
    """P-TA 合成器：GReaT 基座 + PPO 式 GAN 优化。

    训练流程:
      1. GReaT: 将表格行序列化为文本，微调 LLM
      2. 分类器: 训练二分类判别器区分真实/合成样本
      3. GAN: 用判别器反馈作为奖励信号，PPO 式优化生成器
    """

    def __init__(
        self,
        integer_columns: list = None,
        random_state: int = 42,
        llm_model: str = "gpt2",
        great_epochs: int = 3,
        gan_epochs: int = 3,
        classifier_epochs: int = 3,
        batch_size: int = 8,
        num_remove: int = 2,
        max_length: int = 128,
        device: str | None = None,
    ):
        """
        Parameters
        ----------
        llm_model : 基座 LLM，如 "gpt2", "distilgpt2"
        great_epochs : GReaT 微调轮数
        gan_epochs : GAN PPO 优化轮数
        classifier_epochs : 判别器训练轮数
        batch_size : 训练批量大小
        num_remove : 每次遮盖的特征数（用于数据增强和采样）
        max_length : tokenizer 最大长度
        device : "cuda", "cpu" 或 None (自动检测)
        """
        super().__init__(integer_columns=integer_columns, random_state=random_state)
        self.llm_model = llm_model
        self.great_epochs = great_epochs
        self.gan_epochs = gan_epochs
        self.classifier_epochs = classifier_epochs
        self.batch_size = batch_size
        self.num_remove = num_remove
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._columns = []

    # -------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame):
        self._columns = [c for c in df.columns]
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        df = df.copy()
        df["formatted_text"] = df.apply(_format_row, axis=1)
        self._train_df = df
        self._all_texts = df["formatted_text"].tolist()

        self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 1. GReaT 微调
        self._train_great(df)
        # 2. 训练判别器
        self._train_classifier(df)
        # 3. GAN PPO 优化
        self._train_gan(df)

    # -------------------------------------------------------------------
    # GReaT 微调
    # -------------------------------------------------------------------

    def _train_great(self, df: pd.DataFrame):
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=self.random_state)

        train_texts = train_df["formatted_text"].tolist()
        val_texts = val_df["formatted_text"].tolist()

        def tokenize(examples):
            return self._tokenizer(examples["text"], padding="max_length",
                                   truncation=True, max_length=self.max_length)

        train_dataset = HFDataset.from_dict({"text": train_texts})
        val_dataset = HFDataset.from_dict({"text": val_texts})

        tokenized = DatasetDict({
            "train": train_dataset.map(tokenize, batched=True),
            "validation": val_dataset.map(tokenize, batched=True),
        })

        self._model = AutoModelForCausalLM.from_pretrained(self.llm_model).to(self.device)

        training_args = TrainingArguments(
            output_dir="./pta_checkpoints",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.great_epochs,
            weight_decay=0.01,
            save_total_limit=2,
            logging_dir="./pta_logs",
            logging_steps=50,
            report_to="none",
        )

        trainer = _GreatTrainer(
            model=self._model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=self._tokenizer,
        )
        trainer.can_return_loss = True
        trainer.train()

    # -------------------------------------------------------------------
    # 判别器训练
    # -------------------------------------------------------------------

    def _train_classifier(self, df: pd.DataFrame):
        generated_texts = self._generate_texts(len(df))

        real_texts = df["formatted_text"].tolist()
        texts = real_texts + generated_texts
        labels = [1] * len(real_texts) + [0] * len(generated_texts)

        dataset = _TextDataset(texts, labels, self._tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        self._classifier = _TextClassifier(model_name=self.llm_model).to(self.device)
        optimizer = AdamW(self._classifier.parameters(), lr=5e-5)
        loss_fn = nn.CrossEntropyLoss()

        self._classifier.train()
        for epoch in range(self.classifier_epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                lbls = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self._classifier(input_ids, attention_mask)
                loss = loss_fn(outputs, lbls)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    # -------------------------------------------------------------------
    # GAN PPO 优化
    # -------------------------------------------------------------------

    def _train_gan(self, df: pd.DataFrame):
        self._model.train()
        self._classifier.eval()

        for epoch in range(self.gan_epochs):
            total_loss = 0
            for _, row in df.iterrows():
                input_text = row["formatted_text"]
                corrupted_text, missing_slots = _remove_random_values(input_text, self.num_remove)
                filled_text = self._fill_missing_values(corrupted_text, missing_slots)
                pred = self._classify_text(filled_text)

                if pred == 0:  # 被判为假 → 优化生成器
                    optimizer = AdamW(self._model.parameters(), lr=5e-5)
                    optimizer.zero_grad()

                    input_ids = self._tokenizer(
                        corrupted_text, return_tensors="pt"
                    ).input_ids.to(self.device)
                    labels = input_ids.clone().to(self.device)

                    outputs = self._model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

    # -------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------

    def _fill_missing_values(self, corrupted_text: str, missing_slots: dict[int, str]) -> str:
        """用 LLM 填充遮盖的特征值。"""
        tokens = corrupted_text.split(", ")
        new_tokens = tokens[:]

        for idx, col_name in missing_slots.items():
            prompt = f"{col_name} is"
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

            with torch.no_grad():
                output = self._model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 5,
                    pad_token_id=self._tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                )

            generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
            generated_value = generated_text.replace(f"{col_name} is", "").strip().split(",")[0]
            new_tokens[idx] = f"{col_name} is {generated_value}"

        return ", ".join(new_tokens)

    def _classify_text(self, text: str) -> int:
        """判别器预测：1=真实, 0=合成"""
        encoding = self._tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self._classifier(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            return torch.argmax(probs, dim=-1).item()

    def _generate_texts(self, n: int) -> list[str]:
        """生成 n 条合成文本（遮盖-填充真实样本）。"""
        source_texts = self._all_texts
        if len(source_texts) < n:
            source_texts = source_texts * (n // len(source_texts) + 1)
        selected = random.sample(source_texts, n)

        generated = []
        for text in selected:
            corrupted, missing = _remove_random_values(text, self.num_remove)
            filled = self._fill_missing_values(corrupted, missing)
            generated.append(filled)
        return generated

    # -------------------------------------------------------------------
    # Sample
    # -------------------------------------------------------------------

    def _sample(self, n_samples: int) -> pd.DataFrame:
        torch.manual_seed(self.random_state + 1)
        np.random.seed(self.random_state + 1)
        random.seed(self.random_state + 1)

        texts = self._generate_texts(n_samples)
        rows = [_parse_text(t) for t in texts]
        return pd.DataFrame(rows, columns=self._columns)
