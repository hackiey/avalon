"""Critic 模型定义 — 基于 base LLM + 线性 value head。

输入: tokenized prompt (游戏状态描述)
输出: scalar V(s)，估计该状态下玩家的预期回报

基于同一个 base model (如 Qwen2.5-7B-Instruct)，
使用 AutoModelForSequenceClassification (num_labels=1) 实现。
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from typing import Optional


class CriticModel(nn.Module):
    """Critic (Value Function) 模型。

    在 base LLM 上加一个线性 value head，输出 scalar V(s)。
    使用 HuggingFace AutoModelForSequenceClassification 后端，num_labels=1。
    """

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = 1
        if config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id

        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
            **kwargs,
        )

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self._init_value_head()

    def _init_value_head(self):
        """将 value head 权重初始化为接近 0 的小值。

        这样冷启动时 V(s) ≈ 0，GAE 退化为纯 discounted reward，
        是合理的初始行为。
        """
        # 不同模型架构的 head 名称可能不同
        for name, param in self.model.named_parameters():
            if "score" in name or "classifier" in name:
                if param.dim() >= 2:
                    nn.init.normal_(param, mean=0.0, std=0.01)
                else:
                    nn.init.zeros_(param)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播，输出 V(s) 标量。

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            values: (batch_size,) — 每个输入的 V(s) 预测值
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # logits shape: (batch_size, 1)
        values = outputs.logits.squeeze(-1)
        return values

    def save_pretrained(self, save_directory: str):
        """保存模型到目录。"""
        self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        gradient_checkpointing: bool = False,
    ) -> "CriticModel":
        """从已训练的 checkpoint 加载。"""
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.model_name_or_path = model_path

        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 1
        if config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id

        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map

        instance.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            **kwargs,
        )

        if gradient_checkpointing:
            instance.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        return instance


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """加载 tokenizer 并确保 pad_token 设置正确。"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
