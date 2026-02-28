"""SFT 微调 — 在 NLRL 合成数据上微调目标模型。

输入: SFT parquet（training/nlrl/synthesize.py 输出）
输出: HuggingFace 格式 checkpoint（可直接用于 self_play.sh base_model）

训练细节:
  - 使用 trl SFTTrainer
  - 仅对 response 部分计算 loss（prompt token 全部 mask 为 -100）
  - 通过 apply_chat_template 构建完整对话文本
  - 支持 bf16 和梯度累积

用法:
    python -m training.nlrl.sft_train \\
        --train_parquet data/round_1/sft_data/train.parquet \\
        --val_parquet data/round_1/sft_data/test.parquet \\
        --model_path /path/to/base_model \\
        --output_dir experiments/nlrl_v1/checkpoints/round_1 \\
        --epochs 2 --lr 2e-5 --batch_size 2 --grad_accum 8 \\
        --max_seq_length 16384 --bf16
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig


def load_parquet_as_dataset(path: str) -> Dataset:
    """加载 parquet 为 HuggingFace Dataset。"""
    df = pd.read_parquet(path)
    return Dataset.from_pandas(df)


def build_formatting_func(tokenizer, max_seq_length: int):
    """构建 SFT 格式化函数：将 (prompt, response) 转换为完整对话文本。

    使用 apply_chat_template 处理 tools、tool_calls 等格式，
    返回 input_ids + labels（response 部分不 mask）。
    """

    def format_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt_data = json.loads(sample["prompt"])
            response_data = json.loads(sample["response"])
        except (json.JSONDecodeError, TypeError, KeyError):
            return {"input_ids": [], "attention_mask": [], "labels": []}

        messages: List[Dict] = prompt_data.get("messages", [])
        tools: Optional[List[Dict]] = prompt_data.get("tools") or None

        if not messages:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        # 构建 prompt 文本（含 generation prompt，即到 <|im_start|>assistant\n）
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        # 构建完整文本（prompt + response）
        try:
            full_messages = messages + [response_data]
            full_text = tokenizer.apply_chat_template(
                full_messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        # 验证 full_text 以 prompt_text 开头
        if not full_text.startswith(prompt_text):
            # Chat template 不一致，跳过此样本
            return {"input_ids": [], "attention_mask": [], "labels": []}

        # Tokenize prompt（用于确定 mask 边界）
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]

        # Tokenize 完整文本
        full_enc = tokenizer(
            full_text,
            add_special_tokens=False,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        full_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        # Labels: prompt 部分 mask 为 -100，response 部分保留
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        assert len(labels) == len(full_ids), "labels length mismatch"

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return format_sample


class DataCollatorForSFT:
    """带动态 padding 的 SFT data collator。"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 过滤空样本
        features = [f for f in features if f.get("input_ids")]
        if not features:
            # 返回一个占位 batch（不应发生）
            return {
                "input_ids": torch.zeros(1, 1, dtype=torch.long),
                "attention_mask": torch.zeros(1, 1, dtype=torch.long),
                "labels": torch.full((1, 1), -100, dtype=torch.long),
            }

        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_masks, labels_list = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_masks.append(f["attention_mask"] + [0] * pad_len)
            labels_list.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


def train(args):
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {args.model_path}...")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # 加载数据
    print(f"Loading train data from {args.train_parquet}...")
    train_dataset = load_parquet_as_dataset(args.train_parquet)
    print(f"  Train: {len(train_dataset)} rows")

    eval_dataset = None
    if args.val_parquet and os.path.exists(args.val_parquet):
        eval_dataset = load_parquet_as_dataset(args.val_parquet)
        print(f"  Val:   {len(eval_dataset)} rows")

    # 预处理：tokenize + label mask
    format_fn = build_formatting_func(tokenizer, args.max_seq_length)
    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(
        format_fn,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
        num_proc=4,
    )
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"  After filtering: {len(train_dataset)} train samples")

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            format_fn,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval",
            num_proc=4,
        )
        eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    collator = DataCollatorForSFT(pad_token_id=tokenizer.pad_token_id)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        fp16=False,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=eval_dataset is not None,
        report_to=["wandb"] if not args.no_wandb else [],
        run_name=os.path.basename(args.output_dir),
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print(f"\nStarting SFT training:")
    print(f"  Model:        {args.model_path}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Train samples:{len(train_dataset)}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  LR:           {args.lr}")
    print(f"  Batch size:   {args.batch_size} × {args.grad_accum} (grad_accum)")
    print(f"  Max seq len:  {args.max_seq_length}")
    print()

    trainer.train()

    # 保存 HuggingFace 格式
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="SFT 微调（NLRL 合成数据）")
    parser.add_argument("--train_parquet", type=str, required=True)
    parser.add_argument("--val_parquet", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_wandb", action="store_true", help="禁用 wandb")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
