"""Critic 训练脚本 — 用 MSE Loss 训练 V(s) 价值函数。

输入: parquet 数据（含 prompt + discounted_return G_t）
Loss: MSE(V(prompt), G_t)

G_t 的计算在 training/advantage/compute.py 中完成:
    G_T = R_final
    G_t = γ · G_{t+1}   (中间 reward = 0)

每轮 self-play 训练 critic 几个 epoch，保存到指定目录。

用法:
    python -m training.critic.train \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --data_file data/critic_train.parquet \
        --output_dir training/self_play/critic \
        --epochs 3 \
        --lr 1e-5
"""

import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from training.critic.model import CriticModel, load_tokenizer


class CriticDataset(Dataset):
    """Critic 训练数据集。

    每条样本包含:
    - prompt: chat messages (JSON string)
    - discounted_return: G_t 值 (float)
    """

    def __init__(self, data_file: str, tokenizer, max_length: int = 8192):
        df = pd.read_parquet(data_file)
        self.prompts = df["prompt"].tolist()
        self.returns = df["discounted_return"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # prompt 是 JSON 字符串，解析为 messages + tools
        # 新格式: {"messages": [...], "tools": [...]}
        # 旧格式: [...] (纯 messages 列表)
        prompt = self.prompts[idx]
        if isinstance(prompt, str):
            prompt_data = json.loads(prompt)
        else:
            prompt_data = prompt

        if isinstance(prompt_data, dict):
            messages = prompt_data["messages"]
            tools = prompt_data.get("tools") or None
        else:
            # 旧格式兼容: 直接就是 messages 列表
            messages = prompt_data
            tools = None

        # 用 chat template 转换为文本 (含 tool 定义)
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False
        )

        # tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "target_value": self.returns[idx],
        }


def collate_fn(batch, pad_token_id: int):
    """动态 padding 到 batch 内最大长度。"""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    target_values = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        target_values.append(item["target_value"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "target_values": torch.tensor(target_values, dtype=torch.float32),
    }


def train(args):
    """训练 Critic 模型。"""
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = load_tokenizer(args.model_path)

    print(f"Loading critic model from {args.model_path}...")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = CriticModel(
        model_name_or_path=args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if args.device == "auto" else None,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if args.device != "auto":
        device = torch.device(args.device)
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    print(f"Loading data from {args.data_file}...")
    dataset = CriticDataset(args.data_file, tokenizer, max_length=args.max_length)
    print(f"Dataset size: {len(dataset)}")

    grad_accum_steps = args.gradient_accumulation_steps
    micro_batch_size = args.batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_optimizer_steps = (len(dataloader) // grad_accum_steps) * args.epochs
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    loss_fn = torch.nn.MSELoss()

    effective_batch = micro_batch_size * grad_accum_steps
    print(f"\nStarting training: {args.epochs} epochs, "
          f"micro_batch={micro_batch_size}, grad_accum={grad_accum_steps}, "
          f"effective_batch={effective_batch}, "
          f"{total_optimizer_steps} optimizer steps")
    if args.gradient_checkpointing:
        print("Gradient checkpointing: enabled")

    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True
        )

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_values = batch["target_values"].to(device)

            predicted_values = model(input_ids, attention_mask).float()
            loss = loss_fn(predicted_values, target_values)
            loss = loss / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps
            num_batches += 1

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            progress_bar.set_postfix(
                loss=f"{loss.item() * grad_accum_steps:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} - avg loss: {avg_loss:.4f}")

    # 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving critic model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Critic training complete!")


def main():
    parser = argparse.ArgumentParser(description="训练 Critic (Value Function) 模型")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Base model 路径（首次训练）或已训练的 critic checkpoint 路径（继续训练）",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="训练数据 parquet 文件（含 prompt 和 discounted_return 列）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument("--epochs", type=int, default=3, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup 比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--max_length", type=int, default=8192, help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--device", type=str, default="auto", help="设备 (auto/cuda/cpu)"
    )
    parser.add_argument("--bf16", action="store_true", help="使用 bfloat16")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="启用 gradient checkpointing 降低显存占用",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="梯度累积步数（effective_batch = batch_size × 此值）",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
