"""Critic 推理脚本 — 对一批游戏决策点计算 V(s)。

加载训练好的 Critic 模型，对 JSONL 轨迹文件中的每个决策点
计算价值估计 V(s)，输出 JSON 文件。

输入: JSONL 轨迹文件（与游戏导出格式相同）
输出: JSON 文件，每个决策附带 V(s) 值

用法:
    python -m training.rl.critic.infer \
        --model_path training/self_play/critic \
        --input_jsonl data/trajectories.jsonl \
        --output_json data/values.json \
        --batch_size 8
"""

import argparse
import json
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from training.rl.critic.model import CriticModel, load_tokenizer


class DecisionDataset(Dataset):
    """从 JSONL 轨迹中提取的决策点数据集，用于 Critic 推理。"""

    def __init__(
        self,
        decisions: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 8192,
    ):
        self.decisions = decisions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.decisions)

    def __getitem__(self, idx):
        decision = self.decisions[idx]
        messages = decision["messages"]
        tools = decision.get("tools") or None

        # 用 chat template 转换为文本 (含 tool 定义)
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False
        )

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
            "index": idx,
        }


def collate_fn(batch, pad_token_id: int):
    """动态 padding。"""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    indices = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        indices.append(item["index"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "indices": indices,
    }


def extract_decisions_from_trajectories(
    input_jsonl: str,
) -> List[Dict[str, Any]]:
    """从 JSONL 轨迹文件提取所有决策点。

    返回列表，每个元素包含:
    - game_id: 游戏 ID
    - player_seat: 玩家座位号
    - seq_num: 决策序号
    - action_type: 决策类型
    - messages: chat messages (用于 tokenize)
    """
    decisions = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectory = json.loads(line)
            game_id = trajectory.get("game_id", "")

            for decision in trajectory.get("decisions", []):
                llm_input = decision.get("llm_input", {})
                messages = llm_input.get("messages", [])
                if not messages:
                    continue
                tools = llm_input.get("tools", [])

                decisions.append({
                    "game_id": game_id,
                    "player_seat": decision.get("player_seat", 0),
                    "seq_num": decision.get("seq_num", 0),
                    "action_type": decision.get("action_type", ""),
                    "round_num": decision.get("round_num", 1),
                    "messages": messages,
                    "tools": tools,
                })

    return decisions


def infer(args):
    """对所有决策点进行 Critic 推理。"""
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = load_tokenizer(args.model_path)

    print(f"Loading critic model from {args.model_path}...")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = CriticModel.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if args.device == "auto" else None,
    )

    if args.device != "auto":
        device = torch.device(args.device)
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    model.eval()

    # 提取决策点
    print(f"Extracting decisions from {args.input_jsonl}...")
    decisions = extract_decisions_from_trajectories(args.input_jsonl)
    print(f"Found {len(decisions)} decision points")

    if not decisions:
        print("No decisions found. Exiting.")
        return

    # 创建数据集和 dataloader
    dataset = DecisionDataset(decisions, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 推理
    print("Running critic inference...")
    values = [None] * len(decisions)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Critic inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices = batch["indices"]

            predicted_values = model(input_ids, attention_mask)

            for i, idx in enumerate(indices):
                values[idx] = predicted_values[i].item()

    # 组装输出：按 (game_id, player_seat) 分组的 V(s) 值
    results = []
    for i, decision in enumerate(decisions):
        results.append({
            "game_id": decision["game_id"],
            "player_seat": decision["player_seat"],
            "seq_num": decision["seq_num"],
            "action_type": decision["action_type"],
            "round_num": decision["round_num"],
            "value": values[i],
        })

    # 保存结果
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} value predictions to {args.output_json}")

    # 打印统计
    valid_values = [v for v in values if v is not None]
    if valid_values:
        import statistics
        print(f"\nValue statistics:")
        print(f"  Count: {len(valid_values)}")
        print(f"  Mean:  {statistics.mean(valid_values):.4f}")
        print(f"  Std:   {statistics.stdev(valid_values) if len(valid_values) > 1 else 0:.4f}")
        print(f"  Min:   {min(valid_values):.4f}")
        print(f"  Max:   {max(valid_values):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Critic 推理 — 计算每个决策点的 V(s)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的 Critic 模型路径",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="输入的 JSONL 轨迹文件",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="输出 JSON 文件路径（含 V(s) 值）",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="推理 batch size")
    parser.add_argument("--max_length", type=int, default=8192, help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--device", type=str, default="auto", help="设备 (auto/cuda/cpu)"
    )
    parser.add_argument("--bf16", action="store_true", help="使用 bfloat16")

    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()
