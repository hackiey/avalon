"""用 Teacher LLM + 更新后的策略重新生成游戏决策，构建 SFT 训练数据。

输入:
  - annotated.jsonl（带 nlrl_annotation 的轨迹文件）
  - strategies/ 目录（每角色的策略文档 .md 文件）

输出:
  - sft_data/train.parquet + test.parquet（与 training/data/preprocess.py 格式兼容）

核心思路:
  对每个决策，将更新后的策略注入 system message，让 Teacher LLM 重新生成该决策。
  Teacher 拥有 oracle 视角，策略文档告诉它该如何扮演该角色。
  最终得到 (game_state_prompt, strategy_guided_response) 训练对。

用法:
    python -m training.nlrl.synthesize \\
        --annotated_jsonl data/round_1/annotated.jsonl \\
        --strategies_dir experiments/nlrl_v1/strategies/round_1 \\
        --output_dir data/round_1/sft_data \\
        --teacher_provider openai --teacher_model gpt-4o \\
        --concurrency 10
"""

import argparse
import asyncio
import json
import os
import random
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.asyncio import tqdm as atqdm

from game.roles import is_evil_role
from training.nlrl.strategy import load_strategy, get_initial_strategy, _make_client


# ============================================================
# Prompt 构建
# ============================================================

_STRATEGY_HEADER = "=== STRATEGY GUIDE (follow this carefully) ==="
_STRATEGY_FOOTER = "=== END OF STRATEGY GUIDE ==="


def _inject_strategy_into_messages(
    messages: List[Dict[str, Any]],
    strategy: str,
) -> List[Dict[str, Any]]:
    """将策略文档注入 system message 头部。

    如果 messages 第一条是 system，则在其内容前面加策略文档。
    否则在列表头部插入新的 system message。
    """
    strategy_block = f"{_STRATEGY_HEADER}\n{strategy}\n{_STRATEGY_FOOTER}"

    messages = list(messages)  # shallow copy
    if messages and messages[0].get("role") == "system":
        original_system = messages[0].get("content", "")
        messages[0] = {
            "role": "system",
            "content": f"{strategy_block}\n\n{original_system}",
        }
    else:
        messages.insert(0, {"role": "system", "content": strategy_block})

    return messages


def _parse_teacher_response(response) -> Optional[Dict[str, Any]]:
    """解析 teacher LLM 的原始响应为 assistant message dict。

    处理两种情况:
    1. 纯文本响应（discussion 类决策）
    2. tool_calls 响应（vote/propose/assassinate 类决策）

    Returns:
        assistant message dict，可用于 apply_chat_template
    """
    msg = response.choices[0].message

    # 提取 tool_calls（如果有）
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )

    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None)

    result: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        result["tool_calls"] = tool_calls
    if reasoning:
        result["reasoning_content"] = reasoning

    # 如果 content 和 tool_calls 都为空，说明响应无效
    if not content and not tool_calls:
        return None

    return result


async def synthesize_one(
    decision: Dict[str, Any],
    trajectory: Dict[str, Any],
    strategy: str,
    semaphore: asyncio.Semaphore,
    client,
    model: str,
    temperature: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """用 Teacher LLM 为单个决策重新生成回复。

    Returns:
        {"prompt": ..., "response": ..., "ground_truth": ...} 或 None（失败时）
    """
    async with semaphore:
        llm_input = decision.get("llm_input", {})
        original_messages = llm_input.get("messages", [])
        tools = llm_input.get("tools", [])

        if not original_messages:
            return None

        # 将策略注入 system message
        messages = _inject_strategy_into_messages(original_messages, strategy)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await client.chat.completions.create(**kwargs)
            assistant_msg = _parse_teacher_response(response)
            if assistant_msg is None:
                return None
        except Exception as e:
            print(f"  [synthesize] Failed: {e}")
            return None

        # 构建训练样本（与 training/data/preprocess.py 格式一致）
        players = {p["seat"]: p for p in trajectory.get("players", [])}
        seat = decision.get("player_seat", 0)
        player = players.get(seat, {})
        role = player.get("role", "")
        player_team = "evil" if is_evil_role(role) else "good"

        gt = {
            "game_id": trajectory.get("game_id", ""),
            "player_seat": seat,
            "player_role": role,
            "player_team": player_team,
            "game_winner": trajectory.get("winner", ""),
            "action_type": decision.get("action_type", ""),
            "seq_num": decision.get("seq_num", 0),
        }

        # prompt 用注入策略后的 messages（包含策略指导）
        prompt_data: Dict[str, Any] = {"messages": messages}
        if tools:
            prompt_data["tools"] = tools

        return {
            "data_source": "avalon_nlrl",
            "prompt": json.dumps(prompt_data, ensure_ascii=False),
            "response": json.dumps(assistant_msg, ensure_ascii=False),
            "reward_model": {
                "data_source": "avalon_nlrl",
                "ground_truth": json.dumps(gt, ensure_ascii=False),
            },
        }


async def synthesize_trajectories(
    annotated_jsonl: str,
    strategies_dir: str,
    output_dir: str,
    teacher_provider: str,
    teacher_model: str,
    concurrency: int = 10,
    train_ratio: float = 0.9,
    temperature: float = 0.7,
    seed: int = 42,
) -> None:
    """对所有轨迹中的决策重新生成回复，输出 SFT parquet。"""
    random.seed(seed)

    # 加载轨迹
    trajectories = []
    with open(annotated_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    print(f"Loaded {len(trajectories)} trajectories")

    # 加载策略（每角色）
    strategies: Dict[str, str] = {}
    all_roles = ["merlin", "loyal_servant", "assassin", "minion"]
    for role in all_roles:
        strategies[role] = load_strategy(role, strategies_dir)
    print(f"Loaded strategies for: {list(strategies.keys())}")

    client = _make_client(teacher_provider)
    semaphore = asyncio.Semaphore(concurrency)

    # 构建所有合成任务
    tasks = []
    for trajectory in trajectories:
        players = {p["seat"]: p for p in trajectory.get("players", [])}
        for decision in trajectory.get("decisions", []):
            seat = decision.get("player_seat", 0)
            role = players.get(seat, {}).get("role", "").lower()
            strategy = strategies.get(role, get_initial_strategy(role))
            tasks.append(
                synthesize_one(
                    decision, trajectory, strategy, semaphore, client, teacher_model, temperature
                )
            )

    print(f"Synthesizing {len(tasks)} decisions with concurrency={concurrency}...")
    results = await atqdm.gather(*tasks, desc="Synthesizing")

    samples = [r for r in results if r is not None]
    skipped = len(results) - len(samples)
    print(f"Synthesis complete: {len(samples)} samples ({skipped} failed/skipped)")

    if not samples:
        print("WARNING: No samples generated!")
        return

    # 打乱并分割
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # 保存 parquet
    os.makedirs(output_dir, exist_ok=True)
    train_df = pd.DataFrame(train_samples)
    test_df = pd.DataFrame(test_samples)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Test:  {test_path}  ({len(test_df)} rows)")

    # 打印样本信息
    if samples:
        s = samples[0]
        prompt_data = json.loads(s["prompt"])
        resp_data = json.loads(s["response"])
        print(f"\nSample:")
        print(f"  prompt: {len(prompt_data.get('messages', []))} messages, "
              f"{len(prompt_data.get('tools', []))} tools")
        print(f"  response: tool_calls={bool(resp_data.get('tool_calls'))}, "
              f"content={bool(resp_data.get('content'))}")


def main():
    parser = argparse.ArgumentParser(description="用 Teacher LLM + 更新策略重新生成决策 → SFT parquet")
    parser.add_argument("--annotated_jsonl", type=str, required=True)
    parser.add_argument("--strategies_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--teacher_provider", type=str, default="openai")
    parser.add_argument("--teacher_model", type=str, default="gpt-4o")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(
        synthesize_trajectories(
            annotated_jsonl=args.annotated_jsonl,
            strategies_dir=args.strategies_dir,
            output_dir=args.output_dir,
            teacher_provider=args.teacher_provider,
            teacher_model=args.teacher_model,
            concurrency=args.concurrency,
            train_ratio=args.train_ratio,
            temperature=args.temperature,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
