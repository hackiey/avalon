"""JSONL 轨迹 + GAE advantage → Verl 训练用 parquet 格式转换。

将自博弈产生的游戏轨迹转换为 Verl PPO 训练所需的 parquet 格式。
每个 LLM 决策点拆分为一个训练样本。

支持两种 LLM 输出格式:
  1. tool_calls 格式: LLM 通过 function calling 返回结构化动作
     (vote_team, propose_team, vote_quest, assassinate, speak, update_memory)
  2. content 格式: LLM 直接返回文本内容 (旧格式兼容)

输入:
  - JSONL 轨迹文件（每行一局游戏，由 exporter 导出）
  - GAE advantage JSON 文件（可选，由 training.advantage.compute 生成）

输出:
  - train.parquet + test.parquet（Verl 标准格式）

Parquet 列:
  - data_source: "avalon" (顶层列，veRL reward_manager 需要)
  - prompt: JSON string, 格式为 {"messages": [...], "tools": [...]}
      - messages: OpenAI chat messages 列表
      - tools: OpenAI function calling 工具定义列表 (如有)
  - response: JSON string, 格式为 OpenAI assistant message:
      {"role": "assistant", "content": "...", "tool_calls": [...]}
      下游训练代码可用 tokenizer.apply_chat_template() 处理
  - reward_model: struct{data_source, ground_truth} (嵌套列，veRL reward_manager 需要)
  - discounted_return: 折扣回报 G_t (float, 用于 critic 训练)

注意: 不包含 extra_info 列，避免 parquet dict 序列化问题。
Verl 的 rl_dataset 会自动使用默认值 {}。

用法:
    python -m training.data.preprocess \\
        --input_jsonl data/trajectories.jsonl \\
        --output_dir data/processed \\
        --advantages_file data/advantages.json \\
        --train_ratio 0.9
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import pandas as pd

from game.roles import is_evil_role


def _count_response_tokens(response_json: str, tokenizer) -> int:
    """计算 response 的 token 数（与 VeRL 截断逻辑对齐）。

    从 response JSON 中提取实际文本内容并 tokenize，
    不含 JSON 语法开销，更接近 apply_chat_template 的真实 token 数。
    """
    try:
        msg = json.loads(response_json)
    except (json.JSONDecodeError, TypeError):
        return len(tokenizer.encode(response_json, add_special_tokens=False))

    parts = []
    if msg.get("reasoning_content"):
        parts.append(msg["reasoning_content"])
    if msg.get("content"):
        parts.append(msg["content"])
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            parts.append(json.dumps(tc, ensure_ascii=False))

    text = "\n".join(parts) if parts else response_json
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_trajectories(input_jsonl: str) -> List[Dict[str, Any]]:
    """加载 JSONL 轨迹文件。"""
    trajectories = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories


def load_advantages(advantages_file: str) -> Dict[tuple, Dict[str, float]]:
    """加载预计算的 advantage 和 discounted return。

    Returns:
        {(game_id, player_seat, seq_num): {"advantage": float, "discounted_return": float}}
    """
    with open(advantages_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    lookup = {}
    for item in results:
        key = (item["game_id"], item["player_seat"], item["seq_num"])
        lookup[key] = {
            "advantage": item["advantage"],
            "discounted_return": item["discounted_return"],
        }
    return lookup


def format_tool_calls_openai(
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """将内部 tool_calls 格式转换为 OpenAI 标准 assistant message 格式。

    内部格式 (来自 llm_output):
        {"name": "vote_team", "arguments": {"approve": true}, "id": "call_xxx"}

    OpenAI 标准格式 (用于 apply_chat_template):
        {"id": "call_xxx", "type": "function",
         "function": {"name": "vote_team", "arguments": "{\"approve\": true}"}}
    """
    formatted = []
    for i, tc in enumerate(tool_calls):
        args = tc.get("arguments", {})
        formatted.append(
            {
                "id": tc.get("id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": (
                        json.dumps(args, ensure_ascii=False)
                        if isinstance(args, dict)
                        else str(args)
                    ),
                },
            }
        )
    return formatted


def extract_response(decision: Dict[str, Any]) -> str:
    """从 decision 中提取 LLM 的完整回复，构建 assistant message。

    支持两种 LLM 输出格式:
      1. tool_calls 格式: 通过 function calling 返回结构化动作
      2. content 格式: 直接返回文本内容

    Returns:
        JSON 字符串，表示 OpenAI 格式的 assistant message:
        {"role": "assistant", "content": "...", "tool_calls": [...]}
        可直接用于 tokenizer.apply_chat_template()。
    """
    llm_output = decision.get("llm_output", {})

    if llm_output:
        has_tool_calls = bool(llm_output.get("tool_calls"))
        has_content = bool(llm_output.get("content"))
        has_reasoning = bool(llm_output.get("reasoning_content"))

        if has_tool_calls or has_content:
            msg: Dict[str, Any] = {"role": "assistant"}

            # 推理模型的思考过程 (DeepSeek-R1 等)
            if has_reasoning:
                msg["reasoning_content"] = llm_output["reasoning_content"]

            if has_tool_calls:
                # 主要输出: tool_calls (投票、组队、刺杀等动作)
                msg["tool_calls"] = format_tool_calls_openai(
                    llm_output["tool_calls"]
                )
                # tool_calls 模式下 content 通常为 None/空
                msg["content"] = llm_output.get("content") or ""
            else:
                # 纯文本输出 (如 discussion 阶段的 speak)
                msg["content"] = llm_output["content"]

            return json.dumps(msg, ensure_ascii=False)

    # Fallback: 从 decision 字段重建 (旧数据兼容)
    action_type = decision.get("action_type", "")

    if decision.get("vote") is not None:
        tool_name = "vote_quest" if action_type == "quest_vote" else "vote_team"
        key = "success" if tool_name == "vote_quest" else "approve"
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": format_tool_calls_openai(
                [{"name": tool_name, "arguments": {key: decision["vote"]}}]
            ),
        }
        return json.dumps(msg, ensure_ascii=False)

    if decision.get("team") is not None:
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": format_tool_calls_openai(
                [{"name": "propose_team", "arguments": {"team": decision["team"]}}]
            ),
        }
        return json.dumps(msg, ensure_ascii=False)

    if decision.get("target") is not None:
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": format_tool_calls_openai(
                [{"name": "assassinate", "arguments": {"target": decision["target"]}}]
            ),
        }
        return json.dumps(msg, ensure_ascii=False)

    if decision.get("content"):
        msg = {"role": "assistant", "content": decision["content"]}
        return json.dumps(msg, ensure_ascii=False)

    return ""


def build_samples(
    trajectories: List[Dict[str, Any]],
    advantage_lookup: Optional[Dict[tuple, Dict[str, float]]] = None,
    tokenizer=None,
) -> List[Dict[str, Any]]:
    """将轨迹转换为 Verl 训练样本。

    每个 LLM 决策点 → 一个训练样本。
    """
    samples = []
    skipped = 0

    for trajectory in trajectories:
        game_id = trajectory.get("game_id", "")
        winner = trajectory.get("winner", "")
        players = trajectory.get("players", [])
        player_map = {p["seat"]: p for p in players}

        for decision in trajectory.get("decisions", []):
            player_seat = decision.get("player_seat", 0)
            player = player_map.get(player_seat, {})
            role = player.get("role", "")
            player_team = "evil" if is_evil_role(role) else "good"
            seq_num = decision.get("seq_num", 0)

            # --- Prompt: chat messages + tool definitions ---
            llm_input = decision.get("llm_input", {})
            messages = llm_input.get("messages", [])
            if not messages:
                skipped += 1
                continue
            tools = llm_input.get("tools", [])

            # --- Response: LLM 回复 (assistant message with tool_calls/content) ---
            response = extract_response(decision)
            if not response:
                skipped += 1
                continue

            # --- Ground truth: 用于奖励计算 ---
            gt = {
                "game_id": game_id,
                "player_seat": player_seat,
                "player_role": role,
                "player_team": player_team,
                "game_winner": winner,
                "action_type": decision.get("action_type", ""),
                "seq_num": seq_num,
            }

            llm_output = decision.get("llm_output", {})
            if llm_output.get("error") == "tool_call_parse_error":
                gt["format_error"] = True

            # 存储原始 response 的 token 数（VeRL 截断前的真实长度，用于长度惩罚）
            if tokenizer is not None:
                gt["response_token_count"] = _count_response_tokens(
                    response, tokenizer
                )

            # 注入预计算的 advantage（如果有）
            discounted_return = 0.0
            if advantage_lookup:
                key = (game_id, player_seat, seq_num)
                adv_info = advantage_lookup.get(key, {})
                if "advantage" in adv_info:
                    gt["precomputed_advantage"] = adv_info["advantage"]
                discounted_return = adv_info.get("discounted_return", 0.0)

            # --- 构建样本 ---
            # veRL 的 reward_manager 从 non_tensor_batch["reward_model"] 取数据:
            #   data_item.non_tensor_batch["reward_model"]["ground_truth"]
            # 因此需要用嵌套 dict 的 reward_model 列，而非独立的列。
            #
            # 注意: 不包含 extra_info 列!
            # veRL 的 rl_dataset.py 对 extra_info 做 .get("index", 0)，
            # 如果存为 JSON string 会导致 AttributeError。
            # 不包含该列时，veRL 会使用默认值 {}，不会报错。
            # prompt 格式: {"messages": [...], "tools": [...]}
            # 下游训练代码可解析后调用:
            #   tokenizer.apply_chat_template(
            #       prompt_data["messages"],
            #       tools=prompt_data.get("tools"),
            #       add_generation_prompt=True,
            #   )
            prompt_data: Dict[str, Any] = {"messages": messages}
            if tools:
                prompt_data["tools"] = tools

            sample = {
                "data_source": "avalon",
                "prompt": json.dumps(prompt_data, ensure_ascii=False),
                "response": response,
                "reward_model": {
                    "data_source": "avalon",
                    "ground_truth": json.dumps(gt, ensure_ascii=False),
                },
                "discounted_return": discounted_return,
            }

            samples.append(sample)

    if skipped:
        print(f"  Skipped {skipped} decisions (no messages or response)")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="JSONL 轨迹 → Verl parquet 格式转换"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="输入的 JSONL 轨迹文件",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录（生成 train.parquet 和 test.parquet）",
    )
    parser.add_argument(
        "--advantages_file",
        type=str,
        default=None,
        help="预计算的 advantage JSON 文件（可选）",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="训练集比例 (默认 0.9)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径（用于 tokenizer 计算 response token 数，启用长度惩罚时必需）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    args = parser.parse_args()

    # 加载 tokenizer（用于计算 response token 数）
    tokenizer = None
    if args.model_path:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        print(f"  Tokenizer loaded: {type(tokenizer).__name__}")

    # 加载轨迹
    print(f"Loading trajectories from {args.input_jsonl}...")
    trajectories = load_trajectories(args.input_jsonl)
    print(f"  Loaded {len(trajectories)} games")

    # 加载预计算 advantage（如果有）
    advantage_lookup = None
    if args.advantages_file:
        print(f"Loading advantages from {args.advantages_file}...")
        advantage_lookup = load_advantages(args.advantages_file)
        print(f"  Loaded {len(advantage_lookup)} advantage values")

    # 构建训练样本
    print("Building training samples...")
    samples = build_samples(trajectories, advantage_lookup, tokenizer=tokenizer)
    print(f"  Created {len(samples)} training samples")

    if not samples:
        print("WARNING: No samples created! Check input data.")
        return

    # 打乱并分割
    random.seed(args.seed)
    random.shuffle(samples)

    split_idx = int(len(samples) * args.train_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

    # 保存为 parquet
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_samples)
    test_df = pd.DataFrame(test_samples)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Test:  {test_path} ({len(test_df)} rows)")

    # 打印样本信息
    if train_samples:
        sample = train_samples[0]
        rm = sample["reward_model"]
        print(f"\nSample entry:")
        print(f"  data_source: {sample['data_source']}")
        print(f"  prompt length: {len(sample['prompt'])} chars")

        # 解析 prompt 结构
        prompt_data = json.loads(sample["prompt"])
        if isinstance(prompt_data, dict):
            n_msgs = len(prompt_data.get("messages", []))
            n_tools = len(prompt_data.get("tools", []))
            print(f"  prompt: {n_msgs} messages, {n_tools} tools")
            if n_tools > 0:
                tool_names = [t["function"]["name"] for t in prompt_data["tools"]
                              if "function" in t]
                print(f"  tool names: {tool_names}")
        else:
            # 旧格式: messages list
            print(f"  prompt: {len(prompt_data)} messages (legacy format)")

        # 解析 response 结构
        try:
            resp_data = json.loads(sample["response"])
            if isinstance(resp_data, dict):
                has_tc = bool(resp_data.get("tool_calls"))
                has_ct = bool(resp_data.get("content"))
                has_rs = bool(resp_data.get("reasoning_content"))
                print(f"  response: tool_calls={has_tc}, content={has_ct}, "
                      f"reasoning={has_rs}")
                if has_tc:
                    tc_names = [tc["function"]["name"]
                                for tc in resp_data["tool_calls"]]
                    print(f"  response tool_calls: {tc_names}")
            else:
                print(f"  response: {sample['response'][:100]}...")
        except (json.JSONDecodeError, TypeError):
            print(f"  response: {sample['response'][:100]}...")

        gt = json.loads(rm["ground_truth"])
        print(f"  ground_truth keys: {list(gt.keys())}")
        if "precomputed_advantage" in gt:
            print(f"  precomputed_advantage: {gt['precomputed_advantage']:.4f}")
        if "response_token_count" in gt:
            print(f"  response_token_count: {gt['response_token_count']}")
        print(f"  discounted_return: {sample['discounted_return']:.4f}")

    # 打印 response token 分布（用于校准长度惩罚阈值）
    if tokenizer and samples:
        token_counts = []
        for s in samples:
            gt_data = json.loads(s["reward_model"]["ground_truth"])
            if "response_token_count" in gt_data:
                token_counts.append(gt_data["response_token_count"])
        if token_counts:
            token_counts.sort()
            n = len(token_counts)
            print(f"\nResponse token distribution ({n} samples):")
            print(f"  min:  {token_counts[0]}")
            print(f"  p25:  {token_counts[n // 4]}")
            print(f"  p50:  {token_counts[n // 2]}")
            print(f"  p75:  {token_counts[3 * n // 4]}")
            print(f"  p90:  {token_counts[int(n * 0.9)]}")
            print(f"  p95:  {token_counts[int(n * 0.95)]}")
            print(f"  p99:  {token_counts[int(n * 0.99)]}")
            print(f"  max:  {token_counts[-1]}")
            over_8k = sum(1 for t in token_counts if t > 8192)
            print(f"  >8192: {over_8k} ({over_8k / n * 100:.1f}%)")

    # 打印列信息
    print(f"\nParquet columns: {list(train_df.columns)}")
    print(f"Column dtypes:")
    for col in train_df.columns:
        print(f"  {col}: {train_df[col].dtype}")


if __name__ == "__main__":
    main()
