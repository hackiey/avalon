"""用 Teacher LLM 标注游戏决策的质量。

输入: trajectories.jsonl（game/rollout.py 导出格式）
输出: annotated.jsonl（每个 decision 加 nlrl_annotation 字段）

标注结构:
    {
        "quality": "good|neutral|bad",
        "reason": "一句话解释",
        "lesson": "可操作的策略建议（quality=bad 时必填）"
    }

Teacher LLM 拥有 oracle 视角（所有角色信息），因此能够准确判断每个决策的质量。

用法:
    python -m training.nlrl.annotate \\
        --input_jsonl data/round_1/trajectories.jsonl \\
        --output_jsonl data/round_1/annotated.jsonl \\
        --teacher_provider openai --teacher_model gpt-4o \\
        --max_per_role 80 --concurrency 10
"""

import argparse
import asyncio
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from tqdm.asyncio import tqdm as atqdm

from training.nlrl.strategy import _make_client


# ============================================================
# Prompt 构建
# ============================================================

_ACTION_TYPE_DESC = {
    "discussion": "general discussion",
    "leader_discussion": "discussion as round leader",
    "team_vote": "vote on the proposed team",
    "quest_vote": "vote on quest success/failure",
    "team_selection": "propose a team for the quest",
    "assassination": "assassinate a player",
    "assassination_discussion": "discuss before assassination",
}


def _format_llm_output(llm_output: Any) -> str:
    """将 llm_output 格式化为可读文本。"""
    if not llm_output:
        return "(no output)"
    if isinstance(llm_output, str):
        return llm_output[:500]

    parts = []
    if llm_output.get("reasoning_content"):
        parts.append(f"[Thinking]: {llm_output['reasoning_content'][:200]}...")
    if llm_output.get("content"):
        parts.append(f"[Response]: {llm_output['content'][:300]}")
    if llm_output.get("tool_calls"):
        for tc in llm_output["tool_calls"]:
            name = tc.get("name", "unknown")
            args = tc.get("arguments", {})
            parts.append(f"[Action]: {name}({json.dumps(args, ensure_ascii=False)})")
    return "\n".join(parts) if parts else "(empty)"


def _format_last_user_message(messages: List[Dict]) -> str:
    """提取最后一条 user message 的内容（代表当前决策请求）。"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content[:800] if len(content) > 800 else content
    return "(no user message)"


def build_annotation_prompt(
    decision: Dict[str, Any],
    trajectory: Dict[str, Any],
) -> Tuple[str, str]:
    """构建标注 prompt（system + user）。

    Returns:
        (system_message, user_message)
    """
    players = {p["seat"]: p for p in trajectory.get("players", [])}
    winner = trajectory.get("winner", "unknown")
    merlin_assassinated = trajectory.get("merlin_assassinated", False)

    seat = decision.get("player_seat", 0)
    player = players.get(seat, {})
    role = player.get("role", "unknown")
    action_type = decision.get("action_type", "unknown")
    round_num = decision.get("round_num", 1)

    # Oracle 视角：列出所有玩家真实角色
    all_roles = []
    for s in sorted(players.keys()):
        p = players[s]
        team = "EVIL" if p.get("team", "") == "evil" or p.get("role", "") in {"assassin", "minion"} else "GOOD"
        all_roles.append(f"  Player {s} ({p.get('name', f'P{s}')}): {p.get('role', 'unknown')} [{team}]")
    roles_text = "\n".join(all_roles)

    # 游戏结果
    result_text = f"{winner.title()} team won."
    if merlin_assassinated:
        result_text += " Merlin was assassinated."

    # 决策内容
    llm_input = decision.get("llm_input", {})
    messages = llm_input.get("messages", [])
    game_state_text = _format_last_user_message(messages)
    decision_text = _format_llm_output(decision.get("llm_output", {}))

    action_desc = _ACTION_TYPE_DESC.get(action_type, action_type)

    system_msg = (
        "You are an expert Avalon game analyst with full oracle knowledge of all players' roles. "
        "Analyze a specific game decision and evaluate its quality based on optimal play for that role."
    )

    user_msg = f"""## Game Result
{result_text}

## All Players (Oracle View)
{roles_text}

## Decision to Analyze
- **Player**: Player {seat} ({player.get('name', f'P{seat}')}), Role: **{role}** ({'EVIL' if role in {'assassin', 'minion'} else 'GOOD'} team)
- **Round**: {round_num}, **Action**: {action_desc}

## Current Game State (player's perspective)
{game_state_text}

## Decision Made
{decision_text}

## Your Analysis
Respond in JSON only (no markdown fences):
{{
  "quality": "good|neutral|bad",
  "reason": "One sentence: why was this decision good/neutral/bad given the player's role and the game state?",
  "lesson": "Specific, actionable strategy tip for the {role} role to improve this type of decision. Required if quality=bad, empty string otherwise."
}}"""

    return system_msg, user_msg


# ============================================================
# 标注逻辑
# ============================================================

async def annotate_one(
    decision: Dict[str, Any],
    trajectory: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    client,
    model: str,
    temperature: float = 0.3,
) -> Optional[Dict[str, Any]]:
    """标注单个决策，返回 annotation dict 或 None（失败时）。"""
    async with semaphore:
        system_msg, user_msg = build_annotation_prompt(decision, trajectory)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            annotation = json.loads(text)
            # 校验字段
            if "quality" not in annotation:
                return None
            annotation["quality"] = annotation["quality"].lower()
            if annotation["quality"] not in ("good", "neutral", "bad"):
                annotation["quality"] = "neutral"
            return annotation
        except Exception as e:
            print(f"  [annotate] Failed: {e}")
            return None


def _sample_decisions_for_annotation(
    trajectories: List[Dict[str, Any]],
    max_per_role: int,
) -> Dict[str, List[Tuple[int, int]]]:
    """为每个角色随机采样要标注的 (traj_idx, decision_idx) 对。

    Returns:
        {role: [(traj_idx, decision_idx), ...]}
    """
    by_role: Dict[str, List[Tuple[int, int]]] = {}

    for traj_idx, trajectory in enumerate(trajectories):
        players = {p["seat"]: p for p in trajectory.get("players", [])}
        for dec_idx, decision in enumerate(trajectory.get("decisions", [])):
            seat = decision.get("player_seat", 0)
            role = players.get(seat, {}).get("role", "unknown").lower()
            if role not in by_role:
                by_role[role] = []
            by_role[role].append((traj_idx, dec_idx))

    # 每角色随机采样
    sampled: Dict[str, List[Tuple[int, int]]] = {}
    for role, indices in by_role.items():
        if len(indices) <= max_per_role:
            sampled[role] = indices
        else:
            sampled[role] = random.sample(indices, max_per_role)

    return sampled


async def annotate_trajectories(
    input_jsonl: str,
    output_jsonl: str,
    teacher_provider: str,
    teacher_model: str,
    concurrency: int = 10,
    max_per_role: int = 80,
    temperature: float = 0.3,
) -> None:
    """标注整个 JSONL 文件中的决策。

    对每个角色采样最多 max_per_role 个决策进行标注，其余决策写入时不带标注字段。
    """
    # 加载轨迹
    trajectories = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    total_decisions = sum(len(t.get("decisions", [])) for t in trajectories)
    print(f"Loaded {len(trajectories)} trajectories, {total_decisions} total decisions")

    # 采样要标注的 (traj_idx, dec_idx) 对
    sampled_map = _sample_decisions_for_annotation(trajectories, max_per_role)
    annotate_set = set()
    for indices in sampled_map.values():
        annotate_set.update(indices)

    sampled_count = len(annotate_set)
    print(f"Will annotate {sampled_count} decisions (max {max_per_role}/role):")
    for role, indices in sampled_map.items():
        print(f"  {role}: {len(indices)}")

    client = _make_client(teacher_provider)
    semaphore = asyncio.Semaphore(concurrency)

    # 构建所有标注任务
    tasks = []
    task_keys = []
    for traj_idx, traj in enumerate(trajectories):
        for dec_idx in range(len(traj.get("decisions", []))):
            if (traj_idx, dec_idx) in annotate_set:
                tasks.append(
                    annotate_one(
                        traj["decisions"][dec_idx],
                        traj,
                        semaphore,
                        client,
                        teacher_model,
                        temperature,
                    )
                )
                task_keys.append((traj_idx, dec_idx))

    # 并发执行
    print(f"Annotating {len(tasks)} decisions with concurrency={concurrency}...")
    results = await atqdm.gather(*tasks, desc="Annotating")

    # 填充标注结果
    annotation_map: Dict[Tuple[int, int], Optional[Dict]] = {}
    for key, result in zip(task_keys, results):
        annotation_map[key] = result

    success = sum(1 for v in annotation_map.values() if v is not None)
    print(f"Annotation complete: {success}/{len(tasks)} succeeded")

    # 统计各质量分布
    quality_counts: Dict[str, int] = {}
    for ann in annotation_map.values():
        if ann:
            q = ann.get("quality", "unknown")
            quality_counts[q] = quality_counts.get(q, 0) + 1
    print(f"Quality distribution: {quality_counts}")

    # 写入输出 JSONL（添加 nlrl_annotation 字段）
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for traj_idx, trajectory in enumerate(trajectories):
            updated_decisions = []
            for dec_idx, decision in enumerate(trajectory.get("decisions", [])):
                decision = dict(decision)
                annotation = annotation_map.get((traj_idx, dec_idx))
                if annotation is not None:
                    decision["nlrl_annotation"] = annotation
                updated_decisions.append(decision)
            trajectory = dict(trajectory)
            trajectory["decisions"] = updated_decisions
            f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

    print(f"Saved annotated trajectories to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="用 Teacher LLM 标注游戏决策")
    parser.add_argument("--input_jsonl", type=str, required=True, help="输入 JSONL 轨迹文件")
    parser.add_argument("--output_jsonl", type=str, required=True, help="输出带标注的 JSONL 文件")
    parser.add_argument("--teacher_provider", type=str, default="openai", help="teacher LLM provider")
    parser.add_argument("--teacher_model", type=str, default="gpt-4o", help="teacher LLM 模型名")
    parser.add_argument("--concurrency", type=int, default=10, help="并发请求数")
    parser.add_argument("--max_per_role", type=int, default=80, help="每角色最多标注的决策数")
    parser.add_argument("--temperature", type=float, default=0.3, help="标注温度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    asyncio.run(
        annotate_trajectories(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            teacher_provider=args.teacher_provider,
            teacher_model=args.teacher_model,
            concurrency=args.concurrency,
            max_per_role=args.max_per_role,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
