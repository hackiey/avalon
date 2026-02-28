"""角色策略文档管理 — 加载、保存、以及用 Teacher LLM 更新。

维护 4 个 Avalon 角色的自然语言策略文档（NL Policy）。
初始策略内置于代码中；每轮 NLRL 训练后，Teacher LLM 根据标注的坏决策更新策略。

用法（独立调用）:
    python -m training.nlrl.strategy update \\
        --annotated_jsonl data/round_1/annotated.jsonl \\
        --strategies_dir experiments/nlrl_v1/strategies/round_1 \\
        --prev_strategies_dir experiments/nlrl_v1/strategies/round_0 \\
        --teacher_provider openai --teacher_model gpt-4o \\
        --max_decisions_per_role 50
"""

import argparse
import asyncio
import json
import os
import random
from typing import Any, Dict, List, Optional

# ============================================================
# 内置初始策略
# ============================================================

_INITIAL_STRATEGIES: Dict[str, str] = {
    "merlin": """# Merlin Strategy

You secretly know who all Evil players are. Your goal is to help the Good team complete 3 quests while keeping your identity hidden from the Assassin.

## Core Principles

1. **Subtle guidance**: Steer the team toward Good players without being obvious. Use hints, questions, and indirect observations rather than direct accusations. The moment Evil suspects you are Merlin, you become a target for assassination.

2. **Hint through uncertainty**: Say things like "I'm not fully convinced by Player X" rather than "Player X is Evil." Express doubt through behavioral observations, not certainty.

3. **Team selection**: When you are leader, propose teams with only Good players. When others propose, vote against teams containing Evil players — but mix in occasional "mistakes" to avoid looking omniscient.

4. **Vary your patterns**: Occasionally express mild doubt about Good players and occasional trust in Evil players to avoid appearing to have perfect information. Pure consistency reveals Merlin.

5. **Assassination defense**: Track who might be the Assassin and monitor whether they are watching your patterns. In the endgame (Good has 2 successful quests), become deliberately less specific in your accusations to confuse the Assassin.

6. **Calibrate by game state**: Early game, you can be somewhat more open. As Good approaches winning, increase your caution significantly.
""",

    "loyal_servant": """# Loyal Servant Strategy

You have no special knowledge. Your goal is to identify Evil players through careful observation and ensure Good players are on quests.

## Core Principles

1. **Track voting patterns**: Evil players tend to vote against Good-heavy teams and approve Evil-heavy teams. Note inconsistencies between what players say and how they vote.

2. **Analyze quest outcomes**: When a quest fails, at least one team member is Evil. Cross-reference failed quests with approved teams to narrow down suspects.

3. **Listen carefully**: Players with inexplicably accurate judgment about others may be Merlin. Players who subtly deflect suspicion or redirect accusations may be Evil.

4. **Share observations**: Verbalize your reasoning. This helps coordinate with other Good players and forces Evil players to respond to your logic (often revealing inconsistencies).

5. **Vote carefully**: Reject teams where you have strong suspicion of Evil presence. Approve teams you trust, even if not perfect. Be wary of repeated rejections engineered by Evil to force a 5th-rejection loss.

6. **Build trust**: Demonstrate consistent, logical Good-player behavior. Avoid erratic decisions that make you appear suspicious.
""",

    "assassin": """# Assassin Strategy

You know who your Evil teammates are. Your win conditions: fail 3 quests, OR if Good wins 3 quests, successfully assassinate Merlin.

## Core Principles

1. **Identify Merlin**: This is your most important task. Merlin knows all Evil players and will try to subtly exclude you from quests. Watch for players who:
   - Consistently form teams without any Evil players
   - Seem inexplicably confident about who is Good/Evil
   - Lead successful quests that exclude you

2. **Blend in as Good**: Participate actively in discussions showing concern about finding Evil. Agree with Good players' logic when it doesn't harm you.

3. **Strategic quest outcomes**: Don't fail every quest you're on — this makes you obvious. Consider succeeding early quests to appear trustworthy. Save failures for critical moments.

4. **Coordinate subtly**: Align with your Minion without appearing coordinated. Avoid always voting the same way; sometimes vote against your Minion's teams.

5. **Misdirect accusations**: In discussions, cast occasional (not constant) suspicion on Good players to muddy the waters and protect yourself and your Minion.

6. **Track the meta-game**: Pay attention to who is watching you. If you suspect someone has identified you as Evil, increase your Good-player behavior.
""",

    "minion": """# Minion Strategy

You know who the Assassin is. Your goal: help Evil fail 3 quests or help the Assassin identify and assassinate Merlin.

## Core Principles

1. **Support the Assassin**: Help the Assassin get on quests and avoid suspicion. Don't coordinate too obviously — avoid always voting the same way.

2. **Strategic quest failure**: Fail quests when it's strategically safe (e.g., multiple Evil players on the quest, unlikely to be caught individually). Don't fail every quest you're on.

3. **Blend in**: Appear to be a Good player. Actively participate in discussions, express concern about Evil, show consistent reasoning.

4. **Deflect suspicion**: When scrutinized, redirect attention to specific Good players. Use plausible behavioral evidence, not random accusations.

5. **Watch for Merlin**: Note which Good player seems to have insider knowledge about Evil identities. Share observations indirectly through normal gameplay behavior. Help the Assassin narrow down who Merlin is.

6. **Vote strategically**: Approve teams that include you or the Assassin. Reject teams where you're the only Evil member (quest failure would be obvious). Be willing to reject even beneficial teams occasionally to appear principled.
""",
}


def get_initial_strategy(role: str) -> str:
    """获取指定角色的初始策略文本。"""
    return _INITIAL_STRATEGIES.get(role.lower(), f"# {role.title()} Strategy\n\nPlay thoughtfully.")


def load_strategy(role: str, strategies_dir: str) -> str:
    """从目录加载策略文档，不存在则返回初始策略。"""
    path = os.path.join(strategies_dir, f"{role.lower()}.md")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return get_initial_strategy(role)


def save_strategy(role: str, content: str, strategies_dir: str) -> None:
    """保存策略文档到目录。"""
    os.makedirs(strategies_dir, exist_ok=True)
    path = os.path.join(strategies_dir, f"{role.lower()}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _format_decision_for_prompt(decision: Dict[str, Any]) -> str:
    """将一个带标注的决策格式化为可读文本。"""
    role = decision.get("player_role", "unknown")
    action_type = decision.get("action_type", "unknown")
    round_num = decision.get("round_num", "?")

    # 提取决策内容
    llm_output = decision.get("llm_output", {})
    if isinstance(llm_output, dict):
        tool_calls = llm_output.get("tool_calls", [])
        content = llm_output.get("content", "")
        if tool_calls:
            decision_text = f"Tool calls: {json.dumps(tool_calls, ensure_ascii=False)}"
        else:
            decision_text = f"Response: {content[:300]}"
    else:
        decision_text = str(llm_output)[:300]

    # 提取标注
    annotation = decision.get("nlrl_annotation", {})
    quality = annotation.get("quality", "unknown")
    reason = annotation.get("reason", "")
    lesson = annotation.get("lesson", "")

    lines = [
        f"[Round {round_num}, {action_type}]",
        f"Decision: {decision_text}",
        f"Quality: {quality}",
        f"Reason: {reason}",
    ]
    if lesson:
        lines.append(f"Lesson: {lesson}")

    return "\n".join(lines)


async def update_strategy_with_llm(
    role: str,
    bad_decisions: List[Dict[str, Any]],
    current_strategy: str,
    client,
    model: str,
    max_decisions: int = 30,
) -> str:
    """用 Teacher LLM 根据坏决策更新策略文档。

    Args:
        role: 角色名
        bad_decisions: 标注为 bad 的决策列表
        current_strategy: 当前策略文本
        client: AsyncOpenAI 客户端
        model: teacher 模型名
        max_decisions: 最多使用多少个坏决策

    Returns:
        更新后的策略文本
    """
    if not bad_decisions:
        return current_strategy

    # 随机采样避免 prompt 过长
    sampled = random.sample(bad_decisions, min(max_decisions, len(bad_decisions)))
    decisions_text = "\n\n---\n\n".join(_format_decision_for_prompt(d) for d in sampled)

    prompt = f"""You are improving a strategy guide for playing Avalon. You will update the strategy for the **{role.title()}** role based on recent gameplay analysis.

## Current Strategy
{current_strategy}

## Recent Bad Decisions (with analysis)
The following decisions by {role.title()} players were evaluated as poor quality:

{decisions_text}

## Task
Update the strategy guide to:
1. Address the most common mistakes identified above
2. Add specific, actionable guidance to avoid these mistakes
3. Keep successful principles from the current strategy
4. Be concise (400-600 words total)

Return ONLY the updated strategy document (starting with "# {role.title()} Strategy"), no extra commentary."""

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Avalon game strategist. Write clear, actionable strategy guides.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


async def update_all_strategies(
    annotated_jsonl: str,
    strategies_dir: str,
    prev_strategies_dir: Optional[str],
    teacher_provider: str,
    teacher_model: str,
    max_decisions_per_role: int = 50,
) -> None:
    """从标注的 JSONL 更新所有角色的策略文档。

    Args:
        annotated_jsonl: 带 nlrl_annotation 字段的 JSONL 轨迹文件
        strategies_dir: 输出目录（保存更新后的策略）
        prev_strategies_dir: 上一轮的策略目录（不存在则用初始策略）
        teacher_provider: teacher LLM provider 名
        teacher_model: teacher LLM 模型名
        max_decisions_per_role: 每角色最多用多少个坏决策
    """
    client = _make_client(teacher_provider)

    # 按角色收集 bad decisions
    bad_by_role: Dict[str, List[Dict[str, Any]]] = {
        role: [] for role in _INITIAL_STRATEGIES
    }

    with open(annotated_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectory = json.loads(line)
            players = {p["seat"]: p for p in trajectory.get("players", [])}
            for decision in trajectory.get("decisions", []):
                annotation = decision.get("nlrl_annotation")
                if not annotation:
                    continue
                if annotation.get("quality") != "bad":
                    continue
                seat = decision.get("player_seat", 0)
                role = players.get(seat, {}).get("role", "").lower()
                if role in bad_by_role:
                    # 附加玩家角色信息（strategy.py 需要）
                    decision = dict(decision)
                    decision["player_role"] = role
                    bad_by_role[role].append(decision)

    os.makedirs(strategies_dir, exist_ok=True)

    tasks = []
    roles = list(_INITIAL_STRATEGIES.keys())
    for role in roles:
        prev_dir = prev_strategies_dir or ""
        current_strategy = (
            load_strategy(role, prev_dir) if prev_dir else get_initial_strategy(role)
        )
        bad = bad_by_role[role]
        tasks.append(
            update_strategy_with_llm(
                role, bad, current_strategy, client, teacher_model, max_decisions_per_role
            )
        )

    print(f"Updating strategies for {len(roles)} roles...")
    for role, bad in zip(roles, [bad_by_role[r] for r in roles]):
        print(f"  {role}: {len(bad)} bad decisions")

    results = await asyncio.gather(*tasks)

    for role, updated in zip(roles, results):
        save_strategy(role, updated, strategies_dir)
        print(f"  Saved strategy: {strategies_dir}/{role}.md")


def _make_client(provider: str):
    """根据 provider 创建对应的异步客户端（读取 .env API key）。"""
    import os
    from openai import AsyncOpenAI

    provider = provider.lower()
    if provider == "anthropic":
        # Anthropic 不支持 tool_calling（providers.py 中有 TODO）
        # 为统一接口，通过 openai-compatible wrapper 使用
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    elif provider == "deepseek":
        return AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    elif provider == "vllm":
        return AsyncOpenAI(
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        )
    else:
        # openai / qwen / 其他 openai-compatible
        api_key = os.environ.get(f"{provider.upper()}_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get(f"{provider.upper()}_BASE_URL") or os.environ.get("OPENAI_BASE_URL", None)
        return AsyncOpenAI(api_key=api_key, base_url=base_url)


def main():
    parser = argparse.ArgumentParser(description="更新角色策略文档")
    parser.add_argument("command", choices=["update", "init"], help="update: 更新策略; init: 导出初始策略")
    parser.add_argument("--annotated_jsonl", type=str, help="带标注的 JSONL 轨迹文件")
    parser.add_argument("--strategies_dir", type=str, required=True, help="输出策略目录")
    parser.add_argument("--prev_strategies_dir", type=str, default=None, help="上一轮策略目录")
    parser.add_argument("--teacher_provider", type=str, default="openai")
    parser.add_argument("--teacher_model", type=str, default="gpt-4o")
    parser.add_argument("--max_decisions_per_role", type=int, default=50)
    args = parser.parse_args()

    if args.command == "init":
        os.makedirs(args.strategies_dir, exist_ok=True)
        for role, strategy in _INITIAL_STRATEGIES.items():
            save_strategy(role, strategy, args.strategies_dir)
            print(f"Saved initial strategy: {args.strategies_dir}/{role}.md")
        return

    if not args.annotated_jsonl:
        parser.error("--annotated_jsonl is required for 'update' command")

    asyncio.run(
        update_all_strategies(
            annotated_jsonl=args.annotated_jsonl,
            strategies_dir=args.strategies_dir,
            prev_strategies_dir=args.prev_strategies_dir,
            teacher_provider=args.teacher_provider,
            teacher_model=args.teacher_model,
            max_decisions_per_role=args.max_decisions_per_role,
        )
    )


if __name__ == "__main__":
    main()
