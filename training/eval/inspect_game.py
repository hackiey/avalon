"""单局游戏审查工具：查看模型在一局 Avalon 游戏中的完整输出。

运行一局完整游戏，实时显示每个玩家的 LLM 输入/输出、决策过程和游戏进展。
支持聚焦到某个特定玩家或角色，方便调试模型行为。

用法:
    # 使用已启动的 vLLM 服务
    python -m training.eval.inspect_game \
        --model my_model --provider vllm \
        --port 8000

    # 自动启动 vLLM 服务
    python -m training.eval.inspect_game \
        --model_path /path/to/model \
        --port 8000 --tp 4

    # 只查看特定角色的输出
    python -m training.eval.inspect_game \
        --model my_model --provider vllm \
        --port 8000 --focus_role merlin

    # 只查看特定座位的输出
    python -m training.eval.inspect_game \
        --model my_model --provider vllm \
        --port 8000 --focus_seat 0

    # 保存完整记录到文件
    python -m training.eval.inspect_game \
        --model my_model --provider vllm \
        --port 8000 --save game_log.json

    # 指定随机种子（可复现）
    python -m training.eval.inspect_game \
        --model my_model --provider vllm \
        --port 8000 --seed 42
"""

import argparse
import asyncio
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── ANSI 颜色 ────────────────────────────────────────────────────────────────

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"

    @staticmethod
    def disable():
        for attr in dir(C):
            if attr.isupper() and not attr.startswith("_"):
                setattr(C, attr, "")


# ── 阶段中文名映射 ──────────────────────────────────────────────────────────

PHASE_NAMES = {
    "leader_discussion": "队长发言",
    "discussion": "讨论",
    "team_selection": "组队",
    "team_vote": "投票",
    "quest_execution": "任务执行",
    "assassination_discussion": "刺杀讨论",
    "assassination": "刺杀",
}

TEAM_COLORS = {
    "good": C.BLUE,
    "evil": C.RED,
}

ROLE_DISPLAY = {
    "merlin": ("梅林", C.CYAN),
    "loyal_servant": ("忠臣", C.BLUE),
    "assassin": ("刺客", C.RED),
    "minion": ("爪牙", C.MAGENTA),
}


# ── 数据记录 ──────────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    """一个 turn 的完整记录。"""
    turn_num: int
    seat: int
    player_name: str
    role: str
    team: str
    phase: str
    llm_input: Dict[str, Any] = field(default_factory=dict)
    llm_output: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameRecord:
    """完整游戏记录。"""
    players: List[Dict[str, Any]] = field(default_factory=list)
    turns: List[TurnRecord] = field(default_factory=list)
    quest_results: List[Dict[str, Any]] = field(default_factory=list)
    vote_history: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[str] = None
    assassinated: Optional[int] = None
    seed: Optional[int] = None


# ── 格式化输出 ────────────────────────────────────────────────────────────────

def _hr(char="─", width=90):
    return f"{C.DIM}{char * width}{C.RESET}"


def _section(title: str, char="━", width=90):
    pad = width - len(title) - 4
    left = pad // 2
    right = pad - left
    return f"{C.BOLD}{char * left}  {title}  {char * right}{C.RESET}"


def _wrap(text: str, indent: str = "    ", width: int = 82) -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if not line.strip():
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(line, width=width))
    return "\n".join(f"{indent}{l}" for l in wrapped)


def _role_str(role: str) -> str:
    name, color = ROLE_DISPLAY.get(role, (role, C.WHITE))
    return f"{color}{C.BOLD}{name}{C.RESET}"


def _team_str(team: str) -> str:
    color = TEAM_COLORS.get(team, C.WHITE)
    label = "好人阵营" if team == "good" else "坏人阵营"
    return f"{color}{label}{C.RESET}"


def _player_tag(seat: int, name: str, role: str) -> str:
    role_name, role_color = ROLE_DISPLAY.get(role, (role, C.WHITE))
    return f"{role_color}[P{seat + 1} {name} | {role_name}]{C.RESET}"


def print_game_setup(record: GameRecord):
    """打印游戏初始设置。"""
    print(f"\n{_section('游戏设置')}")
    if record.seed is not None:
        print(f"  随机种子: {C.YELLOW}{record.seed}{C.RESET}")
    print(f"  玩家数量: {len(record.players)}")
    print()

    header = f"  {'座位':<6} {'名称':<12} {'角色':<16} {'阵营':<14} {'模型'}"
    print(header)
    print(f"  {_hr(width=84)}")

    for p in record.players:
        role_name, role_color = ROLE_DISPLAY.get(p["role"], (p["role"], C.WHITE))
        team_color = TEAM_COLORS.get(p["team"], C.WHITE)
        team_label = "好人" if p["team"] == "good" else "坏人"
        print(
            f"  P{p['seat'] + 1:<5}"
            f" {p['name']:<12}"
            f" {role_color}{role_name:<8}{C.RESET}"
            f"       {team_color}{team_label:<6}{C.RESET}"
            f"      {C.DIM}{p.get('model', '?')}{C.RESET}"
        )
    print()


def print_turn(turn: TurnRecord, verbose: bool = False, raw: bool = False):
    """打印单个 turn 的信息。"""
    phase_name = PHASE_NAMES.get(turn.phase, turn.phase)
    tag = _player_tag(turn.seat, turn.player_name, turn.role)

    print(f"\n{_hr()}")
    print(
        f"  {C.BOLD}Turn {turn.turn_num}{C.RESET}  │  "
        f"{C.YELLOW}{phase_name}{C.RESET}  │  "
        f"{tag}"
    )
    print(_hr())

    # LLM 输出
    llm_out = turn.llm_output
    if not llm_out:
        print(f"  {C.DIM}(无 LLM 输出){C.RESET}")
    elif raw:
        # 原始模式：直接打印完整 llm_output dict
        print(f"\n  {C.BOLD}📦 模型原始输出:{C.RESET}")
        raw_json = json.dumps(llm_out, indent=2, ensure_ascii=False)
        print(_wrap(raw_json, indent=f"  {C.DIM}│ {C.RESET}", width=120))
    else:
        # Reasoning (思维链)
        reasoning = llm_out.get("reasoning_content", "")
        if reasoning:
            print(f"\n  {C.MAGENTA}{C.BOLD}💭 思考过程:{C.RESET}")
            print(_wrap(reasoning, indent=f"  {C.DIM}│ {C.RESET}{C.MAGENTA}", width=78))
            print(f"  {C.DIM}│{C.RESET}")

        # Content (发言内容)
        content = llm_out.get("content", "")
        if content:
            print(f"\n  {C.GREEN}{C.BOLD}💬 发言/内容:{C.RESET}")
            print(_wrap(content, indent=f"  {C.DIM}│ {C.RESET}", width=78))

        # Tool calls
        tool_calls = llm_out.get("tool_calls", [])
        if tool_calls:
            print(f"\n  {C.CYAN}{C.BOLD}🔧 工具调用:{C.RESET}")
            for tc in tool_calls:
                name = tc.get("name", "?")
                args = tc.get("arguments", {})
                print(f"  {C.DIM}│{C.RESET} {C.CYAN}{name}{C.RESET}({C.DIM}{json.dumps(args, ensure_ascii=False, indent=None)}{C.RESET})")

    # 动作结果
    action = turn.action
    if action:
        print(f"\n  {C.YELLOW}{C.BOLD}⚡ 执行动作:{C.RESET}")
        for k, v in action.items():
            if k == "team":
                team_display = ", ".join(f"P{s + 1}" for s in v)
                print(f"  {C.DIM}│{C.RESET} 选定队伍: [{team_display}]")
            elif k == "approve":
                vote_str = f"{C.GREEN}赞成 ✓{C.RESET}" if v else f"{C.RED}反对 ✗{C.RESET}"
                print(f"  {C.DIM}│{C.RESET} 投票: {vote_str}")
            elif k == "success":
                result_str = f"{C.GREEN}成功 ✓{C.RESET}" if v else f"{C.RED}破坏 ✗{C.RESET}"
                print(f"  {C.DIM}│{C.RESET} 任务: {result_str}")
            elif k == "target":
                print(f"  {C.DIM}│{C.RESET} 刺杀目标: {C.RED}P{v + 1}{C.RESET}")
            elif k == "content":
                pass  # 已在发言中显示
            else:
                print(f"  {C.DIM}│{C.RESET} {k}: {v}")

    # 详细模式：显示完整 LLM 输入
    if verbose and turn.llm_input:
        print(f"\n  {C.DIM}{C.BOLD}📋 LLM 输入 (详细):{C.RESET}")
        messages = turn.llm_input.get("messages", [])
        for msg in messages:
            role_label = msg.get("role", "?").upper()
            content = msg.get("content", "")
            print(f"  {C.DIM}┌─ {role_label} ─────────────────{C.RESET}")
            print(_wrap(content, indent=f"  {C.DIM}│ {C.RESET}", width=78))
            print(f"  {C.DIM}└────────────────────────{C.RESET}")


def print_round_event(event_type: str, data: Dict[str, Any]):
    """打印回合级别事件（投票结果、任务结果等）。"""
    if event_type == "vote_result":
        approved = data.get("approved", False)
        votes = data.get("votes", {})
        round_num = data.get("round", "?")
        attempt = data.get("attempt", "?")

        status = f"{C.GREEN}通过 ✓{C.RESET}" if approved else f"{C.RED}否决 ✗{C.RESET}"
        approve_count = sum(1 for v in votes.values() if v)
        reject_count = len(votes) - approve_count

        print(f"\n  {C.BOLD}📊 投票结果{C.RESET} (第{round_num}轮 第{attempt}次): {status}")
        print(f"     赞成 {C.GREEN}{approve_count}{C.RESET} / 反对 {C.RED}{reject_count}{C.RESET}")
        vote_detail = []
        for seat_str, v in sorted(votes.items(), key=lambda x: int(x[0])):
            seat = int(seat_str)
            mark = f"{C.GREEN}✓{C.RESET}" if v else f"{C.RED}✗{C.RESET}"
            vote_detail.append(f"P{seat + 1}:{mark}")
        print(f"     {' '.join(vote_detail)}")

    elif event_type == "quest_result":
        success = data.get("success")
        fails = data.get("fail_votes", 0)
        round_num = data.get("round", "?")
        team = data.get("team_members", [])

        status = f"{C.GREEN}成功 ✓{C.RESET}" if success else f"{C.RED}失败 ✗{C.RESET}"
        team_display = ", ".join(f"P{s + 1}" for s in team)

        print(f"\n  {C.BOLD}⚔️  任务结果{C.RESET} (第{round_num}轮): {status}")
        print(f"     队伍: [{team_display}]")
        print(f"     破坏票数: {C.RED}{fails}{C.RESET}")

    elif event_type == "assassination":
        target = data.get("target")
        is_merlin = data.get("is_merlin", False)

        if is_merlin:
            print(f"\n  {C.BG_RED}{C.WHITE}{C.BOLD} 🗡️  刺杀成功！梅林被找到了！(P{target + 1}) {C.RESET}")
        else:
            print(f"\n  {C.BG_GREEN}{C.WHITE}{C.BOLD} 🛡️  刺杀失败！梅林安全了！(目标: P{target + 1}) {C.RESET}")


def print_game_result(record: GameRecord):
    """打印游戏最终结果。"""
    print(f"\n{_section('游戏结束')}")

    if record.winner == "good":
        print(f"\n  {C.BG_BLUE}{C.WHITE}{C.BOLD}  🏆 好人阵营获胜！  {C.RESET}")
    elif record.winner == "evil":
        print(f"\n  {C.BG_RED}{C.WHITE}{C.BOLD}  💀 坏人阵营获胜！  {C.RESET}")
    else:
        print(f"\n  {C.YELLOW}  ❓ 游戏未正常结束  {C.RESET}")

    # 任务回顾
    print(f"\n  {C.BOLD}任务回顾:{C.RESET}")
    for i, qr in enumerate(record.quest_results):
        icon = f"{C.GREEN}✓{C.RESET}" if qr.get("success") else f"{C.RED}✗{C.RESET}"
        team_str = ", ".join(f"P{s+1}" for s in qr.get("team_members", []))
        fails = qr.get("fail_votes", 0)
        fail_info = f" ({C.RED}{fails}张破坏票{C.RESET})" if fails > 0 else ""
        print(f"    第{i + 1}轮 {icon} 队伍[{team_str}]{fail_info}")

    # 统计
    total_turns = len(record.turns)
    total_votes = len(record.vote_history)
    print(f"\n  {C.BOLD}统计:{C.RESET}")
    print(f"    总回合数: {total_turns}")
    print(f"    投票次数: {total_votes}")
    rejected = sum(1 for v in record.vote_history if not v.get("approved"))
    print(f"    否决次数: {rejected}")
    print()


def print_scoreboard(record: GameRecord):
    """打印各玩家评分概览。"""
    print(f"\n{_section('玩家行为概览')}")

    for p in record.players:
        seat = p["seat"]
        role_name, role_color = ROLE_DISPLAY.get(p["role"], (p["role"], C.WHITE))
        team_label = "好人" if p["team"] == "good" else "坏人"

        player_turns = [t for t in record.turns if t.seat == seat]
        discussions = [t for t in player_turns if t.phase in ("discussion", "leader_discussion")]
        votes = [t for t in player_turns if t.phase == "team_vote"]
        quests = [t for t in player_turns if t.phase == "quest_execution"]

        approve_count = sum(1 for t in votes if t.action.get("approve"))
        reject_count = len(votes) - approve_count

        print(f"\n  {role_color}{C.BOLD}P{seat + 1} {p['name']} [{role_name}]{C.RESET}")
        print(f"    发言次数: {len(discussions)}  │  投票: {C.GREEN}{approve_count}赞{C.RESET}/{C.RED}{reject_count}反{C.RESET}  │  执行任务: {len(quests)}次")

        if quests and p["team"] == "evil":
            sabotages = sum(1 for t in quests if not t.action.get("success", True))
            print(f"    破坏任务: {C.RED}{sabotages}{C.RESET}次")
    print()


# ── 游戏执行 ──────────────────────────────────────────────────────────────────

async def run_inspect_game(
    model_name: str,
    provider_name: str,
    player_count: int = 5,
    seed: Optional[int] = None,
    focus_seat: Optional[int] = None,
    focus_role: Optional[str] = None,
    verbose: bool = False,
    raw: bool = False,
    save_path: Optional[str] = None,
) -> GameRecord:
    """运行一局游戏并实时打印每个 turn 的详情。"""
    from game.rollout import GameRollout
    from game.roles import is_evil, get_team
    from server.llm.player import LLMPlayerManager
    from server.models.schemas import PlayerConfig
    from server.storage.memory_repository import InMemoryRepository

    record = GameRecord(seed=seed)

    player_configs = [
        {
            "seat": i,
            "name": f"Player{i + 1}",
            "is_human": False,
            "model": model_name,
            "provider": provider_name,
        }
        for i in range(player_count)
    ]

    rollout = GameRollout(player_count=player_count, seed=seed)
    first_seat, first_phase = rollout.create_and_start(player_configs)

    # 记录玩家信息
    for p in rollout.state.players:
        record.players.append({
            "seat": p.seat,
            "name": p.name,
            "role": p.role.value if p.role else "unknown",
            "team": p.team.value if p.team else "unknown",
            "model": model_name,
        })

    # 判断是否聚焦某个 seat
    focus_seats = set()
    if focus_seat is not None:
        focus_seats.add(focus_seat)
    if focus_role is not None:
        for p in rollout.state.players:
            if p.role and p.role.value == focus_role:
                focus_seats.add(p.seat)

    # 打印初始设置
    print_game_setup(record)
    if focus_seats:
        focused = ", ".join(f"P{s + 1}" for s in sorted(focus_seats))
        print(f"  {C.YELLOW}{C.BOLD}🔍 聚焦玩家: {focused}{C.RESET}\n")

    # 创建 LLM Player Manager
    llm_manager = LLMPlayerManager()
    for player in rollout.state.players:
        if not player.is_human:
            llm_manager.add_player(player)
    await llm_manager.initialize_all(rollout.state)

    repo = InMemoryRepository()

    # 游戏循环
    seat, phase = first_seat, first_phase
    turn_num = 0
    prev_round = 1
    prev_vote_attempt = 1

    while not rollout.is_finished:
        state = rollout.state
        turn_num += 1

        # 检测新回合
        if state.current_round != prev_round:
            print(f"\n{_section(f'第 {state.current_round} 轮', char='═')}")
            prev_round = state.current_round
            prev_vote_attempt = 1

        player = state.get_player(seat)
        if not player:
            break

        role_val = player.role.value if player.role else "unknown"
        team_val = player.team.value if player.team else "unknown"

        turn = TurnRecord(
            turn_num=turn_num,
            seat=seat,
            player_name=player.name,
            role=role_val,
            team=team_val,
            phase=phase,
        )

        llm_player = llm_manager.get_player(seat)
        if not llm_player:
            break

        should_show = not focus_seats or seat in focus_seats

        # 执行各阶段逻辑
        if phase in ("leader_discussion", "discussion"):
            if phase == "leader_discussion":
                call_result = await llm_player.discuss_as_leader(state)
            else:
                call_result = await llm_player.discuss(state)

            content = call_result.result
            turn.llm_input = call_result.llm_input
            turn.llm_output = call_result.llm_output
            turn.action = {"content": content}
            rollout.execute_structured_action(seat, phase, content=content)

        elif phase == "team_selection":
            team, speech, llm_input, llm_output = await llm_player.select_team_final(state)
            turn.llm_input = llm_input
            turn.llm_output = llm_output
            turn.action = {"team": team, "content": speech}
            rollout.execute_structured_action(seat, phase, team=team, content=speech)

        elif phase == "team_vote":
            vote_result = await llm_player.vote(state)
            approve = vote_result.result
            turn.llm_input = vote_result.llm_input
            turn.llm_output = vote_result.llm_output
            turn.action = {"approve": approve}
            rollout.execute_structured_action(seat, phase, approve=approve)

        elif phase == "quest_execution":
            if player and not is_evil(player.role):
                turn.action = {"success": True}
                turn.llm_output = {"content": "(好人自动成功)"}
                rollout.execute_structured_action(seat, phase, success=True)
            else:
                quest_result = await llm_player.execute_quest(state)
                success = quest_result.result
                turn.llm_input = quest_result.llm_input
                turn.llm_output = quest_result.llm_output
                turn.action = {"success": success}
                rollout.execute_structured_action(seat, phase, success=success)

        elif phase == "assassination_discussion":
            call_result = await llm_player.discuss_assassination(state)
            content = call_result.result
            turn.llm_input = call_result.llm_input
            turn.llm_output = call_result.llm_output
            turn.action = {"content": content}
            rollout.execute_structured_action(seat, phase, content=content)

        elif phase == "assassination":
            assassinate_result = await llm_player.assassinate(state)
            target = assassinate_result.result
            turn.llm_input = assassinate_result.llm_input
            turn.llm_output = assassinate_result.llm_output
            turn.action = {"target": target}
            rollout.execute_structured_action(seat, phase, target=target)

        record.turns.append(turn)

        if should_show:
            print_turn(turn, verbose=verbose, raw=raw)

        if rollout.is_finished:
            # 刺杀结果
            if phase == "assassination":
                target_player = state.get_player(target)
                is_merlin = target_player and target_player.role and target_player.role.value == "merlin"
                print_round_event("assassination", {
                    "target": target,
                    "is_merlin": is_merlin,
                })
            break

        # 推进到下一个决策点
        prev_phase = phase
        prev_seat = seat
        next_result = rollout.advance_to_next_decision()
        seat, phase = next_result

        if seat is None:
            break

        # 检测投票结果事件
        if prev_phase == "team_vote" and phase != "team_vote":
            if state.vote_history:
                last_vote = state.vote_history[-1]
                print_round_event("vote_result", {
                    "round": last_vote.round,
                    "attempt": last_vote.attempt,
                    "approved": last_vote.approved,
                    "votes": {str(s): v for s, v in last_vote.votes.items()},
                })

        # 检测任务结果事件
        if prev_phase == "quest_execution" and phase != "quest_execution":
            if state.quest_results:
                last_quest = state.quest_results[-1]
                print_round_event("quest_result", {
                    "round": last_quest.round,
                    "success": last_quest.success,
                    "fail_votes": last_quest.fail_votes,
                    "team_members": last_quest.team_members,
                })

    # 记录最终结果
    record.winner = rollout.winner.value if rollout.winner else None
    record.assassinated = rollout.state.assassinated_player

    for qr in rollout.state.quest_results:
        record.quest_results.append({
            "round": qr.round,
            "success": qr.success,
            "fail_votes": qr.fail_votes,
            "team_members": qr.team_members,
        })

    for vh in rollout.state.vote_history:
        record.vote_history.append({
            "round": vh.round,
            "attempt": vh.attempt,
            "approved": vh.approved,
            "votes": {str(s): v for s, v in vh.votes.items()},
        })

    # 打印总结
    print_scoreboard(record)
    print_game_result(record)

    # 保存到文件
    if save_path:
        save_data = {
            "meta": {
                "model": model_name,
                "provider": provider_name,
                "player_count": player_count,
                "seed": seed,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            "players": record.players,
            "turns": [
                {
                    "turn_num": t.turn_num,
                    "seat": t.seat,
                    "player_name": t.player_name,
                    "role": t.role,
                    "team": t.team,
                    "phase": t.phase,
                    "llm_input": t.llm_input,
                    "llm_output": t.llm_output,
                    "action": t.action,
                }
                for t in record.turns
            ],
            "quest_results": record.quest_results,
            "vote_history": record.vote_history,
            "winner": record.winner,
            "assassinated_player": record.assassinated,
        }

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"  {C.GREEN}完整记录已保存到: {save_path}{C.RESET}\n")

    return record


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="单局游戏审查工具：查看模型在 Avalon 游戏中的完整输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group_model = parser.add_argument_group("模型配置")
    group_model.add_argument(
        "--model", type=str, default=None,
        help="模型名称（已启动的 vLLM served-model-name 或其他 provider 的模型名）",
    )
    group_model.add_argument(
        "--provider", type=str, default=None,
        help="Provider 名称（如 vllm, openai, deepseek）",
    )
    group_model.add_argument(
        "--model_path", type=str, default=None,
        help="模型路径（自动启动 vLLM 服务）",
    )
    group_model.add_argument(
        "--port", type=int, default=8000,
        help="vLLM 服务端口（默认 8000）",
    )
    group_model.add_argument(
        "--tp", type=int, default=1,
        help="vLLM tensor-parallel size（默认 1）",
    )
    group_model.add_argument(
        "--gpu_util", type=float, default=0.85,
        help="vLLM GPU 显存占用率（默认 0.85）",
    )
    group_model.add_argument(
        "--devices", type=str, default=None,
        help="CUDA_VISIBLE_DEVICES（如 '0,1,2,3'）",
    )
    group_model.add_argument(
        "--reasoning_parser", type=str, default=None,
        help="vLLM reasoning parser（如 'deepseek_r1'）",
    )

    group_game = parser.add_argument_group("游戏配置")
    group_game.add_argument(
        "--player_count", type=int, default=5,
        help="玩家数量（默认 5）",
    )
    group_game.add_argument(
        "--seed", type=int, default=None,
        help="随机种子（可复现游戏）",
    )

    group_filter = parser.add_argument_group("输出过滤")
    group_filter.add_argument(
        "--focus_seat", type=int, default=None,
        help="只显示指定座位的输出（0-indexed）",
    )
    group_filter.add_argument(
        "--focus_role", type=str, default=None,
        help="只显示指定角色的输出（如 merlin, assassin）",
    )
    group_filter.add_argument(
        "--verbose", action="store_true",
        help="显示完整 LLM 输入（包含 system prompt 和 user prompt）",
    )
    group_filter.add_argument(
        "--raw", action="store_true",
        help="显示模型原始输出（完整 JSON，不做格式化拆分）",
    )
    group_filter.add_argument(
        "--no_color", action="store_true",
        help="禁用彩色输出",
    )

    group_output = parser.add_argument_group("输出保存")
    group_output.add_argument(
        "--save", type=str, default=None,
        help="保存完整游戏记录到 JSON 文件",
    )

    args = parser.parse_args()

    if args.no_color:
        C.disable()

    # 确定模型和 provider
    model_name = args.model
    provider_name = args.provider
    vllm_proc = None

    if args.model_path:
        # 自动启动 vLLM
        model_name = model_name or "inspect_model"
        provider_name = "vllm_inspect"

        os.environ["AVAILABLE_MODELS"] = f"{model_name}:{provider_name}"
        os.environ["VLLM_INSPECT_API_KEY"] = "vllm"
        os.environ["VLLM_INSPECT_BASE_URL"] = f"http://localhost:{args.port}/v1"

        from training.eval.evaluate import start_vllm, stop_vllm
        vllm_proc = start_vllm(
            model_path=args.model_path,
            served_name=model_name,
            port=args.port,
            tp=args.tp,
            gpu_util=args.gpu_util,
            reasoning_parser=args.reasoning_parser,
            cuda_devices=args.devices,
        )
    elif model_name and provider_name:
        # 使用已有服务，确保环境变量已设置
        env_prefix = provider_name.upper()
        if not os.environ.get(f"{env_prefix}_API_KEY"):
            if "vllm" in provider_name.lower():
                os.environ[f"{env_prefix}_API_KEY"] = "vllm"
                if not os.environ.get(f"{env_prefix}_BASE_URL"):
                    os.environ[f"{env_prefix}_BASE_URL"] = f"http://localhost:{args.port}/v1"

        existing = os.environ.get("AVAILABLE_MODELS", "")
        entry = f"{model_name}:{provider_name}"
        if entry not in existing:
            os.environ["AVAILABLE_MODELS"] = f"{existing},{entry}" if existing else entry
    else:
        parser.error("请指定 --model + --provider，或 --model_path（自动启动 vLLM）")

    # 重新加载 settings
    from importlib import reload
    import server.config
    reload(server.config)
    from server.config import settings as _  # noqa: F401

    print(f"\n{C.BOLD}{C.CYAN}{'═' * 90}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  🔍 Avalon 单局游戏审查工具{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 90}{C.RESET}")
    print(f"  模型: {C.YELLOW}{model_name}{C.RESET}")
    print(f"  Provider: {C.YELLOW}{provider_name}{C.RESET}")
    if args.seed is not None:
        print(f"  种子: {C.YELLOW}{args.seed}{C.RESET}")

    try:
        record = asyncio.run(run_inspect_game(
            model_name=model_name,
            provider_name=provider_name,
            player_count=args.player_count,
            seed=args.seed,
            focus_seat=args.focus_seat,
            focus_role=args.focus_role,
            verbose=args.verbose,
            raw=args.raw,
            save_path=args.save,
        ))
    finally:
        if vllm_proc:
            from training.eval.evaluate import stop_vllm
            stop_vllm(vllm_proc, args.port)


if __name__ == "__main__":
    main()
