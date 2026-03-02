"""Incremental observation builder for multi-turn game context.

Instead of reconstructing the full game state at each turn (prompts.py),
this module builds compact incremental observations that only include
NEW information since the player's last action.

Conversation format (tool-calling protocol):
    [system]:    game rules + role info (reuses prompts.py)
    [user]:      initial observation + first phase instruction
    [assistant]: {tool_calls: [{speak, ...}]}
    [tool]:      environment feedback (events only, no instructions)
    [user]:      next phase instruction
    [assistant]: {tool_calls: [{vote_team, ...}]}
    [tool]:      environment feedback (events only)
    [user]:      next phase instruction
    ...

[tool] responses only contain environment events (what happened).
[user] messages contain action directives (what to do next).
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from game.state import GameState, Player
from game.roles import Role, is_evil
from game.prompts import build_system_prompt, _build_phase_instructions as _raw_phase_instructions


@dataclass
class ObservationTracker:
    """Tracks what a player has already seen to compute incremental deltas."""

    quest_results_seen: int = 0
    vote_history_seen: int = 0
    discussions_seen: int = 0
    assassination_discussions_seen: int = 0
    last_proposed_team: Optional[List[int]] = None
    last_round: int = 0
    initialized: bool = False


def _build_phase_instructions(
    state: GameState,
    player: Player,
    visible_evil: List[int],
    phase: str,
    use_tools: bool,
) -> str:
    """Phase instructions adapted for multi-turn mode.

    Strips:
    - update_memory references (conversation history IS the memory)
    - "同伴的分析" / "同伴的分析总结" sections (assassination
      discussions are delivered via tool responses, not repeated here)
    """
    text = _raw_phase_instructions(state, player, visible_evil, phase, use_tools)
    lines = text.split("\n")
    filtered = []
    in_companion_section = False
    for line in lines:
        if "update_memory" in line:
            continue
        if line.startswith("### 同伴的分析"):
            in_companion_section = True
            continue
        if in_companion_section:
            if line.startswith("- 玩家") or line.strip() == "":
                continue
            in_companion_section = False
        filtered.append(line)
    return "\n".join(filtered)


def build_system_prompt_v2(
    player: Player,
    visible_evil: List[int],
    all_players: List[Player],
) -> str:
    """Build system prompt with a multi-turn conversation note appended."""
    base = build_system_prompt(player, visible_evil, all_players)
    base += """
## 对话模式
本局游戏以多轮对话进行。每次你行动后，你会收到新发生事件的更新。你之前的发言和分析都保留在对话中，可以直接参考。
"""
    return base


def build_observation(
    state: GameState,
    player: Player,
    visible_evil: List[int],
    phase: str,
    tracker: ObservationTracker,
    use_tools: bool = True,
) -> Tuple[str, str]:
    """Build observation and phase instruction separately.

    Returns:
        (events, instruction) — *events* is the environment feedback
        (what happened), *instruction* is the action directive for
        the next phase (what to do).
    """
    if not tracker.initialized:
        return _build_initial(state, player, visible_evil, phase, tracker, use_tools)
    return _build_incremental(state, player, visible_evil, phase, tracker, use_tools)


# ---------------------------------------------------------------------------
# Initial observation (first message of the game for this player)
# ---------------------------------------------------------------------------


def _build_initial(
    state: GameState,
    player: Player,
    visible_evil: List[int],
    phase: str,
    tracker: ObservationTracker,
    use_tools: bool,
) -> Tuple[str, str]:
    parts: List[str] = []

    parts.append("游戏开始！角色已分配完毕。\n")

    team_size = state.rules.quest_team_sizes[state.current_round - 1]
    leader = state.get_leader()
    parts.append(
        f"第{state.current_round}轮开始，"
        f"队长是玩家{state.current_leader + 1}({leader.name})。"
    )
    parts.append(f"本轮需要选择 {team_size} 名队员执行任务。")

    # Discussions that already happened before this player's first turn
    prior = [m for m in state.discussion_history if m.seat != player.seat]
    if prior:
        parts.append("")
        for msg in prior:
            parts.append(
                f"💬 玩家{msg.seat + 1}({msg.player_name}): {msg.content}"
            )

    instruction = _build_phase_instructions(
        state, player, visible_evil, phase, use_tools
    )

    # Sync tracker
    tracker.initialized = True
    tracker.discussions_seen = len(state.discussion_history)
    tracker.last_round = state.current_round
    tracker.last_proposed_team = (
        list(state.proposed_team) if state.proposed_team else None
    )
    return "\n".join(parts), instruction


# ---------------------------------------------------------------------------
# Incremental observation (all subsequent turns)
# ---------------------------------------------------------------------------


def _build_incremental(
    state: GameState,
    player: Player,
    visible_evil: List[int],
    phase: str,
    tracker: ObservationTracker,
    use_tools: bool,
) -> Tuple[str, str]:
    events: List[str] = []

    # Chronological order: oldest events first
    _collect_vote_results(state, tracker, events)
    _collect_quest_results(state, tracker, events)
    _collect_round_transition(state, tracker, events)
    _collect_new_discussions(state, player, tracker, events)
    _collect_team_proposal(state, player, visible_evil, tracker, events)
    _collect_assassination_discussions(state, player, tracker, events)

    instruction = _build_phase_instructions(
        state, player, visible_evil, phase, use_tools
    )

    return "\n".join(events), instruction


# ---------------------------------------------------------------------------
# Event collectors (mutate *events* list and *tracker* in place)
# ---------------------------------------------------------------------------


def _collect_new_discussions(
    state: GameState,
    player: Player,
    tracker: ObservationTracker,
    events: List[str],
) -> None:
    new = state.discussion_history[tracker.discussions_seen :]
    for msg in new:
        if msg.seat == player.seat:
            continue
        events.append(
            f"💬 玩家{msg.seat + 1}({msg.player_name}): {msg.content}"
        )
    tracker.discussions_seen = len(state.discussion_history)


def _collect_team_proposal(
    state: GameState,
    player: Player,
    visible_evil: List[int],
    tracker: ObservationTracker,
    events: List[str],
) -> None:
    if not state.proposed_team:
        return
    if state.proposed_team == tracker.last_proposed_team:
        return

    team_str = ", ".join(f"玩家{s + 1}" for s in state.proposed_team)
    leader = state.get_leader()
    events.append(
        f"\n📋 队长玩家{state.current_leader + 1}({leader.name})"
        f"选择了 [{team_str}] 执行任务。"
    )

    if is_evil(player.role):
        evil_count = sum(
            1 for s in state.proposed_team if s in visible_evil or s == player.seat
        )
        events.append(f"  【己方分析】队伍中有{evil_count}个坏人")
    elif player.role == Role.MERLIN:
        evil_count = sum(1 for s in state.proposed_team if s in visible_evil)
        events.append(f"  【梅林视野】队伍中有{evil_count}个坏人")

    tracker.last_proposed_team = list(state.proposed_team)


def _collect_vote_results(
    state: GameState,
    tracker: ObservationTracker,
    events: List[str],
) -> None:
    new_votes = state.vote_history[tracker.vote_history_seen :]
    for vote in new_votes:
        label = "✅ 通过" if vote.approved else "❌ 否决"
        approvers = [f"玩家{s + 1}" for s, v in vote.votes.items() if v]
        rejecters = [f"玩家{s + 1}" for s, v in vote.votes.items() if not v]

        events.append(f"\n📊 投票结果: {label}")
        events.append(f"  赞成: {', '.join(approvers) if approvers else '无'}")
        events.append(f"  反对: {', '.join(rejecters) if rejecters else '无'}")

        if not vote.approved:
            next_seat = (vote.leader + 1) % state.player_count
            next_player = state.get_player(next_seat)
            name = next_player.name if next_player else "?"
            events.append(
                f"  → 换队长为玩家{next_seat + 1}({name})"
            )
            tracker.last_proposed_team = None

    tracker.vote_history_seen = len(state.vote_history)


def _collect_quest_results(
    state: GameState,
    tracker: ObservationTracker,
    events: List[str],
) -> None:
    new_quests = state.quest_results[tracker.quest_results_seen :]
    for q in new_quests:
        if q.success:
            events.append(f"\n⚔️ 第{q.round}轮任务: ✅ 成功")
        else:
            events.append(
                f"\n⚔️ 第{q.round}轮任务: ❌ 失败（{q.fail_votes}张失败票）"
            )
        events.append(
            f"  当前比分: 好人 {state.good_wins} : {state.evil_wins} 坏人"
        )
    tracker.quest_results_seen = len(state.quest_results)


def _collect_round_transition(
    state: GameState,
    tracker: ObservationTracker,
    events: List[str],
) -> None:
    if state.current_round == tracker.last_round:
        return
    # Only show transition if a quest just completed (not on initial)
    if tracker.quest_results_seen == 0:
        tracker.last_round = state.current_round
        return

    leader = state.get_leader()
    team_size = state.rules.quest_team_sizes[state.current_round - 1]
    events.append(f"\n--- 第{state.current_round}轮开始 ---")
    events.append(f"队长: 玩家{state.current_leader + 1}({leader.name})")
    events.append(f"本轮需要选择 {team_size} 名队员执行任务。")
    tracker.last_round = state.current_round


def _collect_assassination_discussions(
    state: GameState,
    player: Player,
    tracker: ObservationTracker,
    events: List[str],
) -> None:
    new = state.assassination_discussion_history[
        tracker.assassination_discussions_seen :
    ]
    for msg in new:
        if msg.seat == player.seat:
            continue
        events.append(
            f"🗡️ 玩家{msg.seat + 1}({msg.player_name}): {msg.content}"
        )
    tracker.assassination_discussions_seen = len(
        state.assassination_discussion_history
    )
