"""Prompt templates for LLM players in Avalon."""

from typing import List, Optional
from server.game.state import GameState, Player, DiscussionMessage
from server.game.roles import Role, Team, get_role_name_cn, get_team, is_evil
from server.game.rules import get_rules


def build_system_prompt(player: Player, visible_evil: List[int], all_players: List[Player]) -> str:
    """Build the complete system prompt with game rules and player identity.
    
    This contains all static information that doesn't change during the game:
    - Game rules
    - Player's role and team
    - Known information (evil players for Merlin/evil team)
    - Player list
    """
    role_name = get_role_name_cn(player.role)
    team = "好人阵营" if get_team(player.role) == Team.GOOD else "坏人阵营"
    
    # Get game rules based on player count
    rules = get_rules(len(all_players))
    quest_info = []
    for i, size in enumerate(rules.quest_team_sizes):
        info = f"第{i+1}轮需{size}人"
        if rules.two_fails_required[i]:
            info += "(需2张失败票才能失败)"
        else:
            info += "(需1张失败票即可失败)"
        quest_info.append(info)
    quest_rules_str = "，".join(quest_info)
    
    prompt = f"""你是一个正在玩阿瓦隆(Avalon)桌游的玩家。

## 游戏规则
1. 好人阵营想要完成3个任务获胜，坏人阵营想要破坏3个任务或让好人5次投票失败
2. 每轮由队长选择一队人执行任务，所有玩家投票决定是否同意这个队伍
3. 本局游戏共{len(all_players)}人，每轮任务人数限制：{quest_rules_str}
4. 如果队伍被同意，队员执行任务。好人必须选择"成功"，坏人可以选择"成功"或"失败"
5. 如果任务中有对应轮次需要的失败票数（默认1张，若规则3中标注需2张则为2张），任务失败
6. 如果好人完成3个任务，刺客有机会刺杀梅林。如果刺杀成功，坏人获胜

## 特别提示
坏人除了破坏任务，还有一个目标是找出梅林，这样即使好人率先完成3个任务后，还有机会通过刺杀梅林获取胜利。

## 角色说明
- 梅林(Merlin)：好人，知道所有坏人身份，但不能暴露自己否则会被刺杀
- 忠臣(Loyal Servant)：好人，没有特殊能力
- 刺客(Assassin)：坏人，知道其他坏人身份，目标是在好人任务成功后，在游戏结束时正确刺杀梅林获得胜利。
- 爪牙(Minion)：坏人，知道其他坏人身份

## 你的身份
- 你是【玩家{player.seat + 1}】
- 角色：【{role_name}】
- 阵营：【{team}】
"""

    # Add known information based on role
    if visible_evil:
        evil_names = [f"玩家{s + 1}({all_players[s].name})" for s in visible_evil]
        if player.role == Role.MERLIN:
            prompt += f"\n### 梅林视野\n你知道以下玩家是坏人：{', '.join(evil_names)}。\n⚠️ 注意：你不能直接暴露自己是梅林，否则会被刺客刺杀！\n"
        elif is_evil(player.role):
            prompt += f"\n### 坏人同伴\n你的同伴是：{', '.join(evil_names)}。\n记住要互相配合，隐藏身份。\n"

    # Player list
    prompt += f"\n## 玩家列表\n游戏中共有{len(all_players)}名玩家，按顺序围坐一圈：\n"
    for p in all_players:
        marker = ""
        if p.seat == player.seat:
            marker = " 👈 你"
        elif p.seat in visible_evil:
            if is_evil(player.role):
                marker = " [同伴]"
            else:
                marker = " [坏人]"
        prompt += f"- 玩家{p.seat + 1}: {p.name}{marker}\n"

    prompt += """
## 行为准则
1. 根据自己的角色和阵营做出决策
2. 通过讨论分析其他玩家的言行
3. 好人要找出坏人，坏人要隐藏身份并误导好人
4. 发言要自然简洁，像一个真实玩家一样
"""
    
    return prompt


def build_user_prompt(
    state: GameState,
    player: Player,
    visible_evil: List[int],
    phase: str,
    current_memory: str = "",
) -> str:
    """Build the user prompt with dynamic game state.
    
    This contains all dynamic information:
    - Previous memory (accumulated knowledge across rounds)
    - Current round discussions (will be summarized into memory next round)
    - Historical quest results and votes
    - Current phase-specific instructions
    """
    prompt = ""
    
    # Section 1: Previous Memory
    if current_memory:
        prompt += f"""## 你的记忆
{current_memory}

"""

    # Section 2: Game History (Quest results + Vote patterns)
    prompt += _build_history_section(state)
    
    # Section 3: Current Round Discussions
    prompt += _build_current_round_discussions(state, player)
    
    # Section 4: Current Situation
    prompt += _build_current_situation(state, player, visible_evil)
    
    # Section 5: Phase-specific instructions
    prompt += _build_phase_instructions(state, player, visible_evil, phase)
    
    return prompt


def _build_history_section(state: GameState) -> str:
    """Build the historical quest results and vote patterns section."""
    if not state.quest_results and not state.vote_history:
        return ""
    
    prompt = "## 历史记录\n"
    
    # Quest results
    if state.quest_results:
        prompt += "### 任务结果\n"
        for q in state.quest_results:
            result = "✅ 成功" if q.success else f"❌ 失败（{q.fail_votes}张失败票）"
            team_str = ", ".join([f"玩家{m + 1}" for m in q.team_members])
            prompt += f"- 第{q.round}轮：[{team_str}] → {result}\n"
        prompt += "\n"
    
    # Vote history (summarized by round)
    if state.vote_history:
        prompt += "### 投票记录\n"
        for vote in state.vote_history:
            approvers = [f"玩家{s + 1}" for s, v in vote.votes.items() if v]
            rejecters = [f"玩家{s + 1}" for s, v in vote.votes.items() if not v]
            result = "通过" if vote.approved else "否决"
            team_str = ", ".join([f"玩家{m + 1}" for m in vote.proposed_team])
            prompt += f"- 第{vote.round}轮第{vote.attempt}次 [{team_str}]：{result}\n"
            prompt += f"  赞成：{', '.join(approvers) if approvers else '无'}\n"
            prompt += f"  反对：{', '.join(rejecters) if rejecters else '无'}\n"
        prompt += "\n"
    
    return prompt


def _build_current_round_discussions(state: GameState, player: Player) -> str:
    """Build the current round discussions section.
    
    Groups discussions by vote attempt and shows vote results between attempts.
    """
    if not state.discussion_history:
        return ""
    
    # Filter to current round only
    current_round_msgs = [d for d in state.discussion_history if d.round == state.current_round]
    
    if not current_round_msgs:
        return ""
    
    # Get vote results for current round
    current_round_votes = [v for v in state.vote_history if v.round == state.current_round]
    votes_by_attempt = {v.attempt: v for v in current_round_votes}
    
    # Group discussions by attempt
    attempts = sorted(set(msg.attempt for msg in current_round_msgs))
    
    prompt = "## 本轮讨论\n"
    
    for attempt in attempts:
        attempt_msgs = [m for m in current_round_msgs if m.attempt == attempt]
        
        if len(attempts) > 1 or attempt > 1:
            prompt += f"\n### 第{attempt}次投票前的讨论\n"
        
        for msg in attempt_msgs:
            speaker = "你" if msg.seat == player.seat else f"玩家{msg.seat + 1}({msg.player_name})"
            prompt += f"- {speaker}: {msg.content}\n"
        
        # Show vote result for this attempt if it exists and is not the current attempt
        if attempt in votes_by_attempt and attempt < state.vote_attempt:
            vote = votes_by_attempt[attempt]
            result = "✅ 通过" if vote.approved else "❌ 否决"
            approvers = [f"玩家{s + 1}" for s, v in vote.votes.items() if v]
            rejecters = [f"玩家{s + 1}" for s, v in vote.votes.items() if not v]
            prompt += f"\n📊 **投票结果**: {result}\n"
            prompt += f"  赞成: {', '.join(approvers) if approvers else '无'}\n"
            prompt += f"  反对: {', '.join(rejecters) if rejecters else '无'}\n"
    
    prompt += "\n"
    
    return prompt


def _build_current_situation(state: GameState, player: Player, visible_evil: List[int]) -> str:
    """Build the current game situation section."""
    prompt = f"""## 当前局势
- 当前轮次：第{state.current_round}轮
- 任务比分：好人 {state.good_wins} : {state.evil_wins} 坏人
- 队长：玩家{state.current_leader + 1}({state.get_leader().name})
- 投票尝试：{state.vote_attempt}/5
"""
    
    if state.proposed_team:
        team_str = ", ".join([f"玩家{s + 1}" for s in state.proposed_team])
        prompt += f"- 提议队伍：[{team_str}]\n"
        
        # Role-specific team analysis
        if is_evil(player.role):
            evil_on_team = [s for s in state.proposed_team if s in visible_evil or s == player.seat]
            prompt += f"- 【己方分析】队伍中有{len(evil_on_team)}个坏人\n"
        elif player.role == Role.MERLIN:
            evil_on_team = [s for s in state.proposed_team if s in visible_evil]
            prompt += f"- 【梅林视野】队伍中有{len(evil_on_team)}个坏人\n"
    
    prompt += "\n"
    return prompt


def _build_phase_instructions(
    state: GameState,
    player: Player,
    visible_evil: List[int],
    phase: str,
) -> str:
    """Build phase-specific instructions."""
    
    if phase == "team_selection":
        return _get_team_selection_instructions(state, player, visible_evil)
    elif phase == "team_selection_final":
        return _get_team_selection_final_instructions(state, player, visible_evil)
    elif phase == "leader_discussion":
        return _get_leader_discussion_instructions(state, player, visible_evil)
    elif phase == "discussion":
        return _get_discussion_instructions(state, player, visible_evil)
    elif phase == "team_vote":
        return _get_vote_instructions(state, player, visible_evil)
    elif phase == "quest_execution":
        return _get_quest_instructions(state, player, visible_evil)
    elif phase == "assassination_discussion":
        return _get_assassination_discussion_instructions(state, player, visible_evil)
    elif phase == "assassination":
        return _get_assassination_instructions(state, player, visible_evil)
    else:
        return ""


def _get_leader_discussion_instructions(state: GameState, player: Player, visible_evil: List[int]) -> str:
    """Instructions for leader's discussion phase (proposing a team)."""
    team_size = state.rules.quest_team_sizes[state.current_round - 1]
    
    prompt = f"""## 行动：作为队长发言
你是本轮队长，需要选择 **{team_size}** 名队员执行任务。
现在是讨论阶段，你需要先发言，提出你初步考虑的队伍配置，并说明理由。
在大家讨论完之后，你可以根据讨论情况调整最终的队伍选择。

可选玩家：
"""
    for p in state.players:
        marker = ""
        if p.seat == player.seat:
            marker = "（你自己）"
        elif p.seat in visible_evil:
            if is_evil(player.role):
                marker = "（同伴）"
            else:
                marker = "（坏人）"
        prompt += f"- 玩家{p.seat + 1}: {p.name} {marker}\n"
    
    prompt += """
请调用 `speak` 工具发言，说明你初步考虑的队伍配置和理由。
请调用 `update_memory` 工具记录你对局势的分析、各玩家身份推断和策略计划。

注意：现在只是讨论阶段的发言，最终队伍选择会在讨论结束后进行。"""
    
    return prompt


def _get_team_selection_final_instructions(state: GameState, player: Player, visible_evil: List[int]) -> str:
    """Instructions for final team selection after discussion."""
    team_size = state.rules.quest_team_sizes[state.current_round - 1]
    
    prompt = f"""## 行动：确定最终队伍
讨论已经结束，你是本轮队长，现在需要确定最终的 **{team_size}** 名队员。

可选玩家：
"""
    for p in state.players:
        marker = ""
        if p.seat == player.seat:
            marker = "（你自己）"
        elif p.seat in visible_evil:
            if is_evil(player.role):
                marker = "（同伴）"
            else:
                marker = "（坏人）"
        prompt += f"- 玩家{p.seat + 1}: {p.name} {marker}\n"
    
    prompt += """
根据刚才的讨论，决定最终队伍配置。你可以：
- 坚持你之前提议的队伍
- 根据讨论情况调整队伍人选

请调用 `propose_team` 工具选择最终队员。
请调用 `speak` 工具做总结发言，说明你的最终决定和理由。
请调用 `update_memory` 工具记录你的决策理由。"""
    
    return prompt


def _get_team_selection_instructions(state: GameState, player: Player, visible_evil: List[int]) -> str:
    """Instructions for team selection phase (legacy, for compatibility)."""
    team_size = state.rules.quest_team_sizes[state.current_round - 1]
    
    prompt = f"""## 行动：选择队伍并说明理由
你是本轮队长，需要选择 **{team_size}** 名队员执行任务，并向大家解释你的选择理由。

可选玩家：
"""
    for p in state.players:
        marker = ""
        if p.seat == player.seat:
            marker = "（你自己）"
        elif p.seat in visible_evil:
            if is_evil(player.role):
                marker = "（同伴）"
            else:
                marker = "（坏人）"
        prompt += f"- 玩家{p.seat + 1}: {p.name} {marker}\n"
    
    prompt += """
请调用 `propose_team` 工具选择队员。
请调用 `speak` 工具向大家解释你选择这个队伍的理由。
请调用 `update_memory` 工具记录你对局势的分析、各玩家身份推断和策略计划。"""
    
    return prompt


def _get_discussion_instructions(state: GameState, player: Player, visible_evil: List[int]) -> str:
    """Instructions for discussion phase."""
    prompt = """## 行动：发表看法
请对队长提议的队伍配置发表你的看法，支持或反对，并说明理由。

"""
    
    # Role-specific hints
    if player.role == Role.MERLIN:
        prompt += "💡 提示：你知道谁是坏人，但要小心不要太明显地暴露这一点。\n\n"
    elif is_evil(player.role):
        prompt += "💡 提示：考虑如何误导好人，隐藏自己的身份。\n\n"
    
    prompt += """请调用 `speak` 工具发表你的看法。
请调用 `update_memory` 工具记录你对局势的分析、各玩家身份推断和策略计划。"""
    
    return prompt


def _get_vote_instructions(state: GameState, player: Player, visible_evil: List[int]) -> str:
    """Instructions for voting phase."""
    prompt = """## 行动：投票
请决定是否同意当前提议的队伍执行任务。

请调用 `vote_team` 工具进行投票（approve: true/false）。
请调用 `update_memory` 工具记录投票原因、对局势的分析和策略计划。"""
    
    return prompt


def _get_quest_instructions(state: GameState, player: Player, visible_evil: List[int]) -> str:
    """Instructions for quest execution phase."""
    prompt = "## 行动：执行任务\n"
    
    if is_evil(player.role):
        evil_on_team = [s for s in state.proposed_team if s in visible_evil or s == player.seat]
        prompt += f"""你是坏人，可以选择让任务成功或失败。
队伍中共有{len(evil_on_team)}个坏人（包括你自己）。

考虑因素：
- 如果任务失败，可能会暴露身份
- 如果这是关键任务，失败可能帮助坏人获胜
- 当前坏人已经赢了{state.evil_wins}轮

"""
    else:
        prompt += """你是好人，必须选择让任务成功。

"""
    
    prompt += """请调用 `vote_quest` 工具决定任务结果（success: true/false）。
请调用 `update_memory` 工具记录任务决策理由和后续策略。"""
    
    return prompt


def _get_assassination_discussion_instructions(state: GameState, player: Player, visible_evil: List[int]) -> str:
    """Instructions for evil team discussion before assassination."""
    prompt = """## 行动：刺杀前讨论
好人完成了3个任务，但坏人阵营还有最后的机会！
现在是坏人阵营的私密讨论时间，你们需要一起分析谁最可能是梅林。

请根据你在「你的记忆」中积累的信息进行分析和判断。
"""
    
    # Show previous assassination discussion if any
    if state.assassination_discussion_history:
        prompt += "\n### 同伴的分析\n"
        for msg in state.assassination_discussion_history:
            prompt += f"- 玩家{msg.seat + 1}({msg.player_name}): {msg.content}\n"
    
    prompt += """
请调用 `speak` 工具发表你对谁是梅林的分析和判断。
请调用 `update_memory` 工具记录你的推理过程。

注意：这是坏人阵营的私密讨论，好人玩家看不到这些内容。请坦诚分享你的判断！"""
    
    return prompt


def _get_assassination_instructions(state: GameState, player: Player, visible_evil: List[int]) -> str:
    """Instructions for assassination phase."""
    prompt = """## 行动：刺杀梅林
好人完成了3个任务！但作为刺客，你有最后一次机会。
如果你能正确指认梅林，坏人将获得最终胜利！

"""
    
    # Show assassination discussion summary
    if state.assassination_discussion_history:
        prompt += "\n### 同伴的分析总结\n"
        for msg in state.assassination_discussion_history:
            prompt += f"- 玩家{msg.seat + 1}({msg.player_name}): {msg.content}\n"
        prompt += "\n"
    
    prompt += """
请调用 `assassinate` 工具选择刺杀目标（target: 玩家编号）。
请调用 `update_memory` 工具记录你的推理过程。"""
    
    return prompt


# Legacy function compatibility (can be removed if not used elsewhere)
def get_system_prompt() -> str:
    """Deprecated: Use build_system_prompt instead."""
    return """你是一个正在玩阿瓦隆(Avalon)桌游的玩家。"""


def get_role_reveal_prompt(player: Player, visible_evil: List[int], all_players: List[Player]) -> str:
    """Deprecated: Use build_system_prompt instead."""
    return build_system_prompt(player, visible_evil, all_players)
