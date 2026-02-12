"""Game rollout - 封装单局 Avalon 游戏的完整 rollout 逻辑。

将游戏创建、状态推进、动作解析、prompt 构建等核心逻辑抽象为可复用的类。
可以被 VERL multi-turn Interaction 和 batch runner 等不同场景引用。

使用示例（VERL 训练 - 文本模式）:
    rollout = GameRollout(player_count=5)
    first_seat, first_phase = rollout.create_and_start()
    system, user = rollout.build_prompt(first_seat, first_phase)

    # 循环：接收模型输出 -> 执行动作 -> 推进到下一个决策点
    rollout.execute_action(seat, phase, model_output_text)
    next_seat, next_phase = rollout.advance_to_next_decision()

使用示例（Batch Runner - 结构化模式，配合 LLMPlayer tool calling）:
    rollout = GameRollout(player_count=5)
    first_seat, first_phase = rollout.create_and_start(player_configs=[...])
    system, user = rollout.build_prompt(first_seat, first_phase, use_tools=True)

    # 循环：调用 LLM -> 执行结构化动作 -> 推进到下一个决策点
    rollout.execute_structured_action(seat, phase, content="...", team=[0,2], approve=True)
    next_seat, next_phase = rollout.advance_to_next_decision()
"""

import re
import random
import logging
from typing import List, Tuple, Optional, Dict, Any

from game.engine import GameEngine
from game.state import GameState, GamePhase, GameStatus
from game.roles import Team, Role, get_team, is_evil
from game.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


class GameRollout:
    """封装单局 Avalon 游戏的 rollout 逻辑。

    职责:
        - 游戏创建和启动
        - 推进游戏到下一个决策点
        - 解析文本输出或结构化结果并执行游戏动作
        - 为指定玩家构建 prompt
        - 计算奖励分数

    不包含:
        - LLM 调用（由外部驱动）
        - 数据库/Socket.IO 等 I/O
        - 多局管理
    """

    def __init__(self, player_count: int = 5, seed: Optional[int] = None):
        self.player_count = player_count
        self.seed = seed
        self.engine: Optional[GameEngine] = None

    @property
    def state(self) -> GameState:
        """获取当前游戏状态。"""
        assert self.engine is not None, "Game not created yet. Call create_and_start() first."
        return self.engine.state

    @property
    def is_finished(self) -> bool:
        """游戏是否已结束。"""
        return self.engine is not None and self.state.status == GameStatus.FINISHED

    @property
    def winner(self) -> Optional[Team]:
        """获胜方（游戏未结束时为 None）。"""
        return self.state.winner if self.engine else None

    # ========================================================================
    # 游戏生命周期
    # ========================================================================

    def create_and_start(
        self,
        player_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[int, str]:
        """创建并启动游戏，返回第一个决策点 (seat, phase)。

        Args:
            player_configs: 可选的玩家配置列表。每个 dict 可包含:
                seat, name, is_human, model, provider。
                如果为 None，则创建默认的 AI 玩家。

        Returns:
            (first_seat, first_phase) - 第一个需要做决策的玩家座位和阶段
        """
        if self.seed is not None:
            random.seed(self.seed)

        if player_configs is None:
            player_configs = [
                {"seat": i, "name": f"AI_{i+1}", "is_human": False}
                for i in range(self.player_count)
            ]

        self.engine = GameEngine.create_game(self.player_count, player_configs)
        self.engine.start_game()

        # 进入讨论阶段
        self.engine.proceed_to_discussion()
        first_speaker = self.engine.next_discussion_speaker()

        # 确定第一个 turn 的阶段
        if first_speaker == self.state.current_leader:
            first_phase = "leader_discussion"
        else:
            first_phase = "discussion"

        return first_speaker, first_phase

    # ========================================================================
    # Prompt 构建
    # ========================================================================

    def build_prompt(
        self,
        seat: int,
        phase: str,
        use_tools: bool = False,
        current_memory: str = "",
    ) -> Tuple[str, str]:
        """为指定玩家在当前阶段构建 (system, user) prompt 对。

        完全复用 game/prompts.py 中的统一 prompt 逻辑。

        Args:
            seat: 玩家座位号
            phase: 当前游戏阶段
            use_tools: True 生成 tool call 格式指令（server/batch 模式），
                       False 生成纯文本格式指令（VERL 训练模式）
            current_memory: 玩家的累积记忆（用于 server/batch 模式的 LLMPlayer）

        Returns:
            (system_prompt, user_prompt)
        """
        player = self.state.get_player(seat)
        visible_evil = self.state.get_visible_evil_players(seat)

        system = build_system_prompt(player, visible_evil, self.state.players)
        user = build_user_prompt(
            state=self.state,
            player=player,
            visible_evil=visible_evil,
            phase=phase,
            current_memory=current_memory,
            use_tools=use_tools,
        )
        return system, user

    # ========================================================================
    # 动作执行 — 文本模式（VERL 训练用）
    # ========================================================================

    def execute_action(self, seat: int, phase: str, content: str) -> bool:
        """解析模型文本输出并执行对应的游戏动作。

        根据当前阶段，将自然语言解析为具体的游戏操作。
        解析失败时会使用 fallback 策略（如随机选队、默认投赞成）。

        Args:
            seat: 执行动作的玩家座位号
            phase: 当前游戏阶段
            content: 模型的文本输出

        Returns:
            True 表示解析和执行成功，False 表示使用了 fallback
        """
        if phase in ("leader_discussion", "discussion"):
            self.engine.add_discussion(seat, content)
            return True

        elif phase == "team_selection":
            team = self.parse_team(content, self.state.player_count)
            required_size = self.engine.get_quest_team_size()

            if team and len(team) == required_size:
                success = self.engine.select_team(team)
                if success:
                    return True

            # 解析失败：随机选队
            logger.warning(
                f"Team parse failed for Player {seat + 1}, using random team. "
                f"Content: {content[:100]}"
            )
            random_team = self._random_team(required_size)
            self.engine.select_team(random_team)
            return False

        elif phase == "team_vote":
            approve = self.parse_vote(content)
            self.engine.cast_vote(seat, approve)
            return True

        elif phase == "quest_execution":
            player = self.state.get_player(seat)
            if player and not is_evil(player.role):
                # 好人必须选择成功
                self.engine.cast_quest_vote(seat, True)
                return True
            else:
                success = self.parse_quest_vote(content)
                self.engine.cast_quest_vote(seat, success)
                return True

        elif phase == "assassination_discussion":
            self.engine.add_assassination_discussion(seat, content)
            return True

        elif phase == "assassination":
            target = self.parse_assassination_target(content, self.state.player_count)
            if target is not None:
                self.engine.assassinate(target)
                return True
            else:
                # 解析失败：随机刺杀一个好人
                logger.warning(
                    f"Assassination target parse failed, using random target. "
                    f"Content: {content[:100]}"
                )
                good_players = [
                    p.seat for p in self.state.players
                    if p.role and get_team(p.role) == Team.GOOD
                ]
                if good_players:
                    self.engine.assassinate(random.choice(good_players))
                return False

        return False

    # ========================================================================
    # 动作执行 — 结构化模式（Batch Runner / Server 用）
    # ========================================================================

    def execute_structured_action(self, seat: int, phase: str, **kwargs) -> None:
        """使用已解析的结构化数据执行游戏动作。

        适用于 LLM tool calling 模式，结果已由 LLMPlayer 解析为结构化数据。

        Args:
            seat: 执行动作的玩家座位号
            phase: 当前游戏阶段
            **kwargs: 阶段特定参数:
                - discussion/leader_discussion: content (str)
                - team_selection: team (List[int]), content (str, optional speech)
                - team_vote: approve (bool)
                - quest_execution: success (bool)
                - assassination_discussion: content (str)
                - assassination: target (int, seat index)
        """
        if phase in ("leader_discussion", "discussion"):
            self.engine.add_discussion(seat, kwargs.get("content", ""))

        elif phase == "team_selection":
            team = kwargs["team"]
            # 如果有队长总结发言，先添加讨论
            speech = kwargs.get("content")
            if speech:
                self.engine.add_discussion(seat, speech)
            self.engine.select_team(team)

        elif phase == "team_vote":
            self.engine.cast_vote(seat, kwargs["approve"])

        elif phase == "quest_execution":
            self.engine.cast_quest_vote(seat, kwargs["success"])

        elif phase == "assassination_discussion":
            self.engine.add_assassination_discussion(seat, kwargs.get("content", ""))

        elif phase == "assassination":
            self.engine.assassinate(kwargs["target"])

    # ========================================================================
    # 游戏推进
    # ========================================================================

    def advance_to_next_decision(self) -> Tuple[Optional[int], Optional[str]]:
        """推进游戏到下一个决策点。

        处理需要收集所有玩家动作后才能推进的阶段（投票、任务执行），
        自动解析中间结果，直到找到下一个需要模型输入的决策点。

        Returns:
            (seat, phase) - 下一个决策点的玩家座位和阶段
            (None, None) - 游戏结束
        """
        state = self.state

        # 讨论阶段：获取下一个发言者
        if state.phase == GamePhase.DISCUSSION:
            next_speaker = self.engine.next_discussion_speaker()
            if next_speaker is not None:
                if next_speaker == state.current_leader:
                    return next_speaker, "leader_discussion"
                else:
                    return next_speaker, "discussion"
            else:
                # 讨论结束，进入组队阶段
                self.engine.proceed_to_team_selection()
                return state.current_leader, "team_selection"

        # 投票阶段：逐个玩家投票
        elif state.phase == GamePhase.TEAM_VOTE:
            for p in state.players:
                if p.seat not in state.current_votes:
                    return p.seat, "team_vote"

            # 所有投票已收集，解析投票结果
            self.engine.resolve_vote()

            if state.status == GameStatus.FINISHED:
                return None, None  # 5 次投票失败，坏人获胜

            # 投票通过或否决后，继续推进
            return self.advance_to_next_decision()

        # 任务执行阶段：逐个队员投票
        elif state.phase == GamePhase.QUEST_EXECUTION:
            for seat in state.proposed_team:
                if seat not in state.current_quest_votes:
                    return seat, "quest_execution"

            # 所有任务投票已收集，解析任务结果
            self.engine.resolve_quest()

            if state.status == GameStatus.FINISHED:
                return None, None  # 坏人赢了 3 轮

            # 检查是否进入刺杀讨论阶段
            if state.phase == GamePhase.ASSASSINATION_DISCUSSION:
                return self.advance_to_next_decision()

            # 继续下一轮
            return self.advance_to_next_decision()

        # 刺杀讨论阶段
        elif state.phase == GamePhase.ASSASSINATION_DISCUSSION:
            next_speaker = self.engine.next_assassination_discussion_speaker()
            if next_speaker is not None:
                return next_speaker, "assassination_discussion"
            else:
                # 讨论结束，进入刺杀阶段
                self.engine.proceed_to_assassination()
                assassin_seat = self.engine.get_assassin_seat()
                if assassin_seat is not None:
                    return assassin_seat, "assassination"
                return None, None

        # 刺杀阶段
        elif state.phase == GamePhase.ASSASSINATION:
            assassin_seat = self.engine.get_assassin_seat()
            if assassin_seat is not None:
                return assassin_seat, "assassination"
            return None, None

        # 游戏结束
        elif state.phase == GamePhase.GAME_OVER:
            return None, None

        logger.warning(f"Unexpected phase: {state.phase}")
        return None, None

    # ========================================================================
    # 奖励计算
    # ========================================================================

    def calculate_reward(self) -> float:
        """计算游戏结束时的最终奖励。

        当前策略：只要游戏正常结束就给 +1.0。
        PPO 的 critic 会学到不同局面的价值差异。
        """
        if self.state.winner is None:
            return 0.0
        return 1.0

    # ========================================================================
    # 文本解析（静态方法，可独立使用）
    # ========================================================================

    @staticmethod
    def parse_team(content: str, player_count: int) -> Optional[List[int]]:
        """从模型输出中解析队伍选择。

        支持格式：
        - 【队伍：玩家1, 玩家3, 玩家5】
        - 队伍：玩家1、玩家3、玩家5
        - 选择玩家1,3,5
        - 数字序列
        """
        pattern = r'队伍[：:]\s*(.+?)(?:\n|$|】)'
        match = re.search(pattern, content)
        if match:
            team_str = match.group(1)
        else:
            team_str = content

        numbers = re.findall(r'玩家\s*(\d+)', team_str)
        if not numbers:
            numbers = re.findall(r'(\d+)', team_str)

        if numbers:
            team = []
            for n in numbers:
                seat = int(n) - 1  # 玩家编号从 1 开始，座位从 0 开始
                if 0 <= seat < player_count and seat not in team:
                    team.append(seat)
            return team if team else None

        return None

    @staticmethod
    def parse_vote(content: str) -> bool:
        """从模型输出中解析投票（赞成/反对）。"""
        content_lower = content.strip()

        if content_lower.startswith("【赞成】") or content_lower.startswith("赞成"):
            return True
        if content_lower.startswith("【反对】") or content_lower.startswith("反对"):
            return False

        approve_keywords = ["赞成", "同意", "支持", "通过", "approve", "yes"]
        reject_keywords = ["反对", "不同意", "否决", "拒绝", "reject", "no"]

        for kw in approve_keywords:
            if kw in content_lower[:50]:
                return True
        for kw in reject_keywords:
            if kw in content_lower[:50]:
                return False

        # 默认赞成（避免总是否决导致 5 次失败）
        return True

    @staticmethod
    def parse_quest_vote(content: str) -> bool:
        """从模型输出中解析任务投票（成功/失败）。"""
        content_lower = content.strip()

        if content_lower.startswith("【失败】") or content_lower.startswith("失败"):
            return False
        if content_lower.startswith("【成功】") or content_lower.startswith("成功"):
            return True

        fail_keywords = ["失败", "破坏", "sabotage", "fail"]
        for kw in fail_keywords:
            if kw in content_lower[:50]:
                return False

        # 默认成功
        return True

    @staticmethod
    def parse_assassination_target(content: str, player_count: int) -> Optional[int]:
        """从模型输出中解析刺杀目标。

        支持格式：
        - 【刺杀：玩家3】
        - 刺杀玩家3
        - 玩家3
        """
        pattern = r'刺杀[：:]*\s*玩家\s*(\d+)'
        match = re.search(pattern, content)
        if match:
            seat = int(match.group(1)) - 1
            if 0 <= seat < player_count:
                return seat

        numbers = re.findall(r'玩家\s*(\d+)', content)
        if numbers:
            seat = int(numbers[0]) - 1
            if 0 <= seat < player_count:
                return seat

        return None

    # ========================================================================
    # 内部辅助
    # ========================================================================

    def _random_team(self, size: int) -> List[int]:
        """生成一个随机队伍作为 fallback。"""
        seats = [p.seat for p in self.state.players]
        return random.sample(seats, min(size, len(seats)))
