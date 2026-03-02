"""Multi-turn LLM player with incremental context and tool calling.

Conversation format follows the standard OpenAI tool-calling protocol:

    [system]    — game rules + role (fixed)
    [user]      — initial observation + first phase instruction
    [assistant] — {tool_calls: [{speak, ...}]}
    [tool]      — environment feedback (events only, no instructions)
    [user]      — next phase instruction
    [assistant] — {tool_calls: [{vote_team, ...}]}
    [tool]      — environment feedback (events only)
    [user]      — next phase instruction
    ...

[tool] responses only contain environment events (what happened).
[user] messages contain action directives (what to do next).

All game-action tools (speak, propose_team, vote_team, vote_quest,
assassinate) are retained.  update_memory is removed — the conversation
history itself serves as the player's memory.

Public API is identical to LLMPlayer / LLMPlayerManager so the batch
runner can switch between the two via a config flag.
"""

import json
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio

from server.llm.base import LLMProvider, Message, ToolCallParseError
from server.llm.providers import create_provider
from server.llm.tools import game_tools
from server.llm.player import LLMCallResult
from game.state import GameState, Player
from game.roles import is_evil
from game.prompts_v2 import (
    build_system_prompt_v2,
    build_observation,
    ObservationTracker,
)


# Tool lists per action (update_memory excluded)
_TOOLS_SPEAK = ["speak"]
_TOOLS_TEAM = ["propose_team", "speak"]
_TOOLS_VOTE = ["vote_team"]
_TOOLS_QUEST = ["vote_quest"]
_TOOLS_ASSASSINATE = ["assassinate"]


class LLMPlayerV2:
    """LLM player that accumulates multi-turn conversation with tool calls."""

    def __init__(self, player: Player, provider: LLMProvider, use_tools: bool = True):
        self.player = player
        self.provider = provider
        self.use_tools = use_tools
        self._initialized = False
        self.memory: str = ""

        self._system_prompt: str = ""
        self._visible_evil: List[int] = []

        self.conversation: List[Message] = []
        self.tracker = ObservationTracker()

    @classmethod
    def create(cls, player: Player, use_tools: bool = True) -> Optional["LLMPlayerV2"]:
        if player.is_human or not player.provider or not player.model_name:
            return None
        provider = create_provider(player.provider, player.model_name)
        if not provider:
            return None
        return cls(player, provider, use_tools=use_tools)

    async def initialize(self, state: GameState):
        if self._initialized:
            return
        self._visible_evil = state.get_visible_evil_players(self.player.seat)
        self._system_prompt = build_system_prompt_v2(
            self.player, self._visible_evil, state.players
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def _all_messages(self) -> List[Message]:
        return [Message(role="system", content=self._system_prompt)] + self.conversation

    def _add_observation(self, state: GameState, phase: str) -> None:
        """Add observation to conversation.

        First turn  → [user] message (events + instruction combined).
        Later turns → [tool] responses with events only,
                       then [user] message with phase instruction.
        """
        is_first = not self.tracker.initialized
        events, instruction = build_observation(
            state, self.player, self._visible_evil, phase, self.tracker, self.use_tools
        )
        if is_first:
            combined = f"{events}\n\n{instruction}" if events else instruction
            self.conversation.append(Message(role="user", content=combined))
        else:
            self._inject_tool_responses(events)
            self.conversation.append(Message(role="user", content=instruction))

    def _inject_tool_responses(self, events: str) -> None:
        """Attach [tool] responses to the last assistant's tool_calls.

        For single tool_call  → the events text goes directly.
        For multiple tool_calls → earlier ones get "OK", the last one
        carries the events.
        If no pending tool_calls exist (e.g. after a plain-text fallback),
        the events are folded into the subsequent [user] message by the
        caller.
        """
        for i in range(len(self.conversation) - 1, -1, -1):
            msg = self.conversation[i]
            if msg.role == "assistant" and msg.tool_calls:
                for j, tc in enumerate(msg.tool_calls):
                    is_last = j == len(msg.tool_calls) - 1
                    self.conversation.append(Message(
                        role="tool",
                        content=(events or "OK") if is_last else "OK",
                        tool_call_id=tc["id"],
                        name=tc["function"]["name"],
                    ))
                return
            if msg.role == "assistant":
                break

    def _record_assistant_response(self, result: Union[str, Dict[str, Any]]) -> None:
        """Record the assistant's response in conversation history.

        Only stores the [assistant] message (with tool_calls if present).
        The corresponding [tool] responses are added later by
        ``_inject_tool_responses`` when the next action is requested.
        """
        if isinstance(result, dict) and result.get("tool_calls"):
            api_tool_calls = self._to_api_tool_calls(result["tool_calls"])
            self.conversation.append(Message(
                role="assistant",
                content=result.get("content", ""),
                tool_calls=api_tool_calls,
            ))
        else:
            content = result if isinstance(result, str) else result.get("content", "") if isinstance(result, dict) else ""
            self.conversation.append(Message(role="assistant", content=content or "..."))

    def _add_fallback_response(self, text: str) -> None:
        self.conversation.append(Message(role="assistant", content=text))

    @staticmethod
    def _to_api_tool_calls(tool_calls_data: List[Dict]) -> List[Dict[str, Any]]:
        """Convert provider tool-call dicts to OpenAI API format for message history."""
        return [
            {
                "id": tc.get("id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            }
            for i, tc in enumerate(tool_calls_data)
        ]

    # ------------------------------------------------------------------
    # Result processing (same logic as LLMPlayer, minus update_memory)
    # ------------------------------------------------------------------

    @staticmethod
    def _process_result(
        result: Union[str, Dict[str, Any]], main_tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        main_args: Dict[str, Any] = {}
        if isinstance(result, dict) and "tool_calls" in result:
            for tc in result["tool_calls"]:
                if main_tool_name and tc["name"] == main_tool_name:
                    main_args = tc["arguments"]
        if isinstance(result, dict):
            if result.get("content"):
                main_args["content"] = result["content"]
        elif isinstance(result, str) and result:
            main_args["content"] = result
        return main_args

    @staticmethod
    def _build_llm_output(result: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(result, str):
            return {"content": result}
        out: Dict[str, Any] = {}
        if isinstance(result, dict):
            for key in ("content", "reasoning_content", "tool_calls"):
                if key in result:
                    out[key] = result[key]
        return out

    def _build_llm_input(
        self, messages: List[Message], tools: List[Dict]
    ) -> Dict[str, Any]:
        serialized = []
        for m in messages:
            msg: Dict[str, Any] = {"role": m.role}
            if m.role == "tool":
                msg["content"] = m.content or ""
                if m.tool_call_id:
                    msg["tool_call_id"] = m.tool_call_id
            elif m.role == "assistant" and m.tool_calls:
                if m.content:
                    msg["content"] = m.content
                msg["tool_calls"] = m.tool_calls
            else:
                msg["content"] = m.content or ""
            serialized.append(msg)
        return {"messages": serialized, "tools": tools}

    # ------------------------------------------------------------------
    # Public API — identical signatures to LLMPlayer
    # ------------------------------------------------------------------

    async def discuss_as_leader(self, state: GameState) -> LLMCallResult:
        self._add_observation(state, "leader_discussion")
        messages = self._all_messages()
        tools = game_tools.to_openai_format(_TOOLS_SPEAK)
        llm_input = self._build_llm_input(messages, tools)

        try:
            result = await self.provider.generate(messages, temperature=0.8, tools=tools)
            self._record_assistant_response(result)
            llm_output = self._build_llm_output(result)
            args = self._process_result(result, "speak")
            content = args.get("content", "")
            if not content:
                team_size = state.rules.quest_team_sizes[state.current_round - 1]
                content = f"作为队长，我需要选择{team_size}名队员执行任务。让我听听大家的意见。"
            return LLMCallResult(result=content, llm_input=llm_input, llm_output=llm_output)
        except ToolCallParseError as e:
            e.llm_input = llm_input
            raise
        except Exception as e:
            fallback = "作为队长，我会仔细考虑队伍组成。请大家发表意见。"
            self._add_fallback_response(fallback)
            print(f"Error in discuss_as_leader: {e}")
            return LLMCallResult(result=fallback, llm_input=llm_input, llm_output={"error": str(e)})

    async def select_team_final(
        self, state: GameState
    ) -> Tuple[List[int], str, Dict[str, Any], Dict[str, Any]]:
        self._add_observation(state, "team_selection_final")
        messages = self._all_messages()
        tools = game_tools.to_openai_format(_TOOLS_TEAM)
        llm_input = self._build_llm_input(messages, tools)

        try:
            result = await self.provider.generate(messages, temperature=0.7, tools=tools)
            self._record_assistant_response(result)
            llm_output = self._build_llm_output(result)

            team_args = self._process_result(result, "propose_team")
            raw_team = team_args.get("team", [])
            team = [
                p - 1
                for p in raw_team
                if isinstance(p, int) and 1 <= p <= len(state.players)
            ]

            team_size = state.rules.quest_team_sizes[state.current_round - 1]
            if len(team) != team_size:
                import random
                available = [p.seat for p in state.players]
                team = [self.player.seat]
                if self.player.seat in available:
                    available.remove(self.player.seat)
                team.extend(random.sample(available, min(len(available), team_size - 1)))

            speech_args = self._process_result(result, "speak")
            speech = speech_args.get("content", "")
            if not speech:
                team_display = ", ".join(f"玩家{s + 1}" for s in team)
                speech = f"综合大家的意见，我最终决定选择 [{team_display}] 执行任务。"

            return team, speech, llm_input, llm_output
        except ToolCallParseError as e:
            e.llm_input = llm_input
            raise
        except Exception as e:
            import random
            print(f"Error in select_team_final: {e}")
            team_size = state.rules.quest_team_sizes[state.current_round - 1]
            team = random.sample([p.seat for p in state.players], team_size)
            team_display = ", ".join(f"玩家{s + 1}" for s in team)
            speech = f"我决定选择 [{team_display}] 执行任务。"
            self._add_fallback_response(speech)
            return team, speech, llm_input, {"error": str(e)}

    async def select_team(
        self, state: GameState
    ) -> Tuple[List[int], str, Dict[str, Any], Dict[str, Any]]:
        return await self.select_team_final(state)

    async def discuss(self, state: GameState) -> LLMCallResult:
        self._add_observation(state, "discussion")
        messages = self._all_messages()
        tools = game_tools.to_openai_format(_TOOLS_SPEAK)
        llm_input = self._build_llm_input(messages, tools)

        try:
            result = await self.provider.generate(messages, temperature=0.8, tools=tools)
            self._record_assistant_response(result)
            llm_output = self._build_llm_output(result)
            args = self._process_result(result, "speak")
            content = args.get("content", "")
            if not content:
                content = "我需要更多信息来做出判断。"
            return LLMCallResult(result=content, llm_input=llm_input, llm_output=llm_output)
        except ToolCallParseError as e:
            e.llm_input = llm_input
            raise
        except Exception as e:
            fallback = "我同意大家的看法，让我们继续观察。"
            self._add_fallback_response(fallback)
            print(f"Error in discuss: {e}")
            return LLMCallResult(result=fallback, llm_input=llm_input, llm_output={"error": str(e)})

    async def vote(self, state: GameState) -> LLMCallResult:
        self._add_observation(state, "team_vote")
        messages = self._all_messages()
        tools = game_tools.to_openai_format(_TOOLS_VOTE)
        llm_input = self._build_llm_input(messages, tools)

        try:
            result = await self.provider.generate(messages, temperature=0.5, tools=tools)
            self._record_assistant_response(result)
            llm_output = self._build_llm_output(result)
            args = self._process_result(result, "vote_team")
            approve = args.get("approve", True)
            return LLMCallResult(result=approve, llm_input=llm_input, llm_output=llm_output)
        except ToolCallParseError as e:
            e.llm_input = llm_input
            raise
        except Exception as e:
            self._add_fallback_response("...")
            print(f"Error in vote: {e}")
            return LLMCallResult(result=True, llm_input=llm_input, llm_output={"error": str(e)})

    async def execute_quest(self, state: GameState) -> LLMCallResult:
        self._add_observation(state, "quest_execution")
        messages = self._all_messages()
        tools = game_tools.to_openai_format(_TOOLS_QUEST)
        llm_input = self._build_llm_input(messages, tools)

        try:
            result = await self.provider.generate(messages, temperature=0.5, tools=tools)
            self._record_assistant_response(result)
            llm_output = self._build_llm_output(result)
            args = self._process_result(result, "vote_quest")
            success = args.get("success", True)
            return LLMCallResult(result=success, llm_input=llm_input, llm_output=llm_output)
        except ToolCallParseError as e:
            e.llm_input = llm_input
            raise
        except Exception as e:
            default = not is_evil(self.player.role)
            self._add_fallback_response("...")
            print(f"Error in execute_quest: {e}")
            return LLMCallResult(result=default, llm_input=llm_input, llm_output={"error": str(e)})

    async def discuss_assassination(self, state: GameState) -> LLMCallResult:
        self._add_observation(state, "assassination_discussion")
        messages = self._all_messages()
        tools = game_tools.to_openai_format(_TOOLS_SPEAK)
        llm_input = self._build_llm_input(messages, tools)

        try:
            result = await self.provider.generate(messages, temperature=0.8, tools=tools)
            self._record_assistant_response(result)
            llm_output = self._build_llm_output(result)
            args = self._process_result(result, "speak")
            content = args.get("content", "")
            if not content:
                content = "我需要仔细回顾一下大家的表现再做判断。"
            return LLMCallResult(result=content, llm_input=llm_input, llm_output=llm_output)
        except ToolCallParseError as e:
            e.llm_input = llm_input
            raise
        except Exception as e:
            fallback = "让我想想谁的行为最像梅林..."
            self._add_fallback_response(fallback)
            print(f"Error in discuss_assassination: {e}")
            return LLMCallResult(result=fallback, llm_input=llm_input, llm_output={"error": str(e)})

    async def assassinate(self, state: GameState) -> LLMCallResult:
        self._add_observation(state, "assassination")
        messages = self._all_messages()
        tools = game_tools.to_openai_format(_TOOLS_ASSASSINATE)
        llm_input = self._build_llm_input(messages, tools)

        try:
            result = await self.provider.generate(messages, temperature=0.3, tools=tools)
            self._record_assistant_response(result)
            llm_output = self._build_llm_output(result)
            args = self._process_result(result, "assassinate")
            raw_target = args.get("target", 1)
            target = raw_target - 1 if isinstance(raw_target, int) else 0

            valid_targets = [
                p.seat for p in state.players
                if p.seat not in self._visible_evil and p.seat != self.player.seat
            ]
            if target not in valid_targets and valid_targets:
                import random
                target = random.choice(valid_targets)

            return LLMCallResult(result=target, llm_input=llm_input, llm_output=llm_output)
        except ToolCallParseError as e:
            e.llm_input = llm_input
            raise
        except Exception as e:
            import random
            valid_targets = [
                p.seat for p in state.players
                if p.seat not in self._visible_evil and p.seat != self.player.seat
            ]
            target = random.choice(valid_targets) if valid_targets else 0
            self._add_fallback_response("...")
            print(f"Error in assassinate: {e}")
            return LLMCallResult(result=target, llm_input=llm_input, llm_output={"error": str(e)})


class LLMPlayerManagerV2:
    """Drop-in replacement for LLMPlayerManager using multi-turn players."""

    def __init__(self, use_tools: bool = True):
        self.players: Dict[int, LLMPlayerV2] = {}
        self.use_tools = use_tools

    def add_player(self, player: Player) -> bool:
        llm_player = LLMPlayerV2.create(player, use_tools=self.use_tools)
        if llm_player:
            self.players[player.seat] = llm_player
            return True
        return False

    def get_player(self, seat: int) -> Optional[LLMPlayerV2]:
        return self.players.get(seat)

    async def initialize_all(self, state: GameState):
        tasks = [p.initialize(state) for p in self.players.values()]
        await asyncio.gather(*tasks)

    def is_human(self, seat: int) -> bool:
        return seat not in self.players
