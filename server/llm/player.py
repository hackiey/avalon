"""LLM player wrapper for game interactions."""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import asyncio

from server.llm.base import LLMProvider, Message
from server.llm.providers import create_provider
from server.llm.prompts import build_system_prompt, build_user_prompt
from server.llm.tools import game_tools
from server.game.state import GameState, Player


@dataclass
class LLMCallResult:
    """Result of an LLM call with full details."""
    result: Any  # The processed result (varies by action type)
    llm_input: Dict[str, Any] = field(default_factory=dict)  # Messages sent to LLM
    llm_output: Dict[str, Any] = field(default_factory=dict)  # Raw LLM response


class LLMPlayer:
    """Wrapper for an LLM player in the game."""
    
    def __init__(self, player: Player, provider: LLMProvider):
        self.player = player
        self.provider = provider
        self._initialized = False
        self.memory: str = ""
        
        # Cached system prompt (built once during initialization)
        self._system_prompt: str = ""
        self._visible_evil: List[int] = []
    
    @classmethod
    def create(cls, player: Player) -> Optional["LLMPlayer"]:
        """Create an LLM player from a Player object."""
        if player.is_human or not player.provider or not player.model_name:
            return None
        
        provider = create_provider(player.provider, player.model_name)
        if not provider:
            return None
        
        return cls(player, provider)
    
    async def initialize(self, state: GameState):
        """Initialize the player with their role information."""
        if self._initialized:
            return
        
        self._visible_evil = state.get_visible_evil_players(self.player.seat)
        self._system_prompt = build_system_prompt(
            self.player, 
            self._visible_evil, 
            state.players
        )
        
        self._initialized = True
    
    def _build_messages(self, state: GameState, phase: str) -> List[Message]:
        """Build the message list with system prompt and user prompt."""
        user_prompt = build_user_prompt(
            state=state,
            player=self.player,
            visible_evil=self._visible_evil,
            phase=phase,
            current_memory=self.memory,
        )
        
        return [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_prompt),
        ]
    
    def _build_llm_input(self, messages: List[Message], tools: List[Dict]) -> Dict[str, Any]:
        """Build complete LLM input including messages and tools."""
        return {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "tools": tools,
        }
    
    def _process_result(self, result: Union[str, Dict[str, Any]], main_tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Process the LLM result, updating memory and returning main tool args."""
        main_args = {}
        
        # 1. Handle Tool Calls (from Dict)
        if isinstance(result, dict) and "tool_calls" in result:
            for tool_call in result["tool_calls"]:
                name = tool_call["name"]
                args = tool_call["arguments"]
                
                if name == "update_memory":
                    self.memory = args.get("memory", "")
                elif main_tool_name and name == main_tool_name:
                    main_args = args
        
        # 2. Handle Content (Pure text or mixed)
        if isinstance(result, dict):
            if "content" in result and result["content"]:
                main_args["content"] = result["content"]
        elif isinstance(result, str) and result:
             main_args["content"] = result
            
        return main_args
    
    def _build_llm_output(self, result: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build structured LLM output from raw result."""
        if isinstance(result, str):
            return {"content": result}
        
        output = {}
        if isinstance(result, dict):
            if "content" in result:
                output["content"] = result["content"]
            if "reasoning_content" in result:
                output["reasoning_content"] = result["reasoning_content"]
            if "tool_calls" in result:
                output["tool_calls"] = result["tool_calls"]
        
        return output

    async def discuss_as_leader(self, state: GameState) -> LLMCallResult:
        """Leader speaks first during discussion phase, proposing a team configuration.
        
        This happens BEFORE the final team selection. The leader can propose
        a team and explain their reasoning. After all players discuss, 
        the leader will make the final team selection (which can be different).
        
        Returns:
            LLMCallResult with the discussion content and LLM call details.
        """
        messages = self._build_messages(state, "leader_discussion")
        tools = game_tools.to_openai_format(["speak", "update_memory"])
        llm_input = self._build_llm_input(messages, tools)
        
        try:
            result = await self.provider.generate(messages, temperature=0.8, tools=tools)
            
            # Build llm_output from raw result
            llm_output = self._build_llm_output(result)
            
            args = self._process_result(result, "speak")
            content = args.get("content", "")
            
            if not content:
                team_size = state.rules.quest_team_sizes[state.current_round - 1]
                content = f"作为队长，我需要选择{team_size}名队员执行任务。让我听听大家的意见。"
            
            return LLMCallResult(result=content, llm_input=llm_input, llm_output=llm_output)
        except Exception as e:
            print(f"Error in discuss_as_leader: {e}")
            return LLMCallResult(
                result="作为队长，我会仔细考虑队伍组成。请大家发表意见。",
                llm_input=llm_input,
                llm_output={"error": str(e)}
            )
    
    async def select_team_final(self, state: GameState) -> Tuple[List[int], str, Dict[str, Any], Dict[str, Any]]:
        """Make the final team selection after discussion with a summary speech.
        
        This is called AFTER discussion is complete. The leader makes the final
        team choice, which can be different from what they proposed during discussion,
        and gives a summary speech explaining their final decision.
        
        Returns:
            Tuple of (team members as seat indices, summary speech, llm_input, llm_output)
        """
        messages = self._build_messages(state, "team_selection_final")
        tools = game_tools.to_openai_format(["propose_team", "speak", "update_memory"])
        llm_input = self._build_llm_input(messages, tools)
        
        try:
            result = await self.provider.generate(messages, temperature=0.7, tools=tools)
            
            # Build llm_output from raw result
            llm_output = self._build_llm_output(result)
            
            # Process propose_team tool
            team_args = self._process_result(result, "propose_team")
            raw_team = team_args.get("team", [])
            
            # Convert 1-indexed player numbers to 0-indexed seat numbers
            team = [p - 1 for p in raw_team if isinstance(p, int) and 1 <= p <= len(state.players)]
            
            # Validate team
            team_size = state.rules.quest_team_sizes[state.current_round - 1]
            if len(team) != team_size:
                # Fallback: select self + random others
                import random
                available = [p.seat for p in state.players]
                team = [self.player.seat]
                if self.player.seat in available:
                    available.remove(self.player.seat)
                team.extend(random.sample(available, min(len(available), team_size - 1)))
            
            # Process speak tool for summary speech
            speech_args = self._process_result(result, "speak")
            speech = speech_args.get("content", "")
            if not speech:
                team_display = ", ".join([f"玩家{s + 1}" for s in team])
                speech = f"综合大家的意见，我最终决定选择 [{team_display}] 执行任务。"
            
            return team, speech, llm_input, llm_output
        except Exception as e:
            print(f"Error in select_team_final: {e}")
            # Fallback
            import random
            team_size = state.rules.quest_team_sizes[state.current_round - 1]
            team = random.sample([p.seat for p in state.players], team_size)
            team_display = ", ".join([f"玩家{s + 1}" for s in team])
            return team, f"我决定选择 [{team_display}] 执行任务。", llm_input, {"error": str(e)}
    
    async def select_team(self, state: GameState) -> Tuple[List[int], str, Dict[str, Any], Dict[str, Any]]:
        """Select a team for the quest (legacy method for compatibility).
        
        Returns:
            Tuple of (team members as seat indices, speech explaining the choice, llm_input, llm_output)
        """
        messages = self._build_messages(state, "team_selection")
        tools = game_tools.to_openai_format(["propose_team", "speak", "update_memory"])
        llm_input = self._build_llm_input(messages, tools)
        
        try:
            result = await self.provider.generate(messages, temperature=0.7, tools=tools)
            
            # Build llm_output from raw result
            llm_output = self._build_llm_output(result)
            
            # Process propose_team tool
            team_args = self._process_result(result, "propose_team")
            raw_team = team_args.get("team", [])
            
            # Convert 1-indexed player numbers to 0-indexed seat numbers
            team = [p - 1 for p in raw_team if isinstance(p, int) and 1 <= p <= len(state.players)]
            
            # Validate team
            team_size = state.rules.quest_team_sizes[state.current_round - 1]
            if len(team) != team_size:
                # Fallback: select self + random others
                import random
                available = [p.seat for p in state.players]
                team = [self.player.seat]
                if self.player.seat in available:
                    available.remove(self.player.seat)
                team.extend(random.sample(available, min(len(available), team_size - 1)))
            
            # Process speak tool
            speech_args = self._process_result(result, "speak")
            speech = speech_args.get("content", "")
            if not speech:
                team_display = ", ".join([f"玩家{s + 1}" for s in team])
                speech = f"我选择了 [{team_display}] 来执行这次任务。"
            
            return team, speech, llm_input, llm_output
        except Exception as e:
            print(f"Error in select_team: {e}")
            # Fallback
            import random
            team_size = state.rules.quest_team_sizes[state.current_round - 1]
            team = random.sample([p.seat for p in state.players], team_size)
            team_display = ", ".join([f"玩家{s + 1}" for s in team])
            return team, f"我选择了 [{team_display}] 来执行这次任务。", llm_input, {"error": str(e)}
    
    async def discuss(self, state: GameState) -> LLMCallResult:
        """Generate a discussion message.
        
        Returns:
            LLMCallResult with the discussion content and LLM call details.
        """
        messages = self._build_messages(state, "discussion")
        tools = game_tools.to_openai_format(["speak", "update_memory"])
        llm_input = self._build_llm_input(messages, tools)
        
        try:
            result = await self.provider.generate(messages, temperature=0.8, tools=tools)
            
            # Build llm_output from raw result
            llm_output = self._build_llm_output(result)
            
            args = self._process_result(result, "speak")
            content = args.get("content", "")
            
            if not content:
                content = "我需要更多信息来做出判断。"
            
            return LLMCallResult(result=content, llm_input=llm_input, llm_output=llm_output)
        except Exception as e:
            print(f"Error in discuss: {e}")
            return LLMCallResult(
                result="我同意大家的看法，让我们继续观察。",
                llm_input=llm_input,
                llm_output={"error": str(e)}
            )
    
    async def vote(self, state: GameState) -> LLMCallResult:
        """Vote on a proposed team.
        
        Returns:
            LLMCallResult with the vote decision and LLM call details.
        """
        messages = self._build_messages(state, "team_vote")
        tools = game_tools.to_openai_format(["vote_team", "update_memory"])
        llm_input = self._build_llm_input(messages, tools)
        
        try:
            result = await self.provider.generate(messages, temperature=0.5, tools=tools)
            
            # Build llm_output from raw result
            llm_output = self._build_llm_output(result)
            
            args = self._process_result(result, "vote_team")
            approve = args.get("approve", True)
            
            return LLMCallResult(result=approve, llm_input=llm_input, llm_output=llm_output)
        except Exception as e:
            print(f"Error in vote: {e}")
            return LLMCallResult(
                result=True,  # Default to approve
                llm_input=llm_input,
                llm_output={"error": str(e)}
            )
    
    async def execute_quest(self, state: GameState) -> LLMCallResult:
        """Decide whether to succeed or fail the quest.
        
        Returns:
            LLMCallResult with the quest vote decision and LLM call details.
        """
        messages = self._build_messages(state, "quest_execution")
        tools = game_tools.to_openai_format(["vote_quest", "update_memory"])
        llm_input = self._build_llm_input(messages, tools)
        
        try:
            result = await self.provider.generate(messages, temperature=0.5, tools=tools)
            
            # Build llm_output from raw result
            llm_output = self._build_llm_output(result)
            
            args = self._process_result(result, "vote_quest")
            success = args.get("success", True)
            
            return LLMCallResult(result=success, llm_input=llm_input, llm_output=llm_output)
        except Exception as e:
            print(f"Error in execute_quest: {e}")
            # Good players always succeed
            from server.game.roles import is_evil
            return LLMCallResult(
                result=not is_evil(self.player.role),
                llm_input=llm_input,
                llm_output={"error": str(e)}
            )
    
    async def discuss_assassination(self, state: GameState) -> LLMCallResult:
        """Discuss who might be Merlin before assassination (evil team only).
        
        Returns:
            LLMCallResult with the discussion content and LLM call details.
        """
        messages = self._build_messages(state, "assassination_discussion")
        tools = game_tools.to_openai_format(["speak", "update_memory"])
        llm_input = self._build_llm_input(messages, tools)
        
        try:
            result = await self.provider.generate(messages, temperature=0.8, tools=tools)
            
            # Build llm_output from raw result
            llm_output = self._build_llm_output(result)
            
            args = self._process_result(result, "speak")
            content = args.get("content", "")
            
            if not content:
                content = "我需要仔细回顾一下大家的表现再做判断。"
            
            return LLMCallResult(result=content, llm_input=llm_input, llm_output=llm_output)
        except Exception as e:
            print(f"Error in discuss_assassination: {e}")
            return LLMCallResult(
                result="让我想想谁的行为最像梅林...",
                llm_input=llm_input,
                llm_output={"error": str(e)}
            )

    async def assassinate(self, state: GameState) -> LLMCallResult:
        """Choose a target for assassination.
        
        Returns:
            LLMCallResult with the target seat (0-indexed) and LLM call details.
        """
        messages = self._build_messages(state, "assassination")
        tools = game_tools.to_openai_format(["assassinate", "update_memory"])
        llm_input = self._build_llm_input(messages, tools)
        
        try:
            result = await self.provider.generate(messages, temperature=0.3, tools=tools)
            
            # Build llm_output from raw result
            llm_output = self._build_llm_output(result)
            
            args = self._process_result(result, "assassinate")
            raw_target = args.get("target", 1)
            
            # Convert 1-indexed player number to 0-indexed seat number
            target = raw_target - 1 if isinstance(raw_target, int) else 0
            
            # Validate target
            valid_targets = [p.seat for p in state.players 
                          if p.seat not in self._visible_evil and p.seat != self.player.seat]
            if target not in valid_targets and valid_targets:
                import random
                target = random.choice(valid_targets)
            
            return LLMCallResult(result=target, llm_input=llm_input, llm_output=llm_output)
        except Exception as e:
            print(f"Error in assassinate: {e}")
            # Random target from good players
            import random
            valid_targets = [p.seat for p in state.players 
                          if p.seat not in self._visible_evil and p.seat != self.player.seat]
            target = random.choice(valid_targets) if valid_targets else 0
            return LLMCallResult(result=target, llm_input=llm_input, llm_output={"error": str(e)})


class LLMPlayerManager:
    """Manages all LLM players in a game."""
    
    def __init__(self):
        self.players: Dict[int, LLMPlayer] = {}  # seat -> LLMPlayer
    
    def add_player(self, player: Player) -> bool:
        """Add an LLM player. Returns True if successfully added."""
        llm_player = LLMPlayer.create(player)
        if llm_player:
            self.players[player.seat] = llm_player
            return True
        return False
    
    def get_player(self, seat: int) -> Optional[LLMPlayer]:
        """Get an LLM player by seat."""
        return self.players.get(seat)
    
    async def initialize_all(self, state: GameState):
        """Initialize all LLM players with their role information."""
        tasks = []
        for llm_player in self.players.values():
            tasks.append(llm_player.initialize(state))
        await asyncio.gather(*tasks)
    
    def is_human(self, seat: int) -> bool:
        """Check if a seat is a human player (not LLM managed)."""
        return seat not in self.players
