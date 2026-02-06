"""Game tools for LLM players to interact with the game."""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str  # "integer", "boolean", "string", "array"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    items: Optional[Dict[str, Any]] = None  # For array types
    minimum: Optional[int] = None
    maximum: Optional[int] = None


@dataclass
class Tool:
    """A tool that can be used by the LLM player."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            
            if param.enum:
                prop["enum"] = param.enum
            
            if param.items:
                prop["items"] = param.items
            
            if param.minimum is not None:
                prop["minimum"] = param.minimum
            
            if param.maximum is not None:
                prop["maximum"] = param.maximum
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }


class GameTools:
    """Manages game tools available to LLM players."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_initial_tools()
    
    def _register_initial_tools(self):
        """Register the initial set of game tools."""
        
        # Tool 1: Propose team for quest
        propose_team_tool = Tool(
            name="propose_team",
            description="作为当前回合的队长，提议执行任务的队员名单。你需要选择指定数量的玩家（包括自己或不包括）来执行本轮任务。",
            parameters=[
                ToolParameter(
                    name="team",
                    type="array",
                    description="选中执行任务的玩家编号列表。例如 [1, 3, 4] 表示选择玩家1、玩家3、玩家4。",
                    items={"type": "integer"},
                ),
            ]
        )
        self.register_tool(propose_team_tool)
        
        # Tool 2: Vote on team
        vote_team_tool = Tool(
            name="vote_team",
            description="对当前提议的任务队伍进行投票。所有玩家都需要投票决定是否同意这个队伍执行任务。",
            parameters=[
                ToolParameter(
                    name="approve",
                    type="boolean",
                    description="是否同意这个队伍执行任务。true 表示赞成，false 表示反对。",
                ),
            ]
        )
        self.register_tool(vote_team_tool)
        
        # Tool 3: Vote on quest (success or fail)
        vote_quest_tool = Tool(
            name="vote_quest",
            description="作为任务执行者，决定任务是否成功。好人阵营必须投成功票，坏人阵营可以选择投成功或失败票。",
            parameters=[
                ToolParameter(
                    name="success",
                    type="boolean",
                    description="任务是否成功。true 表示任务成功，false 表示任务失败（只有坏人可以投失败票）。",
                ),
            ]
        )
        self.register_tool(vote_quest_tool)

        # Tool 4: Assassinate (Merlin)
        assassinate_tool = Tool(
            name="assassinate",
            description="刺客专用：在好人完成3个任务后，刺客有一次机会刺杀梅林。如果刺杀成功，坏人反败为胜。",
            parameters=[
                ToolParameter(
                    name="target",
                    type="integer",
                    description="被刺杀目标的玩家编号。例如 3 表示刺杀玩家3。",
                ),
            ]
        )
        self.register_tool(assassinate_tool)
    
        # Tool 5: Update memory
        update_memory_tool = Tool(
            name="update_memory",
            description="更新你的记忆，保存对当前局势的分析、对各玩家的判断、你的策略计划等。这些记忆会在下次行动时作为参考。请在每次决策后调用此工具，记录重要的信息和思考。",
            parameters=[
                ToolParameter(
                    name="memory",
                    type="string",
                    description="你想要记住的内容。可以包括：局势分析、对各玩家身份的推断、投票/任务结果的解读、下一步策略计划等。建议结构化记录，方便后续回顾。",
                ),
            ]
        )
        self.register_tool(update_memory_tool)
        
        # Tool 6: Speak in discussion
        speak_tool = Tool(
            name="speak",
            description="在讨论阶段发表你的看法。你的发言会被所有玩家看到，请谨慎措辞。",
            parameters=[
                ToolParameter(
                    name="content",
                    type="string",
                    description="你要说的话。根据你的角色立场，可以表达对队伍的支持或反对，分析其他玩家的行为，或者提出自己的见解。",
                ),
            ]
        )
        self.register_tool(speak_tool)
    
    def register_tool(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool. Returns True if the tool was found and removed."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_tool_names(self) -> List[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())
    
    def to_openai_format(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function calling format.
        
        Args:
            tool_names: Optional list of tool names to include. If None, all tools are included.
            
        Returns:
            List of tools in OpenAI format.
        """
        if tool_names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[name] for name in tool_names if name in self._tools]
        
        return [tool.to_openai_format() for tool in tools]
    
    def get_tools_for_phase(self, phase: str) -> List[str]:
        """Get the tool names available for a specific game phase.
        
        Args:
            phase: The game phase (e.g., "team_selection", "team_vote", "quest_execution")
            
        Returns:
            List of available tool names for this phase.
        """
        phase_tools = {
            "team_selection": ["propose_team"],
            "team_vote": ["vote_team"],
            "quest_execution": ["vote_quest"],
        }
        return phase_tools.get(phase, [])


# Global instance for convenience
game_tools = GameTools()


def get_game_tools() -> GameTools:
    """Get the global GameTools instance."""
    return game_tools
