"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class Message:
    """A chat message (supports tool calling multi-turn conversations).

    For regular messages:
        Message(role="user", content="hello")

    For assistant messages with tool calls:
        Message(role="assistant", content="", tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "speak", "arguments": '{"content": "hi"}'}
        }])

    For tool response messages:
        Message(role="tool", content="OK", tool_call_id="call_1", name="speak")
    """

    role: str  # "system", "user", "assistant", "tool"
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ToolCallParseError(Exception):
    """vLLM/serving layer failed to parse the model's tool call output.

    Typically caused by repeated/malformed JSON in <tool_call> blocks.
    Carries raw content and llm_input for error logging and training data.
    """

    def __init__(self, raw_content: str):
        self.raw_content = raw_content
        self.llm_input: Dict[str, Any] = {}
        super().__init__(f"Malformed tool call: {raw_content[:200]}")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
    
    from typing import List, Dict, Any, Optional, Union, Union
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 8192,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Generate a response from the LLM.
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: Optional list of tools/functions
            
        Returns:
            - If tools are used: A dictionary containing tool arguments (and optionally content/reasoning).
            - If tools are NOT used: A string containing the text response.
        """
        pass
