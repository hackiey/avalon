"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", "assistant"
    content: str


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
