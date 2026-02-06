"""LLM provider implementations."""

import json
import re
from typing import List, Dict, Any, Optional, Union

import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from server.llm.base import LLMProvider, Message
from server.config import settings, LLMProviderConfig


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (also works for OpenAI-compatible APIs like DeepSeek)."""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, provider_name: str = "openai"):
        super().__init__(api_key, model, base_url)
        self.provider_name = provider_name
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 24000,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Dict[str, Any]]:
        kwargs = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"enable_thinking": True}
        }
        
        if tools:
            kwargs["tools"] = tools
            
            # Check provider to decide on tool_choice
            if self.provider_name and "qwen" in self.provider_name.lower():
                # Qwen models may not support tool_choice or prefer not having it explicitly
                pass
            else:
                # For other providers, explicitly set tool_choice
                kwargs["tool_choice"] = "auto"

        print(f"[DEBUG] OpenAI Request: {kwargs}")
        try:
            response = await self.client.chat.completions.create(**kwargs)
            print(f"[DEBUG] OpenAI Response: {response}")
            
            message = response.choices[0].message
            
            # 1. Extract reasoning_content (DeepSeek style or standard)
            reasoning_content = getattr(message, "reasoning_content", None)
            
            # 2. Extract content
            content = message.content or ""
            
            # 3. Handle tool calls
            if message.tool_calls:
                # If tool calls are present, return a dictionary containing all info
                # We need to support multiple tool calls (e.g. action + update_memory)
                
                # Transform tool calls into a list of dicts
                tool_calls_data = []
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        print(f"[ERROR] Failed to parse tool arguments: {tc.function.arguments}")
                        args = {}
                    
                    tool_calls_data.append({
                        "name": tc.function.name,
                        "arguments": args,
                        "id": tc.id
                    })

                # We return a structured dict that the Player can parse
                result = {
                    "content": content,
                    "tool_calls": tool_calls_data
                }

                # Add metadata to result
                if reasoning_content:
                    result["reasoning_content"] = reasoning_content
                
                return result
                
            # No tool calls (e.g. discuss phase) -> return pure text content
            return content
            
        except Exception as e:
            print(f"Error in generate: {e}")
            if tools:
                return {}
            return ""


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        super().__init__(api_key, model, base_url)
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 8192,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Dict[str, Any]]:
        # Anthropic uses a different message format
        system_message = None
        chat_messages = []
        
        for m in messages:
            if m.role == "system":
                system_message = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_message or "",
            "messages": chat_messages,
            "temperature": temperature,
        }
        
        if tools:
            # Convert OpenAI format tools to Anthropic format if needed
            # For now, we assume tools are passed in a compatible way or ignored
            # TODO: Implement tool conversion
            pass
            
        response = await self.client.messages.create(**kwargs)
        
        return response.content[0].text if response.content else ""


def create_provider(provider_name: str, model: str) -> Optional[LLMProvider]:
    """Create an LLM provider instance for the given provider and model."""
    providers = settings.get_llm_providers()
    
    if provider_name not in providers:
        return None
    
    config = providers[provider_name]
    
    if model not in config.models:
        return None
    
    if provider_name == "openai":
        return OpenAIProvider(config.api_key, model, config.base_url, provider_name="openai")
    elif provider_name == "anthropic":
        return AnthropicProvider(config.api_key, model)
    elif provider_name == "deepseek":
        # DeepSeek uses OpenAI-compatible API
        return OpenAIProvider(config.api_key, model, config.base_url, provider_name="deepseek")
    else:
        # Default to OpenAI-compatible API
        return OpenAIProvider(config.api_key, model, config.base_url, provider_name=provider_name)
