"""Patch VeRL 的 RLHFDataset 以支持 tool-calling (function calling) 格式。

问题:
  我们的 Avalon 游戏中，LLM 通过 function calling 返回动作 (vote_team,
  propose_team 等)。preprocess.py 将 prompt 存为:
      '{"messages": [...], "tools": [...]}'
  但 VeRL 的 RLHFDataset 期望 prompt 是原生 messages 列表，
  且 apply_chat_template() 不传 tools 参数。

解决方案:
  1. 包装 tokenizer.apply_chat_template()，使其能解析 JSON 字符串格式
     的 prompt，自动提取 messages 和 tools
  2. 重写 _build_messages()，从 prompt 中提取 tools 并注入
     extra_info.tools_kwargs，供 rollout 生成时使用
  3. 通过 thread-local 在 _build_messages 和 apply_chat_template 之间
     传递 tools 信息

用法 (在 run_ppo.py 中):
    from training.rl.verl_extensions.tool_chat_patch import patch_rlhf_dataset_for_tools
    patch_rlhf_dataset_for_tools()
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Thread-local: 在 _build_messages → apply_chat_template 之间传递 tools
_tls = threading.local()


def _parse_prompt(prompt_data) -> Tuple[Any, Optional[List[Dict]]]:
    """解析 prompt 数据，分离 messages 和 tools。

    支持格式:
      - JSON string: '{"messages": [...], "tools": [...]}'
      - JSON string: '[{"role": "user", ...}]'
      - dict: {"messages": [...], "tools": [...]}
      - list: [{"role": "user", ...}]

    Returns:
        (messages, tools): messages 列表和 tools 列表 (tools 可能为 None)
    """
    # 解析 JSON 字符串
    if isinstance(prompt_data, str):
        try:
            prompt_data = json.loads(prompt_data)
        except (json.JSONDecodeError, TypeError):
            return prompt_data, None

    # dict 格式: {"messages": [...], "tools": [...]}
    if isinstance(prompt_data, dict) and "messages" in prompt_data:
        messages = prompt_data["messages"]
        tools = prompt_data.get("tools") or None
        return messages, tools

    # list 格式: [{"role": "user", ...}] (标准 messages)
    return prompt_data, None


def _wrap_tokenizer_for_tools(tokenizer):
    """包装 tokenizer.apply_chat_template() 以支持 tool-calling 格式。

    处理两种调用场景:
      1. 输入为 JSON 字符串 (来自 maybe_filter_out_long_prompts):
         直接解析并提取 messages 和 tools
      2. 输入为 messages 列表 (来自 __getitem__ 经 _build_messages):
         从 thread-local 获取 tools (由 _build_messages 设置)
    """
    if getattr(tokenizer, "_avalon_tools_wrapped", False):
        return  # 已经包装过

    original_act = tokenizer.apply_chat_template

    def wrapped_apply_chat_template(conversation, *args, **kwargs):
        tools = None

        # 场景1: JSON 字符串 (maybe_filter_out_long_prompts 路径)
        if isinstance(conversation, str):
            conversation, tools = _parse_prompt(conversation)

        # 场景2: dict 格式 (直接传入未解析的 prompt)
        elif isinstance(conversation, dict) and "messages" in conversation:
            conversation, tools = _parse_prompt(conversation)

        # 注入从 prompt 解析出的 tools
        if tools and "tools" not in kwargs:
            kwargs["tools"] = tools

        # 从 thread-local 获取 tools (由 patched _build_messages 设置)
        tls_tools = getattr(_tls, "tools", None)
        if tls_tools and "tools" not in kwargs:
            kwargs["tools"] = tls_tools

        return original_act(conversation, *args, **kwargs)

    tokenizer.apply_chat_template = wrapped_apply_chat_template
    tokenizer._avalon_tools_wrapped = True
    logger.info("Wrapped tokenizer.apply_chat_template for tool-calling support")


def patch_rlhf_dataset_for_tools():
    """Patch VeRL 的 RLHFDataset 以支持 Avalon tool-calling prompts。

    必须在 VeRL 训练启动前调用 (在 run_ppo.py 中 import 即可)。

    修改内容:
      1. __init__: 在初始化时包装 tokenizer
      2. _build_messages: 解析 JSON prompt，提取 tools，
         设置 thread-local 和 extra_info.tools_kwargs
    """
    from verl.utils.dataset.rl_dataset import RLHFDataset

    _original_init = RLHFDataset.__init__
    _original_build_messages = RLHFDataset._build_messages

    def patched_init(self, data_files, tokenizer, config, processor=None, **kwargs):
        """在初始化时包装 tokenizer (在 _read_files_and_tokenize 之前)。"""
        _wrap_tokenizer_for_tools(tokenizer)
        if processor is not None and hasattr(processor, "apply_chat_template"):
            _wrap_tokenizer_for_tools(processor)
        _original_init(self, data_files, tokenizer, config, processor=processor, **kwargs)

    def patched_build_messages(self, example):
        """解析 tool-calling prompt 格式，提取 tools。

        1. 从 prompt (可能是 JSON 字符串) 中解析出 messages 和 tools
        2. 设置 thread-local tools (供 wrapped apply_chat_template 使用)
        3. 注入 tools_kwargs 到 row_dict (供 rollout generation 使用)
        """
        prompt_data = example.pop(self.prompt_key)
        messages, tools = _parse_prompt(prompt_data)

        # 设置 thread-local tools (供 wrapped apply_chat_template 使用)
        _tls.tools = tools

        # 注入 tools_kwargs 到 row_dict (供 VeRL rollout generation 使用)
        if tools:
            extra = example.get("extra_info") or {}
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except (json.JSONDecodeError, TypeError):
                    extra = {}
            extra["tools_kwargs"] = {"tools": tools}
            extra["need_tools_kwargs"] = True
            example["extra_info"] = extra

        # 多模态处理 (委托给原始实现)
        if self.image_key in example or self.video_key in example:
            example[self.prompt_key] = messages
            result = _original_build_messages(self, example)
            return result

        return messages

    RLHFDataset.__init__ = patched_init
    RLHFDataset._build_messages = patched_build_messages

    logger.info("Patched RLHFDataset for Avalon tool-calling support")
