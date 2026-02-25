"""Avalon 游戏奖励函数 - Verl 自定义奖励接口。

支持两种模式:
1. 预计算 advantage 模式: 如果 ground_truth 中有 precomputed_advantage，
   使用该值作为基础 advantage（由外部 GAE 计算管线注入），
   并叠加响应质量调节信号（格式/截断/长度惩罚）。
2. 回退模式: 使用最终游戏胜负作为奖励信号：
    - 玩家所在阵营获胜 -> +1.0
    - 玩家所在阵营失败 -> -1.0

Verl 接口:
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
"""

import json
from typing import Any, Dict


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
) -> float:
    """Avalon 奖励函数 - Verl RewardManager 调用入口。

    仅基于游戏最终胜负给予奖励，不做过程奖励。

    Args:
        data_source: 数据集名称 ("avalon")
        solution_str: 模型生成的响应文本（detokenized）
        ground_truth: 真值信息（包含角色、胜负等）
        extra_info: 额外信息（可选）

    Returns:
        奖励分数: 阵营获胜 +1.0，阵营失败 -1.0
    """
    # 解析 ground_truth
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except json.JSONDecodeError:
            return 0.0
    elif isinstance(ground_truth, dict):
        gt = ground_truth
    else:
        return 0.0

    # 如果有预计算的 advantage（由外部 GAE 管线注入），直接返回
    if "precomputed_advantage" in gt:
        return gt["precomputed_advantage"]

    # 回退: 使用游戏胜负奖励
    player_team = gt.get("player_team", "")
    game_winner = gt.get("game_winner", "")

    if not player_team or not game_winner:
        return 0.0

    if player_team == game_winner:
        return 1.0
    else:
        return -1.0
