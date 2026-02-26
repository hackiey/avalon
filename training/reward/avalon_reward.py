"""Avalon 游戏奖励函数 - Verl 自定义奖励接口。

支持两种模式:
1. 预计算 advantage 模式: 如果 ground_truth 中有 precomputed_advantage，
   使用该值作为基础 advantage（由外部 GAE 计算管线注入），
   并叠加响应长度惩罚。
2. 回退模式: 使用最终游戏胜负作为奖励信号：
    - 玩家所在阵营获胜 -> +1.0
    - 玩家所在阵营失败 -> -1.0

响应长度惩罚:
  当回复过长（接近或超过 max_response_length 8192 tokens）时施加幂次惩罚，
  防止模型生成冗长回复导致截断。

  惩罚公式: MAX * ((tokens - START) / (CAP - START)) ^ POWER
  默认 POWER=2（二次方），对中等长度宽容，对接近截断严厉。

  通过环境变量配置（单位: token 数）:
    LEN_PENALTY_START  开始惩罚的 token 数 (默认 5000)
    LEN_PENALTY_CAP    满额惩罚的 token 数 (默认 8192)
    LEN_PENALTY_MAX    最大惩罚值          (默认 -1.0, 负值 = 惩罚)
    LEN_PENALTY_POWER  惩罚曲线幂次        (默认 2.0, 1=线性, 2=二次方)

  ground_truth 中需有 response_token_count 字段（由 preprocess --model_path 注入）。

Verl 接口:
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
"""

import json
import os
from typing import Any

_LEN_PENALTY_START = int(os.environ.get("LEN_PENALTY_START", "5000"))
_LEN_PENALTY_CAP = int(os.environ.get("LEN_PENALTY_CAP", "8192"))
_LEN_PENALTY_MAX = float(os.environ.get("LEN_PENALTY_MAX", "-1.0"))
_LEN_PENALTY_POWER = float(os.environ.get("LEN_PENALTY_POWER", "2.0"))


def _response_length_penalty(gt: dict) -> float:
    """Power-curve penalty based on response token count.

    Returns 0 if response_token_count is not present in ground_truth
    (backward compatible with data preprocessed without --model_path).
    """
    if _LEN_PENALTY_MAX >= 0:
        return 0.0

    token_count = gt.get("response_token_count", 0)
    if token_count <= 0:
        return 0.0
    if token_count <= _LEN_PENALTY_START:
        return 0.0
    if token_count >= _LEN_PENALTY_CAP:
        return _LEN_PENALTY_MAX

    ratio = (token_count - _LEN_PENALTY_START) / (_LEN_PENALTY_CAP - _LEN_PENALTY_START)
    return _LEN_PENALTY_MAX * (ratio ** _LEN_PENALTY_POWER)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
) -> float:
    """Avalon 奖励函数 - Verl RewardManager 调用入口。

    Args:
        data_source: 数据集名称 ("avalon")
        solution_str: 模型生成的响应文本（detokenized）
        ground_truth: 真值信息（包含角色、胜负等）
        extra_info: 额外信息（可选）

    Returns:
        奖励分数（含长度惩罚）
    """
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except json.JSONDecodeError:
            return 0.0
    elif isinstance(ground_truth, dict):
        gt = ground_truth
    else:
        return 0.0

    length_penalty = _response_length_penalty(gt)

    if "precomputed_advantage" in gt:
        return gt["precomputed_advantage"] + length_penalty

    player_team = gt.get("player_team", "")
    game_winner = gt.get("game_winner", "")

    if not player_team or not game_winner:
        return 0.0

    base_reward = 1.0 if player_team == game_winner else -1.0
    return base_reward + length_penalty
