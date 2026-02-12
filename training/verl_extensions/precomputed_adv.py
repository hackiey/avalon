"""自定义 Verl advantage estimator — 使用预计算的 advantage。

将外部计算好的 episode 级 GAE advantage 注入 Verl 训练流程。
reward function 已经通过 ground_truth["precomputed_advantage"] 返回了
预计算的 advantage 值，这个 estimator 直接使用它，
不再需要 Verl 内置的 token 级 GAE 或 critic 模型。

注册方式:
    在训练脚本启动前 import 本模块即可:
    import training.verl_extensions.precomputed_adv

Verl 配置:
    algorithm.adv_estimator=precomputed
"""

import torch
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("precomputed")
def compute_precomputed_advantage(token_level_rewards, response_mask, **kwargs):
    """使用预计算的 advantage。

    reward function (avalon_reward.py) 返回的 score 就是预计算的 advantage。
    Verl 会将这个 score 作为 token_level_rewards 传入。

    我们把 scalar reward 广播到 response 的所有 token 上，
    这样每个 token 的 advantage 都相同（整个 response 共享同一个 advantage）。

    Args:
        token_level_rewards: (batch_size, seq_len) — Verl 传入的 token 级 reward。
            对于我们的场景，reward 集中在 response 的最后一个 token 上。
        response_mask: (batch_size, seq_len) — response 部分的 mask。

    Returns:
        advantages: (batch_size, seq_len) — 广播后的 advantage
        returns: (batch_size, seq_len) — 这里直接复用 advantage 作为 returns
    """
    # token_level_rewards 中，非零值通常只在最后一个 token
    # 我们需要将 per-sequence 的 advantage 广播到所有 response token
    # 先提取每个 sequence 的 reward sum（即预计算的 advantage）
    per_seq_advantage = (token_level_rewards * response_mask).sum(dim=-1, keepdim=True)

    # 广播到所有 response token
    advantages = per_seq_advantage * response_mask

    # returns 复用 advantages（因为 Verl 的 value loss 不再需要）
    returns = advantages.clone()

    return advantages, returns
