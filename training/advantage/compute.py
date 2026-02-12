"""GAE (Generalized Advantage Estimation) 计算脚本。

从 JSONL 轨迹 + Critic V(s) 推理结果，计算 episode 级别的 GAE advantage。

核心逻辑:
1. 从 JSONL 读取轨迹 + 从 critic 推理结果读取 V(s)
2. 按 (game_id, player_seat) 分组，每组构成一个 episode
3. 组内按 seq_num 排序
4. 计算 GAE:
    r_t = 0 for t < T, r_T = +1/-1 (game outcome)
    V(s_{T+1}) = 0 (terminal)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        advantage = delta + gamma * lam * next_advantage
5. 同时计算 discounted return: G_t = r_t + γ·G_{t+1} (用于 critic 训练)

输出: JSON 文件，每个决策附带 advantage 和 discounted_return

用法:
    python -m training.advantage.compute \
        --input_jsonl data/trajectories.jsonl \
        --values_json data/values.json \
        --output_json data/advantages.json \
        --gamma 0.99 --lam 0.95
"""

import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from game.roles import is_evil_role


def load_trajectories(input_jsonl: str) -> List[Dict[str, Any]]:
    """加载 JSONL 轨迹文件。"""
    trajectories = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories


def load_values(values_json: str) -> Dict[str, float]:
    """加载 Critic V(s) 推理结果。

    返回 {(game_id, player_seat, seq_num): value} 的查找表。
    """
    with open(values_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    lookup = {}
    for item in results:
        key = (item["game_id"], item["player_seat"], item["seq_num"])
        lookup[key] = item["value"]

    return lookup


def get_player_reward(
    trajectory: Dict[str, Any],
    player_seat: int,
    players: List[Dict[str, Any]],
) -> float:
    """根据游戏结果计算玩家的最终奖励。

    阵营获胜 → +1.0, 阵营失败 → -1.0
    """
    winner = trajectory.get("winner", "")
    if not winner:
        return 0.0

    # 找到该玩家的角色
    player_role = ""
    for p in players:
        if p.get("seat") == player_seat:
            player_role = p.get("role", "")
            break

    if not player_role:
        return 0.0

    player_team = "evil" if is_evil_role(player_role) else "good"
    return 1.0 if player_team == winner else -1.0


def compute_gae_for_episode(
    episode_decisions: List[Dict[str, Any]],
    value_lookup: Dict[str, float],
    final_reward: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """对一个 episode (一个玩家在一局游戏中的所有决策) 计算 GAE。

    Args:
        episode_decisions: 按 seq_num 排序的决策列表
        value_lookup: V(s) 查找表
        final_reward: 该玩家的最终奖励 (+1/-1)
        gamma: 折扣因子
        lam: GAE lambda

    Returns:
        advantages: 每个决策点的 GAE advantage
        discounted_returns: 每个决策点的 discounted return G_t (用于 critic 训练)
    """
    T = len(episode_decisions)
    if T == 0:
        return [], []

    # 构建 rewards 和 values 数组
    rewards = [0.0] * T
    values = [0.0] * T

    # 最后一个决策点获得 final reward
    rewards[T - 1] = final_reward

    for i, d in enumerate(episode_decisions):
        key = (d["game_id"], d["player_seat"], d["seq_num"])
        values[i] = value_lookup.get(key, 0.0)

    # --- 计算 GAE ---
    advantages = [0.0] * T
    next_value = 0.0  # V(s_{T+1}) = 0 (terminal state)
    next_advantage = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = delta + gamma * lam * next_advantage
        next_value = values[t]
        next_advantage = advantages[t]

    # --- 计算 Discounted Returns G_t ---
    discounted_returns = [0.0] * T
    next_return = 0.0

    for t in reversed(range(T)):
        discounted_returns[t] = rewards[t] + gamma * next_return
        next_return = discounted_returns[t]

    return advantages, discounted_returns


def compute_advantages(
    input_jsonl: str,
    values_json: str,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> List[Dict[str, Any]]:
    """计算所有决策点的 GAE advantage 和 discounted return。

    Args:
        input_jsonl: JSONL 轨迹文件路径
        values_json: Critic V(s) 推理结果路径
        gamma: 折扣因子
        lam: GAE lambda

    Returns:
        决策列表，每个决策附带 advantage 和 discounted_return
    """
    trajectories = load_trajectories(input_jsonl)
    value_lookup = load_values(values_json)

    print(f"Loaded {len(trajectories)} trajectories")
    print(f"Loaded {len(value_lookup)} value predictions")

    # 按 (game_id, player_seat) 分组
    episodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    game_meta: Dict[str, Dict[str, Any]] = {}  # game_id -> trajectory metadata

    for trajectory in trajectories:
        game_id = trajectory.get("game_id", "")
        game_meta[game_id] = trajectory

        for decision in trajectory.get("decisions", []):
            player_seat = decision.get("player_seat", 0)
            key = f"{game_id}_{player_seat}"
            episodes[key].append({
                "game_id": game_id,
                "player_seat": player_seat,
                "seq_num": decision.get("seq_num", 0),
                "action_type": decision.get("action_type", ""),
                "round_num": decision.get("round_num", 1),
            })

    print(f"Found {len(episodes)} episodes (game x player)")

    # 对每个 episode 计算 GAE
    results = []
    total_decisions = 0

    for episode_key, decisions in episodes.items():
        # 按 seq_num 排序
        decisions.sort(key=lambda d: d["seq_num"])
        game_id = decisions[0]["game_id"]
        player_seat = decisions[0]["player_seat"]

        # 获取该玩家的最终奖励
        trajectory = game_meta[game_id]
        players = trajectory.get("players", [])
        final_reward = get_player_reward(trajectory, player_seat, players)

        # 计算 GAE
        advantages, discounted_returns = compute_gae_for_episode(
            decisions, value_lookup, final_reward, gamma, lam
        )

        # 组装结果
        for i, d in enumerate(decisions):
            results.append({
                "game_id": d["game_id"],
                "player_seat": d["player_seat"],
                "seq_num": d["seq_num"],
                "action_type": d["action_type"],
                "round_num": d["round_num"],
                "advantage": advantages[i],
                "discounted_return": discounted_returns[i],
            })
            total_decisions += 1

    print(f"Computed advantages for {total_decisions} decision points")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="计算 episode 级别的 GAE advantage"
    )

    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="输入的 JSONL 轨迹文件",
    )
    parser.add_argument(
        "--values_json",
        type=str,
        required=True,
        help="Critic V(s) 推理结果 JSON 文件",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="输出 JSON 文件路径（含 advantage 和 discounted_return）",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="折扣因子 γ (默认 0.99)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="GAE lambda λ (默认 0.95)",
    )

    args = parser.parse_args()

    results = compute_advantages(
        input_jsonl=args.input_jsonl,
        values_json=args.values_json,
        gamma=args.gamma,
        lam=args.lam,
    )

    # 保存结果
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {args.output_json}")

    # 打印统计
    if results:
        advs = [r["advantage"] for r in results]
        rets = [r["discounted_return"] for r in results]
        print(f"\nAdvantage statistics:")
        print(f"  Count: {len(advs)}")
        print(f"  Mean:  {sum(advs)/len(advs):.4f}")
        print(f"  Min:   {min(advs):.4f}")
        print(f"  Max:   {max(advs):.4f}")
        print(f"\nDiscounted return statistics:")
        print(f"  Mean:  {sum(rets)/len(rets):.4f}")
        print(f"  Min:   {min(rets):.4f}")
        print(f"  Max:   {max(rets):.4f}")


if __name__ == "__main__":
    main()
