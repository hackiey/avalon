"""每轮 Self-Play 游戏统计 — 从 JSONL 轨迹计算胜率等指标。

输出三部分:
  1. 当前轮的详细统计（打印到控制台）
  2. 一行 JSON 追加到汇总文件（用于跨轮趋势追踪）
  3. 上报 wandb（可选，通过 --wandb_project 启用）

用法:
    python -m training.stats.game_stats \
        --input_jsonl data/round_1/trajectories.jsonl \
        --round 1 \
        --summary_file experiments/my_exp/round_stats.jsonl \
        --wandb_project avalon-self-play \
        --experiment_name self_play_v3
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List


def compute_stats(trajectories: List[Dict[str, Any]], round_num: int = 0) -> Dict[str, Any]:
    """从轨迹列表计算游戏统计。"""
    total = len(trajectories)
    if total == 0:
        return {"round": round_num, "total_games": 0}

    good_wins = sum(1 for t in trajectories if t.get("winner") == "good")
    evil_wins = sum(1 for t in trajectories if t.get("winner") == "evil")
    merlin_assassinated = sum(1 for t in trajectories if t.get("merlin_assassinated", False))

    total_rounds_list = [t.get("total_rounds", 0) for t in trajectories]
    total_decisions_list = [t.get("total_decisions", 0) for t in trajectories]

    # 任务成功/失败统计
    quest_successes = 0
    quest_failures = 0
    for t in trajectories:
        for qr in t.get("quest_results", []):
            if qr.get("success"):
                quest_successes += 1
            else:
                quest_failures += 1

    # 按角色统计胜率
    role_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"games": 0, "wins": 0})
    for t in trajectories:
        winner = t.get("winner", "")
        for p in t.get("players", []):
            role = p.get("role", "unknown")
            role_stats[role]["games"] += 1
            team = p.get("team", "")
            if not team:
                from game.roles import is_evil_role
                team = "evil" if is_evil_role(role) else "good"
            if team == winner:
                role_stats[role]["wins"] += 1

    role_winrates = {
        role: {
            "games": s["games"],
            "wins": s["wins"],
            "win_rate": round(s["wins"] / s["games"], 4) if s["games"] > 0 else 0,
        }
        for role, s in sorted(role_stats.items())
    }

    stats = {
        "round": round_num,
        "total_games": total,
        "good_wins": good_wins,
        "evil_wins": evil_wins,
        "good_win_rate": round(good_wins / total, 4),
        "evil_win_rate": round(evil_wins / total, 4),
        "merlin_assassinated": merlin_assassinated,
        "merlin_assassination_rate": round(merlin_assassinated / max(evil_wins, 1), 4),
        "avg_rounds": round(sum(total_rounds_list) / total, 2),
        "avg_decisions": round(sum(total_decisions_list) / total, 2),
        "quest_successes": quest_successes,
        "quest_failures": quest_failures,
        "role_winrates": role_winrates,
    }

    return stats


def print_stats(stats: Dict[str, Any]):
    """打印格式化的游戏统计。"""
    print(f"\n{'='*60}")
    print(f"  Round {stats['round']} Game Statistics")
    print(f"{'='*60}")
    print(f"  Total games:          {stats['total_games']}")
    print(f"  Good wins:            {stats['good_wins']} ({stats['good_win_rate']*100:.1f}%)")
    print(f"  Evil wins:            {stats['evil_wins']} ({stats['evil_win_rate']*100:.1f}%)")
    print(f"  Merlin assassinated:  {stats['merlin_assassinated']} "
          f"({stats['merlin_assassination_rate']*100:.1f}% of evil wins)")
    print(f"  Avg rounds/game:      {stats['avg_rounds']}")
    print(f"  Avg decisions/game:   {stats['avg_decisions']}")
    print(f"  Quest success/fail:   {stats['quest_successes']}/{stats['quest_failures']}")

    role_winrates = stats.get("role_winrates", {})
    if role_winrates:
        print(f"\n  {'Role':<20} {'Games':<8} {'Wins':<8} {'Win Rate':<10}")
        print(f"  {'-'*46}")
        for role, rstat in role_winrates.items():
            print(f"  {role:<20} {rstat['games']:<8} {rstat['wins']:<8} "
                  f"{rstat['win_rate']*100:.1f}%")

    print(f"{'='*60}\n")


def log_to_wandb(stats: Dict[str, Any], project: str, experiment_name: str):
    """将统计指标上报到 wandb。

    使用 round 作为 step，使指标在 wandb 中按轮次显示趋势。
    所有 self-play 轮次的统计共享同一个 wandb run（通过 resume=allow 实现）。
    """
    import wandb

    run = wandb.init(
        project=project,
        name=f"{experiment_name}-game-stats",
        id=f"{experiment_name}-game-stats",
        resume="allow",
        tags=["game-stats", experiment_name],
    )

    round_num = stats["round"]

    metrics = {
        "game/good_win_rate": stats["good_win_rate"],
        "game/evil_win_rate": stats["evil_win_rate"],
        "game/total_games": stats["total_games"],
        "game/merlin_assassination_rate": stats["merlin_assassination_rate"],
        "game/avg_rounds": stats["avg_rounds"],
        "game/avg_decisions": stats["avg_decisions"],
        "game/quest_successes": stats["quest_successes"],
        "game/quest_failures": stats["quest_failures"],
    }

    for role, rstat in stats.get("role_winrates", {}).items():
        metrics[f"game/role_winrate/{role}"] = rstat["win_rate"]

    run.log(metrics, step=round_num)
    run.finish()
    print(f"Stats logged to wandb: {project}/{experiment_name}-game-stats (step={round_num})")


def main():
    parser = argparse.ArgumentParser(description="计算每轮 Self-Play 游戏统计")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="输入的 JSONL 轨迹文件")
    parser.add_argument("--round", type=int, default=0,
                        help="当前轮次编号")
    parser.add_argument("--summary_file", type=str, default=None,
                        help="跨轮汇总文件路径（追加模式，每轮一行 JSON）")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project 名称（启用 wandb 上报）")
    parser.add_argument("--experiment_name", type=str, default="self_play",
                        help="实验名称（用于 wandb run name/id）")
    args = parser.parse_args()

    # 加载轨迹
    trajectories = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    # 计算统计
    stats = compute_stats(trajectories, round_num=args.round)

    # 打印
    print_stats(stats)

    # 追加到汇总文件
    if args.summary_file:
        os.makedirs(os.path.dirname(args.summary_file) or ".", exist_ok=True)
        summary = {k: v for k, v in stats.items() if k != "role_winrates"}
        with open(args.summary_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"Stats appended to {args.summary_file}")

    # 上报 wandb
    if args.wandb_project:
        log_to_wandb(stats, args.wandb_project, args.experiment_name)


if __name__ == "__main__":
    main()
