"""评估训练后模型的 Avalon 游戏表现。

将训练后的模型通过 vLLM 部署为 OpenAI 兼容 API，
然后使用现有的 batch runner 运行对战并统计胜率。

用法:
    # 前提: 已通过 vLLM 部署训练后的模型
    # vllm serve checkpoints/avalon-self-play/self-play-ppo-r5/actor/huggingface --port 8000

    # 评估模型（与自身对战）
    python -m training.eval.evaluate \
        --model_name avalon-ppo \
        --provider vllm \
        --num_games 50

    # 与基线模型对战
    python -m training.eval.evaluate \
        --model_name avalon-ppo \
        --provider vllm \
        --baseline_model qwen-plus \
        --baseline_provider qwen \
        --num_games 50

    # 自动启动 vLLM 服务并评估
    python -m training.eval.evaluate \
        --model_path checkpoints/avalon-self-play/self-play-ppo-r5/actor/huggingface \
        --num_games 50 \
        --auto_serve
"""

import argparse
import asyncio
import json
import os
import signal

from game.roles import is_evil_role
import subprocess
import sys
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """评估配置。"""

    # 被评估模型
    model_name: str = "avalon-ppo"
    provider: str = "vllm"

    # 基线模型（如果为空，则评估模型自我对战）
    baseline_model: Optional[str] = None
    baseline_provider: Optional[str] = None

    # 游戏配置
    num_games: int = 50
    player_count: int = 5
    parallel: int = 3

    # vLLM 自动部署
    auto_serve: bool = False
    model_path: Optional[str] = None
    vllm_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.8


@dataclass
class EvalResult:
    """评估结果。"""

    total_games: int = 0
    completed_games: int = 0

    # 整体胜率
    good_wins: int = 0
    evil_wins: int = 0

    # 被评估模型的表现（当与基线对战时）
    eval_model_wins: int = 0
    eval_model_losses: int = 0

    # 详细统计
    merlin_assassinated: int = 0
    five_vote_failures: int = 0

    # 元信息
    eval_model: str = ""
    baseline_model: str = ""
    batch_id: str = ""


def start_vllm_server(
    model_path: str,
    port: int = 8000,
    gpu_memory_utilization: float = 0.8,
) -> subprocess.Popen:
    """启动 vLLM 服务。"""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 等待服务启动
    print("Waiting for vLLM server to start...")
    max_retries = 60
    for i in range(max_retries):
        try:
            import httpx
            response = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if response.status_code == 200:
                print(f"vLLM server ready on port {port}")
                return process
        except Exception:
            pass
        time.sleep(2)

    process.kill()
    raise RuntimeError(f"vLLM server failed to start after {max_retries * 2}s")


def build_models_config(config: EvalConfig) -> List[Tuple[str, str]]:
    """构建模型配置列表。

    如果有基线模型，返回 [eval_model, baseline_model] 交替分配给玩家。
    如果没有基线模型，所有玩家使用同一模型。
    """
    models = [(config.model_name, config.provider)]

    if config.baseline_model and config.baseline_provider:
        models.append((config.baseline_model, config.baseline_provider))

    return models


async def run_evaluation(config: EvalConfig) -> EvalResult:
    """运行评估游戏并收集结果。"""
    # 导入游戏模块（需要从项目根目录运行）
    from server.batch.runner import BatchGameRunner, BatchConfig
    from server.models.database import init_db, get_db

    await init_db()

    models = build_models_config(config)

    batch_config = BatchConfig(
        num_games=config.num_games,
        player_count=config.player_count,
        models=models,
        rotate_models=True,
        parallel=config.parallel,
        batch_tag=f"eval_{config.model_name}",
    )

    runner = BatchGameRunner(batch_config)
    batch_result = await runner.run()

    # 构建评估结果
    result = EvalResult(
        total_games=batch_result.total_games,
        completed_games=batch_result.completed_games,
        good_wins=batch_result.good_wins,
        evil_wins=batch_result.evil_wins,
        eval_model=config.model_name,
        baseline_model=config.baseline_model or config.model_name,
        batch_id=batch_result.batch_id,
    )

    # 如果有基线对战，统计被评估模型的胜率
    if config.baseline_model:
        result = await _compute_model_winrate(result, batch_result, config)

    return result


async def _compute_model_winrate(
    result: EvalResult,
    batch_result: Any,
    config: EvalConfig,
) -> EvalResult:
    """计算被评估模型在与基线对战中的胜率。

    遍历每局游戏，判断被评估模型所在阵营是否获胜。
    """
    from server.storage.repository import GameRepository

    repo = GameRepository()

    for game_id in batch_result.game_ids:
        game = await repo.get_game(game_id, reveal_all=True)
        if not game:
            continue

        winner = game.get("winner", "")
        players = game.get("players", [])

        # 统计被评估模型所在阵营
        for player in players:
            model = player.get("model_name", "")
            if model == config.model_name:
                role = player.get("role", "")
                team = "evil" if is_evil_role(role) else "good"
                if team == winner:
                    result.eval_model_wins += 1
                else:
                    result.eval_model_losses += 1

    return result


def print_results(result: EvalResult):
    """打印评估结果。"""
    print(f"\n{'='*60}")
    print(f"Avalon RL Evaluation Results")
    print(f"{'='*60}")
    print(f"  Eval Model:     {result.eval_model}")
    print(f"  Baseline:       {result.baseline_model}")
    print(f"  Batch ID:       {result.batch_id}")
    print(f"{'='*60}")
    print(f"  Total Games:    {result.total_games}")
    print(f"  Completed:      {result.completed_games}")
    print(f"{'='*60}")

    total = max(1, result.completed_games)
    print(f"  Good Wins:      {result.good_wins} ({result.good_wins/total*100:.1f}%)")
    print(f"  Evil Wins:      {result.evil_wins} ({result.evil_wins/total*100:.1f}%)")

    if result.eval_model_wins + result.eval_model_losses > 0:
        total_decisions = result.eval_model_wins + result.eval_model_losses
        winrate = result.eval_model_wins / total_decisions * 100
        print(f"{'='*60}")
        print(f"  Eval Model Win Rate:")
        print(f"    Wins:   {result.eval_model_wins}/{total_decisions} ({winrate:.1f}%)")
        print(f"    Losses: {result.eval_model_losses}/{total_decisions} ({100-winrate:.1f}%)")

    print(f"{'='*60}\n")


def save_results(result: EvalResult, output_path: str):
    """保存评估结果到 JSON 文件。"""
    data = {
        "eval_model": result.eval_model,
        "baseline_model": result.baseline_model,
        "batch_id": result.batch_id,
        "total_games": result.total_games,
        "completed_games": result.completed_games,
        "good_wins": result.good_wins,
        "evil_wins": result.evil_wins,
        "good_win_rate": f"{result.good_wins/max(1, result.completed_games)*100:.1f}%",
        "evil_win_rate": f"{result.evil_wins/max(1, result.completed_games)*100:.1f}%",
        "eval_model_wins": result.eval_model_wins,
        "eval_model_losses": result.eval_model_losses,
    }

    if result.eval_model_wins + result.eval_model_losses > 0:
        total = result.eval_model_wins + result.eval_model_losses
        data["eval_model_win_rate"] = f"{result.eval_model_wins/total*100:.1f}%"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="评估训练后的 Avalon 模型"
    )

    # 模型配置
    parser.add_argument(
        "--model_name", type=str, default="avalon-ppo",
        help="被评估模型的名称（需与 .env 中配置一致）",
    )
    parser.add_argument(
        "--provider", type=str, default="vllm",
        help="模型 provider (vllm, openai, deepseek, etc.)",
    )

    # 基线模型
    parser.add_argument(
        "--baseline_model", type=str, default=None,
        help="基线模型名称（不设则自我对战）",
    )
    parser.add_argument(
        "--baseline_provider", type=str, default=None,
        help="基线模型 provider",
    )

    # 游戏配置
    parser.add_argument(
        "--num_games", type=int, default=50,
        help="评估游戏数量（默认 50）",
    )
    parser.add_argument(
        "--player_count", type=int, default=5,
        help="每局玩家数量（默认 5）",
    )
    parser.add_argument(
        "--parallel", type=int, default=3,
        help="并行游戏数量（默认 3）",
    )

    # vLLM 自动部署
    parser.add_argument(
        "--auto_serve", action="store_true",
        help="自动启动 vLLM 服务",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="模型路径（用于 --auto_serve）",
    )
    parser.add_argument(
        "--vllm_port", type=int, default=8000,
        help="vLLM 服务端口（默认 8000）",
    )

    # 输出
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果输出 JSON 路径",
    )

    args = parser.parse_args()

    config = EvalConfig(
        model_name=args.model_name,
        provider=args.provider,
        baseline_model=args.baseline_model,
        baseline_provider=args.baseline_provider,
        num_games=args.num_games,
        player_count=args.player_count,
        parallel=args.parallel,
        auto_serve=args.auto_serve,
        model_path=args.model_path,
        vllm_port=args.vllm_port,
    )

    vllm_process = None

    try:
        # 如果需要，启动 vLLM 服务
        if config.auto_serve:
            if not config.model_path:
                print("Error: --model_path is required when using --auto_serve")
                sys.exit(1)
            vllm_process = start_vllm_server(
                config.model_path,
                config.vllm_port,
                config.vllm_gpu_memory_utilization,
            )

        # 运行评估
        result = asyncio.run(run_evaluation(config))

        # 打印结果
        print_results(result)

        # 保存结果
        output_path = args.output or f"training/eval/results_{config.model_name}.json"
        save_results(result, output_path)

    finally:
        # 清理 vLLM 进程
        if vllm_process:
            print("Stopping vLLM server...")
            vllm_process.send_signal(signal.SIGTERM)
            vllm_process.wait(timeout=10)
            print("vLLM server stopped.")


if __name__ == "__main__":
    main()
