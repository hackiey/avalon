#!/usr/bin/env python3
"""
Command-line interface for batch game running and data export.

Usage:
    # Run 100 games
    python run_batch.py run --num-games 100 --models "qwen-plus:qwen"
    
    # Run with multiple models (rotating among players)
    python run_batch.py run --num-games 100 --models "qwen-plus:qwen,gpt-4o:openai"
    
    # Export trajectories from a batch
    python run_batch.py export --batch-id abc123 --output ./data/training.jsonl
    
    # List all batches
    python run_batch.py list
"""

import argparse
import asyncio
import sys
from typing import List, Tuple


def parse_models(models_str: str) -> List[Tuple[str, str]]:
    """Parse model string like 'qwen-plus:qwen,gpt-4o:openai'."""
    models = []
    for item in models_str.split(","):
        item = item.strip()
        if ":" in item:
            name, provider = item.rsplit(":", 1)
            models.append((name.strip(), provider.strip()))
        else:
            # Default provider based on model name
            models.append((item, "openai"))
    return models


async def cmd_run(args):
    """Run a batch of games."""
    from server.batch.runner import BatchGameRunner, BatchConfig
    
    models = parse_models(args.models) if args.models else []
    
    if not models:
        print("Error: --models is required")
        print("Example: --models 'qwen-plus:qwen' or --models 'gpt-4o:openai,claude-3-5-sonnet:anthropic'")
        sys.exit(1)
    
    config = BatchConfig(
        num_games=args.num_games,
        player_count=args.player_count,
        models=models,
        rotate_models=not args.no_rotate,
        parallel=args.parallel,
        batch_tag=args.tag,
    )
    
    runner = BatchGameRunner(config)
    result = await runner.run()
    
    print(f"\nBatch ID: {result.batch_id}")
    print(f"To export: python run_batch.py export --batch-id {result.batch_id} --output ./data/{result.batch_id}.jsonl")


async def cmd_export(args):
    """Export training data from games."""
    from server.batch.exporter import TrainingDataExporter
    
    game_ids = args.game_ids.split(",") if args.game_ids else None
    
    # Default: only export batch rollout data (exclude web UI games)
    exporter = TrainingDataExporter()
    await exporter.export_trajectories(
        output_path=args.output,
        batch_id=args.batch_id,
        tag=args.tag,
        game_ids=game_ids,
        include_web=args.include_web,
    )


async def cmd_list(args):
    """List all batches."""
    from server.batch.exporter import list_batches
    
    batches = await list_batches()
    
    if not batches:
        print("No batches found.")
        return
    
    print(f"\n{'Batch ID':<12} {'Games':<8} {'Done':<8} {'Good':<8} {'Evil':<8} {'Win Rate':<10}")
    print("-" * 60)
    
    for b in batches:
        print(f"{b['batch_id']:<12} {b['total_games']:<8} {b['completed']:<8} "
              f"{b['good_wins']:<8} {b['evil_wins']:<8} {b['good_win_rate']:<10}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch game runner and training data exporter"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a batch of games")
    run_parser.add_argument("-n", "--num-games", type=int, default=100,
                           help="Number of games to run (default: 100)")
    run_parser.add_argument("-p", "--player-count", type=int, default=5,
                           help="Players per game (default: 5)")
    run_parser.add_argument("-m", "--models", type=str, required=True,
                           help="Models to use, format: 'model:provider,model:provider'")
    run_parser.add_argument("--no-rotate", action="store_true",
                           help="Don't rotate models among players")
    run_parser.add_argument("--parallel", type=int, default=1,
                           help="Number of games to run in parallel (default: 1)")
    run_parser.add_argument("--tag", type=str,
                           help="Experiment tag for this batch run (e.g., 'exp_v1')")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export training trajectories")
    export_parser.add_argument("-b", "--batch-id", type=str,
                              help="Batch ID to export")
    export_parser.add_argument("-t", "--tag", type=str,
                              help="Experiment tag to export (e.g., 'experiment_v1')")
    export_parser.add_argument("-g", "--game-ids", type=str,
                              help="Specific game IDs (comma-separated)")
    export_parser.add_argument("-o", "--output", type=str, default="./data/trajectories.jsonl",
                              help="Output file path (default: ./data/trajectories.jsonl)")
    export_parser.add_argument("--include-web", action="store_true",
                              help="Include web UI games (default: only batch rollout data)")
    
    # List command
    subparsers.add_parser("list", help="List all batch runs")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "export":
        asyncio.run(cmd_export(args))
    elif args.command == "list":
        asyncio.run(cmd_list(args))


if __name__ == "__main__":
    main()
