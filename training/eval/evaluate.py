"""受控 A/B 实验评估：逐角色替换 eval 模型，与全 baseline 对照组比较。

评估方法:
    - 对照组: 所有玩家均使用 baseline 模型
    - 实验组: 将指定角色替换为 eval 模型，其余保持 baseline
    - 对每个角色独立比较该角色所在阵营的胜率变化

    这种方法消除了模型-角色分配的随机性（辛普森悖论），
    提供每个角色维度的精确能力评估。

用法:
    python -m training.eval.evaluate \
        --experiment experiments/my_exp \
        --baseline /path/to/baseline_model \
        --num_games 50 --parallel 5 \
        --eval_port 8001 --baseline_port 8002 \
        --tp 4 \
        --eval_devices 4,5,6,7 \
        --baseline_devices 0,1,2,3

    # 只评估指定轮次和角色
    python -m training.eval.evaluate \
        --experiment my_exp \
        --baseline /path/to/baseline_model \
        --num_games 50 --parallel 10 \
        --eval_port 8001 --baseline_port 8002 \
        --tp 4 \
        --eval_devices 4,5,6,7 \
        --baseline_devices 0,1,2,3 \
        --rounds 3 \
        --roles merlin
"""

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 固定 provider alias ──────────────────────────────────────────────────────
_EVAL_ALIAS = "eval_model"
_BASE_ALIAS = "baseline_model"


def _setup_env(eval_port: int, baseline_port: int) -> None:
    """在 server 模块被导入之前设置 provider 路由环境变量。"""
    os.environ["AVAILABLE_MODELS"] = (
        f"{_EVAL_ALIAS}:vllm_eval,{_BASE_ALIAS}:vllm_baseline"
    )
    os.environ["VLLM_EVAL_API_KEY"] = "vllm"
    os.environ["VLLM_EVAL_BASE_URL"] = f"http://localhost:{eval_port}/v1"
    os.environ["VLLM_BASELINE_API_KEY"] = "vllm"
    os.environ["VLLM_BASELINE_BASE_URL"] = f"http://localhost:{baseline_port}/v1"


# ── 数据类 ────────────────────────────────────────────────────────────────────

@dataclass
class GroupResult:
    """单个实验组（对照组或某角色实验组）的结果。"""
    label: str
    completed: int = 0
    good_wins: int = 0
    evil_wins: int = 0
    errors: int = 0


@dataclass
class RoundResult:
    """一个 checkpoint 的完整评估结果。"""
    round_num: int
    checkpoint_path: str
    control: Optional[GroupResult] = None
    experiments: Dict[str, GroupResult] = field(default_factory=dict)


# ── vLLM 进程管理 ─────────────────────────────────────────────────────────────

def start_vllm(
    model_path: str,
    served_name: str,
    port: int,
    tp: int,
    gpu_util: float,
    log_file: Optional[str] = None,
    reasoning_parser: Optional[str] = None,
    cuda_devices: Optional[str] = None,
) -> subprocess.Popen:
    """启动 vLLM OpenAI 兼容服务，等待健康检查通过后返回进程对象。"""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", served_name,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_util),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--trust-remote-code",
    ]
    if reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]

    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    print(f"[vLLM] Starting '{served_name}' on port {port}")
    print(f"[vLLM]   model: {model_path}")
    if cuda_devices is not None:
        print(f"[vLLM]   CUDA_VISIBLE_DEVICES={cuda_devices}")

    log_fh = open(log_file, "w") if log_file else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, env=env)

    print(f"[vLLM] Waiting for port {port} ", end="", flush=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            import httpx
            r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if r.status_code == 200:
                print(" ready.")
                return proc
        except Exception:
            pass
        time.sleep(3)
        print(".", end="", flush=True)

    proc.kill()
    raise RuntimeError(f"vLLM on port {port} failed to start within 600s")


def stop_vllm(proc: subprocess.Popen, port: int) -> None:
    """终止 vLLM 进程并等待 GPU 显存释放。"""
    if proc is None:
        return
    print(f"[vLLM] Stopping server on port {port}...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    for _ in range(30):
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if not result.stdout.strip():
            break
        time.sleep(2)
    print(f"[vLLM] Port {port} released.")


# ── 实验轮次发现 ──────────────────────────────────────────────────────────────

def discover_rounds(
    experiment_dir: Path,
    rounds: Optional[List[int]] = None,
) -> List[Tuple[int, Path]]:
    """扫描 experiments/<exp>/checkpoints/round_N/ 并返回 (round_num, path) 列表。"""
    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    found = []
    for d in sorted(checkpoint_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("round_")):
            continue
        try:
            n = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if rounds is None or n in rounds:
            found.append((n, d))

    if not found:
        raise FileNotFoundError(f"No round checkpoints in {checkpoint_dir}")

    return found


# ── 实验执行 ──────────────────────────────────────────────────────────────────

async def run_group(
    label: str,
    role_model_map: Optional[Dict[str, Tuple[str, str]]],
    num_games: int,
    player_count: int,
    parallel: int,
) -> GroupResult:
    """运行单个实验组（对照组或某角色实验组）。"""
    from server.batch.runner import BatchGameRunner, BatchConfig

    config = BatchConfig(
        num_games=num_games,
        player_count=player_count,
        models=[(_BASE_ALIAS, "vllm_baseline")],
        role_model_map=role_model_map,
        parallel=parallel,
        batch_tag=f"eval_{label}",
        no_mongo=True,
    )

    runner = BatchGameRunner(config)
    batch = await runner.run()

    return GroupResult(
        label=label,
        completed=batch.completed_games,
        good_wins=batch.good_wins,
        evil_wins=batch.evil_wins,
        errors=batch.failed_games,
    )


async def run_round(
    round_num: int,
    checkpoint_path: Path,
    num_games: int,
    player_count: int,
    parallel: int,
    roles: List[str],
) -> RoundResult:
    """运行对照组 + 逐角色实验组。"""
    result = RoundResult(round_num=round_num, checkpoint_path=str(checkpoint_path))

    # 对照组: 全 baseline
    print(f"\n  [control] {num_games} games (all baseline)...")
    result.control = await run_group(
        label=f"r{round_num}_control",
        role_model_map=None,
        num_games=num_games,
        player_count=player_count,
        parallel=parallel,
    )

    # 逐角色实验组
    for role in roles:
        print(f"  [{role}] {num_games} games (eval as {role})...")
        result.experiments[role] = await run_group(
            label=f"r{round_num}_{role}",
            role_model_map={role: (_EVAL_ALIAS, "vllm_eval")},
            num_games=num_games,
            player_count=player_count,
            parallel=parallel,
        )

    return result


# ── 统计计算 ──────────────────────────────────────────────────────────────────

def _pct(num: int, den: int) -> float:
    return round(num / den * 100, 1) if den > 0 else 0.0


def _delta_ci(
    ctrl_wins: int, ctrl_n: int,
    exp_wins: int, exp_n: int,
) -> Dict:
    """计算两个比例之差的点估计和 95% 置信区间。"""
    p1 = ctrl_wins / ctrl_n if ctrl_n > 0 else 0.0
    p2 = exp_wins / exp_n if exp_n > 0 else 0.0
    delta = p2 - p1

    se = math.sqrt(
        p1 * (1 - p1) / max(1, ctrl_n) +
        p2 * (1 - p2) / max(1, exp_n)
    )
    z = 1.96
    ci_lo = delta - z * se
    ci_hi = delta + z * se
    significant = (ci_lo > 0) or (ci_hi < 0)

    return {
        "ctrl_wr": round(p1 * 100, 1),
        "exp_wr": round(p2 * 100, 1),
        "delta_pp": round(delta * 100, 1),
        "ci_95": [round(ci_lo * 100, 1), round(ci_hi * 100, 1)],
        "significant": significant,
    }


def _derive(rr: RoundResult, player_count: int = 5) -> dict:
    """从 RoundResult 计算所有派生统计。"""
    from game.roles import is_evil_role
    from game.rules import get_rules

    ctrl = rr.control
    ctrl_data = {
        "completed": ctrl.completed,
        "errors": ctrl.errors,
        "good_wins": ctrl.good_wins,
        "evil_wins": ctrl.evil_wins,
        "good_win_rate": _pct(ctrl.good_wins, ctrl.completed),
        "evil_win_rate": _pct(ctrl.evil_wins, ctrl.completed),
    }

    # 统计各角色席位数
    rules = get_rules(player_count)
    seat_counts: Dict[str, int] = {}
    for r in rules.roles:
        seat_counts[r.value] = seat_counts.get(r.value, 0) + 1

    roles_data = {}
    for role, exp in rr.experiments.items():
        team = "evil" if is_evil_role(role) else "good"

        if team == "evil":
            stats = _delta_ci(ctrl.evil_wins, ctrl.completed,
                              exp.evil_wins, exp.completed)
        else:
            stats = _delta_ci(ctrl.good_wins, ctrl.completed,
                              exp.good_wins, exp.completed)

        roles_data[role] = {
            "team": team,
            "seats": seat_counts.get(role, 1),
            "completed": exp.completed,
            "errors": exp.errors,
            "good_win_rate": _pct(exp.good_wins, exp.completed),
            "evil_win_rate": _pct(exp.evil_wins, exp.completed),
            **stats,
        }

    # 综合分数: 按席位数加权平均各角色的 delta
    total_seats = sum(d["seats"] for d in roles_data.values())
    composite = (
        sum(d["seats"] * d["delta_pp"] for d in roles_data.values()) / total_seats
        if total_seats > 0 else 0.0
    )

    return {
        "control": ctrl_data,
        "roles": roles_data,
        "composite_score": round(composite, 1),
    }


# ── 输出 ──────────────────────────────────────────────────────────────────────

def _delta_str(v: float) -> str:
    return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"


def print_summary(
    results: List[RoundResult],
    baseline_path: str,
    experiment: str = "",
    player_count: int = 5,
) -> None:
    """打印逐角色 A/B 实验汇总表。"""
    W = 100
    import datetime

    print(f"\n{'='*W}")
    print(f"  Role-Based A/B Evaluation Summary")
    if experiment:
        print(f"  Experiment : {experiment}")
    print(f"  Baseline   : {baseline_path}")
    print(f"  Evaluated  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*W}")

    for rr in results:
        d = _derive(rr, player_count)
        ctrl = d["control"]

        print(f"\n  Round {rr.round_num}  (checkpoint: {rr.checkpoint_path})")
        print(f"  Control: {ctrl['completed']} games, "
              f"Good {ctrl['good_win_rate']}%, Evil {ctrl['evil_win_rate']}%")

        hdr = (f"  {'Role':<16} {'Team':<6} {'Seats':>5} "
               f"{'CtrlWR':>7} {'ExpWR':>7} {'Δ(pp)':>7} "
               f"{'95% CI':>18} {'Sig':>4}")
        print(f"  {'─'*96}")
        print(hdr)
        print(f"  {'─'*96}")

        for role, rd in d["roles"].items():
            ci_str = f"[{_delta_str(rd['ci_95'][0])}, {_delta_str(rd['ci_95'][1])}]"
            sig_str = " *" if rd["significant"] else ""
            print(
                f"  {role:<16} {rd['team']:<6} {rd['seats']:>5} "
                f"{rd['ctrl_wr']:>6.1f}% {rd['exp_wr']:>6.1f}% "
                f"{_delta_str(rd['delta_pp']):>7} {ci_str:>18}{sig_str}"
            )

        print(f"  {'─'*96}")
        print(f"  Composite Score: {_delta_str(d['composite_score'])} pp")

    # 多轮趋势
    if len(results) > 1:
        composites = [_derive(r, player_count)["composite_score"] for r in results]
        trend = (
            "↑ improving" if composites[-1] > composites[0]
            else "↓ declining" if composites[-1] < composites[0]
            else "→ stable"
        )
        best_i = max(range(len(results)), key=lambda i: composites[i])
        print(f"\n  {'─'*96}")
        print(f"  Trend: {trend}   Best round: {results[best_i].round_num}   "
              f"Composite range: [{min(composites):+.1f}, {max(composites):+.1f}]")

    print(f"\n{'='*W}\n")


def save_results(
    results: List[RoundResult],
    output_path: str,
    meta: Optional[dict] = None,
    player_count: int = 5,
) -> None:
    """保存评估结果到 JSON。"""
    rounds_data = []
    for rr in results:
        d = _derive(rr, player_count)
        rounds_data.append({
            "round": rr.round_num,
            "checkpoint": rr.checkpoint_path,
            **d,
        })

    summary = {}
    if results:
        composites = [_derive(r, player_count)["composite_score"] for r in results]
        best_i = max(range(len(results)), key=lambda i: composites[i])
        summary = {
            "best_round": results[best_i].round_num,
            "best_composite_score": composites[best_i],
            "composite_range": [min(composites), max(composites)],
            "trend": (
                "improving" if len(composites) > 1 and composites[-1] > composites[0]
                else "declining" if len(composites) > 1 and composites[-1] < composites[0]
                else "stable"
            ),
        }

    output = {
        "meta": meta or {},
        "summary": summary,
        "rounds": rounds_data,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="受控 A/B 实验评估：逐角色替换 eval 模型 vs 全 baseline"
    )
    parser.add_argument(
        "--experiment", required=True,
        help="实验名称或路径（如 'my_exp' 或 'experiments/my_exp'）",
    )
    parser.add_argument(
        "--baseline", required=True,
        help="baseline 模型路径（通过 vLLM 启动）",
    )
    parser.add_argument(
        "--rounds", type=int, nargs="+", default=None,
        help="指定要评估的轮次（默认: 所有）",
    )
    parser.add_argument(
        "--roles", nargs="+", default=None,
        help="要评估的角色（默认: 所有唯一角色）。如: merlin assassin",
    )
    parser.add_argument("--num_games", type=int, default=50,
                        help="每个实验组的游戏数（默认 50）")
    parser.add_argument("--player_count", type=int, default=5,
                        help="每局玩家数（默认 5）")
    parser.add_argument("--parallel", type=int, default=5,
                        help="并行游戏数（默认 5）")
    parser.add_argument("--eval_port", type=int, default=8001,
                        help="eval 模型的 vLLM 端口（默认 8001）")
    parser.add_argument("--baseline_port", type=int, default=8002,
                        help="baseline 模型的 vLLM 端口（默认 8002）")
    parser.add_argument("--tp", type=int, default=4,
                        help="vLLM tensor-parallel size（默认 4）")
    parser.add_argument("--gpu_util", type=float, default=0.85,
                        help="vLLM GPU 显存占用率（默认 0.85）")
    parser.add_argument("--eval_devices", type=str, default=None,
                        help="eval vLLM 使用的 GPU，如 '4,5,6,7'")
    parser.add_argument("--baseline_devices", type=str, default=None,
                        help="baseline vLLM 使用的 GPU，如 '0,1,2,3'")
    parser.add_argument("--reasoning_parser", type=str, default=None,
                        help="vLLM reasoning parser（如 'deepseek_r1'，可选）")
    parser.add_argument("--output", type=str, default=None,
                        help="结果 JSON 输出路径（默认: <experiment_dir>/eval_results.json）")
    args = parser.parse_args()

    # ── 解析实验目录 ──────────────────────────────────────────────────────────
    exp_path = Path(args.experiment)
    if not exp_path.exists():
        exp_path = Path("experiments") / args.experiment
    if not exp_path.exists():
        print(f"ERROR: Experiment directory not found: {args.experiment}")
        sys.exit(1)

    # ── 在任何 server 模块导入前设置 provider 环境变量 ────────────────────────
    _setup_env(args.eval_port, args.baseline_port)

    # ── 确定要评估的角色 ──────────────────────────────────────────────────────
    from game.rules import get_rules

    if args.roles:
        roles_to_eval = args.roles
    else:
        rules = get_rules(args.player_count)
        roles_to_eval = list(dict.fromkeys(r.value for r in rules.roles))

    # ── 发现轮次 ──────────────────────────────────────────────────────────────
    rounds = discover_rounds(exp_path, args.rounds)
    output_path = args.output or str(exp_path / "eval_results.json")
    log_dir = exp_path / "logs"
    log_dir.mkdir(exist_ok=True)

    groups_per_round = 1 + len(roles_to_eval)  # control + per-role
    total_games_per_round = args.num_games * groups_per_round

    print(f"\n{'='*60}")
    print(f"  Avalon Role-Based A/B Evaluation")
    print(f"  Experiment : {exp_path}")
    print(f"  Baseline   : {args.baseline}")
    print(f"  Rounds     : {[r for r, _ in rounds]}")
    print(f"  Roles      : {roles_to_eval}")
    print(f"  Games/group: {args.num_games}  "
          f"Groups/round: {groups_per_round}  "
          f"Total/round: {total_games_per_round}")
    print(f"  Parallel   : {args.parallel}")
    print(f"  Eval port  : {args.eval_port}  Baseline port: {args.baseline_port}")
    print(f"  TP={args.tp}  gpu_util={args.gpu_util}")
    print(f"{'='*60}\n")

    import datetime
    meta = {
        "experiment": str(exp_path),
        "baseline": args.baseline,
        "method": "role_based_ab_test",
        "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rounds_evaluated": [r for r, _ in rounds],
        "roles_evaluated": roles_to_eval,
        "games_per_group": args.num_games,
        "player_count": args.player_count,
        "parallel": args.parallel,
    }

    all_results: List[RoundResult] = []
    baseline_proc = None

    try:
        # ── baseline 全程运行 ─────────────────────────────────────────────────
        baseline_proc = start_vllm(
            model_path=args.baseline,
            served_name=_BASE_ALIAS,
            port=args.baseline_port,
            tp=args.tp,
            gpu_util=args.gpu_util,
            log_file=str(log_dir / "eval_baseline_vllm.log"),
            reasoning_parser=args.reasoning_parser,
            cuda_devices=args.baseline_devices,
        )

        # ── 逐轮评估 ─────────────────────────────────────────────────────────
        for round_num, checkpoint_path in rounds:
            print(f"\n{'─'*60}")
            print(f"  Round {round_num}: {checkpoint_path}")
            print(f"{'─'*60}")

            eval_proc = None
            try:
                eval_proc = start_vllm(
                    model_path=str(checkpoint_path),
                    served_name=_EVAL_ALIAS,
                    port=args.eval_port,
                    tp=args.tp,
                    gpu_util=args.gpu_util,
                    log_file=str(log_dir / f"eval_r{round_num}_vllm.log"),
                    reasoning_parser=args.reasoning_parser,
                    cuda_devices=args.eval_devices,
                )

                result = asyncio.run(run_round(
                    round_num=round_num,
                    checkpoint_path=checkpoint_path,
                    num_games=args.num_games,
                    player_count=args.player_count,
                    parallel=args.parallel,
                    roles=roles_to_eval,
                ))
                all_results.append(result)

                save_results(all_results, output_path, meta=meta,
                             player_count=args.player_count)

            finally:
                if eval_proc:
                    stop_vllm(eval_proc, args.eval_port)

    finally:
        if baseline_proc:
            stop_vllm(baseline_proc, args.baseline_port)

    # ── 汇总输出 ──────────────────────────────────────────────────────────────
    if all_results:
        print_summary(all_results, args.baseline, experiment=str(exp_path),
                      player_count=args.player_count)
        save_results(all_results, output_path, meta=meta,
                     player_count=args.player_count)


if __name__ == "__main__":
    main()
