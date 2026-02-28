"""veRL PPO 训练 wrapper — 加载 Avalon 配置 + 在 Ray actor 进程中注册自定义扩展。

功能:
  1. 加载 ppo_avalon.yaml 配置文件，转为 Hydra CLI 覆盖注入 sys.argv。
     VeRL 的默认配置 (ppo_trainer.yaml) 完整保留，我们的 yaml 只覆盖
     关心的字段。命令行参数优先级最高（self_play.sh 的动态值可覆盖 yaml）。

  2. 扩展 TaskRunner，在 Ray actor 进程中注册自定义 advantage estimator
     和 tool-calling patch。

用法:
    # 自动加载 ppo_avalon.yaml + 命令行覆盖动态值
    python training/rl/scripts/run_ppo.py \\
        data.train_files=... \\
        actor_rollout_ref.model.path=... \\
        trainer.experiment_name=...

    # 也可用 --avalon-config 指定其他 yaml
    python training/rl/scripts/run_ppo.py \\
        --avalon-config path/to/custom.yaml \\
        data.train_files=...
"""

import os
import sys

# 确保项目根目录在 sys.path 中，使 `import training.xxx` 可用
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _PROJECT_ROOT not in _existing_pythonpath:
    os.environ["PYTHONPATH"] = _PROJECT_ROOT + (":" + _existing_pythonpath if _existing_pythonpath else "")

# === 加载 Avalon YAML 配置，注入为 Hydra CLI 覆盖 ===
_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "training", "rl", "configs", "ppo_avalon.yaml")


_VERL_IGNORED_KEYS = {"self_play", "experiment_name", "length_penalty"}


def _inject_yaml_overrides():
    """加载 YAML 配置并转为 Hydra 命令行覆盖。

    处理流程:
      1. 从 sys.argv 提取 --avalon-config 路径（默认使用 ppo_avalon.yaml）
      2. 解析 YAML，过滤掉 self_play 等非 VeRL 字段
      3. 将嵌套 dict 展平为 key.subkey=value 格式
      4. 注入到 sys.argv 前部（命令行参数排在后面，优先级更高）
    """
    import yaml

    config_path = _DEFAULT_CONFIG
    remaining_args = []

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--avalon-config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i].startswith("--avalon-config="):
            config_path = sys.argv[i].split("=", 1)[1]
            i += 1
        else:
            remaining_args.append(sys.argv[i])
            i += 1

    if not os.path.isfile(config_path):
        print(f"WARNING: Avalon config not found at {config_path}, using VeRL defaults only")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config:
        return

    # 过滤掉非 VeRL 的顶层字段
    for key in _VERL_IGNORED_KEYS:
        config.pop(key, None)

    def _flatten(d, prefix=""):
        overrides = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if v is None or (isinstance(v, str) and v == "???"):
                continue
            elif isinstance(v, dict):
                overrides.extend(_flatten(v, key))
            elif isinstance(v, list):
                items = ",".join(str(x) for x in v)
                overrides.append(f"++{key}=[{items}]")
            elif isinstance(v, bool):
                overrides.append(f"++{key}={'true' if v else 'false'}")
            else:
                overrides.append(f"++{key}={v}")
        return overrides

    yaml_overrides = _flatten(config)
    sys.argv = [sys.argv[0]] + yaml_overrides + remaining_args
    print(f"Loaded Avalon config from {config_path} ({len(yaml_overrides)} overrides)")


_inject_yaml_overrides()

# 在 driver 进程中注册（对 Ray local_mode 或非 Ray 场景有用）
import training.rl.verl_extensions.precomputed_adv  # noqa: F401 — registers "precomputed"

from training.rl.verl_extensions.tool_chat_patch import patch_rlhf_dataset_for_tools
patch_rlhf_dataset_for_tools()  # Patch RLHFDataset 支持 tool-calling prompts

# === 核心修复: 扩展 TaskRunner，确保在 Ray actor 进程中也注册自定义扩展 ===
import verl.trainer.main_ppo as _main_ppo_mod

_OriginalTaskRunner = _main_ppo_mod.TaskRunner


class _AvalonTaskRunner(_OriginalTaskRunner):
    """确保自定义 advantage estimator 在 Ray actor 进程中被注册。

    Verl 的 run_ppo() 通过 ray.remote 创建 TaskRunner actor，
    actor 运行在独立进程中，driver 进程中的 register_adv_est 注册不会传递。
    此类在 run() 方法中重新导入扩展模块来完成注册。
    """

    # 在类定义时捕获项目根目录路径，cloudpickle 序列化时会保留该值
    _project_root = _PROJECT_ROOT

    def run(self, config):
        # Ray actor 是独立进程，需要在此处设置 sys.path 并注册扩展
        import sys
        if self._project_root not in sys.path:
            sys.path.insert(0, self._project_root)

        import training.rl.verl_extensions.precomputed_adv  # noqa: F401 — 注册 "precomputed" estimator
        from training.rl.verl_extensions.tool_chat_patch import patch_rlhf_dataset_for_tools
        patch_rlhf_dataset_for_tools()

        return super().run(config)


# 替换 main_ppo 模块中的 TaskRunner，这样 run_ppo() 会自动使用我们的版本
_main_ppo_mod.TaskRunner = _AvalonTaskRunner

# === 启动 veRL PPO 训练 ===
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
