"""veRL PPO 训练 wrapper — 在 Ray actor 进程中注册自定义扩展。

解决的问题:
  register_adv_est("precomputed") 将自定义 advantage estimator 注册到
  core_algos.ADV_ESTIMATOR_REGISTRY 全局字典中。但 Verl 的 TaskRunner
  运行在独立的 Ray actor 进程中 (ray.remote)，driver 进程中的注册
  不会传递到 Ray actor 进程，导致 get_adv_estimator_fn("precomputed") 失败。

  解决方案: 扩展 TaskRunner，在其 run() 方法中（即 Ray actor 进程内）
  重新导入并注册扩展模块。通过 monkey-patch main_ppo.TaskRunner 实现。

用法 (与 python -m verl.trainer.main_ppo 参数完全相同):
    python training/scripts/run_ppo.py \\
        data.train_files=... \\
        algorithm.adv_estimator=precomputed \\
        ...
"""

import os
import sys

# 确保项目根目录在 sys.path 中，使 `import training.xxx` 可用
# 脚本位于 training/scripts/run_ppo.py，项目根目录在两级之上
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 同时写入 PYTHONPATH 环境变量，确保 Ray worker 子进程也能 import training.xxx
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _PROJECT_ROOT not in _existing_pythonpath:
    os.environ["PYTHONPATH"] = _PROJECT_ROOT + (":" + _existing_pythonpath if _existing_pythonpath else "")

# 在 driver 进程中注册（对 Ray local_mode 或非 Ray 场景有用）
import training.verl_extensions.precomputed_adv  # noqa: F401 — registers "precomputed"

from training.verl_extensions.tool_chat_patch import patch_rlhf_dataset_for_tools
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

        import training.verl_extensions.precomputed_adv  # noqa: F401 — 注册 "precomputed" estimator
        from training.verl_extensions.tool_chat_patch import patch_rlhf_dataset_for_tools
        patch_rlhf_dataset_for_tools()

        return super().run(config)


# 替换 main_ppo 模块中的 TaskRunner，这样 run_ppo() 会自动使用我们的版本
_main_ppo_mod.TaskRunner = _AvalonTaskRunner

# === 启动 veRL PPO 训练 ===
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
