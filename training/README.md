# Avalon RL Training with Verl

使用 [Verl](https://github.com/verl-project/verl) 框架对 Avalon 游戏中的 LLM 智能体进行强化学习训练。

## 目录结构

```
training/
├── data/
│   └── preprocess.py              # JSONL 轨迹 -> Verl parquet 格式转换
├── reward/
│   └── avalon_reward.py           # Avalon 奖励函数（游戏胜负 / 预计算 advantage）
├── critic/
│   ├── model.py                   # Critic 模型定义 (base LLM + value head)
│   ├── train.py                   # Critic 训练 (MSE loss)
│   └── infer.py                   # Critic 推理 V(s)
├── advantage/
│   └── compute.py                 # Episode 级 GAE 计算
├── verl_extensions/
│   └── precomputed_adv.py         # 自定义 Verl advantage estimator
├── configs/
│   └── ppo_avalon.yaml            # 训练配置
├── scripts/
│   └── self_play.sh               # 自博弈循环 (8 步流程)
├── eval/
│   └── evaluate.py                # 训练后模型评估
├── run_batch.py                   # 批量对局 CLI 工具
├── requirements.txt               # 训练依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：请根据你的 CUDA 版本安装对应的 PyTorch 和 vLLM。参考 [Verl 安装指南](https://verl.readthedocs.io/en/latest/start/install.html)。

### 2. 下载基础模型

```bash
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-7B-Instruct')"
```

### 3. 开始训练

自博弈循环会自动完成完整的 8 步流程：

```bash
# 默认: 5 轮自博弈
bash training/scripts/self_play.sh

# 自定义配置
ROUNDS=10 GAMES_PER_ROUND=200 GAE_GAMMA=0.99 GAE_LAM=0.95 \
    bash training/scripts/self_play.sh
```

### 4. 评估训练效果

```bash
python -m training.eval.evaluate \
    --model_path checkpoints/avalon-self-play/self-play-gae-r5/actor/huggingface \
    --num_games 50 \
    --player_count 5
```

## 训练原理

### 为什么不能用 GRPO

GRPO 通过对同一 prompt 多次采样再组内比较来计算优势。但在 Avalon 场景中，reward 来自游戏最终胜负（在数据收集阶段已确定），同一 prompt 的所有采样会得到相同的 reward，导致 GRPO 优势恒为 0，无法学习。

### 为什么需要 Episode-level GAE

Verl 内置的 PPO + GAE 是 **token 级别** 的 —— 在单个 response 内逐 token 计算 advantage，无法跨多个决策点建立时序关联。但 Avalon 的核心挑战是 **credit assignment**：同一局游戏中一个玩家会做出多次决策（讨论、投票、执行任务等），最终只有一个胜负 reward，我们需要区分哪些决策是好的、哪些是坏的。

因此我们在 Verl 外部完成 **episode 级别** 的 GAE 计算，然后将预计算的 advantage 注入 Verl 做 actor 更新。整体训练仍然是 **on-policy** 的自博弈循环（每轮用当前策略采集新数据），只是 GAE 的计算从 Verl 内部移到了外部，以支持跨决策点的 credit assignment：

1. 独立训练一个 Critic 模型估计每个决策点的状态价值 V(s)
2. 用 GAE 算法跨决策点计算 advantage，实现 credit assignment
3. 将预计算的 advantage 注入 Verl 做 actor 更新

### 一局游戏的决策轨迹示例

以 Player 1（好人阵营）为例：

```
t=0: discussion(r1)   r=0,  V(s0)=0.3   δ0 = 0 + γ·V(s1) - V(s0)
t=1: team_vote(r1)    r=0,  V(s1)=0.4   δ1 = 0 + γ·V(s2) - V(s1)
t=2: quest_exec(r1)   r=0,  V(s2)=0.5   δ2 = 0 + γ·V(s3) - V(s2)
t=3: discussion(r2)   r=0,  V(s3)=0.6   δ3 = 0 + γ·V(s4) - V(s3)
t=4: team_vote(r2)    r=0,  V(s4)=0.2   δ4 = 0 + γ·V(s5) - V(s4)  ← 投了坏队,V骤降
t=5: discussion(r3)   r=0,  V(s5)=-0.1  δ5 = 0 + γ·V(s6) - V(s5)
t=6: team_vote(r3)    r=0,  V(s6)=0.1   δ6 = 0 + γ·0 - V(s6)
t=7: (terminal)       r=-1                                           ← 好人输了
```

- `δ4` 是负的（投票后 V 从 0.2 变成 -0.1）→ 这个投票决策会被惩罚
- 这就是 GAE 的 **credit assignment** 能力

## 数据流（每轮 Self-Play）

```
Step 1: vLLM 跑游戏
    ↓
Step 2: 导出 JSONL 轨迹
    ↓
Step 3: Critic 推理 V(s)           ← 为每个决策点估计状态价值
    ↓
Step 4: GAE 计算 advantage          ← 按 (game_id, player_seat) 分组
    ↓
Step 5: 数据预处理 → parquet        ← 注入 precomputed_advantage
    ↓
Step 6: Verl 训练 actor             ← adv_estimator=precomputed
    ↓
Step 7: 训练 Critic                 ← MSE(V(s), G_t)
    ↓
Step 8: 合并 actor checkpoint       ← 下一轮模型
```

每轮结束后，更新后的 actor 用于下一轮的游戏采样，更新后的 critic 用于下一轮的 V(s) 推理。

## 各模块详解

### Critic 模型 (`training/critic/model.py`)

- 基于同一个 base model (Qwen2.5-7B-Instruct) + 线性 value head
- 使用 `AutoModelForSequenceClassification(num_labels=1)` 实现
- 输入: tokenized prompt (游戏状态描述) → 输出: scalar V(s)
- Value head 初始化为接近 0 的小值，确保冷启动时 V(s) ≈ 0

### Critic 训练 (`training/critic/train.py`)

```bash
python -m training.critic.train \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --data_file data/processed/train.parquet \
    --output_dir training/self_play/critic \
    --epochs 3 --lr 1e-5 --bf16
```

- Loss: `MSE(V(prompt), G_t)`
- G_t 是 discounted return：从终端 reward 向前折扣
  - `G_T = R_final`，`G_t = γ · G_{t+1}`（中间 reward = 0）

### Critic 推理 (`training/critic/infer.py`)

```bash
python -m training.critic.infer \
    --model_path training/self_play/critic \
    --input_jsonl data/trajectories.jsonl \
    --output_json data/values.json \
    --batch_size 8 --bf16
```

对 JSONL 轨迹文件中的每个决策点 batch 推理 V(s)。

### GAE 计算 (`training/advantage/compute.py`)

```bash
python -m training.advantage.compute \
    --input_jsonl data/trajectories.jsonl \
    --values_json data/values.json \
    --output_json data/advantages.json \
    --gamma 0.99 --lam 0.95
```

核心逻辑：

1. 按 `(game_id, player_seat)` 分组，每组构成一个 episode
2. 组内按 `seq_num` 排序
3. 计算 GAE advantage 和 discounted return:

```python
# r_t = 0 for t < T, r_T = +1/-1 (game outcome)
# V(s_{T+1}) = 0 (terminal)
for t in reversed(range(T)):
    delta = rewards[t] + gamma * next_value - values[t]
    advantage = delta + gamma * lam * next_advantage
```

### Verl 扩展 (`training/verl_extensions/precomputed_adv.py`)

自定义 advantage estimator，注册为 `precomputed`：

- reward function 返回的就是预计算的 advantage
- 将 per-sequence advantage 广播到所有 response token
- 不需要 Verl 内置的 critic 模型

### 数据预处理 (`training/data/preprocess.py`)

新增 `--advantages_file` 参数：

```bash
python -m training.data.preprocess \
    --input_jsonl data/trajectories.jsonl \
    --output_dir data/processed \
    --advantages_file data/advantages.json
```

当指定 advantage 文件时：
- `ground_truth["precomputed_advantage"]` 被注入每个训练样本
- `discounted_return` 列被加入 parquet（用于 critic 训练）

### 奖励函数 (`training/reward/avalon_reward.py`)

支持两种模式：
1. **预计算 advantage**：如果 `ground_truth` 中有 `precomputed_advantage`，直接返回
2. **回退**：使用游戏胜负奖励（阵营获胜 +1.0 / 失败 -1.0）

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GAE_GAMMA` (γ) | 0.99 | 折扣因子，控制未来 reward 的衰减 |
| `GAE_LAM` (λ) | 0.95 | GAE lambda。λ=1 退化为 Monte Carlo，λ=0 退化为 TD(0) |
| `CRITIC_LR` | 1e-5 | Critic 学习率（比 actor 的 1e-6 大） |
| `CRITIC_EPOCHS` | 3 | 每轮 self-play 的 critic 训练 epoch 数 |
| `CRITIC_BATCH_SIZE` | 4 | Critic 训练和推理的 batch size |
| `TRAIN_EPOCHS` | 20 | 每轮 Verl actor 训练步数 |
| `GAMES_PER_ROUND` | 100 | 每轮自博弈的游戏数量 |
| `ROUNDS` | 5 | 自博弈总轮数 |

## 冷启动

第一轮训练时 Critic 尚未训练，V(s) ≈ 0，GAE 退化为纯 discounted reward：

```
A_t = (γλ)^(T-t) · R_final
```

这本身就是一个合理的起点 —— 越靠近终局的决策获得越大的 advantage 权重。后续轮次随着 Critic 精度提升，credit assignment 会逐步改善，能够识别出关键的中间决策（如投票错误、暴露身份等）。
