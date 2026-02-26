# Avalon LLM Training Engine

一个用于训练和观察 LLM 玩阿瓦隆游戏的引擎。支持多个 LLM 之间的对战、人类玩家参与、实时观战、历史回放、统计分析、批量对局、训练数据导出，以及基于 RL 自博弈的模型训练。

## 功能特性

- **多 LLM 对战**: 支持 OpenAI、Anthropic、DeepSeek、VLLM 等多个 LLM 提供商
- **人类参与**: 人类玩家可以与 AI 一起游戏
- **实时观战**: 通过 Web 界面实时观看游戏进程
- **历史回放**: 回放历史对局，逐步查看游戏过程
- **统计分析**: 查看模型胜率、角色胜率等统计数据
- **批量对局**: 通过 CLI 工具批量运行游戏，支持并行执行
- **训练数据导出**: 将游戏轨迹导出为 JSONL 格式，用于模型训练
- **RL 训练**: 基于自博弈 + Episode-level GAE + 外部 Critic 进行在线策略强化学习（Verl + PPO）

## 快速开始

### 1. 环境配置

复制环境变量配置文件并填写 API 密钥:

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置你的 LLM API 密钥和数据库连接:

```env
# OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODELS=gpt-4o,gpt-4o-mini

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_MODELS=claude-3-5-sonnet-20241022

# DeepSeek (可选)
DEEPSEEK_API_KEY=xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODELS=deepseek-chat

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=avalon
```

### 2. 安装后端依赖

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 启动 MongoDB

确保 MongoDB 已安装并启动:

```bash
# macOS (Homebrew)
brew services start mongodb-community

# 或使用 Docker
docker run -d -p 27017:27017 --name avalon-mongo mongo:latest
```

### 4. 安装前端依赖

```bash
cd web
pnpm install
```

### 5. 启动服务

启动后端服务 (端口 8000):

```bash
uvicorn server.main:asgi_app --host 0.0.0.0 --port 8000
```

启动前端开发服务器 (端口 5173):

```bash
cd web
pnpm dev
```

访问 http://localhost:5173 开始使用。

## 批量对局 & 训练数据导出

使用 `training/run_batch.py` CLI 工具可以批量运行游戏并导出训练数据:

```bash
# 运行 100 局游戏 (单模型)
python -m training.run_batch run -n 100 -m "qwen-plus:qwen"

# 运行 100 局游戏 (多模型轮换)
python -m training.run_batch run -n 100 -m "qwen-plus:qwen,gpt-4o:openai"

# 并行运行 (4 局同时进行)
python -m training.run_batch run -n 100 -m "gpt-4o:openai" --parallel 4

# 不使用 MongoDB，直接输出 JSONL
python -m training.run_batch run -n 100 -m "gpt-4o:openai" --no-mongo --output ./data/games.jsonl

# 带实验标签
python -m training.run_batch run -n 100 -m "gpt-4o:openai" --tag "exp_v1"

# 查看所有批次
python -m training.run_batch list

# 导出训练轨迹
python -m training.run_batch export --batch-id <BATCH_ID> --output ./data/training.jsonl

# 按标签导出
python -m training.run_batch export --tag "exp_v1" --output ./data/exp_v1.jsonl
```

## RL 训练（自博弈）

基于在线自博弈 + Episode-level GAE 进行 PPO 训练。详见 [training/README.md](./training/README.md)。

```bash
# 复制并编辑配置模板
cp training/configs/ppo_avalon.yaml training/configs/my_exp.yaml

# 开始自博弈训练
bash training/scripts/self_play.sh training/configs/my_exp.yaml

# 断点续训
RESUME_FROM_ROUND=3 RESUME_FROM_STEP=5 \
    bash training/scripts/self_play.sh training/configs/my_exp.yaml
```

每轮自博弈自动完成完整流程：

1. 用当前模型通过 vLLM 跑游戏
2. 计算游戏统计（胜率 → wandb）
3. Critic 推理：为每个决策点估计 V(s)
4. Episode-level GAE：带 credit assignment 的 advantage 计算
5. 数据预处理 → parquet（注入预计算 advantage）
6. Verl PPO 训练 actor
7. 训练 Critic
8. 合并 checkpoint → 下一轮模型

所有产出物保存在 `experiments/<experiment_name>/` 下。

## 项目结构

```
avalon/
├── game/                           # 游戏引擎 (独立模块)
│   ├── roles.py                    # 角色定义与阵营逻辑
│   ├── rules.py                    # 规则配置 (5-10 人局)
│   ├── state.py                    # 游戏状态管理
│   ├── engine.py                   # 游戏核心逻辑
│   └── manager.py                  # 游戏管理器 (协调引擎 + LLM + DB)
├── server/                         # Python 后端
│   ├── main.py                     # FastAPI + Socket.IO 入口
│   ├── config.py                   # 环境配置
│   ├── llm/                        # LLM 集成
│   │   ├── base.py                 # 抽象基类
│   │   ├── providers.py            # 多厂商实现
│   │   ├── player.py               # LLM 玩家
│   │   └── tools.py                # LLM 工具/函数调用
│   ├── api/                        # REST API
│   │   ├── batch.py                # 批量操作 API
│   │   ├── config.py               # 配置 API
│   │   ├── games.py                # 游戏 API
│   │   └── stats.py                # 统计 API
│   ├── batch/                      # 批量对局
│   │   ├── runner.py               # 批量运行器
│   │   └── exporter.py             # 训练数据导出
│   ├── socket/                     # Socket.IO 处理
│   │   └── handlers.py             # WebSocket 事件处理
│   ├── models/                     # 数据模型
│   │   ├── database.py             # 数据库初始化
│   │   └── schemas.py              # Pydantic 数据模式
│   └── storage/                    # 数据存储
│       └── repository.py           # 数据仓库
├── training/                       # RL 训练 (Verl + Episode-level GAE)
│   ├── run_batch.py                # 批量对局 CLI 工具
│   ├── data/                       # 数据预处理
│   ├── reward/                     # 奖励函数（GAE advantage + 长度惩罚）
│   ├── critic/                     # Critic 模型 (value head 训练/推理)
│   ├── advantage/                  # Episode 级 GAE 计算
│   ├── stats/                      # 每轮游戏统计 & wandb 上报
│   ├── verl_extensions/            # 自定义 Verl advantage estimator
│   ├── configs/                    # 训练配置模板 (YAML)
│   ├── scripts/                    # 自博弈循环 & PPO wrapper
│   └── eval/                       # 模型评估
├── web/                            # React 前端
│   ├── src/
│   │   ├── App.tsx                 # 应用入口 (路由配置)
│   │   ├── components/             # UI 组件
│   │   ├── pages/                  # 页面
│   │   ├── hooks/                  # 自定义 Hooks
│   │   └── stores/                 # 状态管理
│   └── package.json
├── .env.example                    # 环境变量示例
├── requirements.txt                # Python 依赖
└── README.md
```

## 游戏规则

阿瓦隆是一款社交推理游戏，玩家分为好人阵营和坏人阵营。

### 角色

**好人阵营:**
- 梅林 (Merlin): 知道所有坏人身份，但不能暴露自己
- 忠臣 (Loyal Servant): 没有特殊能力

**坏人阵营:**
- 刺客 (Assassin): 游戏结束时可以刺杀梅林
- 爪牙 (Minion): 知道其他坏人身份

### 游戏流程

1. **分配角色**: 随机分配角色给每个玩家
2. **夜晚阶段**: 玩家获取各自的信息
3. **任务阶段** (重复 5 轮):
   - 队长选择队员
   - 所有玩家讨论
   - 投票决定是否同意队伍
   - 队员执行任务
4. **刺杀阶段**: 如果好人完成 3 个任务，刺客可以尝试刺杀梅林

### 胜利条件

- **好人胜利**: 完成 3 个任务且梅林未被刺杀
- **坏人胜利**: 破坏 3 个任务，或 5 次投票失败，或成功刺杀梅林

## 技术栈

**后端:**
- FastAPI + python-socketio
- Motor (异步 MongoDB 驱动)
- OpenAI / Anthropic SDK
- Pydantic 数据验证

**前端:**
- React 19 + TypeScript
- Vite + Tailwind CSS
- Zustand (状态管理) + Socket.IO Client
- Recharts (统计图表)
- Lucide React (图标)

**RL 训练:**
- Verl (PPO 框架)
- vLLM (自博弈推理)
- PyTorch + HuggingFace Transformers (Critic 模型)
- wandb (实验追踪)

## License

MIT
