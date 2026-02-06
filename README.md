# Avalon LLM Training Engine

一个用于训练和观察 LLM 玩阿瓦隆游戏的引擎。支持多个 LLM 之间的对战、人类玩家参与、实时观战和游戏数据统计。

## 功能特性

- **多 LLM 对战**: 支持 OpenAI、Anthropic、DeepSeek 等多个 LLM 提供商
- **人类参与**: 人类玩家可以与 AI 一起游戏
- **实时观战**: 通过 Web 界面实时观看游戏进程
- **历史回放**: 回放历史对局，逐步查看游戏过程
- **统计分析**: 查看模型胜率、角色胜率等统计数据

## 快速开始

### 1. 环境配置

复制环境变量配置文件并填写 API 密钥:

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置你的 LLM API 密钥:

```env
# OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_MODELS=gpt-4o,gpt-4o-mini

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_MODELS=claude-3-5-sonnet-20241022

# DeepSeek (可选)
DEEPSEEK_API_KEY=xxx
DEEPSEEK_MODELS=deepseek-chat
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

### 3. 安装前端依赖

```bash
cd web
pnpm install
```

### 4. 启动服务

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

## 项目结构

```
avalon/
├── server/                     # Python 后端
│   ├── main.py                 # FastAPI + Socket.IO 入口
│   ├── config.py               # 环境配置
│   ├── game/                   # 游戏引擎
│   │   ├── engine.py           # 游戏核心逻辑
│   │   ├── roles.py            # 角色定义
│   │   ├── state.py            # 游戏状态
│   │   ├── rules.py            # 规则配置
│   │   └── manager.py          # 游戏管理器
│   ├── llm/                    # LLM 集成
│   │   ├── base.py             # 抽象基类
│   │   ├── providers.py        # 多厂商实现
│   │   ├── prompts.py          # Prompt 模板
│   │   └── player.py           # LLM 玩家
│   ├── api/                    # REST API
│   ├── socket/                 # Socket.IO 处理
│   ├── models/                 # 数据模型
│   └── storage/                # 数据存储
├── web/                        # React 前端
│   ├── src/
│   │   ├── components/         # UI 组件
│   │   ├── pages/              # 页面
│   │   ├── hooks/              # 自定义 hooks
│   │   ├── stores/             # 状态管理
│   │   └── lib/                # 工具函数
│   └── package.json
├── .env.example                # 环境变量示例
├── requirements.txt            # Python 依赖
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
- SQLAlchemy + SQLite
- OpenAI / Anthropic SDK

**前端:**
- React 18 + TypeScript
- Vite + Tailwind CSS
- Zustand + Socket.IO Client
- Recharts (统计图表)
- Lucide React (图标)

## License

MIT
