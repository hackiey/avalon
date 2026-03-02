# Avalon LLM Training Engine

An engine for training and observing LLMs playing the Avalon board game. Supports multi-LLM battles, human participation, real-time spectating, game replay, statistics, batch execution, and RL training data export.

[**中文文档 (Chinese)**](./README_CN.md)

## Features

- **Multi-LLM Battles**: Support for OpenAI, Anthropic, DeepSeek, VLLM and more
- **Human Participation**: Play alongside AI agents
- **Real-time Spectating**: Watch games live through the web UI
- **Game Replay**: Step through historical games move by move
- **Statistics**: View win rates by model, role, and more
- **Batch Execution**: Run games in bulk via CLI with parallel support
- **Training Data Export**: Export game trajectories as JSONL for model training
- **RL Training**: On-policy self-play with Episode-level GAE + external Critic (Verl + PPO)
- **Multi-turn Incremental Context (v2)**: Tool-calling based multi-turn conversation mode with incremental observations, designed for agentic RL training

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your LLM API keys and database connection:

```env
# OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODELS=gpt-4o,gpt-4o-mini

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_MODELS=claude-3-5-sonnet-20241022

# DeepSeek (optional)
DEEPSEEK_API_KEY=xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODELS=deepseek-chat

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=avalon
```

### 2. Install Backend Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Start MongoDB

```bash
# macOS (Homebrew)
brew services start mongodb-community

# or using Docker
docker run -d -p 27017:27017 --name avalon-mongo mongo:latest
```

### 4. Install Frontend Dependencies

```bash
cd web
pnpm install
```

### 5. Start Services

Start the backend server (port 8001):

```bash
uvicorn server.main:asgi_app --host 0.0.0.0 --port 8001
```

Start the frontend dev server (port 5173):

```bash
cd web
pnpm dev
```

Visit http://localhost:5173 to get started.

## Batch Games & Training Data Export

Use the `training/run_batch.py` CLI tool to run games in bulk and export training data:

```bash
# Run 100 games (single model)
python -m training.run_batch run -n 100 -m "qwen-plus:qwen"

# Run 100 games (multiple models, rotating)
python -m training.run_batch run -n 100 -m "qwen-plus:qwen,gpt-4o:openai"

# Parallel execution (4 games at once)
python -m training.run_batch run -n 100 -m "gpt-4o:openai" --parallel 4

# Without MongoDB (write directly to JSONL)
python -m training.run_batch run -n 100 -m "gpt-4o:openai" --no-mongo --output ./data/games.jsonl

# With experiment tag
python -m training.run_batch run -n 100 -m "gpt-4o:openai" --tag "exp_v1"

# List all batches
python -m training.run_batch list

# Export training trajectories
python -m training.run_batch export --batch-id <BATCH_ID> --output ./data/training.jsonl

# Export by tag
python -m training.run_batch export --tag "exp_v1" --output ./data/exp_v1.jsonl
```

## RL Training (Self-Play)

Train LLMs via on-policy self-play with Episode-level GAE. See [training/README.md](./training/README.md) for full details.

```bash
# Copy and edit the config template
cp training/configs/ppo_avalon.yaml training/configs/my_exp.yaml

# Start self-play training
bash training/scripts/self_play.sh training/configs/my_exp.yaml

# Resume from a checkpoint
RESUME_FROM_ROUND=3 RESUME_FROM_STEP=5 \
    bash training/scripts/self_play.sh training/configs/my_exp.yaml
```

Each round of self-play runs the full pipeline automatically:

1. Run games with the current model via vLLM
2. Compute game statistics (win rates → wandb)
3. Critic inference: estimate V(s) for each decision point
4. Episode-level GAE: compute advantage with credit assignment
5. Preprocess data → parquet (with precomputed advantage)
6. Verl PPO: train actor
7. Train Critic
8. Merge checkpoint → next round model

All outputs are saved under `experiments/<experiment_name>/`.

## Context Modes

The engine supports two prompt/context modes for LLM players:

### v1: Full-State Prompt (default)

Each decision point reconstructs the complete game state from scratch as a single `[prompt, response]` pair. The model sees the full history every time.

### v2: Multi-turn Incremental Context

Enabled via the "多轮增量上下文 (v2)" toggle in the web UI, or `use_incremental_context=True` in batch config.

The conversation accumulates across the entire game using the standard tool-calling protocol:

```
[system]    — Game rules + role info (fixed)
[user]      — Initial observation + first phase instruction
[assistant] — {tool_calls: [{speak, ...}]}
[tool]      — Environment feedback (events only: votes, quest results, discussions...)
[user]      — Next phase instruction
[assistant] — {tool_calls: [{vote_team, ...}]}
[tool]      — Environment feedback
[user]      — Next phase instruction
...
```

Key design decisions:
- **[tool] = environment feedback**: Only contains events that happened (vote results, quest outcomes, other players' speeches). No action instructions.
- **[user] = action directive**: Only contains the phase instruction telling the model what to do next.
- **Incremental observations**: Each tool response only includes NEW events since the last action, avoiding redundancy.
- **Tool calling retained**: All game tools (`speak`, `propose_team`, `vote_team`, `vote_quest`, `assassinate`) are preserved. Only `update_memory` is removed — the conversation history itself serves as memory.
- **Events in chronological order**: vote results → quest results → round transitions → discussions → team proposals.

Enable v2 mode:

```python
# Batch config
config = BatchConfig(use_incremental_context=True, ...)

# Batch API
POST /api/batch/run
{"use_incremental_context": true, ...}
```

Related files: `game/prompts_v2.py` (incremental observation builder), `server/llm/player_v2.py` (multi-turn player).

## Project Structure

```
avalon/
├── game/                           # Game engine (standalone)
│   ├── roles.py                    # Role definitions & team logic
│   ├── rules.py                    # Rule configuration (5-10 players)
│   ├── state.py                    # Game state management
│   ├── engine.py                   # Core game logic
│   ├── prompts.py                  # Full-state prompt builder (v1)
│   ├── prompts_v2.py               # Incremental observation builder (v2)
│   └── manager.py                  # Game manager (orchestrates engine + LLM + DB)
├── server/                         # Python backend
│   ├── main.py                     # FastAPI + Socket.IO entry point
│   ├── config.py                   # Configuration
│   ├── llm/                        # LLM integration
│   │   ├── base.py                 # Abstract base classes
│   │   ├── providers.py            # Multi-provider support
│   │   ├── player.py               # LLM player (v1, full-state)
│   │   ├── player_v2.py            # LLM player (v2, multi-turn incremental)
│   │   └── tools.py                # LLM tools / function calling
│   ├── api/                        # REST API
│   │   ├── batch.py                # Batch operations API
│   │   ├── config.py               # Config API
│   │   ├── games.py                # Games API
│   │   └── stats.py                # Statistics API
│   ├── batch/                      # Batch execution
│   │   ├── runner.py               # Batch runner
│   │   └── exporter.py             # Training data exporter
│   ├── socket/                     # Socket.IO handlers
│   │   └── handlers.py             # WebSocket event handlers
│   ├── models/                     # Data models
│   │   ├── database.py             # Database initialization
│   │   └── schemas.py              # Pydantic schemas
│   └── storage/                    # Data storage
│       └── repository.py           # Repository
├── training/                       # RL training (Verl + Episode-level GAE)
│   ├── run_batch.py                # Batch game CLI tool
│   ├── data/                       # Data preprocessing
│   ├── reward/                     # Reward functions (GAE + length penalty)
│   ├── critic/                     # Critic model (value head train/infer)
│   ├── advantage/                  # Episode-level GAE computation
│   ├── stats/                      # Per-round game statistics & wandb logging
│   ├── verl_extensions/            # Custom Verl advantage estimator
│   ├── configs/                    # Training config templates (YAML)
│   ├── scripts/                    # Self-play loop & PPO wrapper
│   └── eval/                       # Model evaluation
├── web/                            # React frontend
│   ├── src/
│   │   ├── App.tsx                 # App entry (routing)
│   │   ├── components/             # UI components
│   │   ├── pages/                  # Pages
│   │   ├── hooks/                  # Custom hooks
│   │   └── stores/                 # State management
│   └── package.json
├── .env.example                    # Environment variables example
├── requirements.txt                # Python dependencies
└── README.md
```

## Game Rules

Avalon is a social deduction game where players are divided into Good and Evil teams.

### Roles

**Good Team:**
- **Merlin**: Knows all Evil players, but must stay hidden
- **Loyal Servant**: No special abilities

**Evil Team:**
- **Assassin**: Can attempt to assassinate Merlin at the end
- **Minion**: Knows other Evil players

### Game Flow

1. **Role Assignment**: Roles are randomly assigned to each player
2. **Night Phase**: Players receive their role-specific information
3. **Quest Phase** (repeated for 5 rounds):
   - The leader selects team members
   - All players discuss
   - Vote to approve or reject the team
   - Team members execute the quest
4. **Assassination Phase**: If Good completes 3 quests, the Assassin may attempt to kill Merlin

### Victory Conditions

- **Good wins**: Complete 3 quests and Merlin survives
- **Evil wins**: Fail 3 quests, or 5 consecutive vote rejections, or successfully assassinate Merlin

## Tech Stack

**Backend:**
- FastAPI + python-socketio
- Motor (async MongoDB driver)
- OpenAI / Anthropic SDK
- Pydantic validation

**Frontend:**
- React 19 + TypeScript
- Vite + Tailwind CSS
- Zustand (state management) + Socket.IO Client
- Recharts (charts)
- Lucide React (icons)

**RL Training:**
- Verl (PPO framework)
- vLLM (self-play inference)
- PyTorch + HuggingFace Transformers (Critic model)
- wandb (experiment tracking)

## License

MIT
