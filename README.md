# Avalon LLM Training Engine

An engine for training and observing LLMs playing the Avalon board game. Supports multi-LLM battles, human participation, real-time spectating, game replay, statistics, batch execution, and training data export.

[**中文文档 (Chinese)**](./README_CN.md)

## Features

- **Multi-LLM Battles**: Support for OpenAI, Anthropic, DeepSeek, VLLM and more
- **Human Participation**: Play alongside AI agents
- **Real-time Spectating**: Watch games live through the web UI
- **Game Replay**: Step through historical games move by move
- **Statistics**: View win rates by model, role, and more
- **Batch Execution**: Run games in bulk via CLI with parallel support
- **Training Data Export**: Export game trajectories as JSONL for model training

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

Start the backend server (port 8000):

```bash
uvicorn server.main:asgi_app --host 0.0.0.0 --port 8000
```

Start the frontend dev server (port 5173):

```bash
cd web
pnpm dev
```

Visit http://localhost:5173 to get started.

## Batch Games & Training Data Export

Use the `run_batch.py` CLI tool to run games in bulk and export training data:

```bash
# Run 100 games (single model)
python run_batch.py run -n 100 -m "qwen-plus:qwen"

# Run 100 games (multiple models, rotating)
python run_batch.py run -n 100 -m "qwen-plus:qwen,gpt-4o:openai"

# Parallel execution (4 games at once)
python run_batch.py run -n 100 -m "gpt-4o:openai" --parallel 4

# With experiment tag
python run_batch.py run -n 100 -m "gpt-4o:openai" --tag "exp_v1"

# List all batches
python run_batch.py list

# Export training trajectories
python run_batch.py export --batch-id <BATCH_ID> --output ./data/training.jsonl

# Export by tag
python run_batch.py export --tag "exp_v1" --output ./data/exp_v1.jsonl
```

## Project Structure

```
avalon/
├── server/                         # Python backend
│   ├── main.py                     # FastAPI + Socket.IO entry point
│   ├── config.py                   # Configuration
│   ├── game/                       # Game engine
│   │   ├── engine.py               # Core game logic
│   │   ├── manager.py              # Game manager
│   │   ├── roles.py                # Role definitions
│   │   ├── rules.py                # Rule configuration
│   │   └── state.py                # Game state
│   ├── llm/                        # LLM integration
│   │   ├── base.py                 # Abstract base classes
│   │   ├── providers.py            # Multi-provider support
│   │   ├── prompts.py              # Prompt templates
│   │   ├── player.py               # LLM player
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
├── web/                            # React frontend
│   ├── src/
│   │   ├── App.tsx                 # App entry (routing)
│   │   ├── components/             # UI components
│   │   │   ├── Discussion.tsx      # Discussion panel
│   │   │   ├── GameBoard.tsx       # Game board
│   │   │   ├── HumanControls.tsx   # Human player controls
│   │   │   ├── LLMDetailModal.tsx  # LLM detail modal
│   │   │   ├── PlayerCard.tsx      # Player card
│   │   │   ├── QuestTracker.tsx    # Quest tracker
│   │   │   ├── VoteHistory.tsx     # Vote history
│   │   │   └── ui/                 # Base UI component library
│   │   ├── pages/                  # Pages
│   │   │   ├── Home.tsx            # Home
│   │   │   ├── Game.tsx            # Live game
│   │   │   ├── Replay.tsx          # Game replay
│   │   │   └── Stats.tsx           # Statistics
│   │   ├── hooks/                  # Custom hooks
│   │   │   └── useSocket.ts        # Socket.IO hook
│   │   └── stores/                 # State management
│   │       └── gameStore.ts        # Zustand store
│   └── package.json
├── run_batch.py                    # Batch game CLI tool
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

## License

MIT
