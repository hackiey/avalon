import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, History, BarChart3, Users, Bot, User, Plus, Minus, Info } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Badge } from '@/components/ui/Badge';
import { getAvailableModels, getGames, createGame } from '@/lib/api';
import type { ModelInfo, GameSummary, PlayerConfig } from '@/lib/types';
import { cn } from '@/lib/utils';

export function Home() {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [games, setGames] = useState<GameSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  // Game creation form
  const [playerCount, setPlayerCount] = useState(5);
  const [players, setPlayers] = useState<PlayerConfig[]>([]);

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    // Initialize players when player count changes
    const newPlayers: PlayerConfig[] = [];
    for (let i = 0; i < playerCount; i++) {
      const existing = players[i];
      newPlayers.push({
        seat: i,
        name: existing?.name || `玩家${i + 1}`,
        is_human: existing?.is_human || false,
        model: existing?.model || models[0]?.model,
        provider: existing?.provider || models[0]?.provider,
      });
    }
    setPlayers(newPlayers);
  }, [playerCount, models]);

  const loadData = async () => {
    try {
      const [modelsData, gamesData] = await Promise.all([
        getAvailableModels(),
        getGames(),
      ]);
      setModels(modelsData);
      setGames(gamesData);
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  const updatePlayer = (index: number, updates: Partial<PlayerConfig>) => {
    setPlayers((prev) =>
      prev.map((p, i) => (i === index ? { ...p, ...updates } : p))
    );
  };

  const handleGlobalModelChange = (value: string) => {
    if (!value) return;
    const [provider, model] = value.split(':');
    setPlayers((prev) =>
      prev.map((p) => ({
        ...p,
        provider,
        model,
      }))
    );
  };

  const handleCreateGame = async () => {
    setCreating(true);
    try {
      // Validate
      for (const player of players) {
        if (!player.is_human && (!player.model || !player.provider)) {
          alert(`请为 ${player.name} 选择一个模型`);
          setCreating(false);
          return;
        }
      }

      const game = await createGame({
        player_count: playerCount,
        players,
      });

      navigate(`/game/${game.id}`);
    } catch (error) {
      console.error('Failed to create game:', error);
      alert('创建游戏失败');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">Avalon LLM Arena</h1>
        <p className="text-muted-foreground text-lg">
          观察 AI 玩家在阿瓦隆游戏中的博弈与推理
        </p>
      </div>

      {/* Navigation */}
      <div className="flex justify-center gap-4 mb-8">
        <Button variant="outline" onClick={() => navigate('/stats')}>
          <BarChart3 className="mr-2 h-4 w-4" />
          统计数据
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Create Game */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Play className="h-5 w-5" />
              创建新游戏
            </CardTitle>
            <CardDescription>
              配置玩家数量和 AI 模型
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Player count */}
            <div>
              <label className="text-sm font-medium mb-2 block">
                玩家数量
              </label>
              <div className="flex items-center gap-4">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setPlayerCount((c) => Math.max(5, c - 1))}
                  disabled={playerCount <= 5}
                >
                  <Minus className="h-4 w-4" />
                </Button>
                <span className="text-2xl font-bold w-8 text-center">
                  {playerCount}
                </span>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setPlayerCount((c) => Math.min(6, c + 1))}
                  disabled={playerCount >= 6}
                >
                  <Plus className="h-4 w-4" />
                </Button>
                <span className="text-sm text-muted-foreground">
                  (目前支持 5-6 人)
                </span>
              </div>
            </div>

            {/* Players */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">玩家配置</label>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">全选模型:</span>
                  <Select
                    value=""
                    onChange={(e) => handleGlobalModelChange(e.target.value)}
                    className="w-48 h-8 text-xs py-1"
                  >
                    <option value="" disabled>选择模型...</option>
                    {models.map((m) => (
                      <option
                        key={`${m.provider}:${m.model}`}
                        value={`${m.provider}:${m.model}`}
                      >
                        {m.display_name}
                      </option>
                    ))}
                  </Select>
                </div>
              </div>
              {players.map((player, index) => (
                <div
                  key={index}
                  className="flex items-center gap-3 p-3 rounded-lg border bg-muted/30"
                >
                  <span className="text-sm text-muted-foreground w-8">
                    #{index + 1}
                  </span>
                  <Input
                    value={player.name}
                    onChange={(e) =>
                      updatePlayer(index, { name: e.target.value })
                    }
                    placeholder="玩家名称"
                    className="w-28"
                  />
                  <Button
                    variant={player.is_human ? 'default' : 'outline'}
                    size="sm"
                    onClick={() =>
                      updatePlayer(index, { is_human: !player.is_human })
                    }
                  >
                    {player.is_human ? (
                      <>
                        <User className="h-4 w-4 mr-1" />
                        人类
                      </>
                    ) : (
                      <>
                        <Bot className="h-4 w-4 mr-1" />
                        AI
                      </>
                    )}
                  </Button>
                  {!player.is_human && (
                    <Select
                      value={`${player.provider}:${player.model}`}
                      onChange={(e) => {
                        const [provider, model] = e.target.value.split(':');
                        updatePlayer(index, { provider, model });
                      }}
                      className="flex-1"
                    >
                      {models.map((m) => (
                        <option
                          key={`${m.provider}:${m.model}`}
                          value={`${m.provider}:${m.model}`}
                        >
                          {m.display_name}
                        </option>
                      ))}
                    </Select>
                  )}
                </div>
              ))}
            </div>

            {/* Create button */}
            <Button
              className="w-full"
              size="lg"
              onClick={handleCreateGame}
              disabled={creating || models.length === 0}
            >
              {creating ? '创建中...' : '开始游戏'}
            </Button>

            {models.length === 0 && !loading && (
              <p className="text-sm text-destructive text-center">
                未配置任何 LLM 模型，请检查 .env 文件
              </p>
            )}
          </CardContent>
        </Card>

        {/* Recent Games */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5" />
              历史对局
            </CardTitle>
            <CardDescription>
              查看和回放历史游戏
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center text-muted-foreground py-8">
                加载中...
              </div>
            ) : games.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                暂无历史对局
              </div>
            ) : (
              <div className="space-y-2">
                {games.slice(0, 10).map((game) => (
                  <div
                    key={game.id}
                    className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50 cursor-pointer transition-colors"
                    onClick={() => navigate(`/game/${game.id}`)}
                  >
                    <div className="flex items-center gap-3">
                      <Users className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">
                        {game.player_count}人局
                      </span>
                      <Badge
                        variant={
                          game.status === 'finished'
                            ? game.winner === 'good'
                              ? 'default'
                              : 'destructive'
                            : 'secondary'
                        }
                      >
                        {game.status === 'finished'
                          ? game.winner === 'good'
                            ? '好人胜'
                            : '坏人胜'
                          : game.status === 'in_progress'
                          ? '进行中'
                          : '等待中'}
                      </Badge>
                    </div>
                    <span className="text-sm text-muted-foreground">
                      {new Date(game.created_at).toLocaleDateString('zh-CN')}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
