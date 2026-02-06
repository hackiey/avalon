import { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, Loader2, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { GameBoard } from '@/components/GameBoard';
import { HumanControls } from '@/components/HumanControls';
import { useSocket } from '@/hooks/useSocket';
import { useGameStore } from '@/stores/gameStore';
import { getGame } from '@/lib/api';

export function Game() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const {
    joinGame,
    leaveGame,
    startGame,
    sendDiscussion,
    sendVote,
    sendQuestVote,
    sendTeamSelect,
    sendAssassinationDiscussion,
    sendAssassinate,
  } = useSocket();

  const {
    gameState,
    setGameState,
    humanSeat,
    setHumanSeat,
    godMode,
    setGodMode,
    selectedPlayers,
    togglePlayerSelection,
    clearSelection,
  } = useGameStore();

  const loadGame = useCallback(async (revealAll: boolean = false) => {
    if (!id) return;
    try {
      const game = await getGame(id, revealAll);
      setGameState(game);

      // Find human player
      const human = game.players?.find((p: any) => p.is_human);
      if (human) {
        setHumanSeat(human.seat);
      }
    } catch (err) {
      setError('游戏不存在');
    } finally {
      setLoading(false);
    }
  }, [id, setGameState, setHumanSeat]);

  useEffect(() => {
    if (!id) return;

    loadGame(godMode);

    // Join socket room
    joinGame(id);

    return () => {
      if (id) {
        leaveGame(id);
      }
    };
  }, [id]);

  // Reload game when god mode changes
  useEffect(() => {
    if (id && !loading) {
      loadGame(godMode);
    }
  }, [godMode]);

  const handleToggleGodMode = () => {
    setGodMode(!godMode);
  };

  const handleStartGame = () => {
    if (id) {
      startGame(id);
    }
  };

  const handleTeamSelect = (speech?: string) => {
    if (id && selectedPlayers.length > 0) {
      sendTeamSelect(id, selectedPlayers, speech);
      clearSelection();
    }
  };

  const handleDiscussion = (content: string) => {
    if (id) {
      sendDiscussion(id, content);
    }
  };

  const handleVote = (approve: boolean) => {
    if (id) {
      sendVote(id, approve);
    }
  };

  const handleQuestVote = (success: boolean) => {
    if (id) {
      sendQuestVote(id, success);
    }
  };

  const handleAssassinationDiscussion = (content: string) => {
    if (id) {
      sendAssassinationDiscussion(id, content);
    }
  };

  const handleAssassinate = (target: number) => {
    if (id) {
      sendAssassinate(id, target);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error || !gameState) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <p className="text-destructive">{error || '加载失败'}</p>
        <Button onClick={() => navigate('/')}>返回首页</Button>
      </div>
    );
  }

  const isWaiting = gameState.status === 'waiting';
  const isHumanTurn =
    gameState.waiting_for_human && humanSeat !== null;

  return (
    <div className="container mx-auto py-6 px-4 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <Button variant="ghost" onClick={() => navigate('/')}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          返回
        </Button>

        <div className="flex items-center gap-2">
          {/* God mode toggle */}
          <Button
            variant={godMode ? 'default' : 'outline'}
            size="sm"
            onClick={handleToggleGodMode}
            title={godMode ? '关闭上帝视角' : '开启上帝视角'}
          >
            {godMode ? (
              <Eye className="mr-2 h-4 w-4" />
            ) : (
              <EyeOff className="mr-2 h-4 w-4" />
            )}
            上帝视角
          </Button>

          {isWaiting && (
            <Button onClick={handleStartGame}>
              <Play className="mr-2 h-4 w-4" />
              开始游戏
            </Button>
          )}
        </div>
      </div>

      {/* Game board */}
      <GameBoard
        gameState={gameState}
        humanSeat={humanSeat}
        godMode={godMode}
        selectedPlayers={selectedPlayers}
        onPlayerClick={
          isHumanTurn &&
          (gameState.human_action_type === 'team_selection' ||
            gameState.human_action_type === 'assassinate')
            ? togglePlayerSelection
            : undefined
        }
      />

      {/* Human controls */}
      {isHumanTurn && (
        <div className="mt-6">
          <HumanControls
            gameState={gameState}
            humanSeat={humanSeat}
            selectedPlayers={selectedPlayers}
            onTeamSelect={handleTeamSelect}
            onDiscussion={handleDiscussion}
            onVote={handleVote}
            onQuestVote={handleQuestVote}
            onAssassinationDiscussion={handleAssassinationDiscussion}
            onAssassinate={handleAssassinate}
          />
        </div>
      )}
    </div>
  );
}
