import { useState, useEffect, useMemo } from 'react';
import { Swords, Users, Vote, Target } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { GameState, GamePhase } from '@/lib/types';
import { PlayerCard } from './PlayerCard';
import { QuestTracker } from './QuestTracker';
import { Discussion } from './Discussion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';

interface GameBoardProps {
  gameState: GameState;
  humanSeat?: number | null;
  godMode?: boolean;
  selectedPlayers?: number[];
  onPlayerClick?: (seat: number) => void;
}

const PHASE_NAMES: Record<GamePhase, string> = {
  role_assignment: '角色分配',
  night_phase: '夜晚阶段',
  team_selection: '队长选队',
  discussion: '讨论阶段',
  team_vote: '投票表决',
  quest_execution: '任务执行',
  assassination_discussion: '刺杀讨论',
  assassination: '刺杀阶段',
  game_over: '游戏结束',
};

export function GameBoard({
  gameState,
  humanSeat,
  godMode = false,
  selectedPlayers = [],
  onPlayerClick,
}: GameBoardProps) {
  // Show roles: game finished, or has human player, or god mode enabled
  const showRoles = gameState.status === 'finished' || humanSeat !== null || godMode;
  const [displayRound, setDisplayRound] = useState<number | 'assassination'>(gameState.current_round || 1);

  // Auto-update display round when current round changes or enters assassination phase
  useEffect(() => {
    if (gameState.phase === 'assassination_discussion' || gameState.phase === 'assassination') {
      setDisplayRound('assassination');
    } else if (gameState.current_round) {
      setDisplayRound(gameState.current_round);
    }
  }, [gameState.current_round, gameState.phase]);

  const isAssassination = displayRound === 'assassination';

  // Transform assassination messages
  const assassinationMessages = useMemo(() => {
    if (!isAssassination || !gameState.assassination_discussion_history) return [];
    return gameState.assassination_discussion_history.map(msg => ({
      ...msg,
      round: gameState.current_round,
      attempt: 0,
    }));
  }, [isAssassination, gameState.assassination_discussion_history, gameState.current_round]);

  // Transform assassination result to vote
  const assassinationVotes = useMemo(() => {
    if (!isAssassination || gameState.assassinated_player === undefined) return [];
    
    // Try to find assassin
    const assassin = gameState.players.find(p => p.role === 'assassin');
    if (!assassin) return [];

    return [{
      round: gameState.current_round,
      attempt: 1,
      votes: { [gameState.assassinated_player]: true },
      approved: true,
      assassin_seat: assassin.seat,
    }];
  }, [isAssassination, gameState.assassinated_player, gameState.players, gameState.current_round]);

  return (
    <div className="space-y-6">
      {/* Game header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-bold">阿瓦隆</h2>
          <Badge variant={gameState.status === 'in_progress' ? 'default' : 'secondary'}>
            {PHASE_NAMES[gameState.phase]}
          </Badge>
          {gameState.waiting_for_human && (
            <Badge variant="destructive">等待玩家操作</Badge>
          )}
        </div>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span>第 {gameState.current_round} 轮</span>
          <span>投票尝试: {gameState.vote_attempt}/5</span>
        </div>
      </div>

      {/* Quest tracker */}
      <Card>
        <CardContent className="pt-6">
          <QuestTracker
            questResults={gameState.quest_results}
            currentRound={gameState.current_round}
            playerCount={gameState.player_count}
            selectedRound={displayRound}
            onRoundClick={setDisplayRound}
            phase={gameState.phase}
            assassinatedPlayer={gameState.assassinated_player}
            winner={gameState.winner}
          />
        </CardContent>
      </Card>

      {/* Game status banner */}
      {gameState.phase !== 'role_assignment' && gameState.phase !== 'night_phase' && (
        <Card>
          <CardContent className="py-4">
            {/* 刺杀环节显示 */}
            {displayRound === 'assassination' ? (
              <div className="space-y-4">
                <div className="flex items-center justify-center gap-3">
                  <Target className="h-5 w-5 text-red-500" />
                  <span className="text-lg font-medium">刺杀环节</span>
                </div>
                
                {/* 当前刺杀状态 */}
                <div className="flex items-center justify-center gap-3">
                  {gameState.phase === 'assassination_discussion' && (
                    <span className="text-muted-foreground">坏人正在讨论刺杀目标...</span>
                  )}
                  {gameState.phase === 'assassination' && (
                    <span className="text-muted-foreground">刺客正在选择刺杀目标...</span>
                  )}
                  {gameState.phase === 'game_over' && gameState.assassinated_player !== undefined && (
                    <div className="text-center">
                      <div className={cn(
                        'text-lg font-bold',
                        gameState.winner === 'good' ? 'text-blue-500' : 'text-red-500'
                      )}>
                        {gameState.winner === 'good' ? '刺杀失败，好人胜利!' : '刺杀成功，坏人胜利!'}
                      </div>
                      <div className="text-sm text-muted-foreground mt-1">
                        刺杀目标: <span className="font-medium text-foreground">
                          {gameState.players[gameState.assassinated_player]?.name}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
                
                {/* 无讨论记录时的提示 */}
                {(!gameState.assassination_discussion_history || gameState.assassination_discussion_history.length === 0) && 
                  gameState.phase !== 'game_over' && (
                  <div className="text-center text-muted-foreground text-sm">
                    等待坏人讨论...
                  </div>
                )}
              </div>
            ) : displayRound !== gameState.current_round || 
                (gameState.phase === 'assassination_discussion' || gameState.phase === 'assassination' || gameState.phase === 'game_over') ? (
              /* 历史轮次状态显示 */
              <>
                {(() => {
                  const historyQuestResult = gameState.quest_results.find(q => q.round === displayRound);
                  const historyVotes = gameState.vote_history?.filter(v => v.round === displayRound) || [];
                  const lastVote = historyVotes.length > 0 ? historyVotes[historyVotes.length - 1] : null;
                  
                  return (
                    <div className="space-y-2">
                      <div className="flex items-center justify-center gap-3">
                        <span className="text-muted-foreground">第 {displayRound} 轮历史记录</span>
                      </div>
                      
                      {/* 任务结果 */}
                      {historyQuestResult && historyQuestResult.success !== null && (
                        <div className="flex items-center justify-center gap-3">
                          <Swords className={cn(
                            "h-5 w-5",
                            historyQuestResult.success ? 'text-blue-500' : 'text-red-500'
                          )} />
                          <span className={cn(
                            "font-medium",
                            historyQuestResult.success ? 'text-blue-500' : 'text-red-500'
                          )}>
                            任务{historyQuestResult.success ? '成功' : '失败'}
                            {historyQuestResult.fail_votes > 0 && (
                              <span className="text-muted-foreground ml-1">
                                ({historyQuestResult.fail_votes} 票失败)
                              </span>
                            )}
                          </span>
                        </div>
                      )}
                      
                      {/* 最后一次投票结果 */}
                      {lastVote && (
                        <div className="flex items-center justify-center gap-3">
                          <Vote className={cn(
                            "h-5 w-5",
                            lastVote.approved ? 'text-green-500' : 'text-red-500'
                          )} />
                          <span className={cn(
                            lastVote.approved ? 'text-green-600' : 'text-red-600'
                          )}>
                            第 {lastVote.attempt} 次投票{lastVote.approved ? '通过' : '否决'}
                          </span>
                        </div>
                      )}
                      
                      {/* 队伍成员（从任务结果获取） */}
                      {historyQuestResult && historyQuestResult.team_members && historyQuestResult.team_members.length > 0 && (
                        <div className="text-center text-sm text-muted-foreground">
                          任务队伍: <span className="font-medium text-foreground">
                            {historyQuestResult.team_members
                              .map((seat: number) => gameState.players[seat]?.name)
                              .join(', ')}
                          </span>
                        </div>
                      )}
                      
                      {/* 无数据提示 */}
                      {!historyQuestResult && historyVotes.length === 0 && (
                        <div className="text-center text-muted-foreground">
                          该轮暂无记录
                        </div>
                      )}
                    </div>
                  );
                })()}
              </>
            ) : (
              /* 当前轮次状态显示 */
              <>
                <div className="flex items-center justify-center gap-3">
                  {gameState.phase === 'team_selection' && (
                    <>
                      <Users className="h-5 w-5 text-primary" />
                      <span>队长 <strong>{gameState.players[gameState.current_leader]?.name}</strong> 正在选队</span>
                    </>
                  )}
                  {gameState.phase === 'discussion' && (
                    <>
                      <Swords className="h-5 w-5 text-amber-500" />
                      <span>讨论阶段进行中...</span>
                    </>
                  )}
                  {gameState.phase === 'team_vote' && (
                    <>
                      <Vote className="h-5 w-5 text-blue-500" />
                      <span>投票表决中...</span>
                    </>
                  )}
                  {gameState.phase === 'quest_execution' && (
                    <>
                      <Swords className="h-5 w-5 text-green-500" />
                      <span>任务执行中...</span>
                    </>
                  )}
                </div>
                {/* Proposed team */}
                {gameState.proposed_team.length > 0 && (
                  <div className="mt-2 text-center text-sm text-muted-foreground">
                    队伍成员: <span className="font-medium text-foreground">
                      {gameState.proposed_team
                        .map((seat) => gameState.players[seat]?.name)
                        .join(', ')}
                    </span>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* Main content: Left - Players, Right - Discussion */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left side - Player list */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-lg">角色列表</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {gameState.players.map((player) => (
                <PlayerCard
                  key={player.seat}
                  player={{
                    ...player,
                    is_on_quest: gameState.proposed_team.includes(player.seat),
                    is_leader: player.seat === gameState.current_leader,
                  }}
                  isSelected={selectedPlayers.includes(player.seat)}
                  onClick={onPlayerClick ? () => onPlayerClick(player.seat) : undefined}
                  showRole={showRoles}
                  size="md"
                  layout="horizontal"
                />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Right side - Discussion */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-lg">讨论记录</CardTitle>
          </CardHeader>
          <CardContent>
            <Discussion
              messages={isAssassination ? assassinationMessages : gameState.discussion_history}
              players={gameState.players}
              displayRound={isAssassination ? gameState.current_round : displayRound}
              votes={isAssassination ? assassinationVotes : gameState.vote_history}
              questResults={isAssassination ? [] : gameState.quest_results}
              gameId={gameState.id}
              godMode={godMode}
              voteTitle={isAssassination ? '刺杀执行记录' : undefined}
              isAssassination={isAssassination}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
