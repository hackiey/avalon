import { useRef, useEffect, useMemo, useState } from 'react';
import { MessageCircle, User, Bot, Check, X, Swords, Target } from 'lucide-react';
import { cn, formatTimestamp } from '@/lib/utils';
import type { DiscussionMessage, Player, VoteResult, QuestResult } from '@/lib/types';
import { LLMDetailModal } from '@/components/LLMDetailModal';

type SelectedVote = {
  type: 'team_vote' | 'quest_vote' | 'assassinate';
  roundNum: number;
  attempt: number;
  playerSeat: number;
  playerName: string;
};

interface DiscussionProps {
  messages: DiscussionMessage[];
  players: Player[];
  displayRound: number; // 当前要显示的轮次
  votes?: VoteResult[];
  questResults?: QuestResult[];
  gameId?: string; // 用于获取 LLM 详情
  godMode?: boolean; // 上帝视角模式（数据由后端 reveal_all 控制）
  voteTitle?: string; // 投票区域标题
  isAssassination?: boolean; // 是否是刺杀阶段
}

export function Discussion({ 
  messages, 
  players, 
  displayRound, 
  votes = [], 
  questResults = [], 
  gameId, 
  godMode: _godMode = false,
  voteTitle,
  isAssassination = false
}: DiscussionProps) {
  // godMode is passed for potential future use, data visibility is controlled by backend reveal_all
  void _godMode;
  const scrollRef = useRef<HTMLDivElement>(null);
  const [selectedMessage, setSelectedMessage] = useState<DiscussionMessage | null>(null);
  const [selectedVote, setSelectedVote] = useState<SelectedVote | null>(null);

  const getPlayer = (seat: number) => players.find((p) => p.seat === seat);
  const getPlayerName = (seat: number) =>
    players.find((p) => p.seat === seat)?.name || `玩家${seat}`;

  // Filter messages by display round
  const filteredMessages = useMemo(() => {
    return messages.filter((msg) => (msg.round || 1) === displayRound);
  }, [messages, displayRound]);

  // Filter votes by display round
  const filteredVotes = useMemo(() => {
    return votes.filter((vote) => vote.round === displayRound);
  }, [votes, displayRound]);

  // Get quest result for display round
  const questResult = useMemo(() => {
    return questResults.find((q) => q.round === displayRound);
  }, [questResults, displayRound]);

  // Handle vote button click
  const handleVoteClick = (type: 'team_vote' | 'quest_vote' | 'assassinate', roundNum: number, attempt: number, seat: number) => {
    const player = getPlayer(seat);
    if (!player || player.is_human || !gameId) return;
    
    setSelectedVote({
      type,
      roundNum,
      attempt,
      playerSeat: seat,
      playerName: player.name,
    });
  };

  // For assassination, find the assassin for LLM detail lookup
  const assassinPlayer = isAssassination
    ? players.find(p => p.role === 'assassin')
    : undefined;

  // Scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filteredMessages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-muted-foreground">
        <MessageCircle className="h-8 w-8 mb-2" />
        <p>暂无讨论记录</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Discussion messages */}
      <div
        ref={scrollRef}
        className="space-y-2 max-h-[400px] overflow-y-auto pr-2"
      >
        {filteredMessages.length === 0 ? (
          <div className="text-center text-muted-foreground py-4">
            该轮暂无讨论记录
          </div>
        ) : (
          filteredMessages.map((msg, index) => {
            const player = getPlayer(msg.seat);
            const isHuman = player?.is_human;
            const isClickable = gameId && !isHuman;

            return (
              <div
                key={`${displayRound}-${index}`}
                className={cn(
                  'flex gap-3 p-3 rounded-lg',
                  'bg-muted/50 hover:bg-muted/80 transition-colors',
                  isClickable && 'cursor-pointer hover:ring-2 hover:ring-primary/30'
                )}
                onClick={isClickable ? () => setSelectedMessage(msg) : undefined}
                title={isClickable ? '点击查看 LLM 调用详情' : undefined}
              >
                {/* Avatar */}
                <div
                  className={cn(
                    'flex-shrink-0 flex items-center justify-center rounded-full',
                    'w-8 h-8 bg-background border'
                  )}
                >
                  {isHuman ? (
                    <User className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <Bot className="h-4 w-4 text-muted-foreground" />
                  )}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-sm">
                      {msg.player_name}
                    </span>
                    <span className="text-xs text-muted-foreground ml-auto">
                      {formatTimestamp(msg.timestamp)}
                    </span>
                  </div>
                  <p className="text-sm text-foreground/90 whitespace-pre-wrap">
                    {msg.content}
                  </p>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Vote results for display round */}
      {filteredVotes.length > 0 && (
        <div className="space-y-2 pt-2 border-t">
          <h4 className="text-sm font-medium text-muted-foreground">{voteTitle || '本轮投票结果'}</h4>
          {filteredVotes.map((vote, index) => (
            <div
              key={index}
              className={cn(
                'p-3 rounded-lg border',
                vote.approved
                  ? 'border-green-500/30 bg-green-50 dark:bg-green-950/20'
                  : 'border-red-500/30 bg-red-50 dark:bg-red-950/20',
                isAssassination && 'border-red-500/30 bg-red-50 dark:bg-red-950/20'
              )}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">
                  {isAssassination ? '刺杀目标选择' : `第 ${vote.attempt} 次投票`}
                </span>
                <span
                  className={cn(
                    'text-sm font-medium',
                    isAssassination 
                      ? 'text-red-600'
                      : vote.approved ? 'text-green-600' : 'text-red-600'
                  )}
                >
                  {isAssassination 
                    ? '已执行'
                    : vote.approved ? '通过' : '否决'}
                </span>
              </div>

              <div className="flex flex-wrap gap-2">
                {Object.entries(vote.votes).map(([seat, approved]) => {
                  const seatNum = parseInt(seat);
                  const player = getPlayer(seatNum);
                  // For assassination: clickable if the assassin is an LLM player
                  const isClickable = isAssassination
                    ? gameId && assassinPlayer && !assassinPlayer.is_human
                    : gameId && player && !player.is_human;
                  
                  return (
                    <button
                      key={seat}
                      onClick={isClickable ? () => handleVoteClick(
                        isAssassination ? 'assassinate' : 'team_vote',
                        vote.round,
                        vote.attempt,
                        // For assassination: LLM record is under the assassin's seat, not the target
                        isAssassination && assassinPlayer ? assassinPlayer.seat : seatNum
                      ) : undefined}
                      disabled={!isClickable}
                      className={cn(
                        'flex items-center gap-1 px-2 py-1 rounded text-xs transition-all',
                        isAssassination
                          ? 'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300'
                          : approved
                            ? 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300'
                            : 'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300',
                        isClickable && 'cursor-pointer hover:ring-2 hover:ring-primary/30 hover:opacity-80',
                        !isClickable && 'cursor-default'
                      )}
                      title={isClickable ? '点击查看 LLM 调用详情' : undefined}
                    >
                      {isAssassination ? (
                        <Target className="h-3 w-3" />
                      ) : approved ? (
                        <Check className="h-3 w-3" />
                      ) : (
                        <X className="h-3 w-3" />
                      )}
                      <span>{getPlayerName(seatNum)}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Quest vote results for display round */}
      {questResult && questResult.success !== null && Object.keys(questResult.quest_votes).length > 0 && (
        <div className="space-y-2 pt-2 border-t">
          <h4 className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <Swords className="h-4 w-4" />
            本轮任务投票结果
          </h4>
          <div
            className={cn(
              'p-3 rounded-lg border',
              questResult.success
                ? 'border-blue-500/30 bg-blue-50 dark:bg-blue-950/20'
                : 'border-red-500/30 bg-red-50 dark:bg-red-950/20'
            )}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">
                任务成员投票
              </span>
              <span
                className={cn(
                  'text-sm font-medium',
                  questResult.success ? 'text-blue-600' : 'text-red-600'
                )}
              >
                {questResult.success ? '任务成功' : '任务失败'} 
                {questResult.fail_votes > 0 && (
                  <span className="text-muted-foreground ml-1">
                    ({questResult.fail_votes} 票失败)
                  </span>
                )}
              </span>
            </div>

            <div className="flex flex-wrap gap-2">
              {Object.entries(questResult.quest_votes).map(([seat, success]) => {
                const seatNum = parseInt(seat);
                const player = getPlayer(seatNum);
                const isClickable = gameId && player && !player.is_human;
                
                return (
                  <button
                    key={seat}
                    onClick={isClickable ? () => handleVoteClick('quest_vote', questResult.round, 1, seatNum) : undefined}
                    disabled={!isClickable}
                    className={cn(
                      'flex items-center gap-1 px-2 py-1 rounded text-xs transition-all',
                      success
                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300'
                        : 'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300',
                      isClickable && 'cursor-pointer hover:ring-2 hover:ring-primary/30 hover:opacity-80',
                      !isClickable && 'cursor-default'
                    )}
                    title={isClickable ? '点击查看 LLM 调用详情' : undefined}
                  >
                    {success ? (
                      <Check className="h-3 w-3" />
                    ) : (
                      <X className="h-3 w-3" />
                    )}
                    <span>{getPlayerName(seatNum)}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* LLM Detail Modal for Discussion */}
      {selectedMessage && gameId && (
        <LLMDetailModal
          isOpen={!!selectedMessage}
          onClose={() => setSelectedMessage(null)}
          gameId={gameId}
          roundNum={selectedMessage.round ?? displayRound}
          playerSeat={selectedMessage.seat}
          playerName={selectedMessage.player_name}
          timestamp={selectedMessage.timestamp ?? ''}
          actionType={isAssassination ? 'assassination_discussion' : 'discussion'}
          voteAttempt={selectedMessage.attempt}
        />
      )}

      {/* LLM Detail Modal for Votes */}
      {selectedVote && gameId && (
        <LLMDetailModal
          isOpen={!!selectedVote}
          onClose={() => setSelectedVote(null)}
          gameId={gameId}
          roundNum={selectedVote.roundNum}
          playerSeat={selectedVote.playerSeat}
          playerName={selectedVote.playerName}
          actionType={selectedVote.type}
          voteAttempt={selectedVote.attempt}
        />
      )}
    </div>
  );
}
