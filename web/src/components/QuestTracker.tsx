import { cn } from '@/lib/utils';
import type { QuestResult, GamePhase } from '@/lib/types';
import { Check, X, Target } from 'lucide-react';

interface QuestTrackerProps {
  questResults: QuestResult[];
  currentRound: number;
  playerCount: number;
  selectedRound?: number | 'assassination';
  onRoundClick?: (round: number | 'assassination') => void;
  phase?: GamePhase;
  assassinatedPlayer?: number;
  winner?: 'good' | 'evil';
}

// Quest team sizes by player count
const QUEST_SIZES: Record<number, number[]> = {
  5: [2, 3, 2, 3, 3],
  6: [2, 3, 4, 3, 4],
  7: [2, 3, 3, 4, 4],
  8: [3, 4, 4, 5, 5],
  9: [3, 4, 4, 5, 5],
  10: [3, 4, 4, 5, 5],
};

export function QuestTracker({
  questResults,
  currentRound,
  playerCount,
  selectedRound,
  onRoundClick,
  phase,
  assassinatedPlayer,
  winner,
}: QuestTrackerProps) {
  const questSizes = QUEST_SIZES[playerCount] || QUEST_SIZES[5];
  
  // 好人是否完成了3个任务
  const goodWins = questResults.filter(q => q.success === true).length;
  const showAssassination = goodWins >= 3 || phase === 'assassination_discussion' || phase === 'assassination' || phase === 'game_over';

  return (
    <div className="flex items-center gap-2">
      {[1, 2, 3, 4, 5].map((round) => {
        const result = questResults.find((q) => q.round === round);
        const isCurrent = round === currentRound && phase !== 'assassination_discussion' && phase !== 'assassination' && phase !== 'game_over';
        const isSelected = selectedRound === round;

        return (
          <div
            key={round}
            className={cn(
              'relative flex flex-col items-center cursor-pointer',
              'transition-transform hover:scale-105 active:scale-95'
            )}
            onClick={() => onRoundClick?.(round)}
          >
            {/* Quest circle */}
            <div
              className={cn(
                'flex items-center justify-center rounded-full border-2 transition-all',
                'w-12 h-12',
                // 选中状态
                isSelected && 'ring-4 ring-primary/50 border-primary',
                // 当前轮次但未选中
                !isSelected && isCurrent && 'border-primary ring-2 ring-primary/30',
                // 结果显示
                result?.success === true && 'bg-blue-500 border-blue-500',
                result?.success === false && 'bg-red-500 border-red-500',
                // 未进行且非当前轮次
                result === undefined && !isCurrent && !isSelected && 'border-muted-foreground/30'
              )}
            >
              {result?.success === true && (
                <Check className="h-6 w-6 text-white" />
              )}
              {result?.success === false && (
                <X className="h-6 w-6 text-white" />
              )}
              {result === undefined && (
                <span
                  className={cn(
                    'text-lg font-bold',
                    isCurrent || isSelected ? 'text-primary' : 'text-muted-foreground'
                  )}
                >
                  {questSizes[round - 1]}
                </span>
              )}
            </div>

            {/* Round number */}
            <span className="text-xs text-muted-foreground mt-1">
              第{round}轮
            </span>

            {/* Fail votes indicator */}
            {result?.success === false && result.fail_votes > 0 && (
              <span className="text-xs text-red-500">
                {result.fail_votes}票失败
              </span>
            )}
          </div>
        );
      })}
      
      {/* 刺杀环节 */}
      {showAssassination && (
        <>
          {/* 分隔线 */}
          <div className="h-12 w-px bg-muted-foreground/30 mx-2" />
          
          <div
            className={cn(
              'relative flex flex-col items-center cursor-pointer',
              'transition-transform hover:scale-105 active:scale-95'
            )}
            onClick={() => onRoundClick?.('assassination')}
          >
            {/* 刺杀环节圆圈 */}
            <div
              className={cn(
                'flex items-center justify-center rounded-full border-2 transition-all',
                'w-12 h-12',
                // 选中状态
                selectedRound === 'assassination' && 'ring-4 ring-primary/50 border-primary',
                // 当前阶段是刺杀阶段
                selectedRound !== 'assassination' && (phase === 'assassination_discussion' || phase === 'assassination') && 'border-red-500 ring-2 ring-red-500/30',
                // 游戏结束后根据结果显示
                phase === 'game_over' && winner === 'evil' && assassinatedPlayer !== undefined && 'bg-red-500 border-red-500',
                phase === 'game_over' && winner === 'good' && 'bg-blue-500 border-blue-500',
                // 默认状态
                phase !== 'game_over' && selectedRound !== 'assassination' && phase !== 'assassination_discussion' && phase !== 'assassination' && 'border-muted-foreground/30'
              )}
            >
              <Target className={cn(
                'h-6 w-6',
                phase === 'game_over' ? 'text-white' : 
                (phase === 'assassination_discussion' || phase === 'assassination' || selectedRound === 'assassination') ? 'text-red-500' : 'text-muted-foreground'
              )} />
            </div>

            {/* 标签 */}
            <span className="text-xs text-muted-foreground mt-1">
              刺杀
            </span>

            {/* 刺杀结果 */}
            {phase === 'game_over' && assassinatedPlayer !== undefined && (
              <span className={cn(
                'text-xs',
                winner === 'evil' ? 'text-red-500' : 'text-blue-500'
              )}>
                {winner === 'evil' ? '刺杀成功' : '刺杀失败'}
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
}
