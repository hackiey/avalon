import { Check, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { VoteResult, Player } from '@/lib/types';

interface VoteHistoryProps {
  votes: VoteResult[];
  players: Player[];
}

export function VoteHistory({ votes, players }: VoteHistoryProps) {
  if (votes.length === 0) {
    return (
      <div className="text-center text-muted-foreground py-4">
        暂无投票记录
      </div>
    );
  }

  const getPlayerName = (seat: number) =>
    players.find((p) => p.seat === seat)?.name || `玩家${seat}`;

  return (
    <div className="space-y-3">
      {votes.map((vote, index) => (
        <div
          key={index}
          className={cn(
            'p-3 rounded-lg border',
            vote.approved ? 'border-green-500/30 bg-green-50 dark:bg-green-950/20' : 'border-red-500/30 bg-red-50 dark:bg-red-950/20'
          )}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">
              第{vote.round}轮 第{vote.attempt}次投票
            </span>
            <span
              className={cn(
                'text-sm font-medium',
                vote.approved ? 'text-green-600' : 'text-red-600'
              )}
            >
              {vote.approved ? '通过' : '否决'}
            </span>
          </div>

          <div className="flex flex-wrap gap-2">
            {Object.entries(vote.votes).map(([seat, approved]) => (
              <div
                key={seat}
                className={cn(
                  'flex items-center gap-1 px-2 py-1 rounded text-xs',
                  approved
                    ? 'bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300'
                    : 'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300'
                )}
              >
                {approved ? (
                  <Check className="h-3 w-3" />
                ) : (
                  <X className="h-3 w-3" />
                )}
                <span>{getPlayerName(parseInt(seat))}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
