import { useState } from 'react';
import { Send, Check, X, Sword, Target, Users, MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import type { GameState } from '@/lib/types';

interface HumanControlsProps {
  gameState: GameState;
  humanSeat: number | null;
  selectedPlayers: number[];
  onTeamSelect: (speech?: string) => void;
  onDiscussion: (content: string) => void;
  onVote: (approve: boolean) => void;
  onQuestVote: (success: boolean) => void;
  onAssassinationDiscussion: (content: string) => void;
  onAssassinate: (target: number) => void;
}

export function HumanControls({
  gameState,
  humanSeat,
  selectedPlayers,
  onTeamSelect,
  onDiscussion,
  onVote,
  onQuestVote,
  onAssassinationDiscussion,
  onAssassinate,
}: HumanControlsProps) {
  const [discussionInput, setDiscussionInput] = useState('');

  const actionType = gameState.human_action_type;
  const player = humanSeat !== null ? gameState.players[humanSeat] : null;
  const isEvil = player?.team === 'evil';

  // Get required team size for current round
  const QUEST_SIZES: Record<number, number[]> = {
    5: [2, 3, 2, 3, 3],
    6: [2, 3, 4, 3, 4],
  };
  const requiredSize = QUEST_SIZES[gameState.player_count]?.[gameState.current_round - 1] || 2;

  const handleSendDiscussion = () => {
    if (discussionInput.trim()) {
      onDiscussion(discussionInput.trim());
      setDiscussionInput('');
    }
  };

  return (
    <Card className="border-primary">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          {actionType === 'team_selection' && (
            <>
              <Users className="h-5 w-5" />
              选择队伍 (需要 {requiredSize} 人)
            </>
          )}
          {actionType === 'discussion' && (
            <>
              <Send className="h-5 w-5" />
              轮到你发言
            </>
          )}
          {actionType === 'leader_discussion' && (
            <>
              <Send className="h-5 w-5" />
              作为队长发言
            </>
          )}
          {actionType === 'vote' && (
            <>
              <Check className="h-5 w-5" />
              投票表决
            </>
          )}
          {actionType === 'quest' && (
            <>
              <Sword className="h-5 w-5" />
              执行任务
            </>
          )}
          {actionType === 'assassination_discussion' && (
            <>
              <MessageCircle className="h-5 w-5" />
              刺杀讨论
            </>
          )}
          {actionType === 'assassinate' && (
            <>
              <Target className="h-5 w-5" />
              选择刺杀目标
            </>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Team Selection */}
        {actionType === 'team_selection' && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              点击玩家选择队员，已选择: {selectedPlayers.length}/{requiredSize}
            </p>
            <div className="flex flex-wrap gap-2">
              {selectedPlayers.map((seat) => (
                <span
                  key={seat}
                  className="px-2 py-1 bg-primary/10 text-primary rounded text-sm"
                >
                  {gameState.players[seat]?.name}
                </span>
              ))}
            </div>
            {/* 总结发言输入框 */}
            <div className="pt-2 border-t">
              <p className="text-sm text-muted-foreground mb-2">
                总结发言：说明你的最终决定和理由
              </p>
              <Input
                value={discussionInput}
                onChange={(e) => setDiscussionInput(e.target.value)}
                placeholder="综合大家的意见，我最终决定..."
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                  }
                }}
              />
            </div>
            <Button
              onClick={() => {
                onTeamSelect(discussionInput.trim() || undefined);
                setDiscussionInput('');
              }}
              disabled={selectedPlayers.length !== requiredSize}
              className="w-full"
            >
              确认队伍并提交
            </Button>
          </div>
        )}

        {/* Discussion (including leader discussion) */}
        {(actionType === 'discussion' || actionType === 'leader_discussion') && (
          <div className="space-y-3">
            {actionType === 'leader_discussion' && (
              <p className="text-sm text-muted-foreground">
                作为队长，请提出你初步考虑的队伍配置和理由。讨论结束后你可以调整最终队伍。
              </p>
            )}
            <div className="flex gap-2">
              <Input
                value={discussionInput}
                onChange={(e) => setDiscussionInput(e.target.value)}
                placeholder={actionType === 'leader_discussion' ? "提出你的队伍配置建议..." : "输入你的发言..."}
                onKeyDown={(e) => e.key === 'Enter' && handleSendDiscussion()}
              />
              <Button onClick={handleSendDiscussion}>
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

        {/* Vote */}
        {actionType === 'vote' && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              是否同意这个队伍执行任务?
            </p>
            <div className="flex gap-4">
              <Button
                onClick={() => onVote(true)}
                className="flex-1"
                variant="default"
              >
                <Check className="mr-2 h-4 w-4" />
                赞成
              </Button>
              <Button
                onClick={() => onVote(false)}
                className="flex-1"
                variant="destructive"
              >
                <X className="mr-2 h-4 w-4" />
                反对
              </Button>
            </div>
          </div>
        )}

        {/* Quest */}
        {actionType === 'quest' && (
          <div className="space-y-4">
            {isEvil ? (
              <>
                <p className="text-sm text-muted-foreground">
                  作为坏人，你可以选择让任务成功或失败
                </p>
                <div className="flex gap-4">
                  <Button
                    onClick={() => onQuestVote(true)}
                    className="flex-1"
                    variant="default"
                  >
                    <Check className="mr-2 h-4 w-4" />
                    成功
                  </Button>
                  <Button
                    onClick={() => onQuestVote(false)}
                    className="flex-1"
                    variant="destructive"
                  >
                    <X className="mr-2 h-4 w-4" />
                    失败
                  </Button>
                </div>
              </>
            ) : (
              <>
                <p className="text-sm text-muted-foreground">
                  作为好人，你必须让任务成功
                </p>
                <Button onClick={() => onQuestVote(true)} className="w-full">
                  <Check className="mr-2 h-4 w-4" />
                  执行任务
                </Button>
              </>
            )}
          </div>
        )}

        {/* Assassination Discussion */}
        {actionType === 'assassination_discussion' && (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              好人完成了3个任务，现在是坏人阵营的私密讨论时间。请分析谁最可能是梅林，与同伴分享你的判断。
            </p>
            {/* Show previous assassination discussions */}
            {gameState.assassination_discussion_history && gameState.assassination_discussion_history.length > 0 && (
              <div className="bg-muted/50 rounded p-3 space-y-2">
                <p className="text-xs text-muted-foreground">同伴的分析：</p>
                {gameState.assassination_discussion_history.map((msg, idx) => (
                  <div key={idx} className="text-sm">
                    <span className="font-medium">{msg.player_name}:</span> {msg.content}
                  </div>
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <Input
                value={discussionInput}
                onChange={(e) => setDiscussionInput(e.target.value)}
                placeholder="我认为梅林是..."
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    if (discussionInput.trim()) {
                      onAssassinationDiscussion(discussionInput.trim());
                      setDiscussionInput('');
                    }
                  }
                }}
              />
              <Button onClick={() => {
                if (discussionInput.trim()) {
                  onAssassinationDiscussion(discussionInput.trim());
                  setDiscussionInput('');
                }
              }}>
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

        {/* Assassinate */}
        {actionType === 'assassinate' && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              点击玩家选择刺杀目标 (尝试找出梅林)
            </p>
            {/* Show assassination discussion summary */}
            {gameState.assassination_discussion_history && gameState.assassination_discussion_history.length > 0 && (
              <div className="bg-muted/50 rounded p-3 space-y-2">
                <p className="text-xs text-muted-foreground">同伴的分析总结：</p>
                {gameState.assassination_discussion_history.map((msg, idx) => (
                  <div key={idx} className="text-sm">
                    <span className="font-medium">{msg.player_name}:</span> {msg.content}
                  </div>
                ))}
              </div>
            )}
            {selectedPlayers.length > 0 && (
              <Button
                onClick={() => onAssassinate(selectedPlayers[0])}
                className="w-full"
                variant="destructive"
              >
                <Target className="mr-2 h-4 w-4" />
                刺杀 {gameState.players[selectedPlayers[0]]?.name}
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
