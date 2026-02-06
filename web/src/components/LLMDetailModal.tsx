import { useState, useEffect } from 'react';
import { Modal } from '@/components/ui/Modal';
import { cn } from '@/lib/utils';
import { Loader2, MessageSquare, Brain, Wrench, ChevronDown, ChevronRight } from 'lucide-react';

interface LLMDetails {
  id: number;
  action_type: string;
  player_seat: number;
  content: string;
  round_num: number;
  timestamp: string;
  llm_input: {
    messages?: Array<{ role: string; content: string }>;
  } | null;
  llm_output: {
    content?: string;
    reasoning_content?: string;
    tool_calls?: Array<{
      name: string;
      arguments: Record<string, unknown>;
    }>;
    error?: string;
  } | null;
}

type ActionType = 'discussion' | 'team_vote' | 'quest_vote' | 'assassinate' | 'assassination_discussion';

interface LLMDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  gameId: string;
  roundNum: number;
  playerSeat: number;
  playerName: string;
  timestamp?: string;  // Optional for vote actions
  actionType?: ActionType;  // Type of action (default: discussion)
  voteAttempt?: number;  // For team_vote actions
}

export function LLMDetailModal({
  isOpen,
  onClose,
  gameId,
  roundNum,
  playerSeat,
  playerName,
  timestamp,
  actionType = 'discussion',
  voteAttempt,
}: LLMDetailModalProps) {
  const [details, setDetails] = useState<LLMDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    messages: true,
    reasoning: true,
    toolCalls: true,
  });

  useEffect(() => {
    if (isOpen && gameId && roundNum !== undefined && playerSeat !== undefined) {
      // For discussion types, timestamp is required; for vote/action types, attempt is required
      if ((actionType === 'discussion' || actionType === 'assassination_discussion') && timestamp) {
        fetchDetails();
      } else if ((actionType === 'team_vote' || actionType === 'quest_vote' || actionType === 'assassinate') && voteAttempt !== undefined) {
        fetchDetails();
      }
    }
  }, [isOpen, gameId, roundNum, playerSeat, timestamp, actionType, voteAttempt]);

  const fetchDetails = async () => {
    // Validate required params
    if (roundNum === undefined || playerSeat === undefined) {
      setError('缺少必要参数');
      return;
    }
    
    setLoading(true);
    setError(null);
    try {
      let response: Response;
      
      if (actionType === 'discussion' || actionType === 'assassination_discussion') {
        if (!timestamp) {
          setError('缺少时间戳参数');
          return;
        }
        const params = new URLSearchParams({
          round_num: String(roundNum),
          player_seat: String(playerSeat),
          timestamp: timestamp,
          action_type: actionType,
        });
        // Add vote attempt if provided (to distinguish discussions in different vote attempts)
        // Skip when 0 (assassination discussions use attempt=0 as placeholder, but DB stores null)
        if (voteAttempt !== undefined && voteAttempt > 0) {
          params.set('attempt', String(voteAttempt));
        }
        response = await fetch(
          `/api/games/${gameId}/discussion-llm-details?${params}`
        );
      } else {
        // team_vote or quest_vote or assassinate
        const params = new URLSearchParams({
          round_num: String(roundNum),
          attempt: String(voteAttempt ?? 1),
          player_seat: String(playerSeat),
          action_type: actionType === 'assassinate' ? 'assassination' : actionType,
        });
        response = await fetch(
          `/api/games/${gameId}/vote-llm-details?${params}`
        );
      }
      
      if (!response.ok) {
        if (response.status === 404) {
          setError('未找到 LLM 调用详情（可能是人类玩家或数据未保存）');
        } else {
          throw new Error('Failed to fetch LLM details');
        }
        return;
      }
      const data = await response.json();
      setDetails(data);
    } catch (err) {
      setError('获取 LLM 调用详情失败');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  const renderMessages = () => {
    if (!details?.llm_input?.messages) return null;

    return (
      <div className="space-y-2">
        <button
          onClick={() => toggleSection('messages')}
          className="flex items-center gap-2 w-full text-left font-medium text-sm hover:text-primary transition-colors"
        >
          {expandedSections.messages ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          <MessageSquare className="h-4 w-4" />
          输入消息 ({details.llm_input.messages.length})
        </button>

        {expandedSections.messages && (
          <div className="space-y-3 pl-6">
            {details.llm_input.messages.map((msg, index) => (
              <div
                key={index}
                className={cn(
                  'rounded-lg p-3 text-sm',
                  msg.role === 'system'
                    ? 'bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800'
                    : msg.role === 'user'
                    ? 'bg-green-50 dark:bg-green-950/30 border border-green-200 dark:border-green-800'
                    : 'bg-muted'
                )}
              >
                <div className="font-medium text-xs uppercase mb-1 opacity-70">
                  {msg.role === 'system' ? '系统提示' : msg.role === 'user' ? '用户提示' : msg.role}
                </div>
                <pre className="whitespace-pre-wrap font-mono text-xs overflow-x-auto">
                  {msg.content}
                </pre>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const renderReasoning = () => {
    if (!details?.llm_output?.reasoning_content) return null;

    return (
      <div className="space-y-2">
        <button
          onClick={() => toggleSection('reasoning')}
          className="flex items-center gap-2 w-full text-left font-medium text-sm hover:text-primary transition-colors"
        >
          {expandedSections.reasoning ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          <Brain className="h-4 w-4" />
          推理过程
        </button>

        {expandedSections.reasoning && (
          <div className="pl-6">
            <div className="rounded-lg p-3 bg-purple-50 dark:bg-purple-950/30 border border-purple-200 dark:border-purple-800">
              <pre className="whitespace-pre-wrap font-mono text-xs overflow-x-auto">
                {details.llm_output.reasoning_content}
              </pre>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderToolCalls = () => {
    if (!details?.llm_output?.tool_calls?.length) return null;

    return (
      <div className="space-y-2">
        <button
          onClick={() => toggleSection('toolCalls')}
          className="flex items-center gap-2 w-full text-left font-medium text-sm hover:text-primary transition-colors"
        >
          {expandedSections.toolCalls ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          <Wrench className="h-4 w-4" />
          工具调用 ({details.llm_output.tool_calls.length})
        </button>

        {expandedSections.toolCalls && (
          <div className="space-y-2 pl-6">
            {details.llm_output.tool_calls.map((tool, index) => (
              <div
                key={index}
                className="rounded-lg p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800"
              >
                <div className="font-medium text-sm mb-2">{tool.name}</div>
                <pre className="whitespace-pre-wrap font-mono text-xs overflow-x-auto bg-background/50 rounded p-2">
                  {JSON.stringify(tool.arguments, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const renderContent = () => {
    if (!details?.llm_output?.content) return null;

    return (
      <div className="space-y-2">
        <div className="font-medium text-sm flex items-center gap-2">
          <MessageSquare className="h-4 w-4" />
          输出内容
        </div>
        <div className="rounded-lg p-3 bg-muted">
          <p className="text-sm whitespace-pre-wrap">{details.llm_output.content}</p>
        </div>
      </div>
    );
  };

  const getTitle = () => {
    switch (actionType) {
      case 'team_vote':
        return `${playerName} 的队伍投票详情`;
      case 'quest_vote':
        return `${playerName} 的任务投票详情`;
      case 'assassinate':
        return `${playerName} 的刺杀选择详情`;
      case 'assassination_discussion':
        return `${playerName} 的刺杀讨论详情`;
      default:
        return `${playerName} 的发言详情`;
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={getTitle()} size="xl">
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : error ? (
        <div className="text-center py-12 text-muted-foreground">
          {error}
        </div>
      ) : details ? (
        <div className="space-y-6">
          {/* Meta info */}
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>第 {details.round_num} 轮</span>
            <span>动作类型: {details.action_type}</span>
          </div>

          {/* Messages */}
          {renderMessages()}

          {/* Reasoning */}
          {renderReasoning()}

          {/* Tool calls */}
          {renderToolCalls()}

          {/* Content */}
          {renderContent()}

          {/* No LLM data */}
          {!details.llm_input && !details.llm_output && (
            <div className="text-center py-8 text-muted-foreground">
              此发言没有 LLM 调用记录（可能是人类玩家）
            </div>
          )}
        </div>
      ) : null}
    </Modal>
  );
}
