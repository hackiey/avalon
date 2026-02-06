import { User, Crown, Sword } from 'lucide-react';
import { cn, ROLE_NAMES } from '@/lib/utils';
import type { Player, Role } from '@/lib/types';
import { Badge } from '@/components/ui/Badge';

interface PlayerCardProps {
  player: Player;
  isSelected?: boolean;
  onClick?: () => void;
  showRole?: boolean;
  size?: 'sm' | 'md' | 'lg';
  layout?: 'vertical' | 'horizontal';
}

export function PlayerCard({
  player,
  isSelected = false,
  onClick,
  showRole = false,
  size = 'md',
  layout = 'vertical',
}: PlayerCardProps) {
  const sizeClasses = {
    sm: 'p-2',
    md: 'p-3',
    lg: 'p-6',
  };

  const iconSize = {
    sm: 16,
    md: 20,
    lg: 32,
  };

  // Horizontal layout for player list
  if (layout === 'horizontal') {
    return (
      <div
        className={cn(
          'relative flex items-center gap-3 rounded-lg border bg-card transition-all',
          sizeClasses[size],
          onClick && 'cursor-pointer hover:border-primary hover:shadow-md',
          isSelected && 'border-primary ring-2 ring-primary ring-offset-2',
          player.is_on_quest && 'bg-amber-50 dark:bg-amber-950/30'
        )}
        onClick={onClick}
      >
        {/* Avatar */}
        <div
          className={cn(
            'flex-shrink-0 flex items-center justify-center rounded-full bg-muted',
            size === 'sm' ? 'h-8 w-8' : size === 'md' ? 'h-10 w-10' : 'h-16 w-16'
          )}
        >
          {player.is_human ? (
            <User size={iconSize[size]} className="text-muted-foreground" />
          ) : (
            <div className="text-xs font-mono text-muted-foreground">AI</div>
          )}
        </div>

        {/* Player info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm truncate">{player.name}</span>
            {player.is_leader && (
              <Crown className="h-4 w-4 text-amber-500 flex-shrink-0" />
            )}
            {player.is_on_quest && (
              <Sword className="h-4 w-4 text-amber-600 flex-shrink-0" />
            )}
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            {/* Role badge */}
            {showRole && player.role && (
              <Badge
                variant={player.team === 'evil' ? 'destructive' : 'default'}
                className="text-xs"
              >
                {ROLE_NAMES[player.role as Role] || player.role}
              </Badge>
            )}
            {/* Team indicator (if visible but not role) */}
            {showRole && player.team && !player.role && (
              <Badge variant={player.team === 'evil' ? 'destructive' : 'default'} className="text-xs">
                {player.team === 'good' ? '好人' : '坏人'}
              </Badge>
            )}
            {/* Model name */}
            {!player.is_human && player.model_name && (
              <span className="text-xs text-muted-foreground truncate">
                {player.model_name}
              </span>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Vertical layout (original)
  return (
    <div
      className={cn(
        'relative rounded-lg border bg-card transition-all',
        sizeClasses[size],
        onClick && 'cursor-pointer hover:border-primary hover:shadow-md',
        isSelected && 'border-primary ring-2 ring-primary ring-offset-2',
        player.is_on_quest && 'bg-amber-50 dark:bg-amber-950/30'
      )}
      onClick={onClick}
    >
      {/* Leader crown */}
      {player.is_leader && (
        <div className="absolute -top-2 -right-2">
          <Crown className="h-5 w-5 text-amber-500" />
        </div>
      )}

      {/* Player info */}
      <div className="flex flex-col items-center gap-2">
        {/* Avatar */}
        <div
          className={cn(
            'flex items-center justify-center rounded-full bg-muted',
            size === 'sm' ? 'h-8 w-8' : size === 'md' ? 'h-12 w-12' : 'h-16 w-16'
          )}
        >
          {player.is_human ? (
            <User size={iconSize[size]} className="text-muted-foreground" />
          ) : (
            <div className="text-xs font-mono text-muted-foreground">AI</div>
          )}
        </div>

        {/* Name */}
        <div className="text-center">
          <div className="font-medium text-sm">{player.name}</div>
        </div>

        {/* Role (if visible) */}
        {showRole && player.role && (
          <Badge
            variant={player.team === 'evil' ? 'destructive' : 'default'}
            className="text-xs"
          >
            {ROLE_NAMES[player.role as Role] || player.role}
          </Badge>
        )}

        {/* Team indicator (if visible but not role) */}
        {showRole && player.team && !player.role && (
          <Badge variant={player.team === 'evil' ? 'destructive' : 'default'}>
            {player.team === 'good' ? '好人' : '坏人'}
          </Badge>
        )}

        {/* Model name */}
        {!player.is_human && player.model_name && (
          <div className="text-xs text-muted-foreground truncate max-w-full">
            {player.model_name}
          </div>
        )}
      </div>

      {/* Quest indicator */}
      {player.is_on_quest && (
        <div className="absolute bottom-1 right-1">
          <Sword className="h-4 w-4 text-amber-600" />
        </div>
      )}
    </div>
  );
}
