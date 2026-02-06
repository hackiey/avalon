import { HTMLAttributes } from 'react';
import { cn } from '@/lib/utils';

export interface BadgeProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'secondary' | 'outline' | 'destructive' | 'success';
}

function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  return (
    <div
      className={cn(
        'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors',
        'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
        {
          'border-transparent bg-primary text-primary-foreground': variant === 'default',
          'border-transparent bg-secondary text-secondary-foreground': variant === 'secondary',
          'text-foreground': variant === 'outline',
          'border-transparent bg-destructive text-destructive-foreground': variant === 'destructive',
          'border-transparent bg-green-500 text-white': variant === 'success',
        },
        className
      )}
      {...props}
    />
  );
}

export { Badge };
