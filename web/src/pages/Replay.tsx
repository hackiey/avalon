import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Loader2,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent } from '@/components/ui/Card';
import { GameBoard } from '@/components/GameBoard';
import { getGameReplay } from '@/lib/api';
import type { GameState } from '@/lib/types';
import { cn } from '@/lib/utils';

export function Replay() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [states, setStates] = useState<GameState[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;

    const loadReplay = async () => {
      try {
        const replay = await getGameReplay(id);
        setStates(replay);
      } catch (err) {
        setError('无法加载回放数据');
      } finally {
        setLoading(false);
      }
    };

    loadReplay();
  }, [id]);

  // Auto-play effect
  useEffect(() => {
    if (!playing || currentIndex >= states.length - 1) {
      setPlaying(false);
      return;
    }

    const timer = setTimeout(() => {
      setCurrentIndex((i) => i + 1);
    }, 2000);

    return () => clearTimeout(timer);
  }, [playing, currentIndex, states.length]);

  const handlePrev = () => {
    setCurrentIndex((i) => Math.max(0, i - 1));
    setPlaying(false);
  };

  const handleNext = () => {
    setCurrentIndex((i) => Math.min(states.length - 1, i + 1));
    setPlaying(false);
  };

  const handleSeek = (index: number) => {
    setCurrentIndex(index);
    setPlaying(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error || states.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <p className="text-destructive">{error || '无回放数据'}</p>
        <Button onClick={() => navigate('/')}>返回首页</Button>
      </div>
    );
  }

  const currentState = states[currentIndex];

  return (
    <div className="container mx-auto py-6 px-4 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <Button variant="ghost" onClick={() => navigate('/')}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          返回
        </Button>
        <span className="text-muted-foreground">
          回放 {currentIndex + 1} / {states.length}
        </span>
      </div>

      {/* Game board */}
      <GameBoard gameState={currentState} />

      {/* Playback controls */}
      <Card className="mt-6">
        <CardContent className="py-4">
          <div className="flex items-center justify-center gap-4">
            <Button
              variant="outline"
              size="icon"
              onClick={handlePrev}
              disabled={currentIndex === 0}
            >
              <SkipBack className="h-4 w-4" />
            </Button>

            <Button
              size="icon"
              onClick={() => setPlaying(!playing)}
              disabled={currentIndex >= states.length - 1}
            >
              {playing ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </Button>

            <Button
              variant="outline"
              size="icon"
              onClick={handleNext}
              disabled={currentIndex >= states.length - 1}
            >
              <SkipForward className="h-4 w-4" />
            </Button>
          </div>

          {/* Timeline */}
          <div className="mt-4 flex gap-1">
            {states.map((_, index) => (
              <button
                key={index}
                onClick={() => handleSeek(index)}
                className={cn(
                  'flex-1 h-2 rounded-full transition-colors',
                  index === currentIndex
                    ? 'bg-primary'
                    : index < currentIndex
                    ? 'bg-primary/30'
                    : 'bg-muted'
                )}
              />
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
