import { useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { useGameStore } from '@/stores/gameStore';

export function useSocket() {
  const socketRef = useRef<Socket | null>(null);
  const { setGameState, addDiscussion, addAssassinationDiscussion, setConnected } = useGameStore();

  useEffect(() => {
    // Create socket connection
    const socket = io('/', {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('Connected to server');
      setConnected(true);
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from server');
      setConnected(false);
    });

    socket.on('game:state', (data) => {
      console.log('Game state update:', data);
      setGameState(data.state);
    });

    socket.on('game:discussion', (data) => {
      console.log('Discussion:', data);
      addDiscussion({
        seat: data.seat,
        player_name: data.player_name,
        content: data.content,
        round: data.round,
        attempt: data.attempt,
        timestamp: data.timestamp || new Date().toISOString(),
      });
    });

    socket.on('game:vote_result', (data) => {
      console.log('Vote result:', data);
    });

    socket.on('game:quest_result', (data) => {
      console.log('Quest result:', data);
    });

    socket.on('game:assassination_discussion', (data) => {
      console.log('Assassination discussion:', data);
      addAssassinationDiscussion({
        seat: data.seat,
        player_name: data.player_name,
        content: data.content,
        timestamp: data.timestamp || new Date().toISOString(),
      });
    });

    socket.on('game:assassination', (data) => {
      console.log('Assassination:', data);
    });

    return () => {
      socket.disconnect();
    };
  }, [setGameState, addDiscussion, addAssassinationDiscussion, setConnected]);

  const joinGame = useCallback((gameId: string) => {
    socketRef.current?.emit('join_game', { game_id: gameId });
  }, []);

  const leaveGame = useCallback((gameId: string) => {
    socketRef.current?.emit('leave_game', { game_id: gameId });
  }, []);

  const startGame = useCallback((gameId: string) => {
    socketRef.current?.emit('game_start', { game_id: gameId });
  }, []);

  const sendDiscussion = useCallback((gameId: string, content: string) => {
    socketRef.current?.emit('human_discussion', { game_id: gameId, content });
  }, []);

  const sendVote = useCallback((gameId: string, approve: boolean) => {
    socketRef.current?.emit('human_vote', { game_id: gameId, approve });
  }, []);

  const sendQuestVote = useCallback((gameId: string, success: boolean) => {
    socketRef.current?.emit('human_quest', { game_id: gameId, success });
  }, []);

  const sendTeamSelect = useCallback((gameId: string, team: number[], speech?: string) => {
    socketRef.current?.emit('human_team_select', { game_id: gameId, team, speech: speech || '' });
  }, []);

  const sendAssassinationDiscussion = useCallback((gameId: string, content: string) => {
    socketRef.current?.emit('human_assassination_discussion', { game_id: gameId, content });
  }, []);

  const sendAssassinate = useCallback((gameId: string, target: number) => {
    socketRef.current?.emit('human_assassinate', { game_id: gameId, target });
  }, []);

  return {
    socket: socketRef.current,
    joinGame,
    leaveGame,
    startGame,
    sendDiscussion,
    sendVote,
    sendQuestVote,
    sendTeamSelect,
    sendAssassinationDiscussion,
    sendAssassinate,
  };
}
