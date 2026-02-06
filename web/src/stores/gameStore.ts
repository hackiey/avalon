import { create } from 'zustand';
import type { GameState, DiscussionMessage, AssassinationDiscussionMessage } from '@/lib/types';

interface GameStore {
  // Connection state
  connected: boolean;
  setConnected: (connected: boolean) => void;

  // Current game state
  gameState: GameState | null;
  setGameState: (state: GameState) => void;
  clearGameState: () => void;

  // Human player seat (if playing)
  humanSeat: number | null;
  setHumanSeat: (seat: number | null) => void;

  // God mode (reveal all information)
  godMode: boolean;
  setGodMode: (enabled: boolean) => void;

  // Discussion
  addDiscussion: (message: DiscussionMessage) => void;
  
  // Assassination Discussion
  addAssassinationDiscussion: (message: AssassinationDiscussionMessage) => void;

  // UI state
  selectedPlayers: number[];
  togglePlayerSelection: (seat: number) => void;
  clearSelection: () => void;
}

export const useGameStore = create<GameStore>((set) => ({
  // Connection
  connected: false,
  setConnected: (connected) => set({ connected }),

  // Game state
  gameState: null,
  setGameState: (state) =>
    set((currentState) => {
      // When in god mode, preserve player role information from existing state
      // because WebSocket updates don't include reveal_all data
      if (currentState.godMode && currentState.gameState && state.players) {
        const existingPlayers = currentState.gameState.players;
        const mergedPlayers = state.players.map((player) => {
          const existingPlayer = existingPlayers.find((p) => p.seat === player.seat);
          // Preserve role and team if they exist in current state but not in new state
          if (existingPlayer && existingPlayer.role && !player.role) {
            return {
              ...player,
              role: existingPlayer.role,
              team: existingPlayer.team,
            };
          }
          return player;
        });
        return { gameState: { ...state, players: mergedPlayers } };
      }
      return { gameState: state };
    }),
  clearGameState: () => set({ gameState: null }),

  // Human player
  humanSeat: null,
  setHumanSeat: (seat) => set({ humanSeat: seat }),

  // God mode
  godMode: false,
  setGodMode: (enabled) => set({ godMode: enabled }),

  // Discussion
  addDiscussion: (message) =>
    set((state) => ({
      gameState: state.gameState
        ? {
            ...state.gameState,
            discussion_history: [...state.gameState.discussion_history, message],
          }
        : null,
    })),
  
  // Assassination Discussion
  addAssassinationDiscussion: (message) =>
    set((state) => ({
      gameState: state.gameState
        ? {
            ...state.gameState,
            assassination_discussion_history: [
              ...(state.gameState.assassination_discussion_history || []),
              message,
            ],
          }
        : null,
    })),

  // Selection
  selectedPlayers: [],
  togglePlayerSelection: (seat) =>
    set((state) => ({
      selectedPlayers: state.selectedPlayers.includes(seat)
        ? state.selectedPlayers.filter((s) => s !== seat)
        : [...state.selectedPlayers, seat],
    })),
  clearSelection: () => set({ selectedPlayers: [] }),
}));
