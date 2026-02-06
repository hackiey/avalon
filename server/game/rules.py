"""Game rules and configurations for different player counts."""

from dataclasses import dataclass
from typing import List, Dict
from server.game.roles import Role


@dataclass
class GameRules:
    """Rules configuration for a specific player count."""
    player_count: int
    good_count: int
    evil_count: int
    quest_team_sizes: List[int]  # Team size for each of 5 quests
    two_fails_required: List[bool]  # Whether quest requires 2 fails (for 7+ players, quest 4)
    
    @property
    def roles(self) -> List[Role]:
        """Get the list of roles for this player count."""
        roles = []
        
        # Good team: 1 Merlin + rest Loyal Servants
        roles.append(Role.MERLIN)
        for _ in range(self.good_count - 1):
            roles.append(Role.LOYAL_SERVANT)
        
        # Evil team: 1 Assassin + rest Minions
        roles.append(Role.ASSASSIN)
        for _ in range(self.evil_count - 1):
            roles.append(Role.MINION)
        
        return roles


# Game configurations for each player count
GAME_CONFIGS: Dict[int, GameRules] = {
    5: GameRules(
        player_count=5,
        good_count=3,
        evil_count=2,
        quest_team_sizes=[2, 3, 2, 3, 3],
        two_fails_required=[False, False, False, False, False]
    ),
    6: GameRules(
        player_count=6,
        good_count=4,
        evil_count=2,
        quest_team_sizes=[2, 3, 4, 3, 4],
        two_fails_required=[False, False, False, False, False]
    ),
    7: GameRules(
        player_count=7,
        good_count=4,
        evil_count=3,
        quest_team_sizes=[2, 3, 3, 4, 4],
        two_fails_required=[False, False, False, True, False]
    ),
    8: GameRules(
        player_count=8,
        good_count=5,
        evil_count=3,
        quest_team_sizes=[3, 4, 4, 5, 5],
        two_fails_required=[False, False, False, True, False]
    ),
    9: GameRules(
        player_count=9,
        good_count=6,
        evil_count=3,
        quest_team_sizes=[3, 4, 4, 5, 5],
        two_fails_required=[False, False, False, True, False]
    ),
    10: GameRules(
        player_count=10,
        good_count=6,
        evil_count=4,
        quest_team_sizes=[3, 4, 4, 5, 5],
        two_fails_required=[False, False, False, True, False]
    ),
}


def get_rules(player_count: int) -> GameRules:
    """Get the rules for a specific player count."""
    if player_count not in GAME_CONFIGS:
        raise ValueError(f"Invalid player count: {player_count}. Must be between 5 and 10.")
    return GAME_CONFIGS[player_count]


def get_quest_team_size(player_count: int, quest_number: int) -> int:
    """Get the team size for a specific quest (1-indexed)."""
    rules = get_rules(player_count)
    return rules.quest_team_sizes[quest_number - 1]


def requires_two_fails(player_count: int, quest_number: int) -> bool:
    """Check if a quest requires two fails to fail (for 7+ players, quest 4)."""
    rules = get_rules(player_count)
    return rules.two_fails_required[quest_number - 1]
