"""Role definitions for Avalon game."""

from enum import Enum
from typing import List, Set
from dataclasses import dataclass


class Team(str, Enum):
    """The two teams in Avalon."""
    GOOD = "good"
    EVIL = "evil"


class Role(str, Enum):
    """Available roles in the game."""
    # Good team
    MERLIN = "merlin"
    LOYAL_SERVANT = "loyal_servant"
    
    # Evil team
    ASSASSIN = "assassin"
    MINION = "minion"


@dataclass
class RoleInfo:
    """Information about a role."""
    role: Role
    team: Team
    name_cn: str
    description: str
    knows_evil: bool = False  # Can see evil players (like Merlin)
    known_to_merlin: bool = True  # Visible to Merlin as evil
    knows_teammates: bool = False  # Can see other evil players


# Role definitions
ROLE_INFO = {
    Role.MERLIN: RoleInfo(
        role=Role.MERLIN,
        team=Team.GOOD,
        name_cn="梅林",
        description="知道所有坏人身份（除莫德雷德外），但不能暴露自己",
        knows_evil=True
    ),
    Role.LOYAL_SERVANT: RoleInfo(
        role=Role.LOYAL_SERVANT,
        team=Team.GOOD,
        name_cn="忠臣",
        description="亚瑟的忠诚仆人，没有特殊能力"
    ),
    Role.ASSASSIN: RoleInfo(
        role=Role.ASSASSIN,
        team=Team.EVIL,
        name_cn="刺客",
        description="游戏结束时如果好人完成三个任务，可以刺杀梅林",
        known_to_merlin=True,
        knows_teammates=True
    ),
    Role.MINION: RoleInfo(
        role=Role.MINION,
        team=Team.EVIL,
        name_cn="爪牙",
        description="莫德雷德的爪牙，知道其他坏人身份",
        known_to_merlin=True,
        knows_teammates=True
    ),
}


def get_role_info(role: Role) -> RoleInfo:
    """Get information about a role."""
    return ROLE_INFO[role]


def get_team(role: Role) -> Team:
    """Get the team for a role."""
    return ROLE_INFO[role].team


def get_role_name_cn(role: Role) -> str:
    """Get the Chinese name for a role."""
    return ROLE_INFO[role].name_cn


def is_evil(role: Role) -> bool:
    """Check if a role is on the evil team."""
    return ROLE_INFO[role].team == Team.EVIL


def can_see_evil(role: Role) -> bool:
    """Check if a role can see evil players."""
    return ROLE_INFO[role].knows_evil


def knows_teammates(role: Role) -> bool:
    """Check if a role knows their evil teammates."""
    return ROLE_INFO[role].knows_teammates
