"""Core game engine logic."""

from typing import List, Optional, Dict, Any
from server.game.state import GameState, GamePhase, GameStatus, Player, QuestResult, VoteResult, DiscussionMessage
from server.game.rules import get_rules, requires_two_fails
from server.game.roles import Team, Role


class GameEngine:
    """Core game engine that manages game logic."""
    
    def __init__(self, state: GameState):
        self.state = state
    
    @classmethod
    def create_game(cls, player_count: int, player_configs: List[Dict[str, Any]]) -> "GameEngine":
        """Create a new game with the given configuration."""
        state = GameState(player_count=player_count)
        
        # Create players
        for config in player_configs:
            player = Player(
                seat=config["seat"],
                name=config["name"],
                is_human=config.get("is_human", False),
                model_name=config.get("model"),
                provider=config.get("provider"),
            )
            state.players.append(player)
        
        return cls(state)
    
    def start_game(self):
        """Start the game - assign roles and move to night phase."""
        self.state.status = GameStatus.IN_PROGRESS
        self.state.phase = GamePhase.ROLE_ASSIGNMENT
        
        # Assign roles randomly
        self.state.assign_roles()
        
        # Move to night phase
        self.state.phase = GamePhase.NIGHT_PHASE
    
    def proceed_to_discussion(self):
        """Move from night phase to discussion phase.
        
        In the new flow, discussion happens BEFORE team selection.
        The leader speaks first to propose a team, then others discuss.
        After discussion, the leader makes the final team selection.
        """
        self.state.phase = GamePhase.DISCUSSION
        # Start discussion from the leader
        self.state.current_discussion_seat = self.state.current_leader
        self.state.discussion_speakers_count = 0
        self.state.discussion_complete = False
        self.state.proposed_team = []  # No team proposed yet
        self._check_human_action()
    
    def proceed_to_team_selection(self):
        """Move from discussion phase to team selection.
        
        After discussion is complete, the leader makes the final team selection.
        """
        self.state.phase = GamePhase.TEAM_SELECTION
        self._check_human_action()
    
    def select_team(self, team: List[int]) -> bool:
        """Leader selects the final team for the quest.
        
        This is called AFTER discussion is complete.
        The leader can choose a different team from what they initially proposed.
        
        Returns True if team is valid.
        """
        required_size = get_rules(self.state.player_count).quest_team_sizes[self.state.current_round - 1]
        
        if len(team) != required_size:
            return False
        
        # Validate all seats exist
        valid_seats = set(p.seat for p in self.state.players)
        if not all(s in valid_seats for s in team):
            return False
        
        self.state.proposed_team = team
        # Move directly to voting phase (discussion already happened)
        self.state.phase = GamePhase.TEAM_VOTE
        self.state.current_votes = {}
        self._check_human_action()
        
        return True
    
    def add_discussion(self, seat: int, content: str):
        """Add a discussion message from a player."""
        player = self.state.get_player(seat)
        if player:
            message = DiscussionMessage(
                seat=seat,
                player_name=player.name,
                content=content,
                round=self.state.current_round,
                attempt=self.state.vote_attempt,
            )
            self.state.discussion_history.append(message)
    
    def next_discussion_speaker(self) -> Optional[int]:
        """Get the next speaker for discussion, or None if discussion is complete.
        
        Starts from the leader (who proposes a team during discussion),
        then goes around the table for all other players.
        The leader speaks first in the new flow.
        """
        # All players need to speak (player_count total)
        while self.state.discussion_speakers_count < self.state.player_count:
            seat = self.state.current_discussion_seat
            # Move to next seat (circular)
            self.state.current_discussion_seat = (self.state.current_discussion_seat + 1) % self.state.player_count
            
            self.state.discussion_speakers_count += 1
            self.state.current_speaker_seat = seat  # Track current speaker
            return seat
        
        self.state.discussion_complete = True
        self.state.current_speaker_seat = None
        return None
    
    def proceed_to_vote(self):
        """Move from discussion to voting phase."""
        self.state.phase = GamePhase.TEAM_VOTE
        self.state.current_votes = {}
        self._check_human_action()
    
    def cast_vote(self, seat: int, approve: bool):
        """Cast a vote for the proposed team."""
        self.state.current_votes[seat] = approve
    
    def all_votes_cast(self) -> bool:
        """Check if all players have voted."""
        return len(self.state.current_votes) == self.state.player_count
    
    def resolve_vote(self) -> bool:
        """Resolve the vote and return True if team was approved."""
        approvals = sum(1 for v in self.state.current_votes.values() if v)
        approved = approvals > self.state.player_count // 2
        
        # Record vote result
        vote_result = VoteResult(
            round=self.state.current_round,
            attempt=self.state.vote_attempt,
            votes=self.state.current_votes.copy(),
            approved=approved,
            proposed_team=self.state.proposed_team.copy(),
            leader=self.state.current_leader,
        )
        self.state.vote_history.append(vote_result)
        
        if approved:
            # Move to quest execution
            self.state.phase = GamePhase.QUEST_EXECUTION
            self.state.current_quest_votes = {}
            self._check_human_action()
        else:
            # Check for 5 failed votes (evil wins)
            if self.state.vote_attempt >= 5:
                self._evil_wins()
            else:
                # Next leader, try again - go back to discussion phase
                self.state.vote_attempt += 1
                self.state.next_leader()
                self.state.proposed_team = []
                # Reset discussion state for new round of discussion
                self.state.current_discussion_seat = self.state.current_leader
                self.state.discussion_speakers_count = 0
                self.state.discussion_complete = False
                self.state.phase = GamePhase.DISCUSSION
                self._check_human_action()
        
        return approved
    
    def cast_quest_vote(self, seat: int, success: bool):
        """Cast a vote for quest success/failure."""
        self.state.current_quest_votes[seat] = success
    
    def all_quest_votes_cast(self) -> bool:
        """Check if all quest members have voted."""
        return len(self.state.current_quest_votes) == len(self.state.proposed_team)
    
    def resolve_quest(self) -> bool:
        """Resolve the quest and return True if quest succeeded."""
        fail_votes = sum(1 for v in self.state.current_quest_votes.values() if not v)
        
        # Check if quest requires 2 fails
        fails_required = 2 if requires_two_fails(self.state.player_count, self.state.current_round) else 1
        success = fail_votes < fails_required
        
        # Record quest result with individual votes
        quest_result = QuestResult(
            round=self.state.current_round,
            team_size=len(self.state.proposed_team),
            success=success,
            fail_votes=fail_votes,
            team_members=self.state.proposed_team.copy(),
            quest_votes=self.state.current_quest_votes.copy(),
        )
        self.state.quest_results.append(quest_result)
        
        # Check win conditions
        if self.state.good_wins >= 3:
            # Good team completed 3 quests - evil team discusses before assassination
            self.state.phase = GamePhase.ASSASSINATION_DISCUSSION
            self.state.assassination_discussion_history = []
            self.state.assassination_discussion_speakers = []
            self.state.assassination_discussion_complete = False
            self._check_human_action()
        elif self.state.evil_wins >= 3:
            # Evil team failed 3 quests - evil wins
            self._evil_wins()
        else:
            # Continue to next round - start with discussion
            self.state.current_round += 1
            self.state.vote_attempt = 1
            self.state.next_leader()
            self.state.proposed_team = []
            # Start discussion from the new leader
            self.state.current_discussion_seat = self.state.current_leader
            self.state.discussion_speakers_count = 0
            self.state.discussion_complete = False
            self.state.phase = GamePhase.DISCUSSION
            self._check_human_action()
        
        return success
    
    def add_assassination_discussion(self, seat: int, content: str):
        """Add a discussion message from an evil player during assassination discussion."""
        from server.game.state import DiscussionMessage
        player = self.state.get_player(seat)
        if player:
            message = DiscussionMessage(
                seat=seat,
                player_name=player.name,
                content=content,
                round=self.state.current_round,
                attempt=1,
            )
            self.state.assassination_discussion_history.append(message)
            if seat not in self.state.assassination_discussion_speakers:
                self.state.assassination_discussion_speakers.append(seat)
    
    def get_evil_seats(self) -> List[int]:
        """Get all evil player seats."""
        from server.game.roles import is_evil
        return [p.seat for p in self.state.players if p.role and is_evil(p.role)]
    
    def next_assassination_discussion_speaker(self) -> Optional[int]:
        """Get the next evil player to speak during assassination discussion."""
        evil_seats = self.get_evil_seats()
        
        for seat in evil_seats:
            if seat not in self.state.assassination_discussion_speakers:
                return seat
        
        self.state.assassination_discussion_complete = True
        return None
    
    def proceed_to_assassination(self):
        """Move from assassination discussion to assassination phase."""
        self.state.phase = GamePhase.ASSASSINATION
        self._check_human_action()

    def assassinate(self, target_seat: int) -> bool:
        """Assassin attempts to kill Merlin. Returns True if assassination succeeds (evil wins)."""
        target = self.state.get_player(target_seat)
        if target is None:
            return False
        
        self.state.assassinated_player = target_seat
        
        if target.role == Role.MERLIN:
            # Assassination succeeded - evil wins
            self._evil_wins()
            return True
        else:
            # Assassination failed - good wins
            self._good_wins()
            return False
    
    def _evil_wins(self):
        """Set evil team as winner."""
        self.state.winner = Team.EVIL
        self.state.status = GameStatus.FINISHED
        self.state.phase = GamePhase.GAME_OVER
        from datetime import datetime
        self.state.finished_at = datetime.now().isoformat()
    
    def _good_wins(self):
        """Set good team as winner."""
        self.state.winner = Team.GOOD
        self.state.status = GameStatus.FINISHED
        self.state.phase = GamePhase.GAME_OVER
        from datetime import datetime
        self.state.finished_at = datetime.now().isoformat()
    
    def _check_human_action(self):
        """Check if we need to wait for a human player action."""
        self.state.waiting_for_human = False
        self.state.human_action_type = None
        
        if self.state.phase == GamePhase.TEAM_SELECTION:
            leader = self.state.get_leader()
            if leader.is_human:
                self.state.waiting_for_human = True
                self.state.human_action_type = "team_selection"
        
        elif self.state.phase == GamePhase.DISCUSSION:
            # Check if there are still speakers remaining
            if self.state.discussion_speakers_count < self.state.player_count:
                # Find the next speaker
                check_seat = self.state.current_discussion_seat
                current_player = self.state.get_player(check_seat)
                if current_player and current_player.is_human:
                    self.state.waiting_for_human = True
                    # Leader uses leader_discussion, others use discussion
                    if check_seat == self.state.current_leader:
                        self.state.human_action_type = "leader_discussion"
                    else:
                        self.state.human_action_type = "discussion"
        
        elif self.state.phase == GamePhase.TEAM_VOTE:
            for p in self.state.players:
                if p.is_human and p.seat not in self.state.current_votes:
                    self.state.waiting_for_human = True
                    self.state.human_action_type = "vote"
                    break
        
        elif self.state.phase == GamePhase.QUEST_EXECUTION:
            for seat in self.state.proposed_team:
                player = self.state.get_player(seat)
                if player and player.is_human and seat not in self.state.current_quest_votes:
                    self.state.waiting_for_human = True
                    self.state.human_action_type = "quest"
                    break
        
        elif self.state.phase == GamePhase.ASSASSINATION_DISCUSSION:
            # Check if there are still evil players who haven't spoken
            evil_seats = self.get_evil_seats()
            for seat in evil_seats:
                if seat not in self.state.assassination_discussion_speakers:
                    player = self.state.get_player(seat)
                    if player and player.is_human:
                        self.state.waiting_for_human = True
                        self.state.human_action_type = "assassination_discussion"
                    break
        
        elif self.state.phase == GamePhase.ASSASSINATION:
            assassin = next((p for p in self.state.players if p.role == Role.ASSASSIN), None)
            if assassin and assassin.is_human:
                self.state.waiting_for_human = True
                self.state.human_action_type = "assassinate"
    
    def get_assassin_seat(self) -> Optional[int]:
        """Get the seat of the assassin."""
        for p in self.state.players:
            if p.role == Role.ASSASSIN:
                return p.seat
        return None
    
    def get_quest_team_size(self) -> int:
        """Get the required team size for the current quest."""
        return get_rules(self.state.player_count).quest_team_sizes[self.state.current_round - 1]
