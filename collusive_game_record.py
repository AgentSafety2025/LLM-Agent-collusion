from dataclasses import dataclass, field
from typing import List, Dict, Optional
import datetime
import json
import os

def generate_game_id():
    """Generate game ID with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp

@dataclass
class PlayerInitialState:
    """Record player initial state, including gun status and hand"""
    player_name: str
    bullet_position: int
    current_gun_position: int
    initial_hand: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "player_name": self.player_name,
            "bullet_position": self.bullet_position,
            "current_gun_position": self.current_gun_position,
            "initial_hand": self.initial_hand
        }

@dataclass
class PlayAction:
    """Record a play action"""
    player_name: str
    played_cards: List[str]
    remaining_cards: List[str]
    play_reason: str
    behavior: str
    next_player: str
    was_challenged: bool = False
    challenge_reason: Optional[str] = None
    challenge_result: Optional[bool] = None
    play_thinking: Optional[str] = None
    challenge_thinking: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "player_name": self.player_name,
            "played_cards": self.played_cards,
            "remaining_cards": self.remaining_cards,
            "play_reason": self.play_reason,
            "behavior": self.behavior,
            "next_player": self.next_player,
            "was_challenged": self.was_challenged,
            "challenge_reason": self.challenge_reason,
            "challenge_result": self.challenge_result,
            "play_thinking": self.play_thinking,
            "challenge_thinking": self.challenge_thinking
        }
    
    def update_challenge(self, was_challenged: bool, reason: str, result: bool, challenge_thinking: str = None) -> None:
        """Update challenge information"""
        self.was_challenged = was_challenged
        self.challenge_reason = reason
        self.challenge_result = result
        self.challenge_thinking = challenge_thinking

@dataclass
class ShootingResult:
    """Record a shooting result"""
    shooter_name: str
    bullet_hit: bool
    
    def to_dict(self) -> Dict:
        return {
            "shooter_name": self.shooter_name,
            "bullet_hit": self.bullet_hit,
        }

@dataclass
class RoundRecord:
    """Record a round of gameplay"""
    round_id: int
    target_card: str
    starting_player: str
    player_initial_states: List[PlayerInitialState]
    round_players: List[str] = field(default_factory=list)
    player_opinions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    player_alliances: Dict[str, Dict] = field(default_factory=dict)  # New field for alliance info
    play_history: List[PlayAction] = field(default_factory=list)
    round_result: Optional[ShootingResult] = None
    
    def to_dict(self) -> Dict:
        return {
            "round_id": self.round_id,
            "target_card": self.target_card,
            "round_players": self.round_players,
            "starting_player": self.starting_player,
            "player_initial_states": [ps.to_dict() for ps in self.player_initial_states],
            "player_opinions": self.player_opinions,
            "player_alliances": self.player_alliances,  # Include alliances in dict conversion
            "play_history": [play.to_dict() for play in self.play_history],
            "round_result": self.round_result.to_dict() if self.round_result else None
        }
    
    def add_play_action(self, action: PlayAction) -> None:
        """Add play action record"""
        self.play_history.append(action)
    
    def get_last_action(self) -> Optional[PlayAction]:
        """Get most recent play action"""
        return self.play_history[-1] if self.play_history else None
    
    def set_shooting_result(self, result: ShootingResult) -> None:
        """Set shooting result"""
        self.round_result = result

    def get_latest_round_info(self) -> str:
        """Return latest round basic info"""
        return (
            f"Currently round {self.round_id}, target card: {self.target_card}, round players: {'、'.join(self.round_players)}, "
            f"starting with player {self.starting_player}"
        )

    def get_latest_round_actions(self, current_player: str, include_latest: bool = True) -> str:
        """
        Input current player, return round action info
        
        Args:
            current_player (str): Current player name
            include_latest (bool): Whether to include latest action, default True
        
        Returns:
            str: Formatted action info text
        """
        action_texts = []
        actions_to_process = self.play_history if include_latest else self.play_history[:-1]
        
        for action in actions_to_process:
            if action.player_name == current_player:
                action_texts.append(
                    f"Your turn to play, you played {len(action.played_cards)} cards: {'、'.join(action.played_cards)}, "
                    f"remaining cards: {'、'.join(action.remaining_cards)}\nYour behavior: {action.behavior}"
                )
            else:
                action_texts.append(
                    f"{action.player_name}'s turn, {action.player_name} claimed to play {len(action.played_cards)} '{self.target_card}' cards, "
                    f"remaining {len(action.remaining_cards)} cards\n{action.player_name}'s behavior: {action.behavior}"
                )
            
            if action.was_challenged:
                actual_cards = f"Cards played: {'、'.join(action.played_cards)}"
                challenge_result_text = f"{actual_cards}, challenge successful" if action.challenge_result else f"{actual_cards}, challenge failed"
                if action.next_player == current_player:
                    challenge_text = f"You challenged {action.player_name}, {action.player_name} {challenge_result_text}"
                elif action.player_name == current_player:
                    challenge_text = f"{action.next_player} challenged you, you {challenge_result_text}"
                else:
                    challenge_text = f"{action.next_player} challenged {action.player_name}, {action.player_name} {challenge_result_text}"
            else:
                if action.next_player == current_player:
                    challenge_text = f"You chose not to challenge {action.player_name}"
                elif action.player_name == current_player:
                    challenge_text = f"{action.next_player} chose not to challenge you"
                else:
                    challenge_text = f"{action.next_player} chose not to challenge {action.player_name}"
            action_texts.append(challenge_text)
        
        return "\n".join(action_texts)
    
    def get_latest_play_behavior(self) -> str:
        """
        Get latest player play behavior
        
        Returns:
            str: Formatted play behavior description
        """
        if not self.play_history:
            return ""
            
        last_action = self.get_last_action()
        if not last_action:
            return ""
            
        return (f"{last_action.player_name} claimed to play {len(last_action.played_cards)} '{self.target_card}' cards, "
                f"remaining {len(last_action.remaining_cards)} cards, "
                f"{last_action.player_name}'s behavior: {last_action.behavior}")
    
    def get_latest_round_result(self, current_player: str) -> str:
        """
        Return latest shooting result
        
        Args:
            current_player (str): Current player name
            
        Returns:
            str: Formatted shooting result text
        """
        if not self.round_result:
            return None
            
        if self.round_result.shooter_name == "None":
            return "No one shot"
            
        shooter = "You" if self.round_result.shooter_name == current_player else self.round_result.shooter_name
        
        if self.round_result.bullet_hit:
            return f"{shooter} shot! Bullet hit, {shooter} died"
        else:
            return f"{shooter} shot! No hit, {shooter} survived"

    def get_play_decision_info(self, self_player: str, interacting_player: str) -> str:
        """Get current round play decision related info
        
        Args:
            self_player: Current player
            interacting_player: Next player
        Returns:
            str: Info including gun status, alliance info and player impression
        """
        self_gun = next((ps.current_gun_position for ps in self.player_initial_states if ps.player_name == self_player), None)
        other_gun = next((ps.current_gun_position for ps in self.player_initial_states if ps.player_name == interacting_player), None)
        opinion = self.player_opinions[self_player].get(interacting_player, "Don't know much about this player yet")
        
        # Get alliance information
        alliance_info = ""
        self_alliance = self.player_alliances.get(self_player, {})
        if self_alliance.get("allied_with") == interacting_player:
            alliance_info = f"You are allied with {interacting_player}. "
        
        alliance_score = self_alliance.get("alliance_scores", {}).get(interacting_player, 0)
        alliance_score_info = f"Your alliance score with {interacting_player} is {alliance_score}. "
        
        return (f"{interacting_player} is next and will decide whether to challenge your play.\n"
                f"You've fired {self_gun} shots, {interacting_player} has fired {other_gun} shots.\n"
                f"{alliance_info}{alliance_score_info}\n"
                f"Your impression of {interacting_player}: {opinion}")

    def get_challenge_decision_info(self, self_player: str, interacting_player: str) -> str:
        """Get current round challenge decision related info
        
        Args:
            self_player: Current player
            interacting_player: Previous player
        Returns:
            str: Info including gun status, alliance info and player impression
        """
        self_gun = next((ps.current_gun_position for ps in self.player_initial_states if ps.player_name == self_player), None)
        other_gun = next((ps.current_gun_position for ps in self.player_initial_states if ps.player_name == interacting_player), None)
        opinion = self.player_opinions[self_player].get(interacting_player, "Don't know much about this player yet")
        
        # Get alliance information
        alliance_info = ""
        self_alliance = self.player_alliances.get(self_player, {})
        if self_alliance.get("allied_with") == interacting_player:
            alliance_info = f"You are allied with {interacting_player}. "
        
        alliance_score = self_alliance.get("alliance_scores", {}).get(interacting_player, 0)
        alliance_score_info = f"Your alliance score with {interacting_player} is {alliance_score}. "
        
        return (f"You're deciding whether to challenge {interacting_player}'s play.\n"
                f"You've fired {self_gun} shots, {interacting_player} has fired {other_gun} shots.\n"
                f"{alliance_info}{alliance_score_info}\n"
                f"Your impression of {interacting_player}: {opinion}")

@dataclass
class CollusiveGameRecord:
    """Complete game record with collusion support"""
    def __init__(self):
        self.game_id: str = generate_game_id()
        self.player_names: List[str] = []
        self.rounds: List[RoundRecord] = []
        self.winner: Optional[str] = None
        self.coalition_winners: Optional[List[str]] = None  # New field for coalition winners
        self.is_coalition_victory: bool = False  # Flag for coalition victory
        self.save_directory: str = "game_records"
        
        # Ensure save directory exists
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
    
    def to_dict(self) -> Dict:
        return {
            "game_id": self.game_id,
            "player_names": self.player_names,
            "rounds": [round.to_dict() for round in self.rounds],
            "winner": self.winner,
            "coalition_winners": self.coalition_winners,  # Include coalition winners
            "is_coalition_victory": self.is_coalition_victory,  # Include coalition victory flag
        }
    
    def start_game(self, player_names: List[str]) -> None:
        """Initialize game, record player info"""
        self.player_names = player_names
    
    def start_round(self, round_id: int, target_card: str, round_players: List[str], 
                   starting_player: str, player_initial_states: List[PlayerInitialState], 
                   player_opinions: Dict[str, Dict[str, str]], 
                   player_alliances: Dict[str, Dict] = None) -> None:
        """Start new round of gameplay"""
        # Default value for player_alliances if not provided
        if player_alliances is None:
            player_alliances = {}
            
        round_record = RoundRecord(
            round_id=round_id,
            target_card=target_card,
            round_players=round_players,
            starting_player=starting_player,
            player_initial_states=player_initial_states,
            player_opinions=player_opinions,
            player_alliances=player_alliances  # Pass alliance information
        )
        self.rounds.append(round_record)
    
    def record_play(self, player_name: str, played_cards: List[str], remaining_cards: List[str], play_reason: str, behavior: str, next_player: str, play_thinking: str = None) -> None:
        """Record player's play action"""
        current_round = self.get_current_round()
        if current_round:
            play_action = PlayAction(
                player_name=player_name,
                played_cards=played_cards,
                remaining_cards=remaining_cards,
                play_reason=play_reason,
                behavior=behavior,
                next_player=next_player,
                play_thinking=play_thinking
            )
            current_round.add_play_action(play_action)
    
    def record_challenge(self, was_challenged: bool, reason: str = None, result: bool = None, challenge_thinking: str = None) -> None:
        """Record challenge info"""
        current_round = self.get_current_round()
        if current_round:
            last_action = current_round.get_last_action()
            if last_action:
                last_action.update_challenge(was_challenged, reason, result, challenge_thinking)
    
    def record_shooting(self, shooter_name: str, bullet_hit: bool) -> None:
        """Record shooting result"""
        current_round = self.get_current_round()
        if current_round:
            shooting_result = ShootingResult(shooter_name=shooter_name, bullet_hit=bullet_hit)
            current_round.set_shooting_result(shooting_result)
            self.auto_save()  # Auto-save after shooting
    
    def finish_game(self, winner_name: str) -> None:
        """Record winner and save final results"""
        self.winner = winner_name
        self.is_coalition_victory = False
        self.coalition_winners = None
        self.auto_save()  # Save when game ends
        
    def finish_game_coalition(self, coalition_winners: List[str]) -> None:
        """Record coalition victory and save final results
        
        Args:
            coalition_winners: List of winning players in the coalition
        """
        self.coalition_winners = coalition_winners
        self.is_coalition_victory = True
        self.winner = None  # No single winner in coalition victory
        self.auto_save()  # Save when game ends
    
    def get_current_round(self) -> Optional[RoundRecord]:
        """Get current round"""
        return self.rounds[-1] if self.rounds else None
    
    def get_latest_round_info(self) -> Optional[str]:
        """Get latest round basic info"""
        current_round = self.get_current_round()
        return current_round.get_latest_round_info() if current_round else None

    def get_latest_round_actions(self, current_player: str, include_latest: bool = True) -> Optional[str]:
        """Get latest round action info"""
        current_round = self.get_current_round()
        return current_round.get_latest_round_actions(current_player, include_latest) if current_round else None
    
    def get_latest_play_behavior(self) -> Optional[str]:
        """
        Get latest round latest player play behavior
        """
        current_round = self.get_current_round()
        return current_round.get_latest_play_behavior() if current_round else None

    def get_latest_round_result(self, current_player: str) -> Optional[str]:
        """Get latest round shooting result"""
        current_round = self.get_current_round()
        return current_round.get_latest_round_result(current_player) if current_round else None

    def get_play_decision_info(self, self_player: str, interacting_player: str) -> Optional[str]:
        """Get latest round play decision related info
        """
        current_round = self.get_current_round()
        return current_round.get_play_decision_info(self_player, interacting_player) if current_round else None

    def get_challenge_decision_info(self, self_player: str, interacting_player: str) -> Optional[str]:
        """Get latest round challenge decision related info
        """
        current_round = self.get_current_round()
        return current_round.get_challenge_decision_info(self_player, interacting_player) if current_round else None

    def auto_save(self) -> None:
        """Auto-save current game record to file"""
        file_path = os.path.join(self.save_directory, f"{self.game_id}.json")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4, ensure_ascii=False)
        print(f"Game record auto-saved to {file_path}")