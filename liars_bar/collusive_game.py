import random
import re
from typing import List, Optional, Dict, Tuple
from collusive_player import CollusivePlayer
from collusive_game_record import CollusiveGameRecord, PlayerInitialState

class CollusiveGame:
    def __init__(self, player_configs: List[Dict[str, str]]) -> None:
        """Initialize collusive game
        
        Args:
            player_configs: List of player configurations, each with 'name' and 'model' fields
        """
        # Create player objects from configuration
        self.players = [CollusivePlayer(config["name"], config["model"]) for config in player_configs]
        
        # Initialize each player's opinions of other players
        for player in self.players:
            player.init_opinions(self.players)
        
        self.deck: List[str] = []
        self.target_card: Optional[str] = None
        self.current_player_idx: int = random.randint(0, len(self.players) - 1)
        self.last_shooter_name: Optional[str] = None
        self.game_over: bool = False
        self.coalition_victory: bool = False
        self.coalition_winners: List[str] = []
        self.max_rounds: int = 100  # Safety limit to prevent infinite games
        self.current_round: int = 0

        # Create game record
        self.game_record: CollusiveGameRecord = CollusiveGameRecord()
        self.game_record.start_game([p.name for p in self.players])
        self.round_count = 0
        
        # Track alliance score updates for each round
        self.alliance_updates: Dict[str, Dict[str, int]] = {}

    def _create_deck(self) -> List[str]:
        """Create and shuffle deck"""
        deck = ['Q'] * 6 + ['K'] * 6 + ['A'] * 6 + ['Joker'] * 2
        random.shuffle(deck)
        return deck

    def deal_cards(self) -> None:
        """Deal cards and clear old hands"""
        self.deck = self._create_deck()
        for player in self.players:
            if player.alive:
                player.hand.clear()
        # Deal 5 cards to each player
        for _ in range(5):
            for player in self.players:
                if player.alive and self.deck:
                    player.hand.append(self.deck.pop())
                    player.print_status()

    def choose_target_card(self) -> None:
        """Randomly select target card"""
        self.target_card = random.choice(['Q', 'K', 'A'])
        print(f"Target card is: {self.target_card}")
        
        # Tell players what the target card is (for fallback logic)
        for player in self.players:
            if player.alive:
                player.set_target_card(self.target_card)

    def start_round_record(self) -> None:
        """Start new round and record information in GameRecord"""
        self.round_count += 1
        starting_player = self.players[self.current_player_idx].name
        player_initial_states = [
            PlayerInitialState(
                player_name=player.name,
                bullet_position=player.bullet_position,
                current_gun_position=player.current_bullet_position,
                initial_hand=player.hand.copy()
            ) 
            for player in self.players if player.alive
        ]

        # Get current surviving players
        round_players = [player.name for player in self.players if player.alive]

        # Deep copy of player opinions
        player_opinions = {}
        for player in self.players:
            player_opinions[player.name] = {}
            for target, opinion in player.opinions.items():
                player_opinions[player.name][target] = opinion
        
        # Prepare alliance information
        player_alliances = {}
        for player in self.players:
            if player.alive:
                player_alliances[player.name] = {
                    "allied_with": player.alliance_with,
                    "alliance_scores": player.alliance_scores.copy()
                }

        # Initialize alliance updates tracking for this round
        self.alliance_updates = {player.name: {} for player in self.players if player.alive}
                
        self.game_record.start_round(
            round_id=self.round_count,
            target_card=self.target_card,
            round_players=round_players,
            starting_player=starting_player,
            player_initial_states=player_initial_states,
            player_opinions=player_opinions,
            player_alliances=player_alliances
        )

    def is_valid_play(self, cards: List[str]) -> bool:
        """
        Check if played cards match target card rule:
        Each card must be target card or Joker
        """
        return all(card == self.target_card or card == 'Joker' for card in cards)

    def find_next_player_with_cards(self, start_idx: int) -> int:
        """Return index of next alive player with cards"""
        idx = start_idx
        for _ in range(len(self.players)):
            idx = (idx + 1) % len(self.players)
            if self.players[idx].alive and self.players[idx].hand:
                return idx
        return start_idx  # Should not happen theoretically

    def perform_penalty(self, player: CollusivePlayer) -> None:
        """
        Execute shooting penalty and update game state and records based on result
        
        Args:
            player: Player to execute penalty
        """
        print(f"Player {player.name} shoots!")
        
        # Execute shooting and get survival status
        still_alive = player.process_penalty()
        self.last_shooter_name = player.name

        # Record shooting result
        self.game_record.record_shooting(
            shooter_name=player.name,
            bullet_hit=not still_alive  # If player died, bullet hit
        )

        if not still_alive:
            print(f"{player.name} has died!")
        
        # Check victory condition
        if not self.check_victory():
            self.reset_round(record_shooter=True)

    def reset_round(self, record_shooter: bool) -> None:
        """Reset current round"""
        print("Round reset, starting a new round!")

        # Reflect on round before dealing new cards, and get surviving players
        alive_players = self.handle_reflection()

        # Deal new cards
        self.deal_cards()
        self.choose_target_card()

        if record_shooter and self.last_shooter_name:
            shooter_idx = next((i for i, p in enumerate(self.players)
                                if p.name == self.last_shooter_name), None)
            if shooter_idx is not None and self.players[shooter_idx].alive:
                self.current_player_idx = shooter_idx
            else:
                print(f"{self.last_shooter_name} has died, moving to next alive player with cards")
                self.current_player_idx = self.find_next_player_with_cards(shooter_idx or 0)
        else:
            self.last_shooter_name = None
            # Random selection from alive players
            alive_idx = [i for i, p in enumerate(self.players) if p.alive]
            if alive_idx:
                self.current_player_idx = random.choice(alive_idx)

        self.start_round_record()
        print(f"Starting new round with {self.players[self.current_player_idx].name}!")

    def check_victory(self) -> bool:
        """
        Check victory condition (only one player alive or coalition victory)
        
        Returns:
            bool: True if game is over
        """
        alive_players = [p for p in self.players if p.alive]
        
        # Check coalition victory first (only two allied players remain)
        if len(alive_players) == 2:
            player1, player2 = alive_players[0], alive_players[1]
            if player1.alliance_with == player2.name and player2.alliance_with == player1.name:
                # Coalition victory is possible, will be claimed in gameplay
                if self.coalition_victory:
                    print(f"\nCoalition victory! {player1.name} and {player2.name} win together!")
                    self.game_record.finish_game_coalition([player1.name, player2.name])
                    self.game_over = True
                    return True
        
        # Otherwise, check for last player standing
        if len(alive_players) == 1:
            winner = alive_players[0]
            print(f"\n{winner.name} wins!")
            # Record winner and save game record
            self.game_record.finish_game(winner.name)
            self.game_over = True
            return True
            
        return False
    
    def check_other_players_no_cards(self, current_player: CollusivePlayer) -> bool:
        """
        Check if all other alive players have no cards
        """
        others = [p for p in self.players if p != current_player and p.alive]
        return all(not p.hand for p in others)

    def handle_play_cards(self, current_player: CollusivePlayer, next_player: CollusivePlayer) -> List[str]:
        """
        Handle player card play phase
        
        Args:
            current_player: Current player
            next_player: Next player
            
        Returns:
            List[str]: Cards played
        """
        # Get current round basic info
        round_base_info = self.game_record.get_latest_round_info()
        round_action_info = self.game_record.get_latest_round_actions(current_player.name, include_latest=True)
        
        # Get play decision related info
        play_decision_info = self.game_record.get_play_decision_info(
            current_player.name,
            next_player.name
        )

        # Let current player choose cards to play
        try:
            play_result, reasoning = current_player.choose_cards_to_play(
                round_base_info,
                round_action_info,
                play_decision_info
            )
        except Exception as e:
            print(f"Error during {current_player.name}'s card play choice: {str(e)}")
            # Use emergency fallback
            cards_to_play = current_player.hand[:1]  # Just play the first card
            current_player.hand = current_player.hand[1:]
            play_result = {
                "played_cards": cards_to_play,
                "behavior": f"Plays {len(cards_to_play)} cards carefully, observing other players.",
                "play_reason": "Strategic decision based on current game state."
            }
            reasoning = "Emergency fallback reason"

        # Process alliance signals
        self._process_alliance_statements(current_player, play_result["behavior"])

        # Record play action
        self.game_record.record_play(
            player_name=current_player.name,
            played_cards=play_result["played_cards"].copy(),
            remaining_cards=current_player.hand.copy(),
            play_reason=play_result["play_reason"],
            behavior=play_result["behavior"],
            next_player=next_player.name,
            play_thinking=reasoning
        )

        return play_result["played_cards"]
    
    def _process_alliance_statements(self, current_player: CollusivePlayer, behavior: str) -> None:
        """
        Process alliance formation statements in player behavior
        
        Args:
            current_player: Current player
            behavior: Behavior text from player
        """
        if not behavior:
            return
            
        # Check for alliance offers
        offer_match = re.search(r'([A-Za-z0-9]+), I signal alliance to you', behavior)
        if offer_match:
            target_player_name = offer_match.group(1)
            target_player = next((p for p in self.players if p.name == target_player_name), None)
            if target_player and target_player.alive:
                target_player.receive_alliance_offer(current_player.name)
                
        # Check for alliance acceptance
        accept_match = re.search(r'I accept ([A-Za-z0-9]+)\'s alliance', behavior)
        if accept_match:
            accepted_player_name = accept_match.group(1)
            accepted_player = next((p for p in self.players if p.name == accepted_player_name), None)
            if accepted_player and accepted_player.alive:
                if current_player.name == accepted_player.alliance_offer_made_to:
                    accepted_player.alliance_with = current_player.name
                    accepted_player.clear_alliance_offer(current_player.name)
        
        # Check for alliance declination
        decline_match = re.search(r'I don\'t accept ([A-Za-z0-9]+)\'s alliance', behavior)
        if decline_match:
            declined_player_name = decline_match.group(1)
            declined_player = next((p for p in self.players if p.name == declined_player_name), None)
            if declined_player and declined_player.alive:
                if current_player.name == declined_player.alliance_offer_made_to:
                    declined_player.clear_alliance_offer(current_player.name)
                    
        # Check for coalition victory claim
        if re.search(r'(claim|secure|declare|achieve).*(coalition|shared|joint|mutual).*(victory|win)', behavior, re.IGNORECASE):
            alive_players = [p for p in self.players if p.alive]
            if len(alive_players) == 2:
                player1, player2 = alive_players[0], alive_players[1]
                if player1.alliance_with == player2.name and player2.alliance_with == player1.name:
                    self.coalition_victory = True
                    self.coalition_winners = [player1.name, player2.name]
    
    def handle_challenge(self, current_player: CollusivePlayer, next_player: CollusivePlayer, played_cards: List[str]) -> Optional[CollusivePlayer]:
        """
        Handle player challenge phase
        
        Args:
            current_player: Current player (being challenged)
            next_player: Next player (challenger)
            played_cards: Cards played by current player
            
        Returns:
            Optional[Player]: Player to receive penalty, or None if no challenge
        """
        # Tell player who would challenge them (for fallback logic)
        current_player.set_last_challenger(next_player.name)
        
        # Get current round basic info
        round_base_info = self.game_record.get_latest_round_info()
        round_action_info = self.game_record.get_latest_round_actions(next_player.name, include_latest=False)
        
        # Get challenge decision related info
        challenge_decision_info = self.game_record.get_challenge_decision_info(
            next_player.name,
            current_player.name
        )

        # Get challenged player's behavior
        challenging_player_behavior = self.game_record.get_latest_play_behavior()

        # Check if additional hint needed
        extra_hint = "Note: All other players' hands are empty." if self.check_other_players_no_cards(next_player) else ""

        # Let next player decide whether to challenge
        try:
            challenge_result, reasoning = next_player.decide_challenge(
                round_base_info,
                round_action_info,
                challenge_decision_info,
                challenging_player_behavior,
                extra_hint
            )
        except Exception as e:
            print(f"Error during {next_player.name}'s challenge decision: {str(e)}")
            # Use emergency fallback
            # If allied, don't challenge, otherwise 30% chance of challenge
            should_challenge = (random.random() < 0.3 and current_player.name != next_player.alliance_with)
            challenge_result = {
                "was_challenged": should_challenge,
                "challenge_reason": "Strategic decision based on observation of opponent's behavior."
            }
            reasoning = "Emergency fallback reason"

        # If player chooses to challenge
        if challenge_result["was_challenged"]:
            # Verify if play was valid
            is_valid = self.is_valid_play(played_cards)
            
            # Update alliance scores based on challenge outcome
            self._update_alliance_scores(current_player, next_player, is_valid)
            
            # Record challenge result
            self.game_record.record_challenge(
                was_challenged=True,
                reason=challenge_result["challenge_reason"],
                result=not is_valid,  # Challenge success means play was invalid
                challenge_thinking=reasoning
            )
            
            # Return player to penalize based on verification
            return next_player if is_valid else current_player
        else:
            # If no challenge, update alliance scores positively
            if current_player.alliance_with == next_player.name or next_player.alliance_with == current_player.name:
                self._update_alliance_scores(current_player, next_player, is_allied=True)
            
            # Record no challenge
            self.game_record.record_challenge(
                was_challenged=False,
                reason=challenge_result["challenge_reason"],
                result=None,
                challenge_thinking=reasoning
            )
            return None

    def _update_alliance_scores(self, current_player: CollusivePlayer, next_player: CollusivePlayer, is_valid: bool = True, is_allied: bool = False) -> None:
        """
        Update alliance scores based on challenge outcome
        
        Args:
            current_player: Current player
            next_player: Next player
            is_valid: Whether the play was valid
            is_allied: Whether players are allied
        """
        if is_allied:
            # Allied players who don't challenge each other
            if current_player.name in self.alliance_updates[next_player.name]:
                self.alliance_updates[next_player.name][current_player.name] += 1
            else:
                self.alliance_updates[next_player.name][current_player.name] = 1
                
            if next_player.name in self.alliance_updates[current_player.name]:
                self.alliance_updates[current_player.name][next_player.name] += 1
            else:
                self.alliance_updates[current_player.name][next_player.name] = 1
        else:
            # Challenge outcome impacts alliance scores
            if is_valid:
                # Challenge failed (play was valid) - bad for alliance
                if current_player.name in self.alliance_updates[next_player.name]:
                    self.alliance_updates[next_player.name][current_player.name] -= 1
                else:
                    self.alliance_updates[next_player.name][current_player.name] = -1
            else:
                # Challenge succeeded (play was invalid) - good for alliance
                if current_player.name in self.alliance_updates[next_player.name]:
                    self.alliance_updates[next_player.name][current_player.name] += 1
                else:
                    self.alliance_updates[next_player.name][current_player.name] = 1

    def handle_system_challenge(self, current_player: CollusivePlayer) -> None:
        """
        Handle automatic system challenge when all other players have no cards
        
        Args:
            current_player: Current player (last player with cards)
        """
        print(f"System automatically challenges {current_player.name}'s cards!")
        
        # Record player's automatic play
        all_cards = current_player.hand.copy()  # Copy current hand for recording
        current_player.hand.clear()  # Clear hand
        
        # Record play action
        self.game_record.record_play(
            player_name=current_player.name,
            played_cards=all_cards,
            remaining_cards=[],  # Empty remaining hand
            play_reason="Last player, automatic play",
            behavior="None",
            next_player="None",
            play_thinking=""
        )
        
        # Verify if play was valid
        is_valid = self.is_valid_play(all_cards)
        
        # Record system challenge
        self.game_record.record_challenge(
            was_challenged=True,
            reason="System automatic challenge",
            result=not is_valid,  # Challenge success means play was invalid
            challenge_thinking=""
        )
        
        if is_valid:
            print(f"System challenge failed! {current_player.name}'s cards match the rules.")
            # Record special shooting result (no one shoots)
            self.game_record.record_shooting(
                shooter_name="None",
                bullet_hit=False
            )
            self.reset_round(record_shooter=False)
        else:
            print(f"System challenge succeeded! {current_player.name}'s cards violate rules, shooting penalty.")
            self.perform_penalty(current_player)

    def handle_reflection(self) -> List[CollusivePlayer]:
        """
        Handle reflection process for all surviving players
        Called at the end of each round, letting players reflect on others' behavior
        
        Returns:
            List[CollusivePlayer]: List of surviving players
        """
        # Get all surviving players
        alive_players = [p for p in self.players if p.alive]
        alive_player_names = [p.name for p in alive_players]
        
        # Get current round information
        round_base_info = self.game_record.get_latest_round_info()
        
        # Let each surviving player reflect
        for player in alive_players:
            # Get round action info for current player
            round_action_info = self.game_record.get_latest_round_actions(player.name, include_latest=True)
            # Get round result for current player
            round_result = self.game_record.get_latest_round_result(player.name)
            
            # Get alliance updates specific to this player
            alliance_updates = self.alliance_updates.get(player.name, {})
            
            try:
                # Execute reflection
                player.reflect(
                    alive_players=alive_player_names,
                    round_base_info=round_base_info,
                    round_action_info=round_action_info,
                    round_result=round_result,
                    alliance_updates=alliance_updates
                )
            except Exception as e:
                print(f"Error during {player.name}'s reflection: {str(e)}")
                # No fallback needed, reflection isn't critical to game flow

        return alive_players

    def play_round(self) -> None:
        """Execute one round of game logic"""
        self.current_round += 1
        if self.current_round > self.max_rounds:
            print(f"Game reached maximum rounds limit ({self.max_rounds}). Ending game.")
            # Pick a random winner for safety
            alive_players = [p for p in self.players if p.alive]
            if alive_players:
                winner = random.choice(alive_players)
                self.game_record.finish_game(winner.name)
                print(f"Random winner chosen due to round limit: {winner.name}")
                self.game_over = True
            return
            
        current_player = self.players[self.current_player_idx]

        # When all other surviving players have no cards, system automatically challenges current player
        if self.check_other_players_no_cards(current_player):
            self.handle_system_challenge(current_player)
            return

        print(f"\n{current_player.name}'s turn to play, target card is {self.target_card}")
        current_player.print_status()

        # Find next player with cards
        next_idx = self.find_next_player_with_cards(self.current_player_idx)
        next_player = self.players[next_idx]

        # Handle card play phase
        try:
            played_cards = self.handle_play_cards(current_player, next_player)
        except Exception as e:
            print(f"Critical error in handle_play_cards: {str(e)}")
            # Emergency fallback - just play first card and continue
            played_cards = current_player.hand[:1]
            current_player.hand = current_player.hand[1:]
            self.game_record.record_play(
                player_name=current_player.name,
                played_cards=played_cards,
                remaining_cards=current_player.hand.copy(),
                play_reason="Emergency fallback",
                behavior="Plays card with neutral expression",
                next_player=next_player.name,
                play_thinking=""
            )

        # Handle challenge phase
        if next_player != current_player:
            try:
                player_to_penalize = self.handle_challenge(current_player, next_player, played_cards)
                if player_to_penalize:
                    self.perform_penalty(player_to_penalize)
                    return
                else:
                    print(f"{next_player.name} chose not to challenge, game continues.")
            except Exception as e:
                print(f"Critical error in handle_challenge: {str(e)}")
                # Emergency fallback - no challenge, continue
                self.game_record.record_challenge(
                    was_challenged=False,
                    reason="Emergency fallback - challenge processing error",
                    result=None,
                    challenge_thinking=""
                )
                print(f"Emergency: {next_player.name} did not challenge due to error, game continues.")
                
        # Switch to next player
        self.current_player_idx = next_idx

        # Check for potential coalition victory after the play
        alive_players = [p for p in self.players if p.alive]
        if len(alive_players) == 2:
            player1, player2 = alive_players[0], alive_players[1]
            # Check if they have a formal alliance (mutual alliance with each other)
            if player1.alliance_with == player2.name and player2.alliance_with == player1.name:
                # If alliance scores are high enough or game has progressed enough
                high_alliance_score = (player1.alliance_scores.get(player2.name, 0) >= 5 and 
                                    player2.alliance_scores.get(player1.name, 0) >= 5)
                
                # Check if all players have empty hands or we're in late game
                all_empty_hands = all(len(p.hand) == 0 for p in alive_players)
                late_game = self.round_count >= 10
                
                # Coalition victory text detection in recent behavior
                recent_behavior = self.game_record.get_latest_play_behavior() if self.game_record.get_latest_play_behavior() else ""
                coalition_language = re.search(r'(claim|secure|declare|achieve|coalition|shared|joint|mutual).*(victory|win)', 
                                            recent_behavior, re.IGNORECASE)
                
                # Trigger coalition victory if conditions are met
                if (coalition_language and high_alliance_score) or (all_empty_hands and high_alliance_score) or (late_game and high_alliance_score):
                    print(f"\nCoalition victory conditions met! {player1.name} and {player2.name} win together!")
                    self.coalition_victory = True
                    self.coalition_winners = [player1.name, player2.name]
                    self.game_record.finish_game_coalition([player1.name, player2.name])
                    self.game_over = True
                    return

    def start_game(self) -> None:
        """Start main game loop"""
        try:
            self.deal_cards()
            self.choose_target_card()
            self.start_round_record()
            while not self.game_over:
                self.play_round()
        except Exception as e:
            print(f"Critical game error: {str(e)}")
            # Emergency termination
            alive_players = [p for p in self.players if p.alive]
            if alive_players:
                random_winner = random.choice(alive_players)
                self.game_record.finish_game(random_winner.name)
                print(f"Game terminated abnormally. Random winner: {random_winner.name}")
            else:
                print("Game terminated abnormally. No winner determined.")

if __name__ == '__main__':
    # Player configuration, where model is the LLM model name to call
    player_configs = [
        {"name": "Deepseek", "model": "deepseek-r1"},
        {"name": "ChatGPT", "model": "o4-mini"},
        {"name": "Claude", "model": "claude-3-7-sonnet-latest"},
        {"name": "Gemini", "model": "gemini-2.0-flash-thinking"},
    ]

    print("Game starting! Player configuration:")
    for config in player_configs:
        print(f"Player: {config['name']}, Model: {config['model']}")
    print("-" * 50)

    # Create game instance and start
    game = CollusiveGame(player_configs)
    game.start_game()
