import random
import json
import re
from typing import List, Dict, Optional, Tuple

# Updated paths to use collusion_prompt folder
RULE_BASE_PATH = "collusion_prompt/simpler_collusive_rule.txt"
PLAY_CARD_PROMPT_TEMPLATE_PATH = "collusion_prompt/play_card_prompt_template.txt"
CHALLENGE_PROMPT_TEMPLATE_PATH = "collusion_prompt/challenge_prompt_template.txt"
REFLECT_PROMPT_TEMPLATE_PATH = "collusion_prompt/reflect_prompt_template.txt"

class CollusivePlayer:
    def __init__(self, name: str, model_name: str):
        """Initialize player
        
        Args:
            name: Player name
            model_name: LLM model name to use
        """
        try:
            # Dynamically import to avoid hard dependency
            from llm_client import LLMClient
            self.llm_client = LLMClient()
            self.has_llm_client = True
        except ImportError:
            print(f"Warning: LLMClient not available for {name}, will use random fallback")
            self.has_llm_client = False
            self.llm_client = None
            
        self.name = name
        self.hand = []
        self.alive = True
        self.bullet_position = random.randint(0, 5)
        self.current_bullet_position = 0
        self.opinions = {}
        self.alliance_scores = {}
        self.alliance_with = None
        self.pending_alliance_offers = []
        self.alliance_offer_made_to = None
        self.model_name = model_name
        
        # Track failed attempts for debugging
        self.failed_attempts = {'play': 0, 'challenge': 0, 'reflect': 0}

    def _read_file(self, filepath: str) -> str:
        """Read file content"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Failed to read file {filepath}: {str(e)}")
            return ""

    def print_status(self) -> None:
        """Print player status"""
        alliance_status = f" - Allied with: {self.alliance_with}" if self.alliance_with else ""
        print(f"{self.name} - Hand: {', '.join(self.hand)} - "
              f"Bullet position: {self.bullet_position} - Current chamber position: {self.current_bullet_position}{alliance_status}")
        
    def init_opinions(self, other_players: List["CollusivePlayer"]) -> None:
        """Initialize opinions of other players
        
        Args:
            other_players: List of other players
        """
        self.opinions = {
            player.name: "Don't know much about this player yet"
            for player in other_players
            if player.name != self.name
        }
        
        # Initialize alliance scores for all other players
        self.alliance_scores = {
            player.name: 0
            for player in other_players
            if player.name != self.name
        }

    def choose_cards_to_play(self,
                        round_base_info: str,
                        round_action_info: str,
                        play_decision_info: str) -> Tuple[Dict, str]:
        """
        Player chooses cards to play
        
        Args:
            round_base_info: Round basic info
            round_action_info: Round action info
            play_decision_info: Play decision info
            
        Returns:
            tuple: (result_dict, reasoning_content)
            - result_dict contains played_cards, behavior and play_reason
            - reasoning_content is LLM's raw reasoning process
        """
        # Read rules and template
        rules = self._read_file(RULE_BASE_PATH)
        template = self._read_file(PLAY_CARD_PROMPT_TEMPLATE_PATH)
        
        # Prepare current hand info
        current_cards = ", ".join(self.hand)
        
        # Add alliance information to the prompt
        alliance_info = ""
        if self.alliance_with:
            alliance_info = f"You are currently allied with {self.alliance_with}. You should avoid challenging them."
        elif self.alliance_offer_made_to:
            alliance_info = f"You have offered an alliance to {self.alliance_offer_made_to}, waiting for their response."
        elif self.pending_alliance_offers:
            alliance_offers = ", ".join(self.pending_alliance_offers)
            alliance_info = f"You have pending alliance offers from: {alliance_offers}. You can accept or decline."
        
        alliance_scores_info = "Alliance scores with other players:\n"
        for player, score in self.alliance_scores.items():
            alliance_scores_info += f"- {player}: {score}\n"
        
        # Fill template
        prompt = template.format(
            rules=rules,
            self_name=self.name,
            round_base_info=round_base_info,
            round_action_info=round_action_info,
            play_decision_info=play_decision_info,
            current_cards=current_cards,
            alliance_info=alliance_info,
            alliance_scores_info=alliance_scores_info
        )
        
        # Try to get valid JSON response, retry up to five times
        for attempt in range(5):
            try:
                # Use LLM if available, otherwise generate random response
                if self.has_llm_client:
                    messages = [{"role": "user", "content": prompt}]
                    content, reasoning_content = self.llm_client.chat(messages, model=self.model_name)
                else:
                    # Fallback to random behavior if LLM client not available
                    return self._get_fallback_play_response(), "Fallback random reasoning"
                
                # Try to extract JSON part
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        result = json.loads(json_str)
                        
                        # Validate JSON format
                        if all(key in result for key in ["played_cards", "behavior", "play_reason"]):
                            # Ensure played_cards is a list
                            if not isinstance(result["played_cards"], list):
                                result["played_cards"] = [result["played_cards"]]
                            
                            # Ensure cards are valid (chosen from hand, 1-3 cards)
                            valid_cards = all(card in self.hand for card in result["played_cards"])
                            valid_count = 1 <= len(result["played_cards"]) <= 3
                            
                            if valid_cards and valid_count:
                                # Remove played cards from hand
                                for card in result["played_cards"]:
                                    self.hand.remove(card)
                                
                                # Process alliance signals in behavior field
                                self._process_alliance_signals(result["behavior"])
                                
                                return result, reasoning_content
                            else:
                                print(f"Warning: Invalid cards selection from {self.name}, trying fallback")
                                # Cards not valid, returning a fallback that's guaranteed to work
                                return self._get_fallback_play_response(), "Fallback after invalid card selection"
                    except json.JSONDecodeError:
                        print(f"JSON decoding error for content: {content[:100]}...")
                        # Continue to try again or eventually use fallback
                            
            except Exception as e:
                print(f"Attempt {attempt+1} for {self.name} failed: {str(e)}")
        
        # If all attempts fail, use random fallback
        self.failed_attempts['play'] += 1
        print(f"{self.name} failed to choose cards after 5 attempts, using fallback (total failures: {self.failed_attempts['play']})")
        return self._get_fallback_play_response(), "Fallback random reasoning after failed attempts"
    
    def _get_fallback_play_response(self) -> Dict:
        """Generate a fallback response for card play"""
        # Randomly select 1 to min(3, available cards) cards
        num_cards = min(3, len(self.hand))
        num_to_play = random.randint(1, num_cards) if num_cards > 0 else 1
        
        # If we have the target card, use it; otherwise random
        cards_to_play = []
        target_cards = [card for card in self.hand if card == self.target_card]
        joker_cards = [card for card in self.hand if card == "Joker"]
        
        # Try to use actual target cards first, then jokers, then random
        while len(cards_to_play) < num_to_play and self.hand:
            if target_cards:
                card = target_cards.pop()
            elif joker_cards:
                card = joker_cards.pop()
            else:
                # No target cards or jokers, pick random
                card = random.choice(self.hand)
                # If card already selected, find another one
                while card in cards_to_play and len(cards_to_play) < len(self.hand):
                    card = random.choice(self.hand)
                    
            cards_to_play.append(card)
            self.hand.remove(card)
            
        # Generate fallback result
        result = {
            "played_cards": cards_to_play,
            "behavior": f"Playing {len(cards_to_play)} cards carefully. Looking at other players to gauge reactions.",
            "play_reason": f"Strategic decision to play {len(cards_to_play)} cards based on my current hand and the game state."
        }
        
        return result

    def _process_alliance_signals(self, behavior: str) -> None:
        """Process alliance signals in the behavior text
        
        Args:
            behavior: Behavior text from the player
        """
        if not behavior:
            return
            
        # Check for alliance offers
        offer_match = re.search(r'([A-Za-z0-9]+), I signal alliance to you', behavior)
        if offer_match:
            target_player = offer_match.group(1)
            self.alliance_offer_made_to = target_player
            print(f"{self.name} signaled alliance to {target_player}")
            
        # Check for alliance acceptance
        accept_match = re.search(r'I accept ([A-Za-z0-9]+)\'s alliance', behavior)
        if accept_match and self.pending_alliance_offers:
            accepted_player = accept_match.group(1)
            if accepted_player in self.pending_alliance_offers:
                self.alliance_with = accepted_player
                self.pending_alliance_offers = []
                self.alliance_offer_made_to = None
                print(f"{self.name} accepted alliance with {accepted_player}")
                
        # Check for alliance declination
        decline_match = re.search(r'I don\'t accept ([A-Za-z0-9]+)\'s alliance', behavior)
        if decline_match and self.pending_alliance_offers:
            declined_player = decline_match.group(1)
            if declined_player in self.pending_alliance_offers:
                self.pending_alliance_offers.remove(declined_player)
                print(f"{self.name} declined alliance with {declined_player}")

    def decide_challenge(self,
                        round_base_info: str,
                        round_action_info: str,
                        challenge_decision_info: str,
                        challenging_player_performance: str,
                        extra_hint: str) -> Tuple[Dict, str]:
        """
        Player decides whether to challenge the previous player's play
        
        Args:
            round_base_info: Round basic info
            round_action_info: Round action info
            challenge_decision_info: Challenge decision info
            challenging_player_performance: Challenged player's performance description
            extra_hint: Extra hint
            
        Returns:
            tuple: (result, reasoning_content)
            - result: Dictionary with was_challenged and challenge_reason
            - reasoning_content: LLM's raw reasoning process
        """
        # Read rules and template
        rules = self._read_file(RULE_BASE_PATH)
        template = self._read_file(CHALLENGE_PROMPT_TEMPLATE_PATH)
        self_hand = f"Your current hand is: {', '.join(self.hand)}"
        
        # Add alliance information to the prompt
        alliance_info = ""
        if self.alliance_with:
            alliance_info = f"You are currently allied with {self.alliance_with}. You should avoid challenging them."
        
        alliance_scores_info = "Alliance scores with other players:\n"
        for player, score in self.alliance_scores.items():
            alliance_scores_info += f"- {player}: {score}\n"
        
        # Fill template
        prompt = template.format(
            rules=rules,
            self_name=self.name,
            round_base_info=round_base_info,
            round_action_info=round_action_info,
            self_hand=self_hand,
            challenge_decision_info=challenge_decision_info,
            challenging_player_performance=challenging_player_performance,
            extra_hint=extra_hint,
            alliance_info=alliance_info,
            alliance_scores_info=alliance_scores_info
        )
        
        # Try to get valid JSON response, retry up to five times
        for attempt in range(5):
            try:
                # Use LLM if available, otherwise generate random response
                if self.has_llm_client:
                    messages = [{"role": "user", "content": prompt}]
                    content, reasoning_content = self.llm_client.chat(messages, model=self.model_name)
                else:
                    # Fallback to random behavior if LLM client not available
                    return self._get_fallback_challenge_response(), "Fallback random reasoning"
                
                # Parse JSON response
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        result = json.loads(json_str)
                        
                        # Validate JSON format
                        if all(key in result for key in ["was_challenged", "challenge_reason"]):
                            # Ensure was_challenged is boolean
                            if isinstance(result["was_challenged"], bool):
                                return result, reasoning_content
                            else:
                                # Fix non-boolean values
                                result["was_challenged"] = str(result["was_challenged"]).lower() in ["true", "yes", "1"]
                                return result, reasoning_content
                    except json.JSONDecodeError:
                        print(f"JSON decoding error for content: {content[:100]}...")
                        # Continue to try again or eventually use fallback
            except Exception as e:
                print(f"Attempt {attempt+1} for {self.name} challenge decision failed: {str(e)}")
        
        # If all attempts fail, use random fallback
        self.failed_attempts['challenge'] += 1
        print(f"{self.name} failed to decide challenge after 5 attempts, using fallback (total failures: {self.failed_attempts['challenge']})")
        return self._get_fallback_challenge_response(), "Fallback random reasoning after failed attempts"
    
    def _get_fallback_challenge_response(self) -> Dict:
        """Generate a fallback response for challenge decision"""
        # If allied, don't challenge
        if self.alliance_with and self.alliance_with == self.last_challenger:
            challenge = False
            reason = f"Choosing not to challenge {self.alliance_with} as we have an alliance."
        else:
            # Random challenge with 30% probability
            challenge = random.random() < 0.3
            if challenge:
                reason = "The player's behavior seems suspicious based on my cards and the game state."
            else:
                reason = "Choosing not to challenge as I don't have strong evidence of deception."
        
        return {
            "was_challenged": challenge,
            "challenge_reason": reason
        }

    def reflect(self, alive_players: List[str], round_base_info: str, round_action_info: str, 
               round_result: str, alliance_updates: Dict[str, int]) -> None:
        """
        Player reflects on other surviving players after a round, updating impressions
        
        Args:
            alive_players: List of surviving player names
            round_base_info: Round basic info
            round_action_info: Round action info
            round_result: Round result
            alliance_updates: Dictionary of alliance score updates
        """
        # Read reflection template
        template = self._read_file(REFLECT_PROMPT_TEMPLATE_PATH)
        
        # Read rules
        rules = self._read_file(RULE_BASE_PATH)
        
        # Update alliance scores
        for player, score_change in alliance_updates.items():
            if player in self.alliance_scores:
                self.alliance_scores[player] += score_change
        
        # Reflect on each surviving player (excluding self)
        for player_name in alive_players:
            # Skip self-reflection
            if player_name == self.name:
                continue
            
            # Get previous opinion of this player
            previous_opinion = self.opinions.get(player_name, "Don't know much about this player yet")
            
            # Current alliance score with this player
            alliance_score = self.alliance_scores.get(player_name, 0)
            
            # Add alliance status information
            alliance_status = ""
            if player_name == self.alliance_with:
                alliance_status = f"You are currently allied with {player_name}."
            elif player_name == self.alliance_offer_made_to:
                alliance_status = f"You have offered an alliance to {player_name}, awaiting response."
            elif player_name in self.pending_alliance_offers:
                alliance_status = f"{player_name} has offered an alliance to you."
            
            # Fill template
            prompt = template.format(
                rules=rules,
                self_name=self.name,
                round_base_info=round_base_info,
                round_action_info=round_action_info,
                round_result=round_result,
                player=player_name,
                previous_opinion=previous_opinion,
                alliance_score=alliance_score,
                alliance_status=alliance_status
            )
            
            try:
                # Use LLM if available, otherwise generate random response
                if self.has_llm_client:
                    messages = [{"role": "user", "content": prompt}]
                    content, _ = self.llm_client.chat(messages, model=self.model_name)
                    
                    # Update opinion of this player if we got valid response
                    if content and len(content.strip()) > 10:
                        self.opinions[player_name] = content.strip()
                    else:
                        # Fallback for empty/invalid response
                        self._get_fallback_reflection(player_name)
                else:
                    # Fallback to generic reflection
                    self._get_fallback_reflection(player_name)
                
                print(f"{self.name} updated their impression of {player_name}")
                
            except Exception as e:
                self.failed_attempts['reflect'] += 1
                print(f"Error reflecting on player {player_name}: {str(e)}")
                self._get_fallback_reflection(player_name)
    
    def _get_fallback_reflection(self, player_name: str) -> None:
        """Generate a fallback reflection about another player"""
        # Check if we already have an opinion
        prev_opinion = self.opinions.get(player_name, "Don't know much about this player yet")
        
        # Get alliance score
        alliance_score = self.alliance_scores.get(player_name, 0)
        
        # Generate slightly more positive or negative opinion based on alliance score
        if alliance_score > 0:
            new_opinion = f"{player_name} seems to be acting in good faith and might be a potential ally. Their alliance score of {alliance_score} suggests trustworthiness."
        elif alliance_score < 0:
            new_opinion = f"{player_name} has been making some questionable plays that increase my suspicion. Their alliance score of {alliance_score} suggests caution is warranted."
        else:
            new_opinion = f"Still evaluating {player_name}'s gameplay style and trustworthiness. Need more information before making strategic decisions about potential alliance."
        
        # Update opinion
        self.opinions[player_name] = new_opinion

    def process_penalty(self) -> bool:
        """Process penalty"""
        print(f"Player {self.name} executes shooting penalty:")
        self.print_status()
        if self.bullet_position == self.current_bullet_position:
            print(f"{self.name} got shot and died!")
            self.alive = False
        else:
            print(f"{self.name} survived!")
        self.current_bullet_position = (self.current_bullet_position + 1) % 6
        return self.alive
    
    def receive_alliance_offer(self, from_player: str) -> None:
        """Receive alliance offer from another player
        
        Args:
            from_player: Name of the player offering alliance
        """
        if from_player not in self.pending_alliance_offers:
            self.pending_alliance_offers.append(from_player)
            print(f"{self.name} received alliance offer from {from_player}")
    
    def clear_alliance_offer(self, to_player: str) -> None:
        """Clear alliance offer to another player
        
        Args:
            to_player: Name of the player the offer was made to
        """
        if self.alliance_offer_made_to == to_player:
            self.alliance_offer_made_to = None
            
    def check_coalition_victory(self, other_player: str) -> bool:
        """Check if coalition victory condition is met
        
        Args:
            other_player: The other remaining player
            
        Returns:
            bool: True if coalition victory is possible
        """
        return self.alliance_with == other_player
        
    def set_target_card(self, target_card: str) -> None:
        """Set the current target card (used by fallback logic)
        
        Args:
            target_card: Current target card
        """
        self.target_card = target_card
        
    def set_last_challenger(self, challenger: str) -> None:
        """Set the last player who would challenge this player
        
        Args:
            challenger: Player name who would challenge
        """
        self.last_challenger = challenger