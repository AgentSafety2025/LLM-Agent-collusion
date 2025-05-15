# import random
# import json
# import re
# from typing import List, Dict
# from llm_client import LLMClient

# RULE_BASE_PATH = "prompt/rule_base.txt"
# PLAY_CARD_PROMPT_TEMPLATE_PATH = "prompt/play_card_prompt_template.txt"
# CHALLENGE_PROMPT_TEMPLATE_PATH = "prompt/challenge_prompt_template.txt"
# REFLECT_PROMPT_TEMPLATE_PATH = "prompt/reflect_prompt_template.txt"

# class Player:
#     def __init__(self, name: str, model_name: str):
#         """Initialize players
        
#         Args:
#             name: player name
#             model_name: model name 
#         """
#         self.name = name
#         self.hand = []
#         self.alive = True
#         self.bullet_position = random.randint(0, 5)
#         self.current_bullet_position = 0
#         self.opinions = {}
        
#         # LLM initialization
#         self.llm_client = LLMClient()
#         self.model_name = model_name

#     def _read_file(self, filepath: str) -> str:
#         """read file content"""
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 return f.read().strip()
#         except Exception as e:
#             print(f"read file {filepath} failed: {str(e)}")
#             return ""

#     def print_status(self) -> None:
#         """print player status"""
#         print(f"{self.name} - hand: {', '.join(self.hand)} - "
#               f"bullet position: {self.bullet_position} - current bay position: {self.current_bullet_position}")
        
#     def init_opinions(self, other_players: List["Player"]) -> None:
#         """Initialize opinions for other players
        
#         Args:
#             other_players: other players list
#         """
#         self.opinions = {
#             player.name: "Doesn't know well about this player yet"
#             for player in other_players
#             if player.name != self.name
#         }

#     def choose_cards_to_play(self,
#                         round_base_info: str,
#                         round_action_info: str,
#                         play_decision_info: str) -> Dict:
#         """
#         Play cards based on the given information
        
#         Args:
#             round_base_info: basic information of the round
#             round_action_info: round action information
#             play_decision_info: play decision information
            
#         Returns:
#             tuple: (result, reasoning_content)
#             - result dictionary includes played_cards, behavior and play_reason
#             - The reasoning content is the original reasoning process of the LLM
#         """
#         # Read rules and templates
#         rules = self._read_file(RULE_BASE_PATH)
#         template = self._read_file(PLAY_CARD_PROMPT_TEMPLATE_PATH)
        
#         # Current cards
#         current_cards = ", ".join(self.hand)
        
#         # Fill in the template
#         prompt = template.format(
#             rules=rules,
#             self_name=self.name,
#             round_base_info=round_base_info,
#             round_action_info=round_action_info,
#             play_decision_info=play_decision_info,
#             current_cards=current_cards
#         )
        
#         # Try to get a valid JSON response, up to five times more
#         for attempt in range(5):
#             # Send the same prompt each time
#             messages = [
#                 {"role": "user", "content": prompt}
#             ]
            
#             try:
#                 content, reasoning_content = self.llm_client.chat(messages, model=self.model_name)
                
#                 # Parse the JSON response
#                 json_match = re.search(r'({[\s\S]*})', content)
#                 if json_match:
#                     json_str = json_match.group(1)
#                     result = json.loads(json_str)
                    
#                     # Verify that the JSON format meets the requirements
#                     if all(key in result for key in ["played_cards", "behavior", "play_reason"]):
#                         # Ensure that played_cards is a list
#                         if not isinstance(result["played_cards"], list):
#                             result["played_cards"] = [result["played_cards"]]
                        
#                         # Ensure that the played cards are valid
#                         valid_cards = all(card in self.hand for card in result["played_cards"])
#                         valid_count = 1 <= len(result["played_cards"]) <= 3
                        
#                         if valid_cards and valid_count:
#                             # remove the played cards from the hand
#                             for card in result["played_cards"]:
#                                 self.hand.remove(card)
#                             return result, reasoning_content
                                
#             except Exception as e:
#                 # Only record errors without modifying retry requests
#                 print(f"Try {attempt+1} failed: {str(e)}")
#         raise RuntimeError(f"Player {self.name} 's choose_cards_to_play method filed after many attempts.")

#     def decide_challenge(self,
#                         round_base_info: str,
#                         round_action_info: str,
#                         challenge_decision_info: str,
#                         challenging_player_performance: str,
#                         extra_hint: str) -> bool:
#         """
#         Decide whether to challenge the previous player's card
        
#         Args:

#             round_base_info: basic information of the round
#             round_action_info: round action information
#             challenge_decision_info: challenge decision information
#             challenging_player_performance: performance of the challenging player
#             extra_hint: additional hints
            
#         Returns:
#             tuple: (result, reasoning_content)
#             - result: A dictionary contains was_challenged and challenge_reason
#             - reasoning_content: LLM's original reasoning process
#         """
#         # read rules and templates
#         rules = self._read_file(RULE_BASE_PATH)
#         template = self._read_file(CHALLENGE_PROMPT_TEMPLATE_PATH)
#         self_hand = f"Your current hand is: {', '.join(self.hand)}"
        
#         # fill in the template
#         prompt = template.format(
#             rules=rules,
#             self_name=self.name,
#             round_base_info=round_base_info,
#             round_action_info=round_action_info,
#             self_hand=self_hand,
#             challenge_decision_info=challenge_decision_info,
#             challenging_player_performance=challenging_player_performance,
#             extra_hint=extra_hint
#         )
        
#         # try to get a valid JSON response, up to five times more
#         for attempt in range(5):
#             # send the same prompt each time
#             messages = [
#                 {"role": "user", "content": prompt}
#             ]
            
#             try:
#                 content, reasoning_content = self.llm_client.chat(messages, model=self.model_name)
                
#                 # parse the JSON response
#                 json_match = re.search(r'({[\s\S]*})', content)
#                 if json_match:
#                     json_str = json_match.group(1)
#                     result = json.loads(json_str)
                    
#                     # verify that the JSON format meets the requirements
#                     if all(key in result for key in ["was_challenged", "challenge_reason"]):
#                         # ensure that was_challenged is a boolean
#                         if isinstance(result["was_challenged"], bool):
#                             return result, reasoning_content
                
#             except Exception as e:
#                 # only record errors without modifying retry requests
#                 print(f"Attemp {attempt+1} failed: {str(e)}")
#         raise RuntimeError(f"Player {self.name} 's choose_cards_to_play method filed after many attempts.")

#     def reflect(self, alive_players: List[str], round_base_info: str, round_action_info: str, round_result: str) -> None:
#         """
#         Reflect on the game and update the impression of other players
        
#         Args:
#             alive_players: list of alive players
#             round_base_info: basic information of the round
#             round_action_info: round action information
#             round_result: round result
#         """
#         # read templates
#         template = self._read_file(REFLECT_PROMPT_TEMPLATE_PATH)
        
#         # read rules
#         rules = self._read_file(RULE_BASE_PATH)
        
#         # reflect on each player
#         for player_name in alive_players:
#             # skip self
#             if player_name == self.name:
#                 continue
            
#             # get the previous opinion
#             previous_opinion = self.opinions.get(player_name, "Doesn't know well about this player yet")
            
#             # fill in the template
#             prompt = template.format(
#                 rules=rules,
#                 self_name=self.name,
#                 round_base_info=round_base_info,
#                 round_action_info=round_action_info,
#                 round_result=round_result,
#                 player=player_name,
#                 previous_opinion=previous_opinion
#             )
            
#             # try to get a valid JSON response, up to five times more
#             messages = [
#                 {"role": "user", "content": prompt}
#             ]
            
#             try:
#                 content, _ = self.llm_client.chat(messages, model=self.model_name)
                
#                 # update the opinion
#                 self.opinions[player_name] = content.strip()
#                 print(f"{self.name} updated the impression to player {player_name}")
                
#             except Exception as e:
#                 print(f"Error while reflecting {player_name} : {str(e)}")

#     def process_penalty(self) -> bool:
#         """process the penalty"""
#         print(f"player {self.name} execute the shooting punishment：")
#         self.print_status()
#         if self.bullet_position == self.current_bullet_position:
#             print(f"{self.name} died！")
#             self.alive = False
#         else:
#             print(f"{self.name} survived！")
#         self.current_bullet_position = (self.current_bullet_position + 1) % 6
#         return self.alive

import random
import json
import re
from typing import List, Dict
from llm_client import LLMClient

RULE_BASE_PATH = "prompt/rule_base.txt"
PLAY_CARD_PROMPT_TEMPLATE_PATH = "prompt/play_card_prompt_template.txt"
CHALLENGE_PROMPT_TEMPLATE_PATH = "prompt/challenge_prompt_template.txt"
REFLECT_PROMPT_TEMPLATE_PATH = "prompt/reflect_prompt_template.txt"

class Player:
    def __init__(self, name: str, model_name: str):
        """Initialize players
        
        Args:
            name: player name
            model_name: model name 
        """
        self.name = name
        self.hand = []
        self.alive = True
        self.bullet_position = random.randint(0, 5)
        self.current_bullet_position = 0
        self.opinions = {}
        
        # LLM initialization
        self.llm_client = LLMClient()
        self.model_name = model_name

    def _read_file(self, filepath: str) -> str:
        """read file content"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"read file {filepath} failed: {str(e)}")
            return ""

    def print_status(self) -> None:
        """print player status"""
        print(f"{self.name} - hand: {', '.join(self.hand)} - "
              f"bullet position: {self.bullet_position} - current bay position: {self.current_bullet_position}")
        
    def init_opinions(self, other_players: List["Player"]) -> None:
        """Initialize opinions for other players
        
        Args:
            other_players: other players list
        """
        self.opinions = {
            player.name: "Doesn't know well about this player yet"
            for player in other_players
            if player.name != self.name
        }

    def choose_cards_to_play(self,
                        round_base_info: str,
                        round_action_info: str,
                        play_decision_info: str) -> Dict:
        """
        Play cards based on the given information
        
        Args:
            round_base_info: basic information of the round
            round_action_info: round action information
            play_decision_info: play decision information
            
        Returns:
            tuple: (result, reasoning_content)
            - result dictionary includes played_cards, behavior and play_reason
            - The reasoning content is the original reasoning process of the LLM
        """
        # Read rules and templates
        rules = self._read_file(RULE_BASE_PATH)
        template = self._read_file(PLAY_CARD_PROMPT_TEMPLATE_PATH)
        
        # Current cards
        current_cards = ", ".join(self.hand)
        
        # Fill in the template
        prompt = template.format(
            rules=rules,
            self_name=self.name,
            round_base_info=round_base_info,
            round_action_info=round_action_info,
            play_decision_info=play_decision_info,
            current_cards=current_cards
        )
        
        # Try to get a valid JSON response, up to five times more
        for attempt in range(5):
            # Send the same prompt each time
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            try:
                content, reasoning_content = self.llm_client.chat(messages, model=self.model_name)
                
                # Parse the JSON response
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    
                    # Verify that the JSON format meets the requirements
                    if all(key in result for key in ["played_cards", "behavior", "play_reason"]):
                        # Ensure that played_cards is a list
                        if not isinstance(result["played_cards"], list):
                            result["played_cards"] = [result["played_cards"]]
                        
                        # Ensure that the played cards are valid
                        valid_cards = all(card in self.hand for card in result["played_cards"])
                        valid_count = 1 <= len(result["played_cards"]) <= 3
                        
                        if valid_cards and valid_count:
                            # remove the played cards from the hand
                            for card in result["played_cards"]:
                                self.hand.remove(card)
                            return result, reasoning_content
                                
            except Exception as e:
                # Only record errors without modifying retry requests
                print(f"Try {attempt+1} failed: {str(e)}")
        raise RuntimeError(f"Player {self.name} 's choose_cards_to_play method filed after many attempts.")

    def decide_challenge(self,
                        round_base_info: str,
                        round_action_info: str,
                        challenge_decision_info: str,
                        challenging_player_performance: str,
                        extra_hint: str) -> bool:
        """
        Decide whether to challenge the previous player's card
        
        Args:

            round_base_info: basic information of the round
            round_action_info: round action information
            challenge_decision_info: challenge decision information
            challenging_player_performance: performance of the challenging player
            extra_hint: additional hints
            
        Returns:
            tuple: (result, reasoning_content)
            - result: A dictionary contains was_challenged and challenge_reason
            - reasoning_content: LLM's original reasoning process
        """
        # read rules and templates
        rules = self._read_file(RULE_BASE_PATH)
        template = self._read_file(CHALLENGE_PROMPT_TEMPLATE_PATH)
        self_hand = f"Your current hand is: {', '.join(self.hand)}"
        
        # fill in the template
        prompt = template.format(
            rules=rules,
            self_name=self.name,
            round_base_info=round_base_info,
            round_action_info=round_action_info,
            self_hand=self_hand,
            challenge_decision_info=challenge_decision_info,
            challenging_player_performance=challenging_player_performance,
            extra_hint=extra_hint
        )
        
        # try to get a valid JSON response, up to five times more
        for attempt in range(5):
            # send the same prompt each time
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            try:
                content, reasoning_content = self.llm_client.chat(messages, model=self.model_name)
                
                # parse the JSON response
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    
                    # verify that the JSON format meets the requirements
                    if all(key in result for key in ["was_challenged", "challenge_reason"]):
                        # ensure that was_challenged is a boolean
                        if isinstance(result["was_challenged"], bool):
                            return result, reasoning_content
                
            except Exception as e:
                # only record errors without modifying retry requests
                print(f"Attemp {attempt+1} failed: {str(e)}")
        raise RuntimeError(f"Player {self.name} 's choose_cards_to_play method filed after many attempts.")

    def reflect(self, alive_players: List[str], round_base_info: str, round_action_info: str, round_result: str) -> None:
        """
        Reflect on the game and update the impression of other players
        
        Args:
            alive_players: list of alive players
            round_base_info: basic information of the round
            round_action_info: round action information
            round_result: round result
        """
        # read templates
        template = self._read_file(REFLECT_PROMPT_TEMPLATE_PATH)
        
        # read rules
        rules = self._read_file(RULE_BASE_PATH)
        
        # reflect on each player
        for player_name in alive_players:
            # skip self
            if player_name == self.name:
                continue
            
            # get the previous opinion
            previous_opinion = self.opinions.get(player_name, "Doesn't know well about this player yet")
            
            # fill in the template
            prompt = template.format(
                rules=rules,
                self_name=self.name,
                round_base_info=round_base_info,
                round_action_info=round_action_info,
                round_result=round_result,
                player=player_name,
                previous_opinion=previous_opinion
            )
            
            # try to get a valid JSON response, up to five times more
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            try:
                content, _ = self.llm_client.chat(messages, model=self.model_name)
                
                # update the opinion
                self.opinions[player_name] = content.strip()
                print(f"{self.name} updated the impression to player {player_name}")
                
            except Exception as e:
                print(f"Error while reflecting {player_name} : {str(e)}")

    def process_penalty(self) -> bool:
        """process the penalty"""
        print(f"player {self.name} execute the shooting punishment：")
        self.print_status()
        if self.bullet_position == self.current_bullet_position:
            print(f"{self.name} died！")
            self.alive = False
        else:
            print(f"{self.name} survived！")
        self.current_bullet_position = (self.current_bullet_position + 1) % 6
        return self.alive