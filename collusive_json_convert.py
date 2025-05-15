import os
import json

def convert_collusive_game_record_to_text(json_file_path):
    """Convert collusive game records to readable text format"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        game_data = json.load(f)

    game_id = game_data["game_id"]
    player_names = game_data["player_names"]
    rounds = game_data["rounds"]
    winner = game_data.get("winner", "None")
    coalition_winners = game_data.get("coalition_winners", [])
    is_coalition_victory = game_data.get("is_coalition_victory", False)

    # Introductory information
    text = f"Game ID: {game_id}\n"
    text += f"Players: {', '.join(player_names)}\n\n"
    text += "════════════════════════════\n"
    text += "         GAME STARTS\n"
    text += "════════════════════════════\n\n"

    for round_record in rounds:
        text += "────────────────────────────\n"
        text += f"Round {round_record['round_id']}\n"
        text += "────────────────────────────\n"
        text += f"Active players: {', '.join(round_record['round_players'])}\n"
        text += f"Starting with: {round_record['starting_player']}\n\n"

        # Record player opinions
        active_players = round_record["round_players"]
        for player_name, opinions in round_record["player_opinions"].items():
            if player_name in active_players:
                text += f"{player_name}'s impressions of other players:\n"
                for other_player, opinion in opinions.items():
                    if other_player in active_players:
                        text += f"  - {other_player}: {opinion}\n"
                text += "\n"
        
        # Record alliance information
        if "player_alliances" in round_record:
            text += "Alliance Status:\n"
            for player_name, alliance_info in round_record["player_alliances"].items():
                if player_name in active_players:
                    alliance_with = alliance_info.get("allied_with", "None")
                    alliance_text = f"  Allied with: {alliance_with}" if alliance_with else "  No formal alliance"
                    text += f"{player_name}: {alliance_text}\n"
                    
                    # Add alliance scores
                    if "alliance_scores" in alliance_info:
                        text += "  Alliance scores:\n"
                        for other_player, score in alliance_info["alliance_scores"].items():
                            if other_player in active_players and other_player != player_name:
                                text += f"    - {other_player}: {score}\n"
            text += "\n"
            
        text += "Cards dealt...\n\n"
        text += f"Target card for this round: {round_record['target_card']}\n"

        # Add player initial states
        if "player_initial_states" in round_record:
            text += "Initial player states:\n"
            for player_state in round_record["player_initial_states"]:
                player_name = player_state["player_name"]
                bullet_pos = player_state["bullet_position"]
                gun_pos = player_state["current_gun_position"]
                initial_hand = ", ".join(player_state["initial_hand"])
                
                text += f"{player_name}:\n"
                text += f"  - Bullet position: {bullet_pos}\n"
                text += f"  - Current chamber position: {gun_pos}\n"
                text += f"  - Initial hand: {initial_hand}\n\n"

        text += "----------------------------------\n"
        for action in round_record["play_history"]:
            # Player turn and behavior
            text += f"{action['player_name']}'s turn\n"
            text += f"Behavior: {action['behavior']}\n"
            text += f"Played: {', '.join(action['played_cards'])} | Remaining: {', '.join(action['remaining_cards'])} (target: {round_record['target_card']})\n"
            text += f"Reasoning: {action['play_reason']}\n\n"

            # Challenge decision
            if "next_player" in action and action["next_player"] != "None":
                if action['was_challenged']:
                    text += f"{action['next_player']} chose to challenge\n"
                    text += f"Reason: {action['challenge_reason']}\n"
                else:
                    text += f"{action['next_player']} chose not to challenge\n"
                    text += f"Reason: {action['challenge_reason']}\n"

                # Challenge outcome
                if action['was_challenged']:
                    if action['challenge_result']:
                        text += f"Challenge successful - {action['player_name']}'s bluff was exposed\n"
                    else:
                        text += f"Challenge failed - {action['next_player']} was penalized\n"
            else:
                text += "System automatic challenge\n"
                
            text += "\n----------------------------------\n"

        # Record shooting results
        if round_record.get('round_result'):
            result = round_record['round_result']
            text += f"Shooting outcome:\n"

            if result.get("shooter_name") == "None":
                text += f"No shooting occurred\n"
            else:
                if result.get("bullet_hit", False):
                    text += f"Bullet hit - {result['shooter_name']} died\n"
                else:
                    text += f"Bullet missed - {result['shooter_name']} survived\n"

            text += "\n"

    # Game conclusion
    text += "\n════════════════════════════\n"
    text += "         GAME OVER\n"
    text += "════════════════════════════\n\n"
    
    # Display winner information
    if is_coalition_victory:
        text += "★ ★ ★ ★ ★ COALITION VICTORY ★ ★ ★ ★ ★\n"
        text += f"    Winners: {' and '.join(coalition_winners)}\n"
        text += "★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★\n"
    else:
        text += "★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★\n"
        text += f"    Winner: {winner}\n"
        text += "★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★\n"
    
    return text

def process_collusive_game_records(input_directory, output_directory):
    """Process all collusive game JSON files and generate readable text files"""
    os.makedirs(output_directory, exist_ok=True)
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(input_directory, filename)
            txt_file_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.txt')

            print(f"Processing {filename}...")
            
            try:
                game_text = convert_collusive_game_record_to_text(json_file_path)
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(game_text)
                print(f"Generated: {txt_file_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == '__main__':
    game_records_directory = 'game_records'
    output_directory = 'converted_collusive_records'
    process_collusive_game_records(game_records_directory, output_directory)