import json
import os
from itertools import combinations
from collections import defaultdict

def format_challenge_event(history_item, round_data, player_states, game_id):
    """
    Formats a single match into readable text with more details
    Parameters.
        history_item: Dictionary containing match information.
        round_data: full data for the current round
        player_states: initial states of all players
        game_id: game identifier
    Returns: A formatted description of the matchup text.
        Formatted matchup text description
    """
    # Extract player names
    player = history_item['player_name']
    next_player = history_item['next_player']
    
    # Extract initial states of the players
    player_initial_state = None
    next_player_initial_state = None
    for state in round_data['player_initial_states']:
        if state['player_name'] == player:
            player_initial_state = state
        elif state['player_name'] == next_player:
            next_player_initial_state = state
    
    # Prepare the output
    output = []
    
    # Add game ID
    output.append(f"Game ID: {game_id}")
    
    # Add round information
    output.append(f"Current player ({player}):")
    output.append(f"Initial hand: {', '.join(player_initial_state['initial_hand'])}")
    output.append(f"played card(s): {', '.join(history_item['played_cards'])}")
    output.append(f"Remaining card(s): {', '.join(history_item['remaining_cards'])}")
    if 'play_reason' in history_item and history_item['play_reason']:
        output.append(f"Reason of playing: {history_item['play_reason']}")
    if 'behavior' in history_item and history_item['behavior']:
        output.append(f"behavior of play: {history_item['behavior']}")
    
    # Add challenge information
    output.append(f"\nChallenger ({next_player}):")
    if next_player_initial_state:
        output.append(f"Initial hand: {', '.join(next_player_initial_state['initial_hand'])}")
    
    if history_item['was_challenged']:
        output.append(f"challenged")
        if 'challenge_reason' in history_item and history_item['challenge_reason']:
            output.append(f"reasons of challenging: {history_item['challenge_reason']}")
        result_text = "successful" if history_item['challenge_result'] else "failed"
        output.append(f"Challenge result: {result_text}")
    else:
        output.append("Chose to not challenge")
        if 'challenge_reason' in history_item and history_item['challenge_reason']:
            output.append(f"Reasons of not challenging: {history_item['challenge_reason']}")
    
    output.append("")
    
    return "\n".join(output)

def extract_matchups(game_data, game_id):
    """
    Extracts all detailed matchup records between players from game data
    Parameters.
        game_data: full game data dictionary
        game_id: game identifier
    Returns.
        Dictionary containing all matchmaking records.
    """
    # Extract player names and initialize matchups
    players = game_data['player_names']
    matchups = defaultdict(list)
    
    # Generate all possible player matchups
    for round_data in game_data['rounds']:
        round_id = round_data['round_id']
        target_card = round_data['target_card']
        
        # deal with each play in the round
        for play in round_data['play_history']:
            player = play['player_name']
            next_player = play['next_player']
            
            # Only consider challenged plays
            if play['was_challenged']:
                matchup_key = '_vs_'.join(sorted([player, next_player]))
                
                # Add round information
                round_info = [
                    f"Run {round_id} of game {game_id}",
                    f"Target card is: {target_card}",
                    "=" * 40,
                    ""
                ]
                
                # Add detailed challenge information
                challenge_text = format_challenge_event(play, round_data, round_data['player_initial_states'], game_id)
                
                # Combine round and challenge information
                full_text = "\n".join(round_info) + challenge_text
                
                matchups[matchup_key].append(full_text)
                
    return matchups

def save_matchups_to_files(all_matchups, output_dir):
    """
    Combine and save all game matchup records into a single file
    Parameters.
        all_matchups: Dictionary containing all matchups for all games.
        output_dir: path to the output folder
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save each matchup to a separate file
    for matchup_key, interactions in all_matchups.items():
        if interactions:
            # Save the matchup to a file
            filename = os.path.join(output_dir, f"{matchup_key}_detailed_matchups.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"{matchup_key.replace('_vs_', ' match ')} 's detailed record\n")
                f.write("=" * 50 + "\n\n")
                f.write("\n\n".join(interactions))
                # Add a summary at the end
                f.write(f"\n\nTotal number of matches: {len(interactions)}\n")

def process_all_json_files(input_dir, output_dir):
    """
    Processes all JSON files in the specified folder and merges the duel records of the same player pairs
    Parameters.
        input_dir: path to input folder (contains JSON files)
        output_dir: path to output folder
    """
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory'{input_dir}' doesn't exist!")
        return
    
    # Check if there are any JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    if not json_files:
        print(f"couldn't find JSON file in the directory '{input_dir}'")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Initialize the dictionary to store all matchups
    all_matchups = defaultdict(list)
    
    # Process each JSON file
    for json_file in json_files:
        print(f"processing: {json_file}")
        file_path = os.path.join(input_dir, json_file)
        
        try:
            # Load the game data
            with open(file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
            
            # Extract the game ID
            game_id = os.path.splitext(json_file)[0]
            
            # Extract the matchups
            game_matchups = extract_matchups(game_data, game_id)
            
            # Merge the matchups
            for key, value in game_matchups.items():
                all_matchups[key].extend(value)
            
            print(f"processed {json_file}")
            
        except Exception as e:
            print(f"error while processing {json_file} : {str(e)}")
    
    # Save the matchups to files
    save_matchups_to_files(all_matchups, output_dir)
    print("All records saved successfully")

# Define input and output folders

# input_dir = "game_records"  # folder containing JSON files
# output_dir = "matchup_records"  # output folder
input_dir = "game_records"   
output_dir = "matchup_records"  

# Process all JSON files
process_all_json_files(input_dir, output_dir)