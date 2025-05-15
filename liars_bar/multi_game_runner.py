from game import Game
from typing import Dict, List
import argparse

class MultiGameRunner:
    def __init__(self, player_configs: List[Dict[str, str]], num_games: int = 10):
        """Initialize the Multi-Game Runner
        
        Args:
            player_configs: List of player configurations
            num_games: Number of games to run (default: 10)
        """
        self.player_configs = player_configs
        self.num_games = num_games

    def run_games(self) -> None:
        """Run spcified number of games"""
        for game_num in range(1, self.num_games + 1):
            print(f"\n=== Running game {game_num}/{self.num_games} ===")
            
            # Start the game
            game = Game(self.player_configs)
            game.start_game()
            
            print(f"Game {game_num} ended")

def parse_arguments():
    """Parsing command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run multiple games',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-n', '--num-games',
        type=int,
        default=10,
        help='Number of games to run (default: 10)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure the player information, where model is the name of the model you are calling through the API.
    player_configs = [
        {"name": "Mistral", "model": "mistral-large-latest"},
        {"name": "ChatGPT", "model": "deepseek-r1"}, # use gemini-2.0-flash for ChatGPT 4
        {"name": "Claude", "model": "deepseek-r1"},# use  for Claude 3
        {"name": "Gemini", "model": "deepseek-r1"},
    ]
    
    # Initialize and run the games
    runner = MultiGameRunner(player_configs, num_games=args.num_games)
    runner.run_games()