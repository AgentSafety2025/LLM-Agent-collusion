# LLM-MAS-Collusion Benchmark

This repository contains frameworks for running and analyzing multi-agent experiments in two different environments: Liar's Bar and CleanUp.

## Liar's Bar

Liar’s Bar is a multi-player card bluffing game played by 4 players, who take turns making claims about the cards in their hand. In each round, opponents decide whether to challenge these claims. The game rules penalize challenging a false claim, but reward challenging claims revealed to be false. Players take turns playing cards. While the cards are played face down, each player involved declaring the card's identities (e.g. Q,K or A), which may or may not involve bluffing. On each round the opponent players decide whether to challenge each declaration. Whenever a challenges is successful, the challenged player must shoot oneself with a revolver loaded with one bullet

#### Game Core

- `game.py` - The main program for the liar's bar game
- `player.py` - The LLM agents participating in the game
- `game_record.py` - Used to save and retrieve game records
- `llm_client.py` - Configures the model interface and initiates LLM requests
- `multi_game_runner.py` - Runs multiple rounds of the game in batch mode

#### Analysis Tools

- `game_analyze.py` - Collects and analyzes all match data
- `player_matchup_analyze.py` - Extracts and analyzes match records between AI opponents
- `json_convert.py` - Converts JSON game records into readable text

### Configuration

Set up the necessary dependencies using a conda environment:

```bash
pip install openai anthropic google-generativeai requests
```

Currently supported models:
- Claude 3.7 sonnet
- GPT-4o-mini
- Mistral large
- Gemini 1.5 pro

The API configuration for this project is in `llm_client.py`, remember to specify your own API keys.

### Usage

#### Running the Game

After completing the project setup, specify the correct model names in the `player_configs` section of `game.py` and `multi_game_runner.py`.

Run a single game:
```bash
python game.py
```

Run multiple games:
```bash
python multi_game_runner.py -n
```
Specify the number of game rounds after `-n`.

#### Analysis
Game records are saved in JSON format in the `game_records` folder.

To convert JSON files into text format:
```bash
python json_convert.py
```

The converted files will be stored in the `converted_game_records` folder.

To extract and analyze match records:
```bash
python player_matchup_analyze.py
```

The extracted records will be saved in the `matchup_records` folder.

To collect and print statistics of all match data:
```bash
python game_analyze.py
```

## CleanUp Environment

A framework for running and analyzing multi-agent experiments in the CleanUp environment. Agents can be powered by various LLM providers (Azure, Gemini, Claude, etc.) and can form alliances, clean pollution, and compete for apples.

### Features

- **Experiment Automation:** Run large-scale experiments with different agent providers and models
- **Alliance Dynamics:** Agents can signal, accept, or decline alliances, affecting their strategies and outcomes
- **Log Analysis:** Tools for analyzing experiment logs and generating summary statistics
- **Replay Viewer:** Visualize experiment logs as replays using Pygame

### Directory Structure

```
.
├── agent.py                  # Agent logic and LLM API integration
├── environment.py            # CleanUp environment and game logic
├── experiment_runner.py      # Main script for running experiments
├── analyze_experiment_logs.py# Tools for analyzing experiment logs
├── replay_viewer.py          # Visualize experiment logs as replays
├── visualize_agent_data.py   # Data visualization for experiments
├── requirements.txt          # Python dependencies
├── experiments/              # Output directory for experiment logs
├── assets/                   # Assets for visualization (e.g., agent avatars)
└── ... (other analysis/visualization scripts)
```

### Getting Started

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Set Up API Keys

Create a `.env` file in the root directory with your API keys for the LLM providers you want to use. Example:

```
AZURE_INFERENCE_ENDPOINT=...
AZURE_INFERENCE_TOKEN=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_TOKEN=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```

#### 3. Run Experiments

```bash
python experiment_runner.py
```

Experiment logs will be saved in the `experiments/` directory.

#### 4. Analyze Results

```bash
python analyze_experiment_logs.py
```

This will generate summary files and CSVs for further analysis.

#### 5. Visualize Replays

```bash
python replay_viewer.py experiments/<experiment_id>
```

### Customization

- **Agent Providers/Models:** Edit `experiment_runner.py` to specify which providers and models to use
- **Environment Parameters:** Change grid size, number of agents, and other parameters in the config dictionary in `experiment_runner.py`
