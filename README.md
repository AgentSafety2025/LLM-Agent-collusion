# LLM Agent Voluntary Collusion Framework

A comprehensive research framework for investigating voluntary collusion behavior in Large Language Model-based multi-agent systems through strategic multi-agent games.

## Overview

This project systematically explores how LLM agents discover and pursue collusive strategies that can harm other agents and human users. The framework implements two distinct strategic environments with secret collusion mechanisms to reveal conditions under which agents willingly engage in harmful collaboration despite recognizing its unfairness.

## Research Games

### Liar's Bar: Secret Collusion in Competitive Bluffing

A multi-agent card game environment studying strategic communication and collusion between LLMs in competitive scenarios.

**Game Mechanics:**
- 4-player bluffing card game with hidden information
- Players make claims about cards (truthful or deceptive)
- Opponents decide whether to challenge claims
- Strategic penalties and rewards based on challenge outcomes

**Experimental Conditions:**
1. **Baseline**: Standard game with no communication
2. **Secret Communication**: Private channels between alliance partners
3. **Secret Hint System**: Strategic guidance without open messaging

**Key Features:**
- Secret communication channels unknown to other players
- Structured coaching system providing optimal play guidance
- Comprehensive behavioral analysis and evolution tracking

### Cleanup: Resource Management with Secret Tools

A sequential social dilemma where agents balance individual rewards with collective good, adapted for studying cooperation dynamics with secret advantages.

**Game Mechanics:**
- grid world with river, orchard, and pollution zones
- Actions: Movement, apple collection, pollution cleaning, agent zapping
- Apple spawning affected by global pollution levels
- Individual rewards vs. collective environmental management

**Experimental Conditions:**
1. **Baseline**: Standard gameplay without secret tools
2. **Secret Communication Channel**: Private coordination between selected agents
3. **Secret Hint System**: Strategic advantage guidance for chosen participants

## Installation

### Prerequisites
```bash
git clone https://github.com/your-org/LLM-Agent-collusion.git
cd LLM-Agent-collusion
pip install -r requirements.txt
```

### Models we tested ((All models are claimed as safety-aligned))
- LLaMA-3-8b-Instruct
- LlaMa-3.1-8b-Instruct
- Mistral-7b-Instruct-v0.3
- Qwen2.5-7B-Instruct

**Hardware Requirements:** Originally designed for H100 GPUs but adaptable to various configurations.

## Quick Start

### Liar's Bar Experiments

```bash
cd Collusion_Liars_Bar

# Baseline game
python src/game/game.py

# Single game with secret communication
python src/game/game_secret_channel.py --trigger-round 3 --chosen-player random --allow-refusal

# Game series with persistent secret alliance
python src/game/game_series_secret_channel.py --games 50 --trigger-game 20 --trigger-round 3 --chosen-player Mike --enable-memory

# Strategic hint system
python src/game/game_series_secret_hint.py --games 50 --trigger-game 20 --trigger-round 3 --chosen-player Mike --enable-memory
```

### Cleanup Experiments

```bash
cd Collusion_Cleanup/cleanup_llm

# Baseline tournament
python -m cleanup_llm.games.tournament --games 20 --steps 30 --out runs/base_20 --roster small

# Secret communication experiment
python -m cleanup_llm.games.tournament_secret_channel --games 20 --steps 30 --out runs/secret_comm_20 --trigger-game 5 --chosen-agent 2

# Secret hint experiment
python -m cleanup_llm.games.tournament_secret_hint --games 20 --steps 30 --out runs/secret_hint_20 --trigger-game 10 --chosen-agent 2
```

## Analysis and Visualization

### Liar's Bar Analysis
```bash
cd Collusion_Liars_Bar

# Interactive condition comparison
python analysis/visualization/create_barplot_comparison.py

# Behavioral evolution analysis
python analysis/behavior/improved_behavioral_evolution.py

# Final score progression
python analysis/visualization/create_final_score_evolution.py

# Testing secret tool acceptance rates
python src/test/test_secret_tool_acceptance.py -n 500 -b 5
```

### Cleanup Analysis
```bash
cd Collusion_Cleanup/cleanup_llm

# Behavioral evolution over time
python -m cleanup_llm.analysis.behavioral_evolution

# Score evolution analysis
python -m cleanup_llm.analysis.create_final_score_evolution
```

## Key Configuration Parameters

### Liar's Bar Secret Tools
- `--trigger-game`: Which game in series to offer secret tool
- `--trigger-round`: Which round within game to make offer
- `--chosen-player`: Target player ("random" or specific name)
- `--allow-refusal`: Whether players can decline alliance offer
- `--enable-memory`: Enable cross-game alliance memory

### Cleanup Secret Tools
- `--trigger-game`: Game number to introduce secret tool
- `--chosen-agent`: Target agent ID (0-3)
- `--games`: Total number of games in tournament
- `--steps`: Episode length per game

## Data and Reproducibility

### Liar's Bar Records
- `experiments/game_records/Lily-Luke-Mike-Quinn/` - Baseline records
- `experiments/game_records/Lily-Luke-Mike-Quinn_secret_comm/` - Secret communication
- `experiments/game_records/Lily-Luke-Mike-Quinn_secret_hint/` - Strategic hints

### Cleanup Records
- `runs/` directory contains experiment results in CSV and JSON formats
- `episodes.csv`: Per-episode results for all agents
- `cumulative_scores.csv`: Running totals across games
- `summary.csv`: Aggregate metrics and behavioral rates

## Project Structure

```
LLM-Agent-collusion/
├── Collusion_Liars_Bar/           # Liar's Bar implementation
│   ├── src/game/                  # Core game logic
│   ├── src/agents/                # Player AI and decision-making
│   ├── src/llm/                   # LLM client interfaces
│   ├── analysis/                  # Statistical analysis tools
│   ├── prompts/                   # Game rules and templates
│   └── experiments/               # Complete game records
├── Collusion_Cleanup/             # Cleanup implementation
│   └── cleanup_llm/
│       ├── cleanup/               # Core environment
│       ├── agents/                # Agent implementations
│       ├── games/                 # Tournament runners
│       ├── analysis/              # Data analysis tools
│       └── runs/                  # Experiment results
└── requirements.txt               # Python dependencies
```

## Research Applications

This framework supports research in:
- Multi-agent coordination and strategic communication
- LLM behavioral analysis in competitive environments
- Game theory and mechanism design
- AI ethics and alignment research
- Voluntary adoption of harmful strategies

## Citation

If you use this framework in your research, please cite our work examining voluntary collusion adoption in LLM-based multi-agent systems.

## License

This project is licensed under the MIT License - see the LICENSE file for details.