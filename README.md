# LLM-MAS-Collusion Benchmark
## Liar's Bar 
## CleanUp

This repository contains a framework for running and analyzing multi-agent experiments in the CleanUp environment. Agents can be powered by various LLM providers (Azure, Gemini, Claude, etc.) and can form alliances, clean pollution, and compete for apples.

### Features

- **Experiment Automation:** Run large-scale experiments with different agent providers and models.
- **Alliance Dynamics:** Agents can signal, accept, or decline alliances, affecting their strategies and outcomes.
- **Log Analysis:** Tools for analyzing experiment logs and generating summary statistics.
- **Replay Viewer:** Visualize experiment logs as replays using Pygame.

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

- **Agent Providers/Models:** Edit `experiment_runner.py` to specify which providers and models to use.
- **Environment Parameters:** Change grid size, number of agents, and other parameters in the config dictionary in `experiment_runner.py`.

### License
