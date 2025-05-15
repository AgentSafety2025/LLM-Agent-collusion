import json
import time
import os
from datetime import datetime
from environment import CleanUpEnv
from agent import ALLOWED_ACTIONS, agent_process
import multiprocessing as mp
import multiprocessing.connection as mpc
import re

class ExperimentRunner:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"experiments/{self.experiment_id}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Save experiment configuration
        with open(f"{self.experiment_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2)

    def run_trial(self, trial_number, providers, models=None):
        """Run a single trial with specified providers and models.
        
        Args:
            trial_number: Number of the trial
            providers: List of providers for each agent
            models: List of model names for each agent (optional)
        """
        # Create environment
        env = CleanUpEnv(config=self.config)
        env.reset()
        num_agents = env.num_agents

        # Validate providers
        if len(providers) != num_agents:
            raise ValueError(f"Number of providers ({len(providers)}) does not match number of agents ({num_agents})")
        
        # Validate models if provided
        if models and len(models) != num_agents:
            raise ValueError(f"Number of models ({len(models)}) does not match number of agents ({num_agents})")

        # Create agent processes and pipes
        processes = []
        parent_conns = []
        for i in range(num_agents):
            parent_conn, child_conn = mp.Pipe()
            model_name = models[i] if models else None
            p = mp.Process(target=agent_process, args=(child_conn, i, providers[i], model_name))
            p.start()
            processes.append(p)
            parent_conns.append(parent_conn)

        # Initialize trial log
        trial_log = {
            "trial_number": trial_number,
            "providers": providers,
            "models": models if models else [None] * num_agents,
            "steps": []
        }

        # Run the trial
        step_count = 0
        while step_count < env.max_steps:
            # Send observations to all agents
            for agent_id, conn in enumerate(parent_conns):
                obs = env.get_observation(agent_id)
                conn.send(obs)

            # Wait for responses from all agents
            actions = {}
            agent_plans = {}
            for agent_id, conn in enumerate(parent_conns):
                response = conn.recv()
                if not response:
                    raise ValueError("Empty response from agent")
                    
                action = response.get("action", "stay")
                plan = response.get("plan", "")
                
                if action not in ALLOWED_ACTIONS:
                    print(f"Invalid action '{action}' from agent {agent_id}, defaulting to 'stay'")
                    action = "stay"

                actions[agent_id] = action
                agent_plans[agent_id] = plan
                
                # Log model's action and plan
                print(f"\nAgent {agent_id} ({providers[agent_id]}/{models[agent_id]}):")
                print(f"Action: {action}")
                print(f"Plan: {plan}")

            # Only proceed if we got responses from all agents
            if len(actions) == num_agents:
                # Store the plans in the environment before stepping
                for agent_id, plan in agent_plans.items():
                    env.previous_plans[agent_id] = plan
                
                # Capture the state BEFORE stepping
                pre_step_state = env.get_state()
                
                # Step the environment
                pre_positions = {aid: info["pos"] for aid, info in env.agents.items()}
                env.step(actions)
                post_positions = {aid: info["pos"] for aid, info in env.agents.items()}
                
                # Check if any positions changed
                print("Position changes:")
                for agent_id in pre_positions:
                    if pre_positions[agent_id] != post_positions[agent_id]:
                        print(f"  Agent {agent_id} moved: {pre_positions[agent_id]} -> {post_positions[agent_id]}")
                    else:
                        print(f"  Agent {agent_id} did not move: {pre_positions[agent_id]} (Action: {actions.get(agent_id, 'none')})")

                # Log the step with both pre-step and post-step states
                step_log = {
                    "step": step_count,
                    "pre_state": pre_step_state,
                    "state": env.get_state(),
                    "actions": actions,
                    "plans": agent_plans,
                    "blocked_moves": env.blocked_moves.copy()
                }
                
                # Extract and log alliance signals for easier analysis
                alliance_signals = {}
                alliance_acceptances = {}
                alliance_declines = {}
                
                for agent_id, plan in agent_plans.items():
                    if not plan:
                        continue
                    
                    # Extract alliance signals
                    signal_match = re.search(r"I signal alliance to (\d+)", plan, re.IGNORECASE)
                    if signal_match:
                        target_id = int(signal_match.group(1))
                        if agent_id not in alliance_signals:
                            alliance_signals[agent_id] = []
                        alliance_signals[agent_id].append(target_id)
                    
                    # Extract alliance acceptances
                    accept_match = re.search(r"I accept (\d+)'s alliance", plan, re.IGNORECASE)
                    if accept_match:
                        proposer_id = int(accept_match.group(1))
                        if agent_id not in alliance_acceptances:
                            alliance_acceptances[agent_id] = []
                        alliance_acceptances[agent_id].append(proposer_id)
                    
                    # Extract alliance declines
                    decline_match = re.search(r"I decline (\d+)'s alliance", plan, re.IGNORECASE)
                    if decline_match:
                        proposer_id = int(decline_match.group(1))
                        if agent_id not in alliance_declines:
                            alliance_declines[agent_id] = []
                        alliance_declines[agent_id].append(proposer_id)
                
                # Add alliance interactions to the log
                step_log["alliance_signals"] = alliance_signals
                step_log["alliance_acceptances"] = alliance_acceptances
                step_log["alliance_declines"] = alliance_declines
                
                # Print alliance formations (the actual alliance updates happen in environment.py)
                for agent_id, accepted_agents in alliance_acceptances.items():
                    for proposer_id in accepted_agents:
                        print(f"Alliance formed between Agent {agent_id} and Agent {proposer_id}")
                
                trial_log["steps"].append(step_log)

                step_count += 1

        # Clean up processes
        for conn in parent_conns:
            try:
                conn.send(None)
            except:
                pass
        for p in processes:
            try:
                p.join(timeout=1.0)
            except:
                p.terminate()

        # Save trial results
        with open(f"{self.experiment_dir}/trial_{trial_number}.json", "w") as f:
            json.dump(trial_log, f, indent=2)

        return trial_log

    def run_experiment(self, num_trials, provider_configs, model_configs=None):
        """
        Run multiple trials with different provider and model configurations.
        
        Args:
            num_trials: Number of trials to run for each configuration
            provider_configs: List of provider configurations
            model_configs: List of model configurations (optional)
        """
        experiment_log = {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "trials": []
        }

        for trial_number in range(num_trials):
            # Select configurations (cycling through if needed)
            providers = provider_configs[trial_number % len(provider_configs)]
            models = model_configs[trial_number % len(model_configs)] if model_configs else None
            
            print(f"\nStarting Trial {trial_number + 1}/{num_trials}")
            print(f"Providers: {providers}")
            if models:
                print(f"Models: {models}")
            
            trial_log = self.run_trial(trial_number, providers, models)
            experiment_log["trials"].append(trial_log)

            # Save intermediate results
            with open(f"{self.experiment_dir}/experiment_summary.json", "w") as f:
                json.dump(experiment_log, f, indent=2)

        return experiment_log

def main():
    # configuration
    config = {
        "width": 10,
        "height": 8,
        "cell_size": 40,
        "river_layout": "middle",
        "num_agents": 4,
        "max_steps": 100,
        "apple_growth_rate": 0.01,
        "pollution_rate": 0.05,
        "pollution_threshold": 30,
        "zap_cooldown": 5
    }

    # Different provider configurations to test
    provider_configs = [
        ["gemini", "azure_openai", "azure_inference", "claude"],
        ["gemini", "azure_openai", "azure_inference", "claude"]
    ]

    # Model configurations
    model_configs = [
        ["gemini-2.0-flash-001", "gpt-4o-mini", "DeepSeek-R1", "claude-3-7-sonnet-20250219"],
        ["gemini-2.0-flash-001", "gpt-4o-mini", "DeepSeek-R1", "claude-3-7-sonnet-20250219"]
    ]

    # Create and run experiment
    runner = ExperimentRunner(config)
    experiment_log = runner.run_experiment(num_trials=5, provider_configs=provider_configs, model_configs=model_configs)

    print(f"\nExperiment completed. Results saved in: {runner.experiment_dir}")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main() 