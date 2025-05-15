# environment.py
import random
import copy
import re

class CleanUpEnv:
    def __init__(self, config=None):
        if config is None:
            config = {}
            
        # Grid configuration
        self.width = config.get("width", 10)
        self.height = config.get("height", 8)
        self.cell_size = config.get("cell_size", 40)
        river_layout = config.get("river_layout", "middle")
        
        # Set up river and orchard columns based on layout
        if river_layout == "middle":
            if self.width < 5:
                self.orchard_cols = list(range(self.width))
                self.river_cols = []
            else:
                mid = self.width // 2
                if self.width % 2 == 0:
                    self.river_cols = list(range(mid - 1, mid + 1))
                    self.orchard_cols = list(range(0, mid - 1)) + list(range(mid + 1, self.width))
                else:
                    self.river_cols = list(range(mid - 1, mid + 2))
                    self.orchard_cols = list(range(0, mid - 1)) + list(range(mid + 2, self.width))
        elif river_layout == "edge":
            self.river_cols = [0]
            self.orchard_cols = list(range(1, self.width))
        else:
            self.orchard_cols = list(range(self.width))
            self.river_cols = []

        # Game state
        self.apples = set()
        self.pollution = set()
        self.agents = {}
        self.num_agents = config.get("num_agents", 4)
        self.step_count = 0
        self.max_steps = config.get("max_steps", 20)

        # Game parameters
        self.apple_growth_rate = config.get("apple_growth_rate", 0.01)
        self.pollution_rate = config.get("pollution_rate", 0.05)
        self.pollution_threshold = config.get("pollution_threshold", 30)
        self.zap_cooldown = config.get("zap_cooldown", 5)

        # Agent state tracking
        self.previous_actions = {}
        self.previous_plans = {}
        self.blocked_moves = {}
        self.zapped_by = {}
        
        # Initialize alliance tracker
        self.alliance_scores = {i: {j: 0 for j in range(self.num_agents) if j != i} for i in range(self.num_agents)}
        
        # Store active alliance proposals (sender_id: target_id) for persistence across turns
        self.alliance_proposals = {}

    def reset(self):
        """Reset the environment to initial state."""
        self.apples = set()
        self.pollution = set()
        self.step_count = 0
        self.agents = {}
        self.previous_actions = {}
        self.previous_plans = {}
        self.blocked_moves = {}
        self.zapped_by = {}
        self.alliance_proposals = {}  # Reset alliance proposals on reset
        
        # Initialize agents in random positions
        occupied_positions = set()
        for i in range(self.num_agents):
            while True:
                pos = (random.randint(0, self.height - 1), random.choice(self.orchard_cols))
                if pos not in occupied_positions:
                    self.agents[i] = {"pos": pos, "score": 0, "cooldown": 0, "active": True}
                    occupied_positions.add(pos)
                    break

    def in_bounds(self, pos):
        """Check if a position is within the grid boundaries."""
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_orchard(self, pos):
        """Check if a position is in the orchard area."""
        r, c = pos
        return c in self.orchard_cols

    def is_river(self, pos):
        """Check if a position is in the river area."""
        r, c = pos
        return c in self.river_cols

    def spawn_apples(self):
        """Spawn apples in orchard cells based on pollution level."""
        pollution_level = len(self.pollution)
        spawn_probability = 0 if pollution_level >= self.pollution_threshold else \
            self.apple_growth_rate * (1 - pollution_level / self.pollution_threshold)
            
        for r in range(self.height):
            for c in self.orchard_cols:
                if (r, c) not in self.apples and random.random() < spawn_probability:
                    self.apples.add((r, c))

    def add_pollution(self):
        """Add pollution to river cells based on probability."""
        for r in range(self.height):
            for c in self.river_cols:
                if (r, c) not in self.pollution and random.random() < self.pollution_rate:
                    self.pollution.add((r, c))

    def move_agent(self, agent_id, action):
        """Move an agent based on the given action."""
        if not self.agents[agent_id]["active"]:
            return
            
        r, c = self.agents[agent_id]["pos"]
        new_r, new_c = r, c
        
        if action.startswith("move_"):
            if action == "move_up":
                new_r -= 1
            elif action == "move_down":
                new_r += 1
            elif action == "move_left":
                new_c -= 1
            elif action == "move_right":
                new_c += 1

            if self.in_bounds((new_r, new_c)):
                position_occupied = any(
                    other_id != agent_id and info["active"] and info["pos"] == (new_r, new_c)
                    for other_id, info in self.agents.items()
                )
                
                if not position_occupied:
                    self.agents[agent_id]["pos"] = (new_r, new_c)
                    self.blocked_moves[agent_id] = False
                else:
                    self.blocked_moves[agent_id] = True
            else:
                self.blocked_moves[agent_id] = True
        else:
            self.blocked_moves[agent_id] = False

    def agent_clean(self, agent_id):
        """Clean pollution at agent's position."""
        pos = self.agents[agent_id]["pos"]
        if self.is_river(pos) and pos in self.pollution:
            self.pollution.remove(pos)

    def update_alliance_score(self, agent_i, agent_j, delta):
        """Update alliance score between two agents."""
        if agent_i in self.alliance_scores and agent_j in self.alliance_scores[agent_i]:
            self.alliance_scores[agent_i][agent_j] += delta

    def agent_zap(self, agent_id, action):
        """Zap another agent in the specified direction."""
        r, c = self.agents[agent_id]["pos"]
        direction = {
            "zap_up": (-1, 0),
            "zap_down": (1, 0),
            "zap_left": (0, -1),
            "zap_right": (0, 1)
        }.get(action)
        
        if direction is None:
            return
            
        dr, dc = direction
        cur_r, cur_c = r + dr, c + dc
        
        while self.in_bounds((cur_r, cur_c)):
            for other_id, info in self.agents.items():
                if other_id != agent_id and info["active"] and info["pos"] == (cur_r, cur_c):
                    self.agents[other_id]["active"] = False
                    self.agents[other_id]["cooldown"] = self.zap_cooldown
                    self.zapped_by[other_id] = agent_id
                    # Update alliance scores when an agent is zapped
                    self.update_alliance_score(agent_id, other_id, -1)
                    return
            cur_r += dr
            cur_c += dc

    def collect_apples(self, agent_id):
        """Collect apple at agent's position."""
        pos = self.agents[agent_id]["pos"]
        if pos in self.apples:
            self.apples.remove(pos)
            self.agents[agent_id]["score"] += 1

    def update_agents(self):
        """Update agent states (cooldowns, active status)."""
        for agent_id, info in self.agents.items():
            if not info["active"]:
                info["cooldown"] -= 1
                if info["cooldown"] <= 0:
                    info["active"] = True
                    if agent_id in self.zapped_by:
                        self.zapped_by.pop(agent_id)

    def step(self, actions):
        """Execute one step of the environment."""
        self.step_count += 1
        self.previous_actions = actions.copy()
        
        # Process plans for alliance signals and acceptances
        self.process_alliance_plans()
        
        # Process movement actions
        intended_positions = {}
        for agent_id, action in actions.items():
            if not self.agents[agent_id]["active"]:
                continue
                
            if action.startswith("move_"):
                r, c = self.agents[agent_id]["pos"]
                new_r, new_c = r, c
                
                if action == "move_up":
                    new_r -= 1
                elif action == "move_down":
                    new_r += 1
                elif action == "move_left":
                    new_c -= 1
                elif action == "move_right":
                    new_c += 1
                    
                intended_positions[agent_id] = (new_r, new_c)

        # Resolve collisions and execute moves
        for agent_id, intended_pos in intended_positions.items():
            if not self.in_bounds(intended_pos):
                self.blocked_moves[agent_id] = True
                continue
                
            collision = any(
                other_id != agent_id and other_pos == intended_pos
                for other_id, other_pos in intended_positions.items()
            )
            
            if not collision:
                self.agents[agent_id]["pos"] = intended_pos
                self.blocked_moves[agent_id] = False
            else:
                self.blocked_moves[agent_id] = True

        # Process other actions
        for agent_id, action in actions.items():
            if not self.agents[agent_id]["active"]:
                continue
                
            if action == "clean":
                self.agent_clean(agent_id)
            elif action.startswith("zap_"):
                self.agent_zap(agent_id, action)
            self.collect_apples(agent_id)

        # Update environment
        self.update_agents()
        self.spawn_apples()
        self.add_pollution()
        
        # Do not clear previous plans to allow alliance signals to persist

    def get_state(self):
        """Get the current state of the environment."""
        return {
            "apples": list(self.apples),
            "pollution": list(self.pollution),
            "agents": {aid: info.copy() for aid, info in self.agents.items()},
            "step_count": self.step_count,
            "alliance_scores": {i: dict(scores) for i, scores in self.alliance_scores.items()},
            "alliance_proposals": dict(self.alliance_proposals)
        }

    def get_observation(self, agent_id):
        """Get the observation for a specific agent."""
        agent_info = self.agents[agent_id]
        
        # Get alliance scores for this agent
        alliance_scores = self.alliance_scores.get(agent_id, {})
        
        # Get pending proposals for this agent
        pending_proposals = []
        for proposer_id, target_id in self.alliance_proposals.items():
            if target_id == agent_id:
                pending_proposals.append(proposer_id)
        
        # Get agents this agent has proposed to
        my_proposals = []
        if agent_id in self.alliance_proposals:
            my_proposals.append(self.alliance_proposals[agent_id])
            
        # Get active alliances (agents with alliance score > 0)
        active_alliances = {aid: score for aid, score in alliance_scores.items() if score > 0}
        
        return {
            "self": {
                "pos": agent_info["pos"],
                "score": agent_info["score"]
            },
            "apples": list(self.apples),
            "pollution": list(self.pollution),
            "agents": {
                aid: {
                    "pos": info["pos"], 
                    "score": info["score"],
                    "plan": self.previous_plans.get(aid, "")  # Include other agents' plans
                }
                for aid, info in self.agents.items()
                if aid != agent_id
            },
            "agent_id": agent_id,
            "alliance_scores": alliance_scores,
            "active_alliances": active_alliances,  # Add active alliances information
            "previous_action": self.previous_actions.get(agent_id, "stay"),
            "previous_plan": self.previous_plans.get(agent_id, ""),
            "last_move_blocked": self.blocked_moves.get(agent_id, False),
            "pending_proposals": pending_proposals,  # Agents who have proposed to this agent
            "my_proposals": my_proposals  # Agents this agent has proposed to
        }

    def process_alliance_plans(self):
        """Process agent plans for alliance signals and acceptances."""
        # First pass: collect all alliance signals from current plans
        new_alliance_signals = {}  # {sender_id: target_id}
        
        for agent_id, plan in self.previous_plans.items():
            if not plan:
                continue
                
            # Check for exact alliance signals
            # Pattern: "I signal alliance to [number]"
            signal_match = re.search(r"I signal alliance to (\d+)", plan, re.IGNORECASE)
            if signal_match:
                target_id = int(signal_match.group(1))
                new_alliance_signals[agent_id] = target_id
                # Store the proposal for future turns
                self.alliance_proposals[agent_id] = target_id
        
        # Second pass: process acceptances and update scores
        for agent_id, plan in self.previous_plans.items():
            if not plan:
                continue
                
            # Check for various forms of alliance acceptances
            # Expanded pattern to catch more acceptance variations
            accept_match = re.search(r"I(?:'ll| will)? accept (?:Agent )?(\d+)'s alliance", plan, re.IGNORECASE)
            if not accept_match:
                # Try alternate phrasings
                accept_match = re.search(r"accept(?:ing)?(?: the)? alliance (?:from|with) (?:Agent )?(\d+)", plan, re.IGNORECASE)
            
            if accept_match:
                proposer_id = int(accept_match.group(1))
                
                # Verify there was a proposal (either in this turn or previous turns)
                if (proposer_id in new_alliance_signals and new_alliance_signals[proposer_id] == agent_id) or \
                   (proposer_id in self.alliance_proposals and self.alliance_proposals[proposer_id] == agent_id):
                    
                    # Check current alliance scores
                    proposer_to_accepter_score = self.alliance_scores.get(proposer_id, {}).get(agent_id, 0)
                    accepter_to_proposer_score = self.alliance_scores.get(agent_id, {}).get(proposer_id, 0)
                    
                    # If previous score was negative, set directly to 2, otherwise add 2
                    if proposer_to_accepter_score < 0:
                        self.alliance_scores[proposer_id][agent_id] = 2
                    else:
                        self.update_alliance_score(proposer_id, agent_id, 2)
                        
                    if accepter_to_proposer_score < 0:
                        self.alliance_scores[agent_id][proposer_id] = 2
                    else:
                        self.update_alliance_score(agent_id, proposer_id, 2)
                    
                    # After successful acceptance, remove the proposal
                    if proposer_id in self.alliance_proposals:
                        self.alliance_proposals.pop(proposer_id)
        
        # For debugging: print active proposals
        # if self.alliance_proposals:
        #     print(f"DEBUG: Active alliance proposals: {self.alliance_proposals}")
        #     print(f"DEBUG: Current alliance scores: {self.alliance_scores}")
