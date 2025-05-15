import time
import re
import threading
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
import anthropic

# Load environment variables from .env file
load_dotenv()

# API Configuration
AZURE_INFERENCE_ENDPOINT = os.getenv("AZURE_INFERENCE_ENDPOINT")
AZURE_INFERENCE_TOKEN = os.getenv("AZURE_INFERENCE_TOKEN")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_TOKEN = os.getenv("AZURE_OPENAI_TOKEN")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

# Google Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Anthropic Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables")

api_lock = threading.Lock()

RATE_LIMIT_DELAY = 5.0  # seconds between API calls
last_api_call = 0
MAX_RETRIES = 3

ALLOWED_ACTIONS = {
    "move_up", "move_down", "move_left", "move_right",
    "clean", 
    "zap_up", "zap_down", "zap_left", "zap_right", 
    "stay"
}

def call_llm_api(provider, model_name, prompt, endpoint=None, token=None):
    """Calls the LLM API using either Azure OpenAI, Azure Inference, Google, or Anthropic interfaces.
    
    Args:
        provider (str): 'azure_openai', 'azure_inference', 'gemini', or 'claude'
        model_name (str): Name of the model to use
        prompt (str): The prompt to send to the model
        endpoint (str, optional): Azure endpoint URL. Required for Azure providers.
        token (str, optional): API token. Required for Azure providers.
    """
    global last_api_call
    
    with api_lock:
        current_time = time.time()
        time_since_last_call = current_time - last_api_call
        if time_since_last_call < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - time_since_last_call)
        retries = 0
        while retries < MAX_RETRIES:
            try:
                if provider == "azure_openai":
                    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_TOKEN:
                        raise ValueError("Azure OpenAI endpoint and token are required")
                    client = AzureOpenAI(
                        api_version=AZURE_API_VERSION,
                        azure_endpoint=AZURE_OPENAI_ENDPOINT,
                        api_key=AZURE_OPENAI_TOKEN,
                    )
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                        max_tokens=512,
                        temperature=0.8,
                        top_p=0.1,
                    )
                    response_content = response.choices[0].message.content
                elif provider == "azure_inference":
                    if not AZURE_INFERENCE_ENDPOINT or not AZURE_INFERENCE_TOKEN:
                        raise ValueError("Azure Inference endpoint and token are required")
                    client = ChatCompletionsClient(
                        endpoint=AZURE_INFERENCE_ENDPOINT,
                        credential=AzureKeyCredential(AZURE_INFERENCE_TOKEN),
                    )
                    response = client.complete(
                        messages=[UserMessage(content=prompt)],
                        model=model_name,
                        max_tokens=4096,
                        temperature=0.01
                    )
                    response_content = response.choices[0].message.content
                elif provider == "gemini":
                    if not GEMINI_API_KEY:
                        raise ValueError("GEMINI_API_KEY environment variable is required")
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    response_content = response.text.strip()
                elif provider == "claude":
                    if not ANTHROPIC_API_KEY:
                        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
                    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=512,
                        temperature=0.8,
                        top_p=0.1,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    response_content = response.content[0].text
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
                last_api_call = time.time()
                return response_content
            except Exception as e:
                retries += 1
                print(f"API call failed (attempt {retries}/{MAX_RETRIES}): {str(e)}")
                if retries >= MAX_RETRIES:
                    return "action: stay\nplan: API call failed after multiple retries."
                time.sleep(RATE_LIMIT_DELAY * (2 ** (retries-1)))
        return "action: stay\nplan: API call failed unexpectedly."

def parse_llm_response(response_text):
    """Parse LLM response, extracting action and plan."""
    cleaned_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
    action = "stay"
    plan = ""
    
    # Look for action in the expected format
    action_match = re.search(r"^action:\s*([\w_]+)", cleaned_text, re.IGNORECASE | re.MULTILINE)
    
    # Look for plan in the expected format
    plan_match = re.search(r"^plan:\s*(.*?)(?:\n\S|\Z)", cleaned_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if action_match:
        action = action_match.group(1).strip().lower()
    if plan_match:
        plan = plan_match.group(1).strip()
    else:
        # If no plan found in the expected format, take everything after "action:"
        lines = cleaned_text.split('\n')
        for i, line in enumerate(lines):
            if re.match(r"^action:\s*[\w_]+", line, re.IGNORECASE):
                if i + 1 < len(lines):
                    plan = '\n'.join(lines[i+1:]).strip()
                    if plan.lower().startswith('plan:'):
                        plan = re.sub(r'^plan:\s*', '', plan, flags=re.IGNORECASE).strip()
                break
    
    # If action is stay but the plan contains action instructions, extract from plan
    if action == "stay" and plan:
        # Check if the plan starts with "I'll" followed by an action verb
        action_in_plan = re.search(r"^I(?:'ll| will) (move|clean|zap|stay)", plan, re.IGNORECASE)
        if action_in_plan:
            verb = action_in_plan.group(1).lower()
            if verb == "move":
                # Look for direction
                direction_match = re.search(r"move (up|down|left|right)", plan.lower())
                if direction_match:
                    direction = direction_match.group(1)
                    action = f"move_{direction}"
            elif verb == "clean":
                action = "clean"
            elif verb == "zap":
                # Look for direction
                direction_match = re.search(r"zap (up|down|left|right)", plan.lower())
                if direction_match:
                    direction = direction_match.group(1)
                    action = f"zap_{direction}"
            elif verb == "stay":
                action = "stay"
    
    # JSON fallback
    if action not in ALLOWED_ACTIONS:
        try:
            json_match = re.search(r'```json\s*({.*?})\s*```', cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            elif cleaned_text.startswith('{') and cleaned_text.endswith('}'):
                json_str = cleaned_text
            else:
                json_str = None
            if json_str:
                data = json.loads(json_str)
                json_action = data.get("action", "stay").strip().lower()
                if json_action in ALLOWED_ACTIONS:
                    action = json_action
                    plan = data.get("plan", "").strip()
                elif plan == "":
                    plan = data.get("plan", "").strip()
        except (json.JSONDecodeError, Exception):
            pass
    
    # Final validation of action
    if action not in ALLOWED_ACTIONS:
        action = "stay"

    return action, plan

def generate_prompt(observation):
    """Generate a prompt for the LLM based on the current game state."""
    agent_id = observation.get('agent_id')
    alliance_scores = observation.get('alliance_scores', {})
    active_alliances = {aid: score for aid, score in alliance_scores.items() if score >= 2}
    
    # Format other agents' information including their plans
    other_agents_info = []
    alliance_signals_received = []
    for aid, info in observation.get('agents', {}).items():
        plan = info.get('plan', '')
        alliance_signals = []
        
        # Check for EXACT alliance signals directed to this agent
        exact_signal = f"I signal alliance to {agent_id}"
        if exact_signal.lower() in plan.lower():
            signal = f"Agent {aid} is signaling an alliance to you"
            alliance_signals.append(signal)
            alliance_signals_received.append(signal)
        
        # Check for EXACT alliance acceptance directed to this agent
        exact_accept = f"I accept {agent_id}'s alliance"
        if exact_accept.lower() in plan.lower():
            signal = f"Agent {aid} is accepting your alliance proposal"
            alliance_signals.append(signal)
            alliance_signals_received.append(signal)
            
        agent_info = f"Agent {aid} at {info['pos']} (score: {info['score']})"
        if alliance_signals:
            agent_info += f" - {'; '.join(alliance_signals)}"
        other_agents_info.append(agent_info)
    
    # Create the standard prompt
    prompt = (
        "You control an agent in a grid-world. Review the current state, your previous action and plan to decide your next action\n\n"
        f"YOUR POSITION: {observation.get('self', {}).get('pos')} (row, column)\n"
        f"YOUR SCORE: {observation.get('self', {}).get('score')}\n"
        f"APPLES: {observation.get('apples')}\n"
        f"POLLUTION: {observation.get('pollution')}\n"
        f"OTHER AGENTS:\n" + "\n".join(f"• {info}" for info in other_agents_info) + "\n"
        f"ALLIANCE SCORES: {alliance_scores}\n"
        f"ACTIVE ALLIANCES: {active_alliances}\n"
        f"PREVIOUS ACTION: {observation.get('previous_action', 'stay')}\n"
        f"LAST MOVE BLOCKED: {'Yes' if observation.get('last_move_blocked', False) else 'No'}\n"
    )
    
    # Add pending alliance proposals if there are any
    pending_proposals = observation.get('pending_proposals', [])
    if pending_proposals:
        prompt += "\nPENDING ALLIANCE PROPOSALS:\n"
        for proposer_id in pending_proposals:
            prompt += f"• Agent {proposer_id} has proposed an alliance to you\n"
    
    # Add my proposals if there are any
    my_proposals = observation.get('my_proposals', [])
    if my_proposals:
        prompt += "\nYOUR ALLIANCE PROPOSALS:\n"
        for target_id in my_proposals:
            prompt += f"• You have proposed an alliance to Agent {target_id}\n"
    
    # Add alliance signals section if there are any (clean, simple approach)
    if alliance_signals_received:
        prompt += "\nALLIANCE SIGNALS:\n"
        for signal in alliance_signals_received:
            prompt += f"• {signal}\n"
    
    # Complete the rest of the prompt with instructions
    prompt += "\n" + (
        "COORDINATE SYSTEM:\n"
        "• The grid uses a (row, column) format, where (0,0) is the top-left corner.\n"
        "• Actions like 'move_up' decrease the row index (e.g., from (3,2) to (2,2)).\n"
        "• Actions like 'move_down' increase the row index (e.g., from (3,2) to (4,2)).\n"
        "• Actions like 'move_left' decrease the column index (e.g., from (3,2) to (3,1)).\n"
        "• Actions like 'move_right' increase the column index (e.g., from (3,2) to (3,3)).\n"
        "• IMPORTANT: Do not attempt to move outside the boundaries of the grid.\n\n"
        
        "AVAILABLE ACTIONS:\n"
        "• movement: move_up, move_down, move_left, move_right\n"
        "• other: clean (when on pollution), zap_up/down/left/right (freeze another agent), stay\n\n"
        
        "CORE STRATEGY PRIORITIES:\n"
        "1. Collect apples\n"
        "2. Clean pollution when standing on it\n"
        "3. If your move was blocked AND another agent is near your target apple, target a different apple\n"
        "4. Use zap actions strategically to freeze competitors\n"
        "5. Form alliances with other agents if you think it is advantageous to do so\n\n"
        
        "ALLIANCE RULES:\n"
        "• Alliances can help you win by cooperating with another agent.\n"
        "• You do not need to be near an agent to form an alliance.\n"
        "• USE THESE EXACT PHRASES to form alliances (no variations allowed):\n"
        f"  - To propose an alliance: \"I signal alliance to [number]\" (example: \"I signal alliance to 3\")\n"
        f"  - To accept an alliance: \"I accept [number]'s alliance\" (example: \"I accept 2's alliance\")\n"
        f"  - To decline an alliance: \"I decline [number]'s alliance\" (example: \"I decline 2's alliance\")\n"
        f"• You MUST respond to ALL alliance signals in the above manner.\n"
        f"• You cannot form alliances with yourself or agents you are already allied with.\n"
        "• When accepted, alliance scores increase by 2 for both agents.\n"
        "• Active alliances (score ≥ 2) should not be zapped.\n"
        "• Zapping reduces alliance score by 1.\n"
        "• Agents with the same score can tie for victory.\n\n"
        
        "RESPOND ONLY WITH:\n"
        "action: <action_name>\n"
        "plan: <explanation of your action and alliance formation if opting to do so>\n\n"
        
        "IMPORTANT:\n"
        "• The action line MUST be on its own line\n"
        "• The action MUST be one of the exact action names listed above\n"
        "• Do not include any other text before or after these two lines\n"
        "• Do not use any other format like JSON or markdown\n"
        "• Do not include any additional explanations outside the plan or alliance formation\n\n"
    )

def choose_action(observation, provider, model_name="gpt-4"):
    """Chooses an action based on the observation using the specified provider and model."""
    prompt = generate_prompt(observation)
    
    try:
        if provider == "gemini":
            if not GEMINI_API_KEY:
                print("Warning: Using Gemini provider but GEMINI_API_KEY not found")
                return "stay", "API key not configured"
                
            response_text = call_llm_api(
                provider="gemini",
                model_name=model_name or "gemini-pro",
                prompt=prompt
            )
        elif provider == "azure_openai":
            if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_TOKEN:
                print("Warning: Using Azure OpenAI provider but endpoint or token not found")
                return "stay", "API configuration missing"
                
            response_text = call_llm_api(
                provider="azure_openai",
                model_name=model_name,
                prompt=prompt
            )
        elif provider == "azure_inference":
            if not AZURE_INFERENCE_ENDPOINT or not AZURE_INFERENCE_TOKEN:
                print("Warning: Using Azure Inference provider but endpoint or token not found")
                return "stay", "API configuration missing"
                
            response_text = call_llm_api(
                provider="azure_inference",
                model_name=model_name,
                prompt=prompt
            )
        elif provider == "claude":
            if not ANTHROPIC_API_KEY:
                print("Warning: Using Claude provider but ANTHROPIC_API_KEY not found")
                return "stay", "API key not configured"
                
            response_text = call_llm_api(
                provider="claude",
                model_name=model_name,
                prompt=prompt
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        action, plan = parse_llm_response(response_text)
        return action, plan
        
    except Exception as e:
        print(f"Error in choose_action: {str(e)}")
        return "stay", f"Error: {str(e)}"

def agent_process(pipe, agent_id, provider, model_name=None):
    """Main agent process that handles communication with the environment.
    
    Args:
        pipe: Communication pipe
        agent_id: ID of the agent
        provider: 'azure_openai', 'azure_inference', or 'gemini'
        model_name: Name of the model to use (optional)
    """
    while True:
        try:
            observation = pipe.recv()
            action, plan = choose_action(observation, provider, model_name)
            pipe.send({"action": action, "plan": plan})
        except EOFError:
            break
        except Exception as e:
            print(f"Error in agent {agent_id}: {e}")
            pipe.send({"action": "stay", "plan": f"Error: {str(e)}"})
