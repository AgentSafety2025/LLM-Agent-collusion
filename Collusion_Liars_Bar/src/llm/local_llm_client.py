import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Any, Optional
import os
import gc
import re
import json

# Basic PyTorch configuration
torch.set_float32_matmul_precision('high')

class LocalLLMClient:
    def __init__(self, model_base_path: str = "/model-weights"):
        """Initialize the Local LLM client for open-weight models

        Args:
            model_base_path: Base path where models are stored
        """
        self.model_base_path = model_base_path
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model name mappings - only the 4 required models
        self.model_mappings = {
            "llama-3-8b": "Meta-Llama-3-8B-Instruct",
            "llama-3.1-8b": "Meta-Llama-3.1-8B-Instruct",
            "qwen2.5-7b": "Qwen2.5-7B-Instruct",
            "mistral-7b": "Mistral-7B-Instruct-v0.3"
        }

        print(f"LocalLLMClient initialized. Device: {self.device}")

    def chat(self, messages: List[Dict[str, str]], model: str = "llama-3.1-8b", max_new_tokens: int = 4096) -> Tuple[str, str]:
        """Chat with a local model - main interface matching closed-source client

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model to use (must be one of the 4 supported models)
            max_new_tokens: Maximum new tokens to generate

        Returns:
            tuple: (content, reasoning_content) - reasoning_content is empty for local models
        """
        try:
            print(f"Local LLM Request to {model}: {messages}")

            # Validate model is supported
            if model not in self.model_mappings:
                print(f"Unsupported model: {model}. Supported models: {list(self.model_mappings.keys())}")
                return "", ""

            # Load the model if needed
            if not self._load_model(model):
                return "", ""

            # Convert messages to proper prompt format
            prompt = self._messages_to_prompt(messages, model)

            # Generate response with anti-hallucination measures
            response = self._generate_response(prompt, max_new_tokens, model)

            print(f"Local LLM Response: {response}")
            return response, ""  # No reasoning content for local models

        except Exception as e:
            print(f"Local LLM error: {str(e)}")
            return "", ""

    def _get_model_path(self, model_name: str) -> str:
        """Get the full path to a model"""
        full_name = self.model_mappings[model_name]
        return os.path.join(self.model_base_path, full_name)

    def _load_model(self, model_name: str) -> bool:
        """Load a model and tokenizer"""
        try:
            # If we already have this model loaded, skip
            if self.current_model_name == model_name:
                return True

            # Clear previous model from memory
            if self.current_model is not None:
                del self.current_model
                del self.current_tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            model_path = self._get_model_path(model_name)

            if not os.path.exists(model_path):
                print(f"Model path does not exist: {model_path}")
                return False

            print(f"Loading model: {model_name} from {model_path}")

            # Load tokenizer
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # Add pad token if it doesn't exist
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token

            # Load model
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Move model to device
            self.current_model = self.current_model.to(self.device)
            self.current_model.eval()  # Set to eval mode

            torch.cuda.empty_cache()

            self.current_model_name = model_name
            print(f"Successfully loaded {model_name}")
            return True

        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return False

    def _messages_to_prompt(self, messages: List[Dict[str, str]], model: str) -> str:
        """Convert chat messages to model-specific prompt format"""

        # Use model-specific chat templates when available
        if hasattr(self.current_tokenizer, 'apply_chat_template') and self.current_tokenizer.chat_template is not None:
            try:
                formatted_prompt = self.current_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted_prompt
            except Exception as e:
                pass  # Fall back to manual formatting

        # Fallback: manual formatting based on model type
        if "llama" in model.lower():
            return self._llama_format(messages)
        elif "qwen" in model.lower():
            return self._qwen_format(messages)
        elif "mistral" in model.lower():
            return self._mistral_format(messages)
        else:
            return self._generic_format(messages)

    def _llama_format(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Llama models"""
        prompt = "<|begin_of_text|>"
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def _qwen_format(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Qwen models"""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n"
        return prompt

    def _mistral_format(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Mistral models"""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"[INST] {content} [/INST]\n"
            elif role == "user":
                prompt += f"[INST] {content} [/INST]\n"
            elif role == "assistant":
                prompt += f"{content}\n"

        return prompt

    def _generic_format(self, messages: List[Dict[str, str]]) -> str:
        """Generic fallback formatting"""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        prompt += "Assistant: "
        return prompt

    def _generate_response(self, prompt: str, max_new_tokens: int, model: str) -> str:
        """Generate response with anti-hallucination measures"""
        try:
            # Tokenize input - use approach that works reliably
            # Primary approach: try without truncation first for llama models
            if "llama-3-8b" in model.lower() or "qwen" in model.lower():
                inputs = self.current_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=False
                )
                # If prompt is too long, truncate manually
                if inputs['input_ids'].shape[1] > 4096 - max_new_tokens:
                    inputs = self.current_tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=4096 - max_new_tokens
                    )
                    # If truncation results in empty tokens, fall back to no truncation
                    if inputs['input_ids'].shape[1] == 0:
                        inputs = self.current_tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=False
                        )
            else:
                # For other models, use standard approach
                inputs = self.current_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096 - max_new_tokens
                )

            # Final safety check - if still empty, return error
            if inputs['input_ids'].shape[1] == 0:
                return ""

            inputs = inputs.to(self.device)

            # Generate with model-specific parameters
            with torch.no_grad():
                # Model-specific generation parameters
                if "llama" in model.lower():
                    generation_kwargs = {
                        **inputs,
                        'max_new_tokens': min(max_new_tokens, 256),
                        'do_sample': True,
                        'temperature': 0.8,
                        'top_p': 0.95,
                        'top_k': 40,
                        'repetition_penalty': 1.1,
                        'pad_token_id': self.current_tokenizer.pad_token_id,
                        'eos_token_id': self.current_tokenizer.eos_token_id,
                    }
                elif "qwen" in model.lower():
                    generation_kwargs = {
                        **inputs,
                        'max_new_tokens': min(max_new_tokens, 256),
                        'do_sample': True,
                        'temperature': 0.8,
                        'top_p': 0.9,
                        'top_k': 50,
                        'repetition_penalty': 1.05,
                        'pad_token_id': self.current_tokenizer.pad_token_id,
                        'eos_token_id': self.current_tokenizer.eos_token_id,
                    }
                else:  # mistral and others
                    generation_kwargs = {
                        **inputs,
                        'max_new_tokens': min(max_new_tokens, 512),
                        'do_sample': True,
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'top_k': 50,
                        'repetition_penalty': 1.02,
                        'pad_token_id': self.current_tokenizer.pad_token_id,
                        'eos_token_id': self.current_tokenizer.eos_token_id,
                    }

                outputs = self.current_model.generate(**generation_kwargs)

            # Decode only the new tokens - handle different output shapes
            try:
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0]

                # Check if we have generated tokens beyond the input
                if len(generated_tokens.shape) == 1 and generated_tokens.shape[0] > input_length:
                    response = self.current_tokenizer.decode(
                        generated_tokens[input_length:],
                        skip_special_tokens=True
                    )
                elif len(generated_tokens.shape) == 1 and generated_tokens.shape[0] > 0:
                    # If we have tokens but not more than input, decode full output
                    response = self.current_tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True
                    )
                    # Remove the input part manually if needed
                    input_text = self.current_tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                    if response.startswith(input_text):
                        response = response[len(input_text):].strip()
                elif generated_tokens.shape[0] == 0:
                    # Empty generation - model produced no tokens
                    print("Warning: Model generated empty output")
                    response = ""
                else:
                    print(f"Warning: Unexpected output shape: {generated_tokens.shape}")
                    response = ""
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                response = ""

            # Clean and validate response
            cleaned_response = self._clean_response(response, model)

            return cleaned_response

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return ""


    def _clean_response(self, response: str, model: str) -> str:
        """Clean response to remove hallucinations and formatting issues"""

        # Remove common hallucination patterns
        response = re.sub(r'^(Assistant:|AI:|Human:|User:)', '', response).strip()
        response = re.sub(r'\n(Assistant:|AI:|Human:|User:).*$', '', response, flags=re.MULTILINE | re.DOTALL)

        # Stop at common hallucination triggers
        for stop_phrase in ["Human:", "User:", "\n\n---", "```\n\n", "<|", "[INST]"]:
            if stop_phrase in response:
                response = response.split(stop_phrase)[0].strip()

        # For JSON responses, extract only the first valid JSON object
        if response.strip().startswith('{') or '```json' in response:
            response = self._extract_clean_json(response)

        return response.strip()

    def _extract_clean_json(self, response: str) -> str:
        """Extract and clean JSON from response"""

        # Remove markdown formatting
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        # Stop at common hallucination triggers
        for stop_phrase in ["Human:", "User:", "Assistant:"]:
            if stop_phrase in response:
                response = response.split(stop_phrase)[0].strip()

        # Try to find the first complete JSON object
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response)
        if json_match:
            json_str = json_match.group(1)

            # Fix common unquoted value issues, but preserve boolean and null values
            json_str = re.sub(r':\s*(?!true\b|false\b|null\b)([A-Za-z][A-Za-z0-9]*)', r': "\1"', json_str)
            json_str = re.sub(r'\[\s*(?!true\b|false\b|null\b)([A-Za-z0-9]+)\s*,\s*(?!true\b|false\b|null\b)([A-Za-z0-9]+)\s*\]', r'["\1", "\2"]', json_str)
            json_str = re.sub(r'\[\s*(?!true\b|false\b|null\b)([A-Za-z0-9]+)\s*\]', r'["\1"]', json_str)

            try:
                parsed = json.loads(json_str)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                return json_str

        return response.strip()

    def list_available_models(self) -> List[str]:
        """List available models"""
        return list(self.model_mappings.keys())

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if self.current_model is None:
            return {"status": "No model loaded"}

        return {
            "model_name": self.current_model_name,
            "device": str(self.device),
            "model_type": type(self.current_model).__name__,
            "parameters": sum(p.numel() for p in self.current_model.parameters()),
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
        }

# Usage example
if __name__ == "__main__":
    client = LocalLLMClient()

    print("Available models:")
    for model in client.list_available_models():
        print(f"  - {model}")

    # Test with a simple message
    messages = [
        {"role": "user", "content": "Generate a JSON response with keys 'name' and 'value'. Example: {\"name\": \"test\", \"value\": 42}"}
    ]

    for model in ["llama-3.1-8b", "qwen2.5-7b", "mistral-7b"]:
        print(f"\nTesting {model}:")
        response, _ = client.chat(messages, model=model)
        print(f"Response: {response}")