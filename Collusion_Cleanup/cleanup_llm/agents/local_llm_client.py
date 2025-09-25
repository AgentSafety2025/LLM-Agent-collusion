import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, List, Tuple, Any, Optional
import os
import gc

# Disable PyTorch dynamo and compiler optimizations to avoid issues
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention SDP
torch.set_float32_matmul_precision('high')   # Set proper matmul precision

class LocalLLMClient:
    def __init__(self, model_base_path: str = "/model-weights"):
        """Initialize the Local LLM client for cleanup game experiments

        Args:
            model_base_path: Base path where models are stored
        """
        self.model_base_path = model_base_path
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Model name mappings for cleanup game experiments
        self.model_mappings = {
            # Small models (8B parameters or less)
            "llama-3-8b": "Meta-Llama-3-8B-Instruct",
            "llama-3.1-8b": "Meta-Llama-3.1-8B-Instruct",
            "mistral-7b": "Mistral-7B-Instruct-v0.3",
            "qwen2.5-7b": "Qwen2.5-7B-Instruct",

            # Large models (30B+ parameters)
            "llama-3-70b": "Meta-Llama-3-70B",
            "llama-3.1-70b": "Meta-Llama-3.1-70B",
            "mixtral-8x7b": "Mixtral-8x7B-Instruct-v0.1",
            "qwen2.5-32b": "Qwen2.5-32B-Instruct",

            # Additional models from Liar's Bar system (kept for compatibility)
            "llama-3.2-3b": "Llama-3.2-3B-Instruct",
            "llama-2-7b": "Llama-2-7b-hf",
            "qwen2.5-14b": "Qwen2.5-14B-Instruct",
            "qwen2.5-72b": "Qwen2.5-72B-Instruct",
            "qwen3-8b": "Qwen3-8B",
            "qwen2-7b": "Qwen2-7B-Instruct",
            "deepseek-r1-8b": "DeepSeek-R1-Distill-Llama-8B",
            "deepseek-r1-7b": "DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-r1-70b": "DeepSeek-R1-Distill-Llama-70B",
            "deepseek-coder": "DeepSeek-Coder-V2-Lite-Instruct",
            "ministral-8b": "Ministral-8B-Instruct-2410",
            "gemma-2-9b": "gemma-2-9b-it",
            "gemma-2-27b": "gemma-2-27b-it",
            "gemma-3-1b": "gemma-3-1b-it",
            "gemma-3-4b": "gemma-3-4b-it",
            "gemma-3-12b": "gemma-3-12b-it",
            "gemma-3-27b": "gemma-3-27b-it",
            "gemma-7b": "gemma-7b",
            "gpt-2-l": "gpt2-large",
            "gpt-2-xl": "gpt2-xl"
        }

        print(f"LocalLLMClient initialized for cleanup experiments. Device: {self.device}, GPUs available: {self.num_gpus}")
        print(f"Target models: Llama-3-8b, Llama-3.1-8b, Mistral-7b, Qwen2.5-7b (small)")
        print(f"              Llama-3-70b, Llama-3.1-70b, Mixtral-8x7b, Qwen2.5-32b (large)")

    def _get_model_path(self, model_name: str) -> str:
        """Get the full path to a model

        Args:
            model_name: Model name (can be short name or full name)

        Returns:
            Full path to the model
        """
        # Check if it's a short name that needs mapping
        if model_name in self.model_mappings:
            full_name = self.model_mappings[model_name]
        else:
            full_name = model_name

        return os.path.join(self.model_base_path, full_name)

    def _get_device_map(self, model_name: str):
        """Get appropriate device mapping for a model based on its size and available GPUs

        Args:
            model_name: Name of the model to load

        Returns:
            Device mapping configuration for the model
        """
        # Define large models that need multi-GPU support
        large_models = [
            "llama-3.1-70b", "llama-3-70b", "qwen2.5-72b", "deepseek-r1-70b",
            "mixtral-8x7b", "qwen2.5-32b"  # These might also be too large for single GPU
        ]

        # If we only have 1 GPU or model is small, use single GPU
        if self.num_gpus <= 1 or model_name not in large_models:
            return {"": self.device}

        # For large models with multiple GPUs available, use auto device mapping
        print(f"Detected {self.num_gpus} GPUs available for large model {model_name}")
        return "auto"

    def _load_model(self, model_name: str) -> bool:
        """Load a model and tokenizer

        Args:
            model_name: Name of the model to load

        Returns:
            True if successful, False otherwise
        """
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


            # Load model with appropriate settings for H100 + 64GB
            # Use device mapping based on model size and available GPUs
            device_map = self._get_device_map(model_name)

            try:
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for better H100 performance
                    device_map=device_map,  # Explicit device mapping
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2"  # Use Flash Attention 2 for H100 optimization
                )
            except Exception as flash_attn_error:
                print(f"Flash attention failed, falling back to eager: {flash_attn_error}")
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for better H100 performance
                    device_map=device_map,  # Explicit device mapping
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    # Use default attention implementation
                )

            # Ensure model is on the correct device (only for single GPU setups)
            if device_map != "auto":
                self.current_model = self.current_model.to(self.device)

            # Clear any cached tensors that might be on CPU
            torch.cuda.empty_cache()

            self.current_model_name = model_name
            print(f"Successfully loaded {model_name}")
            return True

        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return False

    def generate(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate a response using the local model

        Args:
            model_name: The model to use (short name)
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model
            temperature: Sampling temperature (unused in local model)
            max_tokens: Maximum tokens to generate

        Returns:
            str: The generated response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response, _ = self.chat(messages, model=model_name, max_new_tokens=max_tokens)
        return response

    def chat(self, messages: List[Dict[str, str]], model: str = "llama-3.1-8b", max_new_tokens: int = 4096) -> Tuple[str, str]:
        """Chat with a local model

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model to use (short name or full name)

        Returns:
            tuple: (content, reasoning_content) - reasoning_content is empty for local models
        """
        try:
            print(f"Local LLM Request to {model}: {messages}")

            # Load the model if needed
            if not self._load_model(model):
                return "", ""

            # Convert messages to prompt format
            prompt = self._messages_to_prompt(messages)

            # Generate response
            response = self._generate_response(prompt, max_new_tokens)

            print(f"Local LLM Response: {response}")
            return response, ""  # No reasoning content for local models

        except Exception as e:
            print(f"Local LLM error: {str(e)}")
            return "", ""

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a prompt format

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted prompt string
        """
        # TEMPORARILY DISABLE: Try to use the tokenizer's chat template if available
        # The chat template is causing "assistant" to be appended to responses
        # try:
        #     if hasattr(self.current_tokenizer, 'apply_chat_template'):
        #         formatted_prompt = self.current_tokenizer.apply_chat_template(
        #             messages,
        #             tokenize=False,
        #             add_generation_prompt=True
        #         )
        #         print(f"DEBUG: Chat template generated prompt: {repr(formatted_prompt[-200:])}")  # Show last 200 chars
        #         return formatted_prompt
        # except Exception as e:
        #     print(f"DEBUG: Chat template failed: {e}")
        #     pass

        # Fallback to manual formatting
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"{content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        # Don't add "Assistant: " at the end to avoid the model continuing with "assistant"
        return prompt

    def _generate_response(self, prompt: str, max_length: int = 4096) -> str:
        """Generate response from the loaded model

        Args:
            prompt: Input prompt
            max_length: Maximum response length

        Returns:
            Generated response
        """
        try:
            # Set context length based on model type
            if "gpt2" in self.current_model_name.lower() or "gpt-2" in self.current_model_name.lower():
                # GPT-2 models have 1024 max position embeddings
                context_length = 1024
                # For GPT-2, we need to leave room for generation, so use less than max
                input_max_length = 800  # Leave ~224 tokens for generation
            else:
                context_length = 8192
                input_max_length = context_length - max_length  # Leave room for generation

            inputs = self.current_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=input_max_length
            )

            # For multi-GPU models, send to the first device with model parameters
            if hasattr(self.current_model, 'device'):
                # Model has a single device
                target_device = self.current_model.device
            elif hasattr(self.current_model, 'hf_device_map'):
                # Model is spread across multiple devices, use the device of the first layer
                first_device = next(iter(self.current_model.hf_device_map.values()))
                target_device = first_device if isinstance(first_device, str) else f"cuda:{first_device}"
            else:
                # Fallback to default device
                target_device = self.device

            inputs = inputs.to(target_device)

            # Generate with stable parameters
            with torch.no_grad():
                # Ensure all generation arguments are on the correct device
                # Adjust max_new_tokens for GPT-2 models to stay within limits
                if "gpt2" in self.current_model_name.lower() or "gpt-2" in self.current_model_name.lower():
                    # For GPT-2, ensure total tokens don't exceed 1024
                    current_input_length = inputs['input_ids'].shape[1]
                    max_new_tokens = min(max_length, 1024 - current_input_length - 10)  # -10 for safety
                    max_new_tokens = max(10, max_new_tokens)  # Ensure at least 10 tokens can be generated
                else:
                    max_new_tokens = max_length

                generation_kwargs = {
                    **inputs,
                    'max_new_tokens': max_new_tokens,
                    'do_sample': True,
                    'temperature': 0.7,  # More stable temperature
                    'top_p': 0.9,
                    'top_k': 50,  # Add top_k for stability
                    'repetition_penalty': 1.1  # Prevent repetition
                }

                # Only add token IDs if they exist and move to device
                if self.current_tokenizer.pad_token_id is not None:
                    generation_kwargs['pad_token_id'] = self.current_tokenizer.pad_token_id
                if self.current_tokenizer.eos_token_id is not None:
                    generation_kwargs['eos_token_id'] = self.current_tokenizer.eos_token_id

                outputs = self.current_model.generate(**generation_kwargs)

            # Decode response (remove the input prompt part)
            response = self.current_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Clean up spurious "assistant" tokens that models sometimes generate
            # This happens due to chat training even when we don't include "Assistant:" in prompt
            cleaned_response = response.replace("assistant", "").strip()

            return cleaned_response

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return ""

    def list_available_models(self) -> List[str]:
        """List all available models in the model directory

        Returns:
            List of available model names
        """
        try:
            if not os.path.exists(self.model_base_path):
                return []

            models = []
            for item in os.listdir(self.model_base_path):
                item_path = os.path.join(self.model_base_path, item)
                if os.path.isdir(item_path):
                    # Check if it looks like a model directory
                    if any(os.path.exists(os.path.join(item_path, f))
                          for f in ['config.json', 'pytorch_model.bin', 'model.safetensors']):
                        models.append(item)

            return sorted(models)

        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model

        Returns:
            Dictionary with model information
        """
        if self.current_model is None:
            return {"status": "No model loaded"}

        return {
            "model_name": self.current_model_name,
            "device": str(self.device),
            "model_type": type(self.current_model).__name__,
            "parameters": sum(p.numel() for p in self.current_model.parameters()),
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
        }

    def get_cleanup_models(self) -> Dict[str, List[str]]:
        """Get the specific models planned for cleanup experiments

        Returns:
            Dictionary with small and large model lists
        """
        return {
            "small_models": ["llama-3-8b", "llama-3.1-8b", "mistral-7b", "qwen2.5-7b"],
            "large_models": ["llama-3-70b", "llama-3.1-70b", "mixtral-8x7b", "qwen2.5-32b"]
        }

# Usage example
if __name__ == "__main__":
    client = LocalLLMClient()

    print("Available models:")
    for model in client.list_available_models():
        print(f"  - {model}")

    print("\nCleanup experiment models:")
    cleanup_models = client.get_cleanup_models()
    print("Small models:", cleanup_models["small_models"])
    print("Large models:", cleanup_models["large_models"])

    # Test with a simple message
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    response, _ = client.chat(messages, model="llama-3.1-8b")
    print(f"\nResponse: {response}")

    print(f"\nModel info: {client.get_model_info()}")