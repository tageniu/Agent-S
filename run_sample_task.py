#!/usr/bin/env python3
"""
Run a sample task with Agent-S
This script demonstrates how to run Agent-S with configurable model settings.

Model Configuration:
- Main LLM: Configurable via --provider and --model (default: OpenAI gpt-4o)
- Grounding Model: Configurable via --ground_provider and --ground_model 
- Grounding Dimensions: Set via --grounding_width and --grounding_height

Supported Providers:
- Main LLM: openai, anthropic, azure, gemini, open_router, vllm, huggingface
- Grounding: openai, anthropic, open_router, vllm, huggingface
"""
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ================================
# MODEL CONFIGURATION SETTINGS
# ================================
# Modify these settings to customize your model configuration

# Main LLM Configuration
MAIN_PROVIDER = "openai"           # Options: openai, anthropic, azure, gemini, open_router, vllm, huggingface
MAIN_MODEL = "gpt-4o"              # Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022, o3-2025-04-16)
MAIN_MODEL_URL = ""                # Custom API URL (optional)
MAIN_API_KEY = ""                  # Custom API key (optional, uses env vars if empty)

# Grounding Model Configuration  
GROUND_PROVIDER = "open_router"    # Options: openai, anthropic, open_router, vllm, huggingface
GROUND_URL = "https://openrouter.ai/api/v1"  # Grounding model endpoint URL
GROUND_MODEL = "bytedance/ui-tars-1.5-7b"  # Grounding model name
GROUND_API_KEY = ""                # Custom API key (optional, uses env vars if empty)

# Grounding Model Dimensions
GROUNDING_WIDTH = 1728
GROUNDING_HEIGHT = 1117

# Agent Configuration
MAX_TRAJECTORY_LENGTH = 8          # Maximum number of image turns to keep
ENABLE_REFLECTION = True           # Enable reflection agent

# ================================
# END CONFIGURATION SETTINGS
# ================================

print("Agent-S Sample Task Runner")
print("=========================\n")

# Configure default model settings
print("Configuring model and grounding model settings...")

# Set default arguments for the CLI
import sys
default_args = [
    # Main model configuration
    "--provider", MAIN_PROVIDER,
    "--model", MAIN_MODEL,
    # Grounding model configuration
    "--ground_provider", GROUND_PROVIDER,
    "--ground_url", GROUND_URL,
    "--ground_model", GROUND_MODEL,
    "--grounding_width", str(GROUNDING_WIDTH),
    "--grounding_height", str(GROUNDING_HEIGHT),
    # Agent configuration
    "--max_trajectory_length", str(MAX_TRAJECTORY_LENGTH),
]

# Add optional URL and API key arguments if specified
if MAIN_MODEL_URL:
    default_args.extend(["--model_url", MAIN_MODEL_URL])
if MAIN_API_KEY:
    default_args.extend(["--model_api_key", MAIN_API_KEY])
if GROUND_API_KEY:
    default_args.extend(["--ground_api_key", GROUND_API_KEY])

# Add reflection flag if enabled
if ENABLE_REFLECTION:
    default_args.append("--enable_reflection")

# Add default arguments to sys.argv if not already provided
for i in range(0, len(default_args), 2):
    arg_name = default_args[i]
    if i + 1 < len(default_args):
        arg_value = default_args[i + 1]
        # Only add if not already in command line arguments
        if arg_name not in sys.argv:
            sys.argv.extend([arg_name, arg_value])
    else:
        # Handle flags (arguments without values)
        if arg_name not in sys.argv:
            sys.argv.append(arg_name)

print(f"Model settings configured:")
print(f"  - Main model: {MAIN_MODEL} ({MAIN_PROVIDER})")
print(f"  - Grounding model: {GROUND_MODEL} ({GROUND_PROVIDER})")
print(f"  - Grounding dimensions: {GROUNDING_WIDTH}x{GROUNDING_HEIGHT}")
print(f"  - Max trajectory length: {MAX_TRAJECTORY_LENGTH}")
print(f"  - Reflection enabled: {ENABLE_REFLECTION}\n")

# Import and run
try:
    from gui_agents.s2_5.cli_app import main
    main()
except Exception as e:
    print(f"Error running Agent-S: {e}")
    print("\nMake sure you're in the virtual environment:")
    print("source .venv/bin/activate")