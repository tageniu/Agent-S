#!/usr/bin/env python3
"""
Run a sample task with Agent-S
This script demonstrates how to run Agent-S with placeholder API keys
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("Agent-S Sample Task Runner")
print("=========================\n")

# Check if API key is set
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
    print("⚠️  OpenAI API key not set!")
    print("\nTo run Agent-S, you need to set your OpenAI API key.")
    print("Please update the OPENAI_API_KEY in the .env file with your actual API key.")
    print("\nThe .env file has been created with placeholders for all supported API keys.")
    print("\nOnce you have your API key, you can run:")
    print("- agent_s2  # For interactive CLI mode")
    sys.exit(0)

# If API key is set, run the agent
print("API key detected. Starting Agent-S...\n")

# Import and run
try:
    from gui_agents.s2.cli_app import main
    main()
except Exception as e:
    print(f"Error running Agent-S: {e}")
    print("\nMake sure you're in the virtual environment:")
    print("source .venv/bin/activate")