#!/usr/bin/env python3
"""
Demo setup script for Agent-S
This demonstrates the basic setup without making actual API calls
"""
import os
import platform

print("Agent-S Setup Demo")
print("==================")

# 1. Check Python version
import sys
print(f"Python version: {sys.version}")
if sys.version_info < (3, 8):
    print("WARNING: Python 3.8+ is recommended")

# 2. Check platform
system = platform.system().lower()
if system == "darwin":
    current_platform = "darwin"
    print("Platform: macOS")
elif system == "windows":
    current_platform = "windows"
    print("Platform: Windows")
else:
    current_platform = "linux"
    print("Platform: Linux")

# 3. Check required packages
print("\nChecking installed packages...")
required_packages = [
    "gui_agents",
    "pyautogui",
    "openai",
    "anthropic",
    "pandas",
    "numpy",
    "scikit-learn"
]

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} (not installed)")

# 4. Check screen resolution
try:
    import pyautogui
    screen_width, screen_height = pyautogui.size()
    print(f"\nScreen resolution: {screen_width}x{screen_height}")
except Exception as e:
    print(f"\nCould not get screen resolution: {e}")

# 5. API Key setup instructions
print("\nAPI Key Configuration")
print("====================")
print("To use Agent-S, you'll need to set up API keys for the LLM providers.")
print("\nOption 1: Environment variables (add to ~/.bashrc or ~/.zshrc):")
print("  export OPENAI_API_KEY='your-openai-api-key-here'")
print("  export ANTHROPIC_API_KEY='your-anthropic-api-key-here'  # Optional")
print("  export HF_TOKEN='your-huggingface-token-here'  # Optional")

print("\nOption 2: Set in Python script:")
print("  import os")
print("  os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'")

# 6. Sample usage
print("\nSample Usage")
print("============")
print("Once you have your API keys set up, you can run Agent-S using:")
print("\n1. CLI mode:")
print("   agent_s2 --provider openai --model gpt-4o")
print("\n2. Python SDK:")
print("   python test_agent_s.py")

print("\nPerplexica Setup (Optional)")
print("===========================")
print("For web search capabilities, you can set up Perplexica:")
print("1. Install Docker Desktop")
print("2. cd Perplexica && git submodule update --init")
print("3. Configure Perplexica/config.toml")
print("4. docker compose up -d")
print("5. export PERPLEXICA_URL=http://localhost:3000/api/search")

print("\nSetup complete! Replace the API key placeholders in test_agent_s.py to start using Agent-S.")