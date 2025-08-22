#!/usr/bin/env python3
"""
Quick test script for GAIA benchmark integration.
This runs a single GAIA task to verify the setup works correctly.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path and change to project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv
load_dotenv()


def main():
    print("GAIA Benchmark Quick Test")
    print("=" * 50)
    
    # First, download the dataset if needed
    data_file = Path(__file__).parent / "data" / "gaia_2023_validation_level1.json"
    
    if not data_file.exists():
        print("\n📥 Downloading GAIA dataset...")
        from download_gaia import download_gaia_dataset
        download_gaia_dataset()
    else:
        print("\n✅ GAIA dataset already downloaded")
    
    # Run a single task as a test
    print("\n🚀 Running quick test with 1 task...")
    print("-" * 50)
    
    # Import runner after ensuring dataset exists
    from run_gaia_benchmark import main as run_benchmark
    
    # Set up test arguments
    test_args = [
        "--max_tasks", "1",
        "--provider", "openai",
        "--model", "gpt-5",
        "--temperature", "1.0",
        "--ground_provider", "open_router", 
        "--ground_model", "bytedance/ui-tars-1.5-7b",
        "--ground_url", "https://openrouter.ai/api/v1",
        "--max_steps", "20",
        "--enable_reflection"
    ]
    
    # Add arguments to sys.argv
    sys.argv = [sys.argv[0]] + test_args
    
    try:
        run_benchmark()
        print("\n✅ Quick test completed successfully!")
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())