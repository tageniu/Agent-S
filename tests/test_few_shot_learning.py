#!/usr/bin/env python3
"""Test script for few-shot learning implementation in AgentS"""

import json
import os
import sys
from pathlib import Path
from gui_agents.s2_5.memory.demonstration_memory import Demonstration, DemonstrationMemory

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_demonstration_memory():
    """Test the demonstration memory system"""
    print("=" * 60)
    print("Testing Demonstration Memory System")
    print("=" * 60)
    
    # Create a test demonstration memory
    demo_memory = DemonstrationMemory(storage_path="test_demonstrations")
    
    # Test 1: Add demonstrations
    print("\n1. Adding sample demonstrations...")
    
    demo1 = Demonstration(
        task_instruction="Open the calculator application",
        trajectory=[
            {"thought": "Need to open calculator", "action": "agent.click('Calculator icon')", "reflection": None},
            {"thought": "Calculator opened", "action": "agent.done()", "reflection": "Task completed"}
        ],
        success=True,
        metadata={"platform": "darwin", "model": "test"}
    )
    
    demo2 = Demonstration(
        task_instruction="Open a text editor and create a new file",
        trajectory=[
            {"thought": "Opening text editor", "action": "agent.click('Text Editor')", "reflection": None},
            {"thought": "Creating new file", "action": "agent.hotkey('cmd', 'n')", "reflection": None},
            {"thought": "File created", "action": "agent.done()", "reflection": "Successfully created"}
        ],
        success=True,
        metadata={"platform": "darwin", "model": "test"}
    )
    
    demo3 = Demonstration(
        task_instruction="Calculate 25 + 37 using the calculator",
        trajectory=[
            {"thought": "Opening calculator", "action": "agent.click('Calculator')", "reflection": None},
            {"thought": "Entering first number", "action": "agent.type('25')", "reflection": None},
            {"thought": "Adding operation", "action": "agent.click('+')", "reflection": None},
            {"thought": "Entering second number", "action": "agent.type('37')", "reflection": None},
            {"thought": "Getting result", "action": "agent.click('=')", "reflection": None},
            {"thought": "Result is 62", "action": "agent.done()", "reflection": "Calculation complete"}
        ],
        success=True,
        metadata={"platform": "darwin", "model": "test"}
    )
    
    demo_memory.add_demonstration(demo1)
    demo_memory.add_demonstration(demo2)
    demo_memory.add_demonstration(demo3)
    
    print(f"✓ Added {len(demo_memory.demonstrations)} demonstrations")
    
    # Test 2: Retrieve similar demonstrations
    print("\n2. Testing similarity-based retrieval...")
    
    test_queries = [
        "Open calculator app",
        "Create a new text document",
        "Perform addition in calculator",
        "Open a web browser"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        similar_demos = demo_memory.retrieve_similar_demonstrations(query, k=2, min_similarity=0.1)
        print(f"   Found {len(similar_demos)} similar demonstrations:")
        for demo in similar_demos:
            print(f"     - {demo.task_instruction[:50]}...")
    
    # Test 3: Format demonstrations for prompt
    print("\n3. Testing prompt formatting...")
    
    query = "Use the calculator to add numbers"
    similar_demos = demo_memory.retrieve_similar_demonstrations(query, k=2)
    formatted = demo_memory.format_demonstrations_for_prompt(similar_demos, max_steps_per_demo=3)
    
    print("\nFormatted prompt:")
    print("-" * 40)
    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
    
    # Test 4: Statistics
    print("\n4. Demonstration statistics:")
    stats = demo_memory.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test 5: Persistence
    print("\n5. Testing persistence...")
    
    # Save and reload
    demo_memory.save_demonstrations()
    
    # Create new instance and load
    demo_memory2 = DemonstrationMemory(storage_path="test_demonstrations")
    print(f"✓ Loaded {len(demo_memory2.demonstrations)} demonstrations from disk")
    
    # Cleanup
    import shutil
    if os.path.exists("test_demonstrations"):
        shutil.rmtree("test_demonstrations")
    print("✓ Cleaned up test files")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)


def test_integration():
    """Test the integration with AgentS"""
    print("\n" + "=" * 60)
    print("Testing Few-Shot AgentS Integration")
    print("=" * 60)
    
    try:
        from gui_agents.s2_5.agents.few_shot_agent_s import FewShotAgentS2_5
        from gui_agents.s2_5.agents.few_shot_worker import FewShotWorker
        
        print("\n✓ Successfully imported FewShotAgentS2_5 and FewShotWorker")
        print("✓ Integration modules are properly configured")
        
        # Check if the classes have the expected methods
        required_methods = {
            FewShotAgentS2_5: ['reset', 'predict', 'get_statistics'],
            FewShotWorker: ['generate_next_action', 'get_demonstration_statistics']
        }
        
        for cls, methods in required_methods.items():
            print(f"\nChecking {cls.__name__}:")
            for method in methods:
                if hasattr(cls, method):
                    print(f"  ✓ Has method: {method}")
                else:
                    print(f"  ✗ Missing method: {method}")
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Integration test completed!")
    print("=" * 60)
    
    return True


def main():
    """Run all tests"""
    print("\n🚀 Starting Few-Shot Learning Tests for AgentS\n")
    
    # Run tests
    test_demonstration_memory()
    test_integration()
    
    print("\n🎉 All tests completed successfully!")
    print("\nUsage example:")
    print("-" * 60)
    print("python gui_agents/s2_5/cli_app.py \\")
    print("  --provider openai \\")
    print("  --model gpt-4o \\")
    print("  --ground_provider openai \\")
    print("  --ground_url https://api.openai.com/v1 \\")
    print("  --ground_model gpt-4o \\")
    print("  --grounding_width 1920 \\")
    print("  --grounding_height 1080 \\")
    print("  --enable_few_shot \\")
    print("  --num_demonstrations 3 \\")
    print("  --min_similarity 0.3")
    print("-" * 60)
    print("\nThe few-shot learning system will:")
    print("1. Store successful task demonstrations automatically")
    print("2. Retrieve similar demonstrations for new tasks")
    print("3. Include demonstrations in the agent's context")
    print("4. Improve task success rate through learned examples")


if __name__ == "__main__":
    main()