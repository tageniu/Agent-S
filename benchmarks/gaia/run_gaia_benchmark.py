#!/usr/bin/env python3
"""
Run GAIA benchmark with Agent-S.
This script runs GAIA 2023 validation level 1 tasks using Agent-S.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # Change to project root for imports

from gui_agents.s2_5.agents.agent_s import AgentS2_5
from gui_agents.s2_5.agents.grounding import OSWorldACI
from benchmarks.gaia.scorer import question_scorer


class GAIABenchmarkRunner:
    """Runner for GAIA benchmark tasks with Agent-S."""
    
    def __init__(self, args):
        self.args = args
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.setup_logging()
        
        # Load tasks
        self.tasks = self.load_tasks()
        
        # Initialize agent
        self.agent = self.initialize_agent()
        
    def setup_logging(self):
        """Set up logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.results_dir / f"gaia_run_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load GAIA tasks from JSON file."""
        data_file = Path(__file__).parent / "data" / "gaia_2023_validation_level1.json"
        
        if not data_file.exists():
            self.logger.error(f"Data file not found: {data_file}")
            self.logger.info("Please run download_gaia.py first to download the dataset")
            sys.exit(1)
        
        with open(data_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        
        # Apply task limit if specified
        if self.args.max_tasks:
            tasks = tasks[:self.args.max_tasks]
        
        # Filter by task IDs if specified
        if self.args.task_ids:
            task_id_set = set(self.args.task_ids.split(","))
            tasks = [t for t in tasks if t["task_id"] in task_id_set]
        
        self.logger.info(f"Loaded {len(tasks)} tasks")
        return tasks
    
    def initialize_agent(self) -> AgentS2_5:
        """Initialize the Agent-S instance with configuration."""
        self.logger.info("Initializing Agent-S...")
        
        import platform
        current_platform = platform.system().lower()
        
        # Create engine parameters for main model
        engine_params = {
            "engine_type": self.args.provider,
            "model": self.args.model,
            "temperature": self.args.temperature,
            "api_key": self.args.model_api_key if self.args.model_api_key else None,
        }
        if self.args.model_url:
            engine_params["base_url"] = self.args.model_url
        
        # Create engine parameters for grounding model
        engine_params_for_grounding = {
            "engine_type": self.args.ground_provider,
            "model": self.args.ground_model,
            "base_url": self.args.ground_url,
            "api_key": self.args.ground_api_key if self.args.ground_api_key else None,
            "grounding_width": self.args.grounding_width,
            "grounding_height": self.args.grounding_height,
        }
        
        # Initialize ACI (Agent-Computer Interface)
        aci = OSWorldACI(
            platform=current_platform,
            engine_params_for_generation=engine_params,
            engine_params_for_grounding=engine_params_for_grounding,
            width=self.args.grounding_width,
            height=self.args.grounding_height
        )
        
        # Initialize Agent-S
        agent = AgentS2_5(
            engine_params,
            aci,
            platform=current_platform,
            max_trajectory_length=self.args.max_trajectory_length,
            enable_reflection=self.args.enable_reflection,
        )
        
        self.logger.info("Agent-S initialized successfully")
        return agent
    
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single GAIA task."""
        import io
        import pyautogui
        from PIL import Image
        
        task_id = task["task_id"]
        question = task["question"]
        expected_answer = task.get("final_answer", "")
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running task: {task_id}")
        self.logger.info(f"Question: {question}")
        self.logger.info(f"{'='*60}\n")
        
        start_time = time.time()
        trajectory = []
        agent_answer = ""
        
        try:
            obs = {}
            for step in range(self.args.max_steps):
                # Get screenshot
                screenshot = pyautogui.screenshot()
                screenshot = screenshot.resize(
                    (self.args.grounding_width, self.args.grounding_height), 
                    Image.LANCZOS
                )
                
                # Save screenshot to bytes
                buffered = io.BytesIO()
                screenshot.save(buffered, format="PNG")
                screenshot_bytes = buffered.getvalue()
                obs["screenshot"] = screenshot_bytes
                
                self.logger.info(f"Step {step + 1}/{self.args.max_steps}: Getting next action...")
                
                # Get next action from agent
                info, code = self.agent.predict(instruction=question, observation=obs)
                
                # Log the action
                if code:
                    self.logger.info(f"Action: {code[:100]}...")  # Log first 100 chars
                    trajectory.append({
                        "step": step + 1,
                        "action": code,
                        "info": info
                    })
                
                # Check if task is complete (simplified check)
                if info and info.get("status") == "completed":
                    agent_answer = info.get("answer", "")
                    break
                    
                # Execute the action if provided
                if code:
                    # Check if this is a DONE or FAIL action (returned by agent.done() or agent.fail())
                    if isinstance(code, list) and len(code) == 1 and code[0] in ["DONE", "FAIL"]:
                        if code[0] == "DONE":
                            # Extract answer from info if available
                            if hasattr(self.agent.executor.grounding_agent, 'returned_info'):
                                agent_answer = str(self.agent.executor.grounding_agent.returned_info)
                                self.logger.info(f"Task completed with answer: {agent_answer}")
                            else:
                                agent_answer = ""
                                self.logger.info("Task marked as complete")
                            info = {"status": "completed", "answer": agent_answer}
                            break
                        else:  # FAIL
                            self.logger.info("Task marked as failed")
                            info = {"status": "failed"}
                            break
                    elif isinstance(code, str) and code in ["DONE", "FAIL"]:
                        if code == "DONE":
                            # Extract answer from info if available
                            if hasattr(self.agent.executor.grounding_agent, 'returned_info'):
                                agent_answer = str(self.agent.executor.grounding_agent.returned_info)
                                self.logger.info(f"Task completed with answer: {agent_answer}")
                            else:
                                agent_answer = ""
                                self.logger.info("Task marked as complete")
                            info = {"status": "completed", "answer": agent_answer}
                            break
                        else:  # FAIL
                            self.logger.info("Task marked as failed")
                            info = {"status": "failed"}
                            break
                    else:
                        # Execute normal actions
                        try:
                            # Create a namespace with the agent object for execution context
                            exec_namespace = {
                                'agent': self.agent.executor.grounding_agent,
                                'pyautogui': pyautogui,
                                'time': time,
                                '__builtins__': __builtins__
                            }
                            
                            # Handle both string and list of strings
                            if isinstance(code, list):
                                for action in code:
                                    # Check if this action contains agent.done() or agent.fail()
                                    if "agent.done(" in str(action) or "agent.fail(" in str(action):
                                        # Execute in namespace with agent object
                                        exec(action, exec_namespace)
                                        # Check if it was done or fail
                                        if "agent.done(" in str(action):
                                            # Extract answer from grounding agent
                                            if hasattr(self.agent.executor.grounding_agent, 'returned_info'):
                                                agent_answer = str(self.agent.executor.grounding_agent.returned_info)
                                                self.logger.info(f"Task completed with answer: {agent_answer}")
                                            else:
                                                agent_answer = ""
                                                self.logger.info("Task marked as complete")
                                            info = {"status": "completed", "answer": agent_answer}
                                            break
                                        elif "agent.fail(" in str(action):
                                            self.logger.info("Task marked as failed")
                                            info = {"status": "failed"}
                                            break
                                    else:
                                        exec(action, exec_namespace)
                                        time.sleep(0.5)  # Small delay after each action
                            else:
                                # Check if this action contains agent.done() or agent.fail()
                                if "agent.done(" in str(code) or "agent.fail(" in str(code):
                                    # Execute in namespace with agent object
                                    exec(code, exec_namespace)
                                    # Check if it was done or fail
                                    if "agent.done(" in str(code):
                                        # Extract answer from grounding agent
                                        if hasattr(self.agent.executor.grounding_agent, 'returned_info'):
                                            agent_answer = str(self.agent.executor.grounding_agent.returned_info)
                                            self.logger.info(f"Task completed with answer: {agent_answer}")
                                        else:
                                            agent_answer = ""
                                            self.logger.info("Task marked as complete")
                                        info = {"status": "completed", "answer": agent_answer}
                                        break
                                    elif "agent.fail(" in str(code):
                                        self.logger.info("Task marked as failed")
                                        info = {"status": "failed"}
                                        break
                                else:
                                    exec(code, exec_namespace)
                                    time.sleep(0.5)  # Small delay after action
                        except Exception as e:
                            self.logger.warning(f"Error executing action: {e}")
            
            # Check if answer matches expected
            is_correct = self.check_answer(agent_answer, expected_answer)
            
            execution_time = time.time() - start_time
            
            task_result = {
                "task_id": task_id,
                "question": question,
                "expected_answer": expected_answer,
                "agent_answer": agent_answer,
                "is_correct": is_correct,
                "execution_time": execution_time,
                "trajectory": trajectory,
                "status": "completed"
            }
            
            self.logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            self.logger.info(f"Agent answer: {agent_answer}")
            self.logger.info(f"Correct: {is_correct}")
            
        except Exception as e:
            self.logger.error(f"Error running task {task_id}: {str(e)}")
            task_result = {
                "task_id": task_id,
                "question": question,
                "expected_answer": expected_answer,
                "agent_answer": None,
                "is_correct": False,
                "execution_time": time.time() - start_time,
                "error": str(e),
                "status": "failed"
            }
        
        return task_result
    
    def check_answer(self, agent_answer: str, expected_answer: str) -> bool:
        """Evaluate correctness using the official GAIA scorer."""
        if not expected_answer:
            return False
        return bool(question_scorer(str(agent_answer) if agent_answer is not None else "", str(expected_answer)))
    
    def run_benchmark(self):
        """Run the full benchmark suite."""
        self.logger.info(f"\nStarting GAIA benchmark run with {len(self.tasks)} tasks")
        
        results = []
        correct_count = 0
        
        for i, task in enumerate(self.tasks, 1):
            self.logger.info(f"\nProgress: {i}/{len(self.tasks)}")
            
            # Run the task
            task_result = self.run_task(task)
            results.append(task_result)
            
            if task_result["is_correct"]:
                correct_count += 1
            
            # Save intermediate results
            self.save_results(results)
            
            # Add delay between tasks if specified
            if self.args.delay_between_tasks > 0 and i < len(self.tasks):
                time.sleep(self.args.delay_between_tasks)
        
        # Calculate final metrics
        accuracy = correct_count / len(results) if results else 0
        avg_time = sum(r["execution_time"] for r in results) / len(results) if results else 0
        
        # Print summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("BENCHMARK COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total tasks: {len(results)}")
        self.logger.info(f"Correct: {correct_count}")
        self.logger.info(f"Accuracy: {accuracy:.2%}")
        self.logger.info(f"Average time per task: {avg_time:.2f}s")
        
        # Save final results with summary
        self.save_results(results, include_summary=True)
        
    def save_results(self, results: List[Dict[str, Any]], include_summary: bool = False):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"gaia_results_{timestamp}.json"
        
        output_data = {
            "benchmark": "GAIA 2023 Validation Level 1",
            "timestamp": timestamp,
            "configuration": {
                "model": self.args.model,
                "provider": self.args.provider,
                "temperature": self.args.temperature,
                "ground_model": self.args.ground_model,
                "enable_reflection": self.args.enable_reflection,
                "max_steps": self.args.max_steps,
            },
            "results": results
        }
        
        if include_summary:
            correct_count = sum(1 for r in results if r.get("is_correct", False))
            output_data["summary"] = {
                "total_tasks": len(results),
                "correct": correct_count,
                "accuracy": correct_count / len(results) if results else 0,
                "average_time": sum(r["execution_time"] for r in results) / len(results) if results else 0
            }
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run GAIA benchmark with Agent-S")
    
    # Benchmark configuration
    parser.add_argument("--max_tasks", type=int, help="Maximum number of tasks to run")
    parser.add_argument("--task_ids", type=str, help="Comma-separated list of specific task IDs to run")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps per task")
    parser.add_argument("--delay_between_tasks", type=int, default=2, help="Delay in seconds between tasks")
    
    # Model configuration
    parser.add_argument("--provider", type=str, default="openai", help="Main model provider")
    parser.add_argument("--model", type=str, default="gpt-5", help="Main model name")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for model generation")
    parser.add_argument("--model_url", type=str, help="Custom model API URL")
    parser.add_argument("--model_api_key", type=str, help="Custom model API key")
    
    # Grounding model configuration
    parser.add_argument("--ground_provider", type=str, default="open_router", help="Grounding model provider")
    parser.add_argument("--ground_model", type=str, default="bytedance/ui-tars-1.5-7b", help="Grounding model name")
    parser.add_argument("--ground_url", type=str, default="https://openrouter.ai/api/v1", help="Grounding model URL")
    parser.add_argument("--ground_api_key", type=str, help="Grounding model API key")
    parser.add_argument("--grounding_width", type=int, default=1728, help="Grounding image width")
    parser.add_argument("--grounding_height", type=int, default=1117, help="Grounding image height")
    
    # Agent configuration
    parser.add_argument("--enable_reflection", action="store_true", default=True, help="Enable reflection agent")
    parser.add_argument("--max_trajectory_length", type=int, default=8, help="Maximum trajectory length")
    
    args = parser.parse_args()
    
    # Create and run benchmark
    runner = GAIABenchmarkRunner(args)
    runner.run_benchmark()


if __name__ == "__main__":
    main()