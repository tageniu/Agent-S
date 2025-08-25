import logging
import platform
from typing import Dict, List, Tuple

from gui_agents.s2_5.agents.grounding import ACI
from gui_agents.s2_5.agents.worker import Worker
from gui_agents.s2_5.agents.coding_agent import CodingAgent
from gui_agents.s2_5.core.module import BaseModule

logger = logging.getLogger("desktopenv.agent")


class HybridAgent(BaseModule):
    """A hybrid agent that can switch between GUI operations and code execution"""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
    ):
        """Initialize HybridAgent

        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (darwin, linux, windows)
            max_trajectory_length: Maximum number of image turns to keep
            enable_reflection: Creates a reflection agent to assist the worker agent
        """
        super().__init__(engine_params, platform)
        self.engine_params = engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.reset()

    def reset(self) -> None:
        """Reset agent state and initialize components"""
        # Initialize GUI worker
        self.gui_worker = Worker(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            platform=self.platform,
            max_trajectory_length=self.max_trajectory_length,
            enable_reflection=self.enable_reflection,
        )
        
        # Initialize coding agent
        self.coding_agent = CodingAgent(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            platform=self.platform,
            max_trajectory_length=self.max_trajectory_length,
            enable_reflection=self.enable_reflection,
        )
        
        self.mode = "gui"  # Default mode is GUI
        self.turn_count = 0

    def _select_mode(self, instruction: str, observation: Dict) -> str:
        """Select the appropriate mode (GUI or Coding) based on the task"""
        # For now, we'll use a simple heuristic
        # In a more advanced implementation, we could use an LLM to decide
        
        # Keywords that suggest coding might be needed
        coding_keywords = [
            "script", "code", "program", "calculate", "process data", 
            "analyze", "file", "API", "web", "scrape", "automation",
            "math", "convert", "transform", "sort", "filter", "data analysis",
            "parse", "extract", "download", "upload", "backup"
        ]
        
        instruction_lower = instruction.lower()
        for keyword in coding_keywords:
            if keyword in instruction_lower:
                return "coding"
        
        # Default to GUI for most tasks
        return "gui"

    def _execute_gui_mode(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Execute the task using GUI operations"""
        return self.gui_worker.generate_next_action(instruction, observation)

    def _execute_coding_mode(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Execute the task using code execution"""
        return self.coding_agent.generate_next_action(instruction, observation)

    def generate_next_action(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Generate next action prediction, switching between modes as needed"""
        # Select the appropriate mode
        selected_mode = self._select_mode(instruction, observation)
        self.mode = selected_mode
        
        logger.info(f"Selected mode: {selected_mode}")
        
        # Execute based on selected mode
        if selected_mode == "gui":
            return self._execute_gui_mode(instruction, observation)
        elif selected_mode == "coding":
            return self._execute_coding_mode(instruction, observation)
        else:
            # Fallback to GUI mode
            return self._execute_gui_mode(instruction, observation)