import logging
import platform
from typing import Dict, List, Tuple

from gui_agents.s2_5.agents.grounding import ACI
from gui_agents.s2_5.agents.worker import Worker
from gui_agents.s2_5.agents.hybrid_agent import HybridAgent
from gui_agents.s2_5.agents.coding_agent import CodingAgent

logger = logging.getLogger("desktopenv.agent")


class UIAgent:
    """Base class for UI automation agents"""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
    ):
        """Initialize UIAgent

        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (macos, linux, windows)
        """
        self.engine_params = engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform

    def reset(self) -> None:
        """Reset agent state"""
        pass

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Generate next action prediction

        Args:
            instruction: Natural language instruction
            observation: Current UI state observation

        Returns:
            Tuple containing agent info dictionary and list of actions
        """
        pass


class AgentS2_5(UIAgent):
    """Agent that uses no hierarchy for less inference time"""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
    agent_mode: str = "gui",  # New parameter to select agent mode
    ):
        """Initialize a minimalist AgentS2 without hierarchy

        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (darwin, linux, windows)
            max_trajectory_length: Maximum number of image turns to keep
            enable_reflection: Creates a reflection agent to assist the worker agent
            agent_mode: Agent mode to use ("gui", "hybrid", "coding")
        """

        super().__init__(engine_params, grounding_agent, platform)
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.agent_mode = agent_mode
        self.reset()

    def reset(self) -> None:
        """Reset agent state and initialize components"""
        if self.agent_mode == "hybrid":
            self.executor = HybridAgent(
                engine_params=self.engine_params,
                grounding_agent=self.grounding_agent,
                platform=self.platform,
                max_trajectory_length=self.max_trajectory_length,
                enable_reflection=self.enable_reflection,
            )
        elif self.agent_mode == "coding":
            self.executor = CodingAgent(
                engine_params=self.engine_params,
                grounding_agent=self.grounding_agent,
                platform=self.platform,
                max_trajectory_length=self.max_trajectory_length,
                enable_reflection=self.enable_reflection,
            )
        else:
            # Default to GUI worker
            self.executor = Worker(
                engine_params=self.engine_params,
                grounding_agent=self.grounding_agent,
                platform=self.platform,
                max_trajectory_length=self.max_trajectory_length,
                enable_reflection=self.enable_reflection,
            )

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        # Initialize the three info dictionaries
        executor_info, actions = self.executor.generate_next_action(
            instruction=instruction, obs=observation
        )

        # concatenate the three info dictionaries
        info = {**{k: v for d in [executor_info or {}] for k, v in d.items()}}

        return info, actions
