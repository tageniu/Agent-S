import logging
import platform
from typing import Dict, List, Tuple

from gui_agents.s2_5.agents.few_shot_worker import FewShotWorker
from gui_agents.s2_5.agents.grounding import ACI

logger = logging.getLogger("desktopenv.agent")


class FewShotAgentS2_5:
    """AgentS2.5 enhanced with few-shot learning from demonstrations"""
    
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        enable_few_shot: bool = True,
        demonstration_path: str = "demonstrations",
        num_demonstrations: int = 3,
        min_similarity: float = 0.3,
    ):
        """Initialize FewShotAgentS2_5
        
        Args:
            engine_params: Configuration parameters for the LLM engine
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (darwin, linux, windows)
            max_trajectory_length: Maximum number of image turns to keep
            enable_reflection: Whether to enable reflection agent
            enable_few_shot: Whether to enable few-shot learning
            demonstration_path: Path to store/load demonstrations
            num_demonstrations: Number of demonstrations to retrieve for each task
            min_similarity: Minimum similarity threshold for demonstration retrieval
        """
        self.engine_params = engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.enable_few_shot = enable_few_shot
        self.demonstration_path = demonstration_path
        self.num_demonstrations = num_demonstrations
        self.min_similarity = min_similarity
        
        self.reset()
        
        if self.enable_few_shot:
            stats = self.executor.get_demonstration_statistics()
            logger.info(
                f"Few-shot learning initialized with {stats.get('total_demonstrations', 0)} demonstrations "
                f"(success rate: {stats.get('success_rate', 0):.1%})"
            )
    
    def reset(self) -> None:
        """Reset agent state and initialize components"""
        self.executor = FewShotWorker(
            engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            platform=self.platform,
            max_trajectory_length=self.max_trajectory_length,
            enable_reflection=self.enable_reflection,
            enable_few_shot=self.enable_few_shot,
            demonstration_path=self.demonstration_path,
            num_demonstrations=self.num_demonstrations,
            min_similarity=self.min_similarity,
        )
    
    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Generate next action prediction with few-shot learning
        
        Args:
            instruction: Natural language instruction
            observation: Current UI state observation
            
        Returns:
            Tuple containing agent info dictionary and list of actions
        """
        executor_info, actions = self.executor.generate_next_action(
            instruction=instruction, obs=observation
        )
        
        # Consolidate info
        info = {**executor_info} if executor_info else {}
        
        # Add few-shot statistics to info if enabled
        if self.enable_few_shot:
            info["few_shot_enabled"] = True
            stats = self.executor.get_demonstration_statistics()
            info["demonstration_count"] = stats.get("total_demonstrations", 0)
        
        return info, actions
    
    def get_statistics(self) -> Dict:
        """Get agent statistics including few-shot learning metrics
        
        Returns:
            Dictionary with agent statistics
        """
        stats = {
            "platform": self.platform,
            "reflection_enabled": self.enable_reflection,
            "few_shot_enabled": self.enable_few_shot,
        }
        
        if self.enable_few_shot:
            demo_stats = self.executor.get_demonstration_statistics()
            stats.update(demo_stats)
        
        return stats