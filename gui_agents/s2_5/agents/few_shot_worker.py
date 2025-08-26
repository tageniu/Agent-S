import logging
import textwrap
from typing import Dict, List, Optional, Tuple

from gui_agents.s2_5.agents.grounding import ACI
from gui_agents.s2_5.agents.worker import Worker
from gui_agents.s2_5.memory.demonstration_memory import DemonstrationMemory
from gui_agents.s2_5.memory.procedural_memory import PROCEDURAL_MEMORY
from gui_agents.s2_5.utils.common_utils import (
    call_llm_safe,
    extract_first_agent_function,
    parse_single_code_from_string,
    sanitize_code,
    split_thinking_response,
)

logger = logging.getLogger("desktopenv.agent")


class FewShotWorker(Worker):
    """Worker enhanced with few-shot learning from demonstrations"""
    
    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        enable_few_shot: bool = True,
        demonstration_path: str = "demonstrations",
        num_demonstrations: int = 3,
        min_similarity: float = 0.3,
    ):
        """Initialize FewShotWorker with demonstration learning capabilities
        
        Args:
            engine_params: Parameters for the multimodal engine
            grounding_agent: The grounding agent to use
            platform: OS platform (darwin, linux, windows)
            max_trajectory_length: Amount of images turns to keep
            enable_reflection: Whether to enable reflection
            enable_few_shot: Whether to enable few-shot learning
            demonstration_path: Path to store/load demonstrations
            num_demonstrations: Number of demonstrations to retrieve
            min_similarity: Minimum similarity for demonstration retrieval
        """
        super().__init__(
            engine_params=engine_params,
            grounding_agent=grounding_agent,
            platform=platform,
            max_trajectory_length=max_trajectory_length,
            enable_reflection=enable_reflection,
        )
        
        self.enable_few_shot = enable_few_shot
        self.num_demonstrations = num_demonstrations
        self.min_similarity = min_similarity
        
        if self.enable_few_shot:
            self.demo_memory = DemonstrationMemory(storage_path=demonstration_path)
            logger.info(f"Few-shot learning enabled with {self.demo_memory.get_statistics()['total_demonstrations']} demonstrations")
        else:
            self.demo_memory = None
        
        self.current_task_instruction = None
        self.current_trajectory = []
    
    def reset(self):
        """Reset agent state and initialize components"""
        super().reset()
        self.current_task_instruction = None
        self.current_trajectory = []
    
    def _enhance_prompt_with_demonstrations(self, instruction: str) -> str:
        """Enhance the system prompt with relevant demonstrations
        
        Args:
            instruction: Current task instruction
            
        Returns:
            Enhanced prompt with demonstrations
        """
        if not self.enable_few_shot or not self.demo_memory:
            return ""
        
        # Retrieve similar demonstrations
        similar_demos = self.demo_memory.retrieve_similar_demonstrations(
            task_instruction=instruction,
            k=self.num_demonstrations,
            min_similarity=self.min_similarity
        )
        
        if not similar_demos:
            logger.info("No similar demonstrations found")
            return ""
        
        # Format demonstrations for prompt
        demo_prompt = self.demo_memory.format_demonstrations_for_prompt(
            demonstrations=similar_demos,
            max_steps_per_demo=5
        )
        
        return demo_prompt
    
    def generate_next_action(
        self,
        instruction: str,
        obs: Dict,
    ) -> Tuple[Dict, List]:
        """Generate next action with few-shot learning enhancement
        
        Args:
            instruction: Task instruction
            obs: Current observation
            
        Returns:
            Tuple of executor info and actions
        """
        agent = self.grounding_agent
        
        # Store current task instruction on first turn
        if self.turn_count == 0:
            self.current_task_instruction = instruction
            
            # Get demonstrations and enhance system prompt
            demo_prompt = self._enhance_prompt_with_demonstrations(instruction)
            
            # Load the task into the system prompt with demonstrations
            enhanced_prompt = self.generator_agent.system_prompt.replace(
                "TASK_DESCRIPTION", instruction
            )
            
            if demo_prompt:
                # Insert demonstrations before the response format instructions
                enhanced_prompt = enhanced_prompt.replace(
                    "Your response should be formatted like this:",
                    f"{demo_prompt}\n\nYour response should be formatted like this:"
                )
                logger.info("Enhanced prompt with few-shot demonstrations")
            
            self.generator_agent.add_system_prompt(enhanced_prompt)
        else:
            # Standard prompt update for subsequent turns
            if self.turn_count == 1:  # Only update once after first turn
                self.generator_agent.system_prompt = self.generator_agent.system_prompt.replace(
                    "TASK_DESCRIPTION", instruction
                )
        
        generator_message = (
            ""
            if self.turn_count > 0
            else "The initial screen is provided. No action has been taken yet."
        )
        
        # Get per-step reflection (unchanged from parent)
        reflection = None
        reflection_thoughts = None
        if self.enable_reflection:
            if self.turn_count == 0:
                text_content = textwrap.dedent(
                    f"""
                    Task Description: {instruction}
                    Current Trajectory below:
                    """
                )
                updated_sys_prompt = (
                    self.reflection_agent.system_prompt + "\n" + text_content
                )
                self.reflection_agent.add_system_prompt(updated_sys_prompt)
                self.reflection_agent.add_message(
                    text_content="The initial screen is provided. No action has been taken yet.",
                    image_content=obs["screenshot"],
                    role="user",
                )
            else:
                self.reflection_agent.add_message(
                    text_content=self.worker_history[-1],
                    image_content=obs["screenshot"],
                    role="user",
                )
                full_reflection = call_llm_safe(
                    self.reflection_agent,
                    temperature=self.temperature,
                    use_thinking=self.use_thinking,
                )
                reflection, reflection_thoughts = split_thinking_response(
                    full_reflection
                )
                self.reflections.append(reflection)
                generator_message += f"REFLECTION: You may use this reflection on the previous action and overall trajectory:\n{reflection}\n"
                logger.info("REFLECTION: %s", reflection)
        
        # Add finalized message to conversation
        generator_message += f"\nCurrent Text Buffer = [{','.join(agent.notes)}]\n"
        self.generator_agent.add_message(
            generator_message, image_content=obs["screenshot"], role="user"
        )
        
        # Generate plan
        full_plan = call_llm_safe(
            self.generator_agent,
            temperature=self.temperature,
            use_thinking=self.use_thinking,
        )
        plan, plan_thoughts = split_thinking_response(full_plan)
        self.worker_history.append(plan)
        logger.info("FULL PLAN:\n %s", full_plan)
        self.generator_agent.add_message(plan, role="assistant")
        
        # Parse and execute action
        try:
            agent.assign_coordinates(plan, obs)
            plan_code = parse_single_code_from_string(plan.split("Grounded Action")[-1])
            plan_code = sanitize_code(plan_code)
            plan_code = extract_first_agent_function(plan_code)
            exec_code = eval(plan_code)
            
            # Track trajectory for potential demonstration recording
            self.current_trajectory.append({
                "thought": plan.split("(Next Action)")[1].split("(Grounded Action)")[0].strip() if "(Next Action)" in plan else "",
                "action": plan_code,
                "reflection": reflection,
            })
            
        except Exception as e:
            logger.error("Error in parsing plan code: %s", e)
            plan_code = "agent.wait(1.0)"
            exec_code = eval(plan_code)
            
            self.current_trajectory.append({
                "thought": "Error in action parsing",
                "action": plan_code,
                "reflection": reflection,
            })
        
        executor_info = {
            "full_plan": full_plan,
            "executor_plan": plan,
            "plan_thoughts": plan_thoughts,
            "plan_code": plan_code,
            "reflection": reflection,
            "reflection_thoughts": reflection_thoughts,
        }
        
        # Check if task is completed to record demonstration
        if "agent.done()" in plan_code and self.enable_few_shot and self.demo_memory:
            self._record_demonstration(success=True)
        elif "agent.fail()" in plan_code and self.enable_few_shot and self.demo_memory:
            self._record_demonstration(success=False)
        
        self.turn_count += 1
        self.screenshot_inputs.append(obs["screenshot"])
        self.flush_messages()
        
        return executor_info, [exec_code]
    
    def _record_demonstration(self, success: bool):
        """Record current trajectory as a demonstration
        
        Args:
            success: Whether the task was completed successfully
        """
        if not self.current_task_instruction or not self.current_trajectory:
            return
        
        self.demo_memory.record_trajectory(
            task_instruction=self.current_task_instruction,
            trajectory=self.current_trajectory,
            success=success,
            metadata={
                "platform": self.platform,
                "turn_count": self.turn_count,
                "model": self.engine_params.get("model", "unknown"),
            }
        )
        
        logger.info(f"Recorded demonstration: success={success}, steps={len(self.current_trajectory)}")
    
    def get_demonstration_statistics(self) -> Dict:
        """Get statistics about stored demonstrations
        
        Returns:
            Dictionary with demonstration statistics
        """
        if not self.enable_few_shot or not self.demo_memory:
            return {"few_shot_enabled": False}
        
        stats = self.demo_memory.get_statistics()
        stats["few_shot_enabled"] = True
        return stats