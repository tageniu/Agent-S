# Few-Shot Learning for AgentS

## Overview

This implementation adds few-shot learning capabilities to AgentS, enabling the agent to learn from human demonstrations and improve task accuracy. The system automatically stores successful task trajectories and retrieves similar demonstrations to guide future task execution.

## Key Features

- **In-Context Learning**: Retrieves and includes relevant demonstrations in the agent's prompt
- **Automatic Trajectory Recording**: Stores successful and failed task executions as demonstrations
- **Similarity-Based Retrieval**: Finds the most relevant demonstrations based on task similarity
- **Persistent Storage**: Saves demonstrations to disk for reuse across sessions
- **Seamless Integration**: Works with existing AgentS2.5 architecture without disrupting core functionality

## Architecture

### Components

1. **DemonstrationMemory** (`gui_agents/s2_5/memory/demonstration_memory.py`)
   - Manages storage and retrieval of demonstration trajectories
   - Computes task similarity for retrieval
   - Formats demonstrations for LLM prompts

2. **FewShotWorker** (`gui_agents/s2_5/agents/few_shot_worker.py`)
   - Extends the base Worker class with demonstration capabilities
   - Enhances prompts with retrieved demonstrations
   - Records new trajectories after task completion

3. **FewShotAgentS2_5** (`gui_agents/s2_5/agents/few_shot_agent_s.py`)
   - Main agent class with few-shot learning
   - Coordinates between worker and demonstration memory
   - Provides statistics and monitoring

## Usage

### Command Line

Enable few-shot learning with the `--enable_few_shot` flag:

```bash
python gui_agents/s2_5/cli_app.py \
  --provider openai \
  --model gpt-4o \
  --ground_provider openai \
  --ground_url https://api.openai.com/v1 \
  --ground_model gpt-4o \
  --grounding_width 1920 \
  --grounding_height 1080 \
  --enable_few_shot \
  --demonstration_path "demonstrations" \
  --num_demonstrations 3 \
  --min_similarity 0.3
```

### Parameters

- `--enable_few_shot`: Enable few-shot learning (default: False)
- `--demonstration_path`: Directory to store demonstrations (default: "demonstrations")
- `--num_demonstrations`: Number of demonstrations to retrieve per task (default: 3)
- `--min_similarity`: Minimum similarity threshold for retrieval (default: 0.3)

### Programmatic Usage

```python
from gui_agents.s2_5.agents.few_shot_agent_s import FewShotAgentS2_5
from gui_agents.s2_5.agents.grounding import OSWorldACI

# Initialize grounding agent
grounding_agent = OSWorldACI(...)

# Create few-shot agent
agent = FewShotAgentS2_5(
    engine_params=engine_params,
    grounding_agent=grounding_agent,
    enable_few_shot=True,
    num_demonstrations=3,
    min_similarity=0.3
)

# Use the agent
info, actions = agent.predict(instruction, observation)
```

## How It Works

### 1. Demonstration Recording

When a task completes (successfully or unsuccessfully), the system automatically:
- Captures the task instruction
- Stores the trajectory (thoughts, actions, reflections)
- Records success/failure status
- Saves metadata (platform, model, timestamp)

### 2. Demonstration Retrieval

For new tasks, the system:
- Computes similarity between the new task and stored demonstrations
- Retrieves the k most similar demonstrations above the similarity threshold
- Formats demonstrations for inclusion in the LLM prompt

### 3. Enhanced Execution

The agent:
- Includes retrieved demonstrations in its context
- Uses demonstrations as examples to guide action selection
- Adapts strategies from similar successful tasks

## Performance Benefits

Based on research (LearnAct, AdaptAgent):
- **Single demonstration**: 50-200% relative improvement in task success
- **Multiple demonstrations**: Progressive improvement with more examples
- **Task adaptation**: Faster learning of new similar tasks

## Demonstration Format

Demonstrations are stored as JSON with the following structure:

```json
{
  "task_instruction": "Open the calculator application",
  "trajectory": [
    {
      "thought": "Need to open calculator",
      "action": "agent.click('Calculator icon')",
      "reflection": null
    },
    {
      "thought": "Calculator opened",
      "action": "agent.done()",
      "reflection": "Task completed"
    }
  ],
  "success": true,
  "metadata": {
    "platform": "darwin",
    "model": "gpt-4o",
    "turn_count": 2
  }
}
```

## Testing

Run the test suite to verify the implementation:

```bash
python test_few_shot_learning.py
```

This will test:
- Demonstration storage and retrieval
- Similarity computation
- Prompt formatting
- Integration with AgentS components

## Future Enhancements

Potential improvements for the few-shot learning system:

1. **Advanced Similarity Metrics**: Use embedding models (e.g., sentence-transformers) for better similarity computation
2. **Active Learning**: Prioritize demonstrations that maximize learning
3. **Meta-Learning**: Implement MAML or Reptile for rapid adaptation
4. **Demonstration Curation**: Automatic quality filtering of demonstrations
5. **Multi-Modal Demonstrations**: Include screenshot embeddings in similarity matching

## References

- LearnAct: Learning from Human Demonstrations for GUI Agents (2024)
- AdaptAgent: Few-Shot Adaptation for Web Agents (2024)
- MAML: Model-Agnostic Meta-Learning (Finn et al., 2017)