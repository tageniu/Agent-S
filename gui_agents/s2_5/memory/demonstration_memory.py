import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("desktopenv.agent")


@dataclass
class Demonstration:
    """Single demonstration trajectory"""
    task_instruction: str
    trajectory: List[Dict]
    success: bool
    metadata: Dict = None
    
    def to_dict(self):
        return {
            "task_instruction": self.task_instruction,
            "trajectory": self.trajectory,
            "success": self.success,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class DemonstrationMemory:
    """Manages storage and retrieval of demonstration trajectories for few-shot learning"""
    
    def __init__(self, storage_path: str = "demonstrations", embedding_model: Optional[str] = None):
        """Initialize demonstration memory
        
        Args:
            storage_path: Directory to store demonstrations
            embedding_model: Model name for computing embeddings (optional)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.demonstrations: List[Demonstration] = []
        self.embeddings = []
        self.embedding_model = embedding_model
        self._load_demonstrations()
    
    def _load_demonstrations(self):
        """Load existing demonstrations from storage"""
        demo_file = self.storage_path / "demonstrations.json"
        embedding_file = self.storage_path / "embeddings.pkl"
        
        if demo_file.exists():
            with open(demo_file, "r") as f:
                data = json.load(f)
                self.demonstrations = [Demonstration.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.demonstrations)} demonstrations")
        
        if embedding_file.exists() and self.embedding_model:
            with open(embedding_file, "rb") as f:
                self.embeddings = pickle.load(f)
    
    def save_demonstrations(self):
        """Persist demonstrations to storage"""
        demo_file = self.storage_path / "demonstrations.json"
        embedding_file = self.storage_path / "embeddings.pkl"
        
        with open(demo_file, "w") as f:
            json.dump([d.to_dict() for d in self.demonstrations], f, indent=2)
        
        if self.embeddings:
            with open(embedding_file, "wb") as f:
                pickle.dump(self.embeddings, f)
    
    def add_demonstration(self, demonstration: Demonstration):
        """Add a new demonstration to memory
        
        Args:
            demonstration: Demonstration object to add
        """
        self.demonstrations.append(demonstration)
        
        if self.embedding_model:
            embedding = self._compute_embedding(demonstration.task_instruction)
            self.embeddings.append(embedding)
        
        self.save_demonstrations()
        logger.info(f"Added demonstration for task: {demonstration.task_instruction[:50]}...")
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute text embedding for similarity matching
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Simple TF-IDF or sentence embedding
        # For production, use sentence-transformers or OpenAI embeddings
        words = text.lower().split()
        vocab = list(set(word for demo in self.demonstrations for word in demo.task_instruction.lower().split()))
        if not vocab:
            vocab = words
        
        embedding = np.zeros(len(vocab))
        for word in words:
            if word in vocab:
                embedding[vocab.index(word)] = 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def retrieve_similar_demonstrations(
        self, 
        task_instruction: str, 
        k: int = 3,
        min_similarity: float = 0.3
    ) -> List[Demonstration]:
        """Retrieve k most similar demonstrations for a given task
        
        Args:
            task_instruction: Current task description
            k: Number of demonstrations to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar demonstrations
        """
        if not self.demonstrations:
            return []
        
        # Compute similarity based on task instruction
        similarities = []
        for demo in self.demonstrations:
            similarity = self._compute_similarity(task_instruction, demo.task_instruction)
            similarities.append(similarity)
        
        # Get top-k demonstrations
        indices = np.argsort(similarities)[::-1][:k]
        selected_demos = []
        
        for idx in indices:
            if similarities[idx] >= min_similarity:
                selected_demos.append(self.demonstrations[idx])
                logger.info(f"Retrieved demo with similarity {similarities[idx]:.2f}: {self.demonstrations[idx].task_instruction[:50]}...")
        
        return selected_demos
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple Jaccard similarity for now
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def format_demonstrations_for_prompt(
        self, 
        demonstrations: List[Demonstration],
        max_steps_per_demo: int = 5
    ) -> str:
        """Format demonstrations for inclusion in LLM prompt
        
        Args:
            demonstrations: List of demonstrations to format
            max_steps_per_demo: Maximum number of steps to include per demonstration
            
        Returns:
            Formatted string for prompt
        """
        if not demonstrations:
            return ""
        
        formatted = "Here are similar task demonstrations for reference:\n\n"
        
        for i, demo in enumerate(demonstrations, 1):
            formatted += f"Example {i}: {demo.task_instruction}\n"
            formatted += f"Success: {'Yes' if demo.success else 'No'}\n"
            
            # Include key steps from trajectory
            steps = demo.trajectory[:max_steps_per_demo]
            for j, step in enumerate(steps, 1):
                if "action" in step:
                    formatted += f"  Step {j}: {step.get('thought', '')}\n"
                    formatted += f"  Action: {step['action']}\n"
            
            if len(demo.trajectory) > max_steps_per_demo:
                formatted += f"  ... ({len(demo.trajectory) - max_steps_per_demo} more steps)\n"
            
            formatted += "\n"
        
        formatted += "Use these examples to guide your approach, but adapt to the specific requirements of the current task.\n"
        
        return formatted
    
    def record_trajectory(
        self,
        task_instruction: str,
        trajectory: List[Dict],
        success: bool,
        metadata: Optional[Dict] = None
    ):
        """Record a new trajectory as a demonstration
        
        Args:
            task_instruction: Task description
            trajectory: List of trajectory steps
            success: Whether the task was successful
            metadata: Additional metadata
        """
        demonstration = Demonstration(
            task_instruction=task_instruction,
            trajectory=trajectory,
            success=success,
            metadata=metadata
        )
        
        self.add_demonstration(demonstration)
    
    def get_statistics(self) -> Dict:
        """Get statistics about stored demonstrations
        
        Returns:
            Dictionary with statistics
        """
        if not self.demonstrations:
            return {
                "total_demonstrations": 0,
                "successful_demonstrations": 0,
                "success_rate": 0.0,
                "unique_tasks": 0
            }
        
        successful = sum(1 for d in self.demonstrations if d.success)
        unique_tasks = len(set(d.task_instruction for d in self.demonstrations))
        
        return {
            "total_demonstrations": len(self.demonstrations),
            "successful_demonstrations": successful,
            "success_rate": successful / len(self.demonstrations),
            "unique_tasks": unique_tasks,
            "avg_trajectory_length": np.mean([len(d.trajectory) for d in self.demonstrations])
        }