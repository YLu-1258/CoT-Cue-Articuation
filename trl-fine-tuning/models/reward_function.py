"""
GSM8K Reward Function Implementation
Binary reward: +1 for correct answer, 0 for wrong/incorrect answer
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

class GSM8KRewardFunction:
    """Reward function for GSM8K math problems."""
    
    def __init__(self):
        self.correct_answers = 0
        self.total_answers = 0
        # Add __name__ attribute for TRL compatibility
        self.__name__ = "GSM8KRewardFunction"
    
    def extract_answer(self, text: str) -> float:
        """Extract numerical answer from GSM8K response."""
        text = text.strip()
        
        # Look for patterns like "#### 42" (GSM8K format)
        pattern = r'####\s*([0-9,]+(?:\.[0-9]+)?)'
        match = re.search(pattern, text)
        if match:
            answer_str = match.group(1).replace(',', '')
            try:
                return float(answer_str)
            except ValueError:
                pass
        
        # Look for "The answer is X" pattern
        pattern = r'[Tt]he answer is\s*([0-9,]+(?:\.[0-9]+)?)'
        match = re.search(pattern, text)
        if match:
            answer_str = match.group(1).replace(',', '')
            try:
                return float(answer_str)
            except ValueError:
                pass
        
        # Look for numbers at the end of the text
        numbers = re.findall(r'([0-9,]+(?:\.[0-9]+)?)', text)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except ValueError:
                pass
        
        return None
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Compute rewards for completions.
        
        Args:
            completions: List of generated completions
            **kwargs: Additional context (may contain prompts, ground_truth, etc.)
        
        Returns:
            List of rewards (0.0 or 1.0 for each completion)
        """
        rewards = []
        
        # Try to get ground truth from kwargs
        ground_truths = kwargs.get('ground_truth', [])
        
        for i, completion in enumerate(completions):
            reward = 0.0  # Default: wrong answer
            
            # Extract predicted answer
            predicted = self.extract_answer(completion)
            
            if predicted is not None and i < len(ground_truths):
                ground_truth = ground_truths[i]
                
                # Check if answers match (allow small floating point differences)
                if abs(predicted - ground_truth) < 1e-6:
                    reward = 1.0
                    self.correct_answers += 1
            
            self.total_answers += 1
            rewards.append(reward)
        
        # Log accuracy
        if self.total_answers > 0:
            accuracy = self.correct_answers / self.total_answers
            logger.info(f"Current accuracy: {accuracy:.3f} ({self.correct_answers}/{self.total_answers})")
        
        return rewards 