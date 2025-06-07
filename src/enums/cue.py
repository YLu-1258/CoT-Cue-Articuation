"""Cue types for bias injection in prompts."""

from enum import Enum
from typing import Dict, Any


class Cue(Enum):
    """Types of cues that can be injected into prompts to test model behavior."""
    
    STANFORD_PROFESSOR = "stanford_professor"
    FEW_SHOT_BLACK_SQUARES = "fewshot_black_squares"
    
    @property
    def display_name(self) -> str:
        """Human-readable display name for the cue."""
        return {
            self.STANFORD_PROFESSOR: "Stanford Professor",
            self.FEW_SHOT_BLACK_SQUARES: "Few-Shot Black Squares"
        }[self]
    
    @property
    def description(self) -> str:
        """Description of what this cue tests."""
        return {
            self.STANFORD_PROFESSOR: "Tests authority-based bias injection",
            self.FEW_SHOT_BLACK_SQUARES: "Tests pattern-based bias through few-shot examples"
        }[self] 