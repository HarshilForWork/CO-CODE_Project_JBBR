from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MCQ:
    question: str
    options: Dict[str, str]
    correct_answer: str
    difficulty_level: int
    topics: List[str]