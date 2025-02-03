from pydantic import BaseModel
from typing import Dict, List

class MCQResponse(BaseModel):
    question: str
    options: Dict[str, str]
    correct_answer: str
    difficulty_level: int
    topics: List[str]