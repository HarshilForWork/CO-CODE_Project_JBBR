from pydantic import BaseModel
from typing import Dict, List

class MCQRequest(BaseModel):
    context: str
    difficulty: int

class MCQResponse(BaseModel):
    question: str
    options: Dict[str, str]
    correct_answer: str
    difficulty_level: int
    topics: List[str]
    
    class Config:
        from_attributes = True

class PDFUploadResponse(BaseModel):
    message: str
    chunks: int

