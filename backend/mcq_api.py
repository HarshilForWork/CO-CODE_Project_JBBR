from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging
from Streamlit.mcq import MCQResponse
from .dependencies import get_mcq_generator

router = APIRouter()

class MCQRequest(BaseModel):
    context: str
    difficulty: int

@router.post("/generate", response_model=MCQResponse)
async def generate_mcq(request: MCQRequest):
    try:
        mcq_generator = get_mcq_generator()
        mcq = mcq_generator.generate_mcq(request.context, request.difficulty)
        
        if not mcq:
            raise HTTPException(
                status_code=400, 
                detail="Could not generate a valid MCQ from the given context"
            )
        
        return mcq
        
    except Exception as e:
        logging.error(f"MCQ generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while generating MCQ: {str(e)}"
        )