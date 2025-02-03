from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.dependencies import get_document_processor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload")
async def process_pdf(file: UploadFile = File(...)):
    doc_processor = get_document_processor()
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        logger.info(f"Processing PDF file: {file.filename}")
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
            
        chunks = doc_processor.process_pdf(file_content)
        
        # Convert chunks to a serializable format
        serialized_chunks = []
        for chunk in chunks:
            serialized_chunks.append({
                "page_content": chunk.page_content,
                "metadata": {
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", 0)
                }
            })
        
        logger.info(f"Successfully processed PDF into {len(serialized_chunks)} chunks")
        
        return {
            "message": "PDF processed successfully",
            "chunks": serialized_chunks
        }
        
    except ValueError as e:
        logger.error(f"ValueError in PDF processing: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in PDF processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))