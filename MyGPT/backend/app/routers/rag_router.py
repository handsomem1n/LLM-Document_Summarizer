from fastapi import APIRouter, HTTPException
from ..models.rag_model import DocumentsRequest
from ..services.rag_service import add_new_documents

router = APIRouter()

@router.post("/rag/add", response_model=int)
async def query_endpoint(request: DocumentsRequest):
    try:
        add_new_documents(request.documents)
        return 200
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))