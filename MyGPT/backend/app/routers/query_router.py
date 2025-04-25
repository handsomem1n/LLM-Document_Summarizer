from fastapi import APIRouter, HTTPException
from app.services.query_service import process_query
from app.models.query_model import QueryRequest, QueryResponse

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        # generate_rag_response 호출하여 쿼리 처리
        result = process_query(request.query)

        # result에서 응답 내용과 검색된 문서 반환
        return QueryResponse(
            result=result['result'],  # Ollama 응답 내용
            searched_documents=result['searched_documents']  # 검색된 문서들
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))