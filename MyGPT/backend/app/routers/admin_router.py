from fastapi import APIRouter, HTTPException
from typing import List
from app.services.rag_service import conversation_manager  # 대화 기록 관리 객체
from app.services.rag_service import vector_store  # 벡터 스토어 객체
from langchain.schema import Document

# 어드민 페이지 관련 라우터
admin_router = APIRouter()

# 대화 기록과 검색된 문서 정보를 반환하는 엔드포인트
@admin_router.get("/admin/conversation", response_model=dict)
async def get_conversation():
    try:
        # conversation_manager에서 대화 기록을 가져옴
        conversation = conversation_manager.get_conversation()
        return {"conversation": conversation}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation.")

# 직전 쿼리에서 검색한 문서들을 반환하는 엔드포인트
@admin_router.get("/admin/searched_documents", response_model=List[str])
async def get_searched_documents():
    try:        
        return [f'{doc.metadata["source"]}> {doc.page_content}' for doc in conversation_manager.get_last_doc()]
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to retrieve searched documents.")
