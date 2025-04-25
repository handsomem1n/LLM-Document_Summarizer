from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import query_router, admin_router, rag_router

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (배포 환경에서는 제한하는 것이 좋음)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# Include the router for query-related endpoints
app.include_router(admin_router.admin_router)
app.include_router(rag_router.router)
app.include_router(query_router.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG backend API"}
