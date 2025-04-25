from pydantic import BaseModel

class Document(BaseModel):
    page_content: str
    source: str  # 문서의 출처를 기록하기 위한 필드

class DocumentsRequest(BaseModel):
    documents: list[Document]  # 여러 문서를 한 번에 추가할 수 있는 필드