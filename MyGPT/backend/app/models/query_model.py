from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str
    searched_documents: List[str]

