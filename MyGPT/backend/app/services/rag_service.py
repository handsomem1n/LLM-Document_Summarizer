import ollama
import os
from app.utils.embeddings import embed_query
from app.utils.data_loader import load_documents
from app.services.conversation_manager import ConversationManager
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import schema

from app.models.rag_model import Document

conversation_manager = ConversationManager()

# Chroma 벡터 스토어 생성 및 초기화
def initialize_vector_store(documents):
    # 문서를 여러 작은 텍스트 덩어리로 나누기
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    persist_directory = "data/chroma_db"
    os.makedirs(persist_directory, exist_ok=True)

    # Ollama의 임베딩 함수를 사용하여 Chroma 벡터 스토어 생성
    vector_store = Chroma.from_documents(texts, embed_query, persist_directory=persist_directory)
    return vector_store


def add_new_documents(documents: list[Document]):
    """
    새로 추가된 문서를 Chroma 벡터 스토어에 저장하는 함수.
    """
    try:
        # 문서를 벡터화하여 벡터 스토어에 추가
        # 각 문서의 content를 임베딩하여 벡터화
        embedded_documents = []
        
        for doc in documents:
            with open('data/raw/'+doc.source, 'w', encoding="utf-8") as F:
                F.write(doc.page_content)
            # 문서 임베딩 (metadata는 없으므로 content만 사용)
            embedded_documents.append(
                schema.Document(doc.page_content, metadata={
                    'source': doc.source
                })
            )
        
        # 벡터 스토어에 문서 추가
        vector_store.add_documents(embedded_documents)
        
        # 벡터 스토어 저장
        vector_store.persist()

        print(f"Successfully added {len(documents)} documents to the vector store.")
        
    except Exception as e:
        print(f"Error while adding documents: {e}")
        raise e

def generate_rag_response(query: str):
    # 쿼리를 임베딩 벡터로 변환
    embedded_query = embed_query.embed_query(query)
    
    # 유사한 문서 검색
    retrieved_docs = vector_store.similarity_search(embedded_query)

    seen_sources = set()  # 이미 본 문서의 source를 추적하기 위한 set
    unique_docs = []  # 중복을 제거한 문서들

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "")
        if source not in seen_sources:
            seen_sources.add(source)
            unique_docs.append(doc)
    
    # 검색된 문서를 텍스트로 변환하여 결합
    context_text = "\n\n".join([f'{doc.metadata["source"]}> {doc.page_content}' for doc in unique_docs])
    conversation_manager.set_last_doc(unique_docs)
    
    # 대화 관리에 추가
    conversation_manager.add_message("system", context_text)
    conversation_manager.add_message("user", query)

    # 대화 기록 가져오기
    full_query = conversation_manager.get_conversation()

    # Ollama 모델 호출
    response = ollama.chat(
        model="bllossom",
        messages=full_query
    )

    result_text = response['message']["content"]

    # 대화 기록에 Ollama의 응답 추가
    conversation_manager.add_message("assistant", result_text)
    
    # 검색된 문서 내용과 Ollama 응답 반환
    return {
        "result": result_text,  # Ollama의 응답
        "searched_documents": [doc.page_content for doc in unique_docs]  # 검색된 문서들
    }

# 문서 로드 및 벡터 스토어 초기화
documents = load_documents("data/raw")
vector_store = initialize_vector_store(documents)
