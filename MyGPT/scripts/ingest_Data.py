from app.utils.data_loader import load_documents
from app.utils.embeddings import embed_query
from langchain.vectorstores import VectorStore

def main():
    documents = load_documents()
    vector_store = VectorStore()

    # Process and add each document to vector store
    for doc in documents:
        embedded_doc = embed_query(doc.text)
        vector_store.add(embedded_doc)

    print("Data ingested successfully!")

if __name__ == "__main__":
    main()
