from app.services.rag_service import generate_rag_response

def process_query(query: str):
    # Main logic to handle a user query
    return generate_rag_response(query)
