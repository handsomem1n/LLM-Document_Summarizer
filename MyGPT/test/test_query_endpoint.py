from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_query_endpoint():
    response = client.post("/query", json={"query": "What is RAG?"})
    assert response.status_code == 200
    assert "result" in response.json()
