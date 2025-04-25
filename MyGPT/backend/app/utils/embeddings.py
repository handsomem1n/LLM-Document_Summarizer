from langchain_community.embeddings.ollama import OllamaEmbeddings
from pydantic import ConfigDict

embed_query = OllamaEmbeddings(
    base_url="http://172.17.0.2:11434",
    model="bgm-m3",
    )