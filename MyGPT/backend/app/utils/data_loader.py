import os
from langchain_community.document_loaders import DirectoryLoader

def load_documents(directory_path: str):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        with open(directory_path+'/test.txt', 'x') as F:
            F.write("this is test rag file")
    loader = DirectoryLoader(directory_path)
    documents = loader.load()
    return documents
