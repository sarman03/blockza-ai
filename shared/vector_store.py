import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./data/directory")
    return Chroma(
        client=client,
        collection_name="directory_docs",
        embedding_function=embeddings
    )
