import os
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from dotenv import load_dotenv
import certifi

load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Use ChromaDB's default embedding (no heavy torch dependency)
client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "./chroma_db"))
collection = client.get_or_create_collection(
    name="legal_docs",
    metadata={"hnsw:space": "cosine"}
)

def add_documents(chunks: list[dict]):
    ids = [c['id'] for c in chunks]
    documents = [c['text'] for c in chunks]
    metadatas = [{'doc_id': c['doc_id'], 'chunk_index': c['chunk_index']} for c in chunks]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

def search(query: str, n_results: int = 5) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=n_results)
    output = []
    for i, doc in enumerate(results['documents'][0]):
        output.append({
            'text': doc,
            'metadata': results['metadatas'][0][i]
        })
    return output