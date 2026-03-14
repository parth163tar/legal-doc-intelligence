import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Initialize ChromaDB client (saves to disk automatically)
client = chromadb.PersistentClient(path="./chroma_db")

# Use FREE local embeddings - no API key needed!
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get collection
collection = client.get_or_create_collection(
    name="legal_documents",
    embedding_function=local_ef
)

def add_documents(chunks: list[dict]):
    """Store document chunks in ChromaDB"""
    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[{"doc_id": c["doc_id"],
                   "chunk_index": c["chunk_index"]} for c in chunks]
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")

def search(query: str, n_results: int = 5) -> list[dict]:
    """Search for relevant chunks"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0]
        )
    ]

if __name__ == "__main__":
    # Quick test
    test_chunks = [
        {"id": "test_1", "text": "This agreement is governed by the laws of New York.",
         "doc_id": "test_doc", "chunk_index": 0},
        {"id": "test_2", "text": "The liability cap shall not exceed $1,000,000.",
         "doc_id": "test_doc", "chunk_index": 1},
    ]
    add_documents(test_chunks)
    results = search("What is the liability limit?")
    print("\n🔍 Search Results:")
    for r in results:
        print(f"  → {r['text']}")