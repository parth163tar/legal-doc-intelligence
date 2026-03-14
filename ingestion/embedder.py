import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.chunker import chunk_all_contracts
from retrieval.vector_store import add_documents, search

def ingest_all(filepath: str, batch_size: int = 100):
    """Chunk all contracts and store in ChromaDB"""
    print("📄 Loading and chunking contracts...")
    chunks = chunk_all_contracts(filepath)

    print(f"\n⚡ Ingesting {len(chunks)} chunks in batches of {batch_size}...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        add_documents(batch)
        print(f"  Progress: {min(i+batch_size, len(chunks))}/{len(chunks)}", end='\r')

    print(f"\n✅ All chunks ingested into ChromaDB!")

if __name__ == "__main__":
    ingest_all('data/CUADv1.json')

    print("\n🔍 Testing search...")
    results = search("liability cap indemnification")
    print("\nTop results:")
    for r in results:
        print(f"  → [{r['metadata']['doc_id'][:40]}]")
        print(f"     {r['text'][:120]}\n")