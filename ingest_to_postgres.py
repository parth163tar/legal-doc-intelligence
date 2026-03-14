import os, certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from dotenv import load_dotenv
load_dotenv()

from ingestion.chunker import chunk_all_contracts
from retrieval.vector_store import add_documents, get_chunk_count

print("📄 Loading and chunking contracts...")
chunks = chunk_all_contracts('data/CUADv1.json')
print(f"✅ {len(chunks)} chunks ready")

batch_size = 50
print(f"⚡ Ingesting to PostgreSQL in batches of {batch_size}...")
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    add_documents(batch)
    print(f"  Progress: {min(i+batch_size, len(chunks))}/{len(chunks)}", end='\r')

print(f"\n✅ Done! Total chunks in DB: {get_chunk_count()}")