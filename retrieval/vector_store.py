import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# ── Try PostgreSQL + pgvector first, fallback to ChromaDB ──
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_conn():
        conn = psycopg2.connect(DATABASE_URL)
        register_vector(conn)
        return conn

    def setup_table():
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS legal_chunks (
                id TEXT PRIMARY KEY,
                text TEXT,
                doc_id TEXT,
                chunk_index INTEGER,
                embedding vector(384)
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ PostgreSQL table ready")

    setup_table()

    def add_documents(chunks: list[dict]):
        conn = get_conn()
        cur = conn.cursor()
        texts = [c['text'] for c in chunks]
        embeddings = model.encode(texts)
        for chunk, emb in zip(chunks, embeddings):
            cur.execute("""
                INSERT INTO legal_chunks (id, text, doc_id, chunk_index, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (chunk['id'], chunk['text'], chunk['doc_id'], chunk['chunk_index'], emb.tolist()))
        conn.commit()
        cur.close()
        conn.close()

    def search(query: str, n_results: int = 5) -> list[dict]:
        query_emb = model.encode([query])[0]
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT text, doc_id, chunk_index
            FROM legal_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_emb.tolist(), n_results))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{'text': r[0], 'metadata': {'doc_id': r[1], 'chunk_index': r[2]}} for r in rows]

    def get_chunk_count():
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM legal_chunks")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count

else:
    # Fallback to ChromaDB for local development
    import chromadb
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    collection = client.get_or_create_collection(name="legal_docs", metadata={"hnsw:space": "cosine"})

    def add_documents(chunks: list[dict]):
        ids = [c['id'] for c in chunks]
        documents = [c['text'] for c in chunks]
        metadatas = [{'doc_id': c['doc_id'], 'chunk_index': c['chunk_index']} for c in chunks]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def search(query: str, n_results: int = 5) -> list[dict]:
        results = collection.query(query_texts=[query], n_results=n_results)
        return [{'text': doc, 'metadata': results['metadatas'][0][i]} for i, doc in enumerate(results['documents'][0])]

    def get_chunk_count():
        return collection.count()