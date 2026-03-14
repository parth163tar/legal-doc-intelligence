import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
import cohere
from retrieval.vector_store import search
from dotenv import load_dotenv

load_dotenv()
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
co = cohere.Client(os.getenv('COHERE_API_KEY'))
GROQ_MODEL = 'llama-3.1-8b-instant'

def rerank(query, documents, top_n=3):
    if not documents:
        return []
    results = co.rerank(query=query, documents=documents, model='rerank-english-v3.0', top_n=top_n)
    return [documents[r.index] for r in results.results]

def build_prompt(query, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a legal document analyst. Answer the question using ONLY the contract excerpts below.
If the answer is not in the excerpts, say 'I could not find this information in the provided contracts.'
Always cite which part of the contract your answer comes from.

CONTRACT EXCERPTS:
{context}

QUESTION: {query}

ANSWER:"""

def answer_query(query, n_retrieve=10, n_rerank=3):
    print(f"Searching for: {query}")
    retrieved = search(query, n_results=n_retrieve)
    texts = [r['text'] for r in retrieved]
    print(f"  Retrieved {len(texts)} chunks")

    try:
        reranked = rerank(query, texts, top_n=n_rerank)
        print(f"  Reranked to top {len(reranked)} chunks")
    except Exception as e:
        print(f"  Reranking skipped: {e}")
        reranked = texts[:n_rerank]

    prompt = build_prompt(query, reranked)
    print(f"  Generating answer with Groq...")

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{'role': 'user', 'content': prompt}]
    )

    return {
        'query': query,
        'answer': response.choices[0].message.content,
        'source_chunks': reranked,
        'num_sources': len(reranked)
    }

if __name__ == "__main__":
    questions = [
        "What is the liability cap in these contracts?",
        "What are the termination conditions?",
        "How is intellectual property ownership handled?"
    ]
    for q in questions:
        print("\n" + "="*60)
        result = answer_query(q)
        print(f"\n❓ Question: {result['query']}")
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n📄 Based on {result['num_sources']} contract excerpts")
