import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from groq import Groq
from retrieval.vector_store import search
from dotenv import load_dotenv

load_dotenv()
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
GROQ_MODEL = 'llama-3.1-8b-instant'

CLAUSE_TYPES = [
    "Governing Law", "Termination for Convenience", "Liability Cap",
    "IP Ownership", "Non-Compete", "Confidentiality", "Indemnification",
    "Auto-Renewal", "Assignment", "Dispute Resolution"
]

def extract_clauses_from_text(contract_text):
    prompt = f"""You are a legal contract analyst. Extract key clauses from the contract below.
Respond ONLY with a JSON array, no other text:
[{{"clause_type": "name", "clause_text": "exact text", "location": "section number"}}]

Look for: {", ".join(CLAUSE_TYPES)}

CONTRACT:
{contract_text[:3000]}

JSON:"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{'role': 'user', 'content': prompt}]
    )
    raw = response.choices[0].message.content.strip()
    try:
        start = raw.find('[')
        end = raw.rfind(']') + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        return []
    except json.JSONDecodeError:
        return []

def extract_clauses_from_query(query):
    print(f"\n🔍 Searching contracts for: {query}")
    results = search(query, n_results=5)
    all_clauses = []
    sources = []
    for r in results:
        doc_id = r['metadata']['doc_id']
        sources.append(doc_id)
        clauses = extract_clauses_from_text(r['text'])
        for clause in clauses:
            clause['source_contract'] = doc_id
            all_clauses.append(clause)
    return {'query': query, 'clauses_found': all_clauses, 'contracts_searched': list(set(sources)), 'total_clauses': len(all_clauses)}

if __name__ == "__main__":
    result = extract_clauses_from_query("termination and governing law clauses")
    print(f"\n✅ Found {result['total_clauses']} clauses")
    for clause in result['clauses_found'][:5]:
        print(f"\n  Type: {clause.get('clause_type')}")
        print(f"  Text: {clause.get('clause_text','')[:150]}")
        print(f"  Source: {clause.get('source_contract','')[:50]}")
