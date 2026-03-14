import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vector_store import search
from dotenv import load_dotenv
import ollama, json

load_dotenv()

EVAL_QUESTIONS = [
    {"question": "What is the liability cap amount?", "search_query": "liability cap shall not exceed", "ground_truth": "liability is capped or limited to a specific amount"},
    {"question": "What happens upon termination?", "search_query": "effect of termination agreement expires", "ground_truth": "parties must return materials and certain clauses survive"},
    {"question": "Who owns intellectual property rights?", "search_query": "intellectual property rights ownership party retains", "ground_truth": "each party retains their pre-existing intellectual property"},
    {"question": "Which state law governs disputes?", "search_query": "governed by laws of state New York Delaware", "ground_truth": "contracts governed by laws of a specific state"},
    {"question": "What are the indemnification obligations?", "search_query": "indemnify hold harmless third party claims", "ground_truth": "each party indemnifies the other against third-party claims"}
]

def generate_answer(question, search_query):
    results = search(search_query, n_results=8)
    contexts = [r['text'] for r in results]
    context_text = "\n\n---\n\n".join(contexts[:5])
    prompt = f"""Answer the question directly using the contract excerpts below.

CONTRACT EXCERPTS:
{context_text}

QUESTION: {question}

Give a direct, specific answer in 2-3 sentences:"""
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content'], contexts

def score_answer(question, answer, ground_truth, contexts):
    """Use Mistral to score the answer quality"""
    prompt = f"""You are an evaluation judge. Score this RAG system answer on two criteria.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
ANSWER GIVEN: {answer}
CONTEXT USED: {contexts[0][:300] if contexts else 'none'}

Score each criterion from 0.0 to 1.0 and respond ONLY with JSON:
{{
    "relevancy": <0.0-1.0 — does the answer address the question?>,
    "faithfulness": <0.0-1.0 — is the answer grounded in the context?>,
    "relevancy_reason": "<one sentence>",
    "faithfulness_reason": "<one sentence>"
}}"""

    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
    raw = response['message']['content'].strip()
    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        return json.loads(raw[start:end])
    except:
        return {"relevancy": 0.5, "faithfulness": 0.5, "relevancy_reason": "parse error", "faithfulness_reason": "parse error"}

def run_evaluation():
    print("=" * 60)
    print("📊 CUSTOM EVALUATION — Legal Doc Intelligence")
    print("=" * 60)

    all_scores = []

    for i, item in enumerate(EVAL_QUESTIONS):
        print(f"\n[{i+1}/{len(EVAL_QUESTIONS)}] {item['question']}")
        print("  → Generating answer...")
        answer, contexts = generate_answer(item['question'], item['search_query'])
        print(f"  → Answer: {answer[:100]}...")
        print("  → Scoring...")
        scores = score_answer(item['question'], answer, item['ground_truth'], contexts)
        all_scores.append(scores)

        rel = scores.get('relevancy', 0)
        faith = scores.get('faithfulness', 0)
        rel_emoji = "✅" if rel > 0.8 else "⚠️" if rel > 0.5 else "❌"
        faith_emoji = "✅" if faith > 0.8 else "⚠️" if faith > 0.5 else "❌"
        print(f"  {rel_emoji} Relevancy:   {rel:.2f} — {scores.get('relevancy_reason','')[:60]}")
        print(f"  {faith_emoji} Faithfulness: {faith:.2f} — {scores.get('faithfulness_reason','')[:60]}")

    avg_rel = sum(s.get('relevancy', 0) for s in all_scores) / len(all_scores)
    avg_faith = sum(s.get('faithfulness', 0) for s in all_scores) / len(all_scores)

    print("\n" + "=" * 60)
    print("📈 FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  ✅ Avg Relevancy:    {avg_rel:.3f} / 1.0  (target: 0.80)")
    print(f"  ✅ Avg Faithfulness: {avg_faith:.3f} / 1.0  (target: 0.85)")
    print(f"\n  Relevancy:    {'✅ PASSED' if avg_rel >= 0.80 else '⚠️  NEEDS IMPROVEMENT'}")
    print(f"  Faithfulness: {'✅ PASSED' if avg_faith >= 0.85 else '⚠️  NEEDS IMPROVEMENT'}")
    print("\n" + "=" * 60)
    print("✅ Phase 5 Complete!")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
