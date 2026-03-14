import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
GROQ_MODEL = 'llama-3.1-8b-instant'

RISK_RULES = {
    "Liability Cap": {"high": ["unlimited", "no cap", "no limit"], "low": ["shall not exceed", "limited to", "capped at"]},
    "Non-Compete": {"high": ["worldwide", "perpetual", "unlimited duration"], "low": ["6 months", "1 year", "limited territory"]},
    "Termination for Convenience": {"high": ["immediately", "no notice", "without cause"], "low": ["30 days", "60 days", "written notice"]},
    "Indemnification": {"high": ["unlimited", "all claims", "any losses"], "low": ["limited to", "direct damages only", "capped"]},
    "Auto-Renewal": {"high": ["perpetual", "automatic", "unless terminated"], "low": ["written consent required", "opt-in"]}
}

def rule_based_score(clause_type, clause_text):
    text_lower = clause_text.lower()
    rules = RISK_RULES.get(clause_type, {})
    high_matches = sum(1 for s in rules.get("high", []) if s in text_lower)
    low_matches = sum(1 for s in rules.get("low", []) if s in text_lower)
    if high_matches > low_matches:
        return "HIGH"
    elif low_matches > 0:
        return "LOW"
    return "MEDIUM"

def llm_risk_score(clause_type, clause_text):
    prompt = f"""You are a legal risk analyst. Score this clause risk level.
Clause Type: {clause_type}
Clause Text: {clause_text}

Respond ONLY with JSON:
{{"risk_level": "LOW" or "MEDIUM" or "HIGH", "reason": "one sentence", "red_flags": ["flag1"]}}

JSON:"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{'role': 'user', 'content': prompt}]
    )
    raw = response.choices[0].message.content.strip()
    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except:
        pass
    return {"risk_level": rule_based_score(clause_type, clause_text), "reason": "Rule-based fallback", "red_flags": []}

def score_contract_clauses(clauses, use_llm=True):
    scored = []
    for clause in clauses:
        clause_type = clause.get('clause_type', 'Unknown')
        clause_text = clause.get('clause_text', '')
        score = llm_risk_score(clause_type, clause_text) if use_llm else {"risk_level": rule_based_score(clause_type, clause_text), "reason": "Rule-based", "red_flags": []}
        scored.append({**clause, "risk_level": score.get("risk_level", "MEDIUM"), "risk_reason": score.get("reason", ""), "red_flags": score.get("red_flags", [])})
    return scored

def generate_risk_report(contract_name, scored_clauses):
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for clause in scored_clauses:
        level = clause.get("risk_level", "MEDIUM")
        risk_counts[level] = risk_counts.get(level, 0) + 1
    overall = "HIGH" if risk_counts["HIGH"] > 0 else "MEDIUM" if risk_counts["MEDIUM"] > 0 else "LOW"
    return {"contract": contract_name, "overall_risk": overall, "risk_counts": risk_counts, "total_clauses": len(scored_clauses), "high_risk_clauses": [c for c in scored_clauses if c.get("risk_level") == "HIGH"], "all_clauses": scored_clauses}

if __name__ == "__main__":
    test_clauses = [
        {"clause_type": "Liability Cap", "clause_text": "Liability shall not exceed total amounts paid in last 12 months."},
        {"clause_type": "Non-Compete", "clause_text": "Distributor agrees not to compete worldwide for perpetual duration."},
        {"clause_type": "Termination for Convenience", "clause_text": "Either party may terminate upon 30 days written notice."},
    ]
    scored = score_contract_clauses(test_clauses)
    report = generate_risk_report("TEST CONTRACT", scored)
    print(f"Overall Risk: {report['overall_risk']}")
    for clause in report['all_clauses']:
        emoji = "🔴" if clause['risk_level'] == "HIGH" else "🟡" if clause['risk_level'] == "MEDIUM" else "🟢"
        print(f"{emoji} {clause['clause_type']}: {clause['risk_level']} — {clause['risk_reason']}")
