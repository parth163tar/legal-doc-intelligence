import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Legal Document Intelligence",
    page_icon="⚖️",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Legal AI")
    st.markdown("**Legal Document Intelligence**")
    st.markdown("Powered by Mistral + ChromaDB")
    st.divider()

    # API Health Check
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        if r.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error(f"❌ API Error: {r.status_code}")
    except Exception as e:
        st.warning(f"⚠️ {str(e)[:50]}")
        st.info("API URL: " + API_URL)

    st.divider()
    st.markdown("**Navigation**")
    page = st.radio("", ["💬 Ask Questions", "📄 Analyze Contract", "🔍 Search Contracts"])

st.title("⚖️ Legal Document Intelligence")

# ══════════════════════════════════════════════════════════
# PAGE 1 — QA
# ══════════════════════════════════════════════════════════
if page == "💬 Ask Questions":
    st.subheader("Ask anything about the 510 legal contracts")
    st.markdown("*Powered by RAG — Retrieve → Rerank → Generate*")

    # Example questions
    st.markdown("**Quick examples:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💰 Liability cap?"):
            st.session_state.question = "What is the liability cap in these contracts?"
    with col2:
        if st.button("🚪 Termination?"):
            st.session_state.question = "What are the termination conditions?"
    with col3:
        if st.button("🔒 IP ownership?"):
            st.session_state.question = "Who owns the intellectual property rights?"

    question = st.text_area(
        "Your question:",
        value=st.session_state.get("question", ""),
        height=100,
        placeholder="e.g. What is the governing law in these contracts?"
    )

    col_a, col_b = st.columns([1, 4])
    with col_a:
        n_sources = st.selectbox("Sources", [3, 5, 8], index=0)

    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching 80,532 contract chunks..."):
                try:
                    response = requests.post(
                        f"{API_URL}/qa",
                        json={"question": question, "n_retrieve": 10, "n_rerank": n_sources},
                        timeout=120
                    )
                    result = response.json()

                    st.success("✅ Answer generated!")
                    st.markdown("### 💬 Answer")
                    st.markdown(result["answer"])

                    st.divider()
                    st.markdown(f"### 📄 Sources ({result['num_sources']} contract excerpts used)")
                    for i, chunk in enumerate(result["source_chunks"]):
                        with st.expander(f"Source {i+1}"):
                            st.markdown(chunk[:500])

                except requests.exceptions.Timeout:
                    st.error("⏳ Timeout — Mistral is slow on CPU, try again")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ══════════════════════════════════════════════════════════
# PAGE 2 — CONTRACT ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "📄 Analyze Contract":
    st.subheader("Paste a contract to extract clauses and score risk")

    SAMPLE = (
        "DISTRIBUTOR AGREEMENT\n\n"
        "1. LIABILITY: Total liability shall not exceed $500,000.\n\n"
        "2. TERMINATION: Either party may terminate upon 30 days written notice.\n\n"
        "3. INTELLECTUAL PROPERTY: TechCorp retains all ownership of its products.\n\n"
        "4. NON-COMPETE: Distributor agrees not to distribute competing products for 5 years.\n\n"
        "5. GOVERNING LAW: This Agreement shall be governed by the laws of Delaware."
    )

    contract_name = st.text_input("Contract name:", value="My Contract")
    contract_text = st.text_area(
        "Paste contract text here:",
        value=SAMPLE,
        height=300
    )

    if st.button("⚡ Analyze Contract", type="primary", use_container_width=True):
        if not contract_text.strip():
            st.warning("Please paste contract text")
        else:
            with st.spinner("Extracting clauses and scoring risk..."):
                try:
                    response = requests.post(
                        f"{API_URL}/analyze",
                        json={"contract_text": contract_text, "contract_name": contract_name},
                        timeout=300
                    )
                    result = response.json()

                    # Overall risk banner
                    risk = result["overall_risk"]
                    if risk == "HIGH":
                        st.error(f"🔴 Overall Risk: HIGH")
                    elif risk == "MEDIUM":
                        st.warning(f"🟡 Overall Risk: MEDIUM")
                    else:
                        st.success(f"🟢 Overall Risk: LOW")

                    # Risk breakdown
                    counts = result["risk_counts"]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Clauses", result["total_clauses"])
                    col2.metric("🔴 High Risk", counts.get("HIGH", 0))
                    col3.metric("🟡 Medium Risk", counts.get("MEDIUM", 0))
                    col4.metric("🟢 Low Risk", counts.get("LOW", 0))

                    st.divider()
                    st.markdown("### 📋 Clause Analysis")

                    for clause in result["clauses"]:
                        risk_level = clause.get("risk_level", "MEDIUM")
                        emoji = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"

                        with st.expander(f"{emoji} {clause.get('clause_type', 'Unknown')} — {risk_level}"):
                            st.markdown(f"**Clause Text:** {clause.get('clause_text', '')}")
                            st.markdown(f"**Risk Reason:** {clause.get('risk_reason', '')}")
                            if clause.get('red_flags'):
                                st.markdown(f"**⚠️ Red Flags:** {', '.join(clause['red_flags'])}")

                except requests.exceptions.Timeout:
                    st.error("⏳ Timeout — try a shorter contract")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ══════════════════════════════════════════════════════════
# PAGE 3 — SEARCH
# ══════════════════════════════════════════════════════════
elif page == "🔍 Search Contracts":
    st.subheader("Semantic search across 510 legal contracts")
    st.markdown("*Finds relevant clauses even if you don't use exact keywords*")

    query = st.text_input(
        "Search query:",
        placeholder="e.g. automatic renewal clause"
    )
    n_results = st.slider("Number of results", 3, 10, 5)

    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching..."):
                try:
                    response = requests.post(
                        f"{API_URL}/search",
                        json={"query": query, "n_results": n_results},
                        timeout=30
                    )
                    result = response.json()

                    st.success(f"✅ Found {len(result['results'])} results")

                    for i, r in enumerate(result["results"]):
                        with st.expander(f"Result {i+1} — {r['doc_id'][:60]}"):
                            st.markdown(r["text"])
                            st.caption(f"Chunk {r['chunk_index']} from {r['doc_id']}")

                except Exception as e:
                    st.error(f"❌ Error: {e}")