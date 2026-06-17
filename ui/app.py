"""
Streamlit UI — Information Retrieval System
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="IR Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("Information Retrieval System")
st.markdown("Search across two document collections using multiple retrieval models.")

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search Settings")

    dataset = st.selectbox("Dataset", ["dataset1", "dataset2"])

    model = st.selectbox(
        "Retrieval Model",
        ["bm25", "tfidf", "embedding", "hybrid_serial", "hybrid_parallel"]
    )

    top_k = st.slider("Number of Results", min_value=5, max_value=50, value=10)

    st.subheader("BM25 Parameters")
    k1 = st.slider("k1 (term saturation)", 0.5, 3.0, 1.5, 0.1)
    b  = st.slider("b (length normalization)", 0.0, 1.0, 0.75, 0.05)

    if model in ("hybrid_serial", "hybrid_parallel"):
        st.subheader("Hybrid Settings")
        hybrid_mode   = "serial" if model == "hybrid_serial" else "parallel"
        fusion_method = st.selectbox("Fusion Method", ["rrf", "linear"]) if hybrid_mode == "parallel" else "rrf"
    else:
        hybrid_mode   = "serial"
        fusion_method = "rrf"

    st.subheader("Query Refinement")
    spell_correction = st.checkbox("Spell Correction", value=True)
    use_synonyms     = st.checkbox("Synonym Expansion", value=False)
    use_prf = st.checkbox("Pseudo Relevance Feedback (PRF)",value=False)

# ─── Search Bar ───────────────────────────────────────────────────────────────
query = st.text_input("Enter your search query", placeholder="e.g. climate change effects")

data = None
search_clicked = st.button("Search")

if search_clicked and query:
    with st.spinner("Searching..."):
        try:
            payload = {
                "query": query,
                "dataset": dataset,
                "model": model,
                "top_k": top_k,
                "bm25_k1": k1,
                "bm25_b": b,
                "hybrid_mode": hybrid_mode,
                "fusion_method": fusion_method,
                "use_spell_correction": spell_correction,
                "use_synonyms": use_synonyms,
                "use_prf": use_prf,
            }

            response = requests.post(
                f"{API_URL}/search",
                json=payload,
                timeout=300
            )
            try:
                data = response.json()
            except Exception:
                st.error("Backend did not return valid JSON")
                data = None

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API Gateway")
            data = None

        except Exception as e:
            st.error(f"Error: {e}")
            data = None
# Ranking Table
# ─── Safe check ─────────────────────────────
if data and "results" in data:

    if data.get("corrected_query") and data["corrected_query"] != query:
        st.info(f"Query corrected to: **{data['corrected_query']}**")

    if data.get("expanded_query"):
        st.info(f"Expanded Query: {data['expanded_query']}")

    st.success(f"Found {data['total_results']} results using **{data['model_used']}**")

    import pandas as pd

    table_data = [
        {
            "Rank": r["rank"],
            "Doc ID": r["doc_id"],
            "Score": r["score"]
        }
        for r in data["results"]
    ]

    st.subheader("Ranking Table")
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    for result in data["results"]:
        with st.expander(
            f"Rank {result['rank']} — Doc ID: {result['doc_id']} | Score: {result['score']}"
        ):
            st.write(result.get("snippet", "No snippet available."))

# ─── Suggestions ──────────────────────────────────────────────────────────────
if query and len(query) > 2:
    history = st.session_state.get("history", [])
    if history:
        try:
            params = {"query": query, "history": history}
            resp = requests.get(f"{API_URL}/suggest", params=params, timeout=5)
            suggestions = resp.json().get("suggestions", [])
            if suggestions:
                st.markdown("**Query suggestions:**")
                for s in suggestions:
                    if st.button(s, key=f"sugg_{s}"):
                        st.session_state["query"] = s
        except Exception:
            pass

if query:
    history = st.session_state.get("history", [])
    if query not in history:
        history.append(query)
        st.session_state["history"] = history
