import json
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "knowledge_base.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="ClaimCheck | Evidence Lab",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  .stApp { background: #0b1020; color: #e8edf7; }
  [data-testid="stSidebar"] { background: #10182b; }
  .hero { padding: 1.5rem 1.6rem; border: 1px solid #293550; border-radius: 18px;
          background: linear-gradient(135deg, #151f37 0%, #111827 100%); margin-bottom: 1rem; }
  .eyebrow { text-transform: uppercase; letter-spacing: .16em; color: #8ca7ff;
             font-size: .75rem; font-weight: 700; }
  .hero h1 { margin: .4rem 0; font-size: 2.3rem; color: #f4f7ff; }
  .hero p { color: #b6c3dc; margin-bottom: 0; }
  .pill { display: inline-block; border: 1px solid #34415c; color: #bdc9df;
          padding: .2rem .6rem; border-radius: 999px; font-size: .78rem; margin-right: .35rem; }
  .evidence { background: #111a2d; border: 1px solid #283752; border-radius: 12px;
              padding: 1rem; margin: .55rem 0; }
  .source { color: #8ea4c7; font-size: .82rem; }
  .small-note { color: #91a0ba; font-size: .86rem; }
  div[data-testid="stMetric"] { background: #111a2d; border: 1px solid #283752;
    padding: 1rem; border-radius: 12px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the local sentence-embedding model...")
def load_model():
    return SentenceTransformer(MODEL_NAME)

def load_knowledge_base():
    if not DATA_FILE.exists():
        return []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def split_passages(text, max_chars=650):
    # Lightweight paragraph chunking; preserves sentence text and avoids huge passages.
    chunks = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
        else:
            sentences = paragraph.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|")
            current = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current) + len(sentence) + 1 > max_chars and current:
                    chunks.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            if current:
                chunks.append(current)
    return chunks

def retrieve_evidence(claim, docs, model, top_k=5):
    passages = []
    for doc in docs:
        for idx, passage in enumerate(split_passages(doc.get("text", ""))):
            passages.append({
                "text": passage,
                "source": doc.get("source", "Unknown source"),
                "title": doc.get("title", "Untitled document"),
                "url": doc.get("url", ""),
                "doc_id": doc.get("id", ""),
            })
    if not passages:
        return []
    claim_embedding = model.encode([claim], normalize_embeddings=True)
    passage_embeddings = model.encode(
        [p["text"] for p in passages],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = cosine_similarity(claim_embedding, passage_embeddings)[0]
    for passage, score in zip(passages, scores):
        passage["score"] = float(score)
    passages.sort(key=lambda p: p["score"], reverse=True)
    return passages[:top_k]

def cautious_assessment(results, min_relevance=0.28):
    if not results or results[0]["score"] < min_relevance:
        return (
            "Insufficient evidence",
            "The local collection did not return a sufficiently relevant passage. "
            "This does not mean the claim is false; it means this collection did not provide useful evidence.",
        )
    return (
        "Evidence found — review required",
        "Relevant passages were retrieved, but semantic similarity alone cannot establish whether a claim is true or false. "
        "Review the passages and their sources before drawing a conclusion.",
    )

def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

init_state()
docs = load_knowledge_base()

with st.sidebar:
    st.markdown("## 🔎 ClaimCheck")
    st.caption("Evidence-first claim review")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Enter a claim\n2. Retrieve related passages\n3. Inspect sources\n4. Decide whether the evidence is sufficient")
    st.markdown("---")
    st.metric("Documents in knowledge base", len(docs))
    st.markdown("### Important")
    st.caption("This prototype retrieves relevant text. It does not independently prove truth or falsehood.")
    st.markdown("---")
    st.caption("Local model: all-MiniLM-L6-v2")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Evidence Lab · Local AI/ML Prototype</div>
  <h1>Don't just get an answer.<br>Inspect the evidence.</h1>
  <p>Search a curated collection of documents and review the passages related to a claim.</p>
  <div style="margin-top:1rem">
    <span class="pill">Semantic search</span>
    <span class="pill">Source traceability</span>
    <span class="pill">No paid API</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not docs:
    st.warning("Your knowledge base is empty. Add documents to data/knowledge_base.json, then restart or refresh the app.")

with st.form("claim_form"):
    claim = st.text_area(
        "Claim to investigate",
        placeholder="Example: Python is always faster than C++.",
        height=110,
        help="Enter one factual or technical statement. Results are limited to your local document collection.",
    )
    top_k = st.slider("Number of evidence passages", min_value=3, max_value=10, value=5)
    submitted = st.form_submit_button("🔍 Find evidence", type="primary", use_container_width=True)

if submitted:
    if not claim.strip():
        st.error("Please enter a claim first.")
    elif not docs:
        st.error("Add at least one document to the knowledge base before searching.")
    else:
        try:
            model = load_model()
            with st.spinner("Searching the local knowledge base..."):
                results = retrieve_evidence(claim.strip(), docs, model, top_k=top_k)
                verdict, explanation = cautious_assessment(results)
                record = {
                    "claim": claim.strip(),
                    "verdict": verdict,
                    "explanation": explanation,
                    "results": results,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.last_result = record
                st.session_state.history.insert(0, {
                    "claim": record["claim"],
                    "verdict": verdict,
                    "timestamp": record["timestamp"],
                })
        except Exception as exc:
            st.error("The model could not be loaded or the search failed.")
            st.exception(exc)
            st.info("Check your internet connection for the first model download, then try again.")

record = st.session_state.last_result
if record:
    st.markdown("## Investigation result")
    st.markdown(f"**Claim:** {record['claim']}")
    st.markdown(f"**Assessment:** {record['verdict']}")
    st.info(record["explanation"])

    if record["results"]:
        best_score = record["results"][0]["score"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Passages retrieved", len(record["results"]))
        c2.metric("Top relevance score", f"{best_score:.3f}")
        c3.metric("Checked at", record["timestamp"].split(" ")[1])
        st.caption("Relevance scores rank passages by semantic similarity. They are not truth probabilities or confidence scores.")
        st.markdown("### Retrieved evidence")
        for i, item in enumerate(record["results"], start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {item['title']}**")
                st.markdown(item["text"])
                st.markdown(
                    f"<div class='source'>Source: {item['source']} · Relevance: {item['score']:.3f}</div>",
                    unsafe_allow_html=True,
                )
                if item.get("url"):
                    st.markdown(f"[Open source]({item['url']})")
    else:
        st.warning("No passages were available to display.")

    st.download_button(
        "Download investigation report (JSON)",
        data=json.dumps(record, ensure_ascii=False, indent=2),
        file_name="claimcheck_report.json",
        mime="application/json",
    )

with st.expander("📚 View the knowledge base format"):
    st.markdown("Add entries to `data/knowledge_base.json`. Each entry should have a title, source, text, and optional URL.")
    st.code("""[
  {
    "id": "doc-001",
    "title": "Example document",
    "source": "Course notes",
    "url": "",
    "text": "Paste a paragraph from a trusted, legally reusable document here."
  }
]""", language="json")

with st.expander("🕘 Session history"):
    if st.session_state.history:
        for item in st.session_state.history[:20]:
            st.markdown(f"- **{item['verdict']}** — {item['claim']}  \n  <span class='small-note'>{item['timestamp']}</span>", unsafe_allow_html=True)
    else:
        st.caption("Your searches will appear here during this session.")

st.markdown("---")
st.markdown(
    "<p class='small-note'>ClaimCheck is an educational prototype. Retrieved text may be incomplete or misleading. "
    "Verify important claims against original sources; this app is not a definitive fact-checking authority.</p>",
    unsafe_allow_html=True,
)
