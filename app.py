"""Streamlit chat UI for the Steuer-RAG system.

Run:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Steuer-RAG",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- sidebar ----------

with st.sidebar:
    st.title("🧾 Steuer-RAG")
    st.caption("Bilingual (DE/EN) RAG for the German Steuererklärung")

    st.divider()

    k = st.slider("Retrieved chunks (k)", min_value=3, max_value=20, value=8, step=1)

    source_options = {"All sources": None, "BMF": "bmf", "Elster": "elster", "BZSt": "bzst", "Gesetze": "gesetze"}
    source_label = st.selectbox("Filter by source", list(source_options.keys()))
    source_filter = source_options[source_label]

    lang_options = {"Auto-detect": None, "Deutsch": "de", "English": "en"}
    lang_label = st.selectbox("Language", list(lang_options.keys()))
    lang_filter = lang_options[lang_label]

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Sources: [BMF](https://www.bundesfinanzministerium.de) · "
        "[Elster](https://www.elster.de) · "
        "[BZSt](https://www.bzst.de) · "
        "[Gesetze](https://www.gesetze-im-internet.de)"
    )
    st.caption("Not legal or tax advice. Consult a Steuerberater for individual cases.")

# ---------- load RAG (cached so model loads once) ----------

@st.cache_resource(show_spinner="Loading RAG system…")
def load_ask():
    from steuer_rag.generation.chain import ask
    return ask


ask_fn = load_ask()

# ---------- chat state ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- render history ----------

st.header("Steuer-RAG Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} source(s)", expanded=False):
                for c in msg["citations"]:
                    score_txt = f" · rerank={c['rerank_score']:.2f}" if c.get("rerank_score") else ""
                    st.markdown(
                        f"**[{c['n']}] {c['title'] or c['url']}** — `{c['source']}`{score_txt}  \n"
                        f"[{c['url']}]({c['url']})"
                    )

# ---------- input ----------

if prompt := st.chat_input("Ask about the German Steuererklärung…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching corpus and generating answer…"):
            try:
                result = ask_fn(
                    prompt,
                    k=k,
                    source=source_filter,
                    language=lang_filter,
                )
                answer = result.answer
                citations = [
                    {
                        "n": c.n,
                        "source": c.source,
                        "url": c.url,
                        "title": c.title,
                        "rerank_score": c.rerank_score,
                    }
                    for c in result.citations
                ]
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                citations = []

        st.markdown(answer)
        if citations:
            with st.expander(f"📎 {len(citations)} source(s)", expanded=False):
                for c in citations:
                    score_txt = f" · rerank={c['rerank_score']:.2f}" if c.get("rerank_score") else ""
                    st.markdown(
                        f"**[{c['n']}] {c['title'] or c['url']}** — `{c['source']}`{score_txt}  \n"
                        f"[{c['url']}]({c['url']})"
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
