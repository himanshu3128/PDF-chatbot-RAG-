"""
app.py
------
The Streamlit web interface. Only UI code lives here;
all the RAG logic is in rag_pipeline.py.

Run with:
    streamlit run app.py
"""

import streamlit as st

from rag_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    answer_question,
    get_llm,
    process_pdf,
)

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

DEFAULT_STATE = {
    "vector_store": None,
    "file_id": None,
    "file_name": None,
    "stats": None,
    "error": None,
    "messages": [],
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_state():
    """Forget the current PDF and chat (used when the file is removed)."""
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value if not isinstance(value, list) else []


def show_sources(sources: list[dict]):
    """Show the source pages under an answer, with the retrieved text in an expander."""
    if not sources:
        return
    pages = sorted({source["page"] for source in sources})
    st.caption("📄 Sources: " + ", ".join(f"page {p}" for p in pages))
    with st.expander("Show retrieved text"):
        for source in sources:
            st.markdown(f"**Page {source['page']}** · similarity {source['score']:.2f}")
            st.text(source["text"])


with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is None:
        if st.session_state.file_id is not None:
            reset_state()

    elif uploaded_file.file_id != st.session_state.file_id:
        reset_state()
        st.session_state.file_id = uploaded_file.file_id
        st.session_state.file_name = uploaded_file.name

        with st.spinner("Reading the PDF and building the search index..."):
            try:
                store, n_pages, n_chunks = process_pdf(uploaded_file)
                st.session_state.vector_store = store
                st.session_state.stats = (n_pages, n_chunks)
            except (ValueError, ConnectionError) as e:
                st.session_state.error = str(e)
            except Exception as e:
                st.session_state.error = f"Unexpected error while processing the PDF: {e}"

    if st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.vector_store is not None:
        n_pages, n_chunks = st.session_state.stats
        st.success(f"**{st.session_state.file_name}**\n\n{n_pages} pages → {n_chunks} chunks")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption(
        f"**LLM:** {LLM_MODEL}  \n"
        f"**Embeddings:** {EMBEDDING_MODEL}  \n"
        f"**Chunks:** {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap  \n"
        f"**Top-k:** {TOP_K}"
    )

st.title("📄 PDF Question Answering Chatbot")
st.caption("Answers come only from your uploaded PDF. Runs 100% locally with Ollama.")

if st.session_state.vector_store is None:
    st.info("👈 Upload a PDF in the sidebar to start chatting.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources"))

question = st.chat_input("Ask a question about your PDF...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and writing an answer..."):
            try:
                result = answer_question(question, st.session_state.vector_store, get_llm())
                answer, sources = result["answer"], result["sources"]
            except Exception as e:
                answer, sources = f"⚠️ Sorry, something went wrong: {e}", []
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
