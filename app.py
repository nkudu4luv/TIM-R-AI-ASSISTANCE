import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from handbook import DEFAULT_HANDBOOK_TEXT, find_handbook_path, split_document
from retriever import create_embeddings, generate_answer, retrieve_information

st.set_page_config(
    page_title="TIM-R Academy Handbook Assistant",
    page_icon="📘",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

CLIENT = OpenAI(api_key=API_KEY) if OpenAI and API_KEY else None


@st.cache_resource
def load_handbook():
    handbook_path = find_handbook_path(BASE_DIR)
    if handbook_path is None:
        document = DEFAULT_HANDBOOK_TEXT
        source = "Built-in handbook"
    else:
        document = handbook_path.read_text(encoding="utf-8")
        source = handbook_path.name

    chunks = split_document(document)
    embeddings = create_embeddings(chunks, CLIENT, EMBEDDING_MODEL)
    return chunks, embeddings, source


st.title("TIM-R Academy Handbook Assistant")
st.caption("Ask a question about the school handbook.")

chunks, embeddings, handbook_source = load_handbook()
st.sidebar.subheader("About")
st.sidebar.write(f"Handbook: {handbook_source}")
if CLIENT is None:
    st.sidebar.info("Running without an OpenAI API key. Answers use local fallback mode.")
else:
    st.sidebar.info("OpenAI answers are enabled; local fallback is used if the API is unavailable.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Relevant handbook text"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

question = st.chat_input("Ask about fees, attendance, admission, or school dates")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retrieved = retrieve_information(
        question,
        embeddings,
        chunks,
        client=CLIENT,
        model=EMBEDDING_MODEL,
    )
    answer = generate_answer(question, retrieved, CLIENT, MODEL)
    source_chunks = [item["chunk"][:300] for item in retrieved]
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": source_chunks}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
        if source_chunks:
            with st.expander("Relevant handbook text"):
                for source in source_chunks:
                    st.markdown(f"- {source}")
