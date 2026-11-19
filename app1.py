
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from handbook import (
    DEFAULT_HANDBOOK_TEXT,
    find_handbook_path,
    split_document
)

from retriever import (
    create_embeddings,
    generate_answer,
    retrieve_information
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="TIM-R Academy AI Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. APPLICATION CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

EMBEDDING_MODEL = "text-embedding-3-small"


# ==========================================================
# 3. OPENAI CLIENT
# ==========================================================

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


CLIENT = OpenAI(api_key=API_KEY) if OpenAI and API_KEY else None


# ==========================================================
# 4. CUSTOM CSS - CHATGPT STYLE
# ==========================================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background-color: var(--background-color);
}

/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #f7f7f8;
    border-right: 1px solid #e5e5e5;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #202123;
}

/* Main content */

.block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

/* Main heading */

.main-title {
    font-size: 28px;
    font-weight: 700;
    color: #202123;
    text-align: center;
    margin-top: 100px;
}

.subtitle {
    text-align: center;
    color: #777777;
    font-size: 15px;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Chat input */

[data-testid="stChatInput"] {
    border-radius: 18px;
}

/* Buttons */

.stButton button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 500;
}

/* Suggested questions */

div.stButton > button {
    border: 1px solid #dedede;
    background-color: white;
    color: #333333;
    transition: 0.2s;
}

div.stButton > button:hover {
    border-color: #777777;
    background-color: #f5f5f5;
}

/* Sidebar divider */

[data-testid="stSidebar"] hr {
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Source expanders */

[data-testid="stExpander"] {
    border-radius: 10px;
    border: 1px solid #e5e5e5;
}

/* Status information */

.status-text {
    font-size: 12px;
    color: #888888;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 5. SESSION STATE INITIALIZATION
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0


# ==========================================================
# 6. LOAD HANDBOOK AND CREATE EMBEDDINGS
# ==========================================================

@st.cache_resource
def load_handbook():

    handbook_path = find_handbook_path(BASE_DIR)

    if handbook_path is None:

        document = DEFAULT_HANDBOOK_TEXT
        source = "Built-in handbook"

    else:

        document = handbook_path.read_text(
            encoding="utf-8"
        )

        source = handbook_path.name

    chunks = split_document(document)

    embeddings = create_embeddings(
        chunks,
        CLIENT,
        EMBEDDING_MODEL
    )

    return chunks, embeddings, source


chunks, embeddings, handbook_source = load_handbook()


# ==========================================================
# 7. HELPER FUNCTION: GENERATE AI RESPONSE
# ==========================================================

def process_question(question):

    retrieved = retrieve_information(
        question,
        embeddings,
        chunks,
        client=CLIENT,
        model=EMBEDDING_MODEL
    )

    answer = generate_answer(
        question,
        retrieved,
        CLIENT,
        MODEL
    )

    sources = [
        item["chunk"][:300]
        for item in retrieved
    ]

    return answer, sources


# ==========================================================
# 8. HELPER FUNCTION: SAVE CURRENT CONVERSATION
# ==========================================================

def save_conversation():

    if st.session_state.messages:

        title = "New Conversation"

        for message in st.session_state.messages:

            if message["role"] == "user":

                title = message["content"][:45]
                break

        st.session_state.chat_counter += 1

        st.session_state.chat_history.append({
            "id": st.session_state.chat_counter,
            "title": title,
            "messages": list(st.session_state.messages)
        })


# ==========================================================
# 9. SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        "<h2 style='margin-bottom:0;'>📘 TIM-R Academy</h2>",
        unsafe_allow_html=True
    )

    st.caption("Intelligent Handbook Assistant")

    st.divider()

    # New chat

    if st.button(
        "＋ New Chat",
        width="stretch"
    ):

        save_conversation()

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # Conversation history

    st.subheader("Recent Conversations")

    if st.session_state.chat_history:

        for index, chat in enumerate(
            reversed(st.session_state.chat_history)
        ):

            if st.button(
                f"💬 {chat['title']}",
                key=f"history_{chat['id']}",
                width="stretch"
            ):

                save_conversation()

                st.session_state.messages = list(
                    chat["messages"]
                )

                st.rerun()

    else:

        st.caption("Your conversations will appear here.")

    st.divider()

    # Handbook information

    st.subheader("Knowledge Base")

    st.write(f"📚 Handbook: {handbook_source}")

    st.write(f"🤖 Model: {MODEL}")

    if CLIENT is None:

        st.warning(
            "OpenAI API unavailable. Local fallback mode is active."
        )

    else:

        st.success(
            "AI connection configured"
        )

    st.divider()

    # Theme selection

    st.subheader("Appearance")

    theme = st.selectbox(
        "Choose theme",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1
    )

    st.session_state.theme = theme

    # Clear conversation

    if st.button(
        "🗑️ Clear Current Chat",
        width="stretch"
    ):

        st.session_state.messages = []

        st.rerun()

    # Download conversation

    if st.session_state.messages:

        chat_text = ""

        for message in st.session_state.messages:

            role = message["role"].upper()

            chat_text += f"\n{role}:\n"
            chat_text += message["content"]
            chat_text += "\n\n"

        st.download_button(
            "⬇️ Download Conversation",
            data=chat_text,
            file_name="TIM-R_Academy_Conversation.txt",
            mime="text/plain",
            width="stretch"
        )

    st.divider()

    st.caption("TIM-R Academy AI Assistant v2.0")


# ==========================================================
# 10. DARK AND LIGHT THEME
# ==========================================================

if st.session_state.theme == "Dark":

    st.markdown("""
    <style>

    .stApp {
        background-color: #212121;
        color: #eeeeee;
    }

    [data-testid="stSidebar"] {
        background-color: #171717;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #eeeeee;
    }

    .main-title {
        color: #eeeeee;
    }

    .subtitle {
        color: #bbbbbb;
    }

    div.stButton > button {
        background-color: #303030;
        color: white;
        border-color: #555555;
    }

    </style>
    """, unsafe_allow_html=True)


# ==========================================================
# 11. MAIN HEADER
# ==========================================================

st.markdown(
    """
    <div style="text-align:center;">

        <h2 style="font-weight:700;">
            TIM-R Academy AI Assistant
        </h2>

        <p style="color:gray;">
            Your intelligent school handbook companion.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ==========================================================
# 12. WELCOME SCREEN
# ==========================================================

suggested_question = None

if not st.session_state.messages:

    st.markdown(
        """
        <div class="main-title">
            How can I help you today?
        </div>

        <div class="subtitle">
            Ask me anything about TIM-R Academy.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.write("### Suggested questions")

    col1, col2 = st.columns(2)

    suggested_question = None

    with col1:

        if st.button(
            "📚 Admission requirements",
            width="stretch"
        ):

            suggested_question = (
                "What are the admission requirements?"
            )

        if st.button(
            "💰 School fees",
            width="stretch"
        ):

            suggested_question = (
                "What information is available about school fees?"
            )

    with col2:

        if st.button(
            "📅 Academic calendar",
            width="stretch"
        ):

            suggested_question = (
                "What are the important academic dates?"
            )

        if st.button(
            "📋 Attendance policy",
            width="stretch"
        ):

            suggested_question = (
                "Explain the school's attendance policy."
            )


# ==========================================================
# 13. DISPLAY CHAT HISTORY
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander(
                "📖 Relevant handbook references"
            ):

                for source in message["sources"]:

                    st.markdown(f"- {source}")


# ==========================================================
# 14. CHAT INPUT
# ==========================================================

question = st.chat_input(
    "Message TIM-R Academy AI..."
)

if question is None:
    question = suggested_question


# ==========================================================
# 15. GENERATE RESPONSE
# ==========================================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("TIM-R AI is thinking..."):

            try:

                answer, source_chunks = process_question(
                    question
                )

            except Exception as error:

                answer = (
                    "I encountered an error while processing "
                    "your question. Please try again."
                )

                source_chunks = []

                st.error(str(error))

        st.markdown(answer)

        if source_chunks:

            with st.expander(
                "📖 Relevant handbook references"
            ):

                for source in source_chunks:

                    st.markdown(f"- {source}")

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": source_chunks

    })

    st.rerun()


# ==========================================================
# 16. REGENERATE LAST RESPONSE
# ==========================================================

if (
    len(st.session_state.messages) >= 2
    and st.session_state.messages[-1]["role"] == "assistant"
):

    if st.button("🔄 Regenerate response"):

        last_question = next(
            (
                message["content"]
                for message in reversed(
                    st.session_state.messages[:-1]
                )
                if message["role"] == "user"
            ),
            None
        )

        if last_question:

            st.session_state.messages.pop()

            with st.spinner("Regenerating response..."):

                answer, source_chunks = process_question(
                    last_question
                )

            st.session_state.messages.append({

                "role": "assistant",

                "content": answer,

                "sources": source_chunks

            })

            st.rerun()
