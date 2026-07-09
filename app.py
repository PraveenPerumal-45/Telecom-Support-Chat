import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv

from retriever import retrieve_with_confidence
from rag_chain import generate_answer

load_dotenv()

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FALLBACK_RESPONSE = """
I couldn't find enough information in the Telecom knowledge base to answer your question confidently.

Please contact Telecom Customer Care by calling **611** or use the **MyTelecom App** for further assistance.
"""

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi Calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]

# ---------------------------------------------------------------------
# Streamlit Config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)

# ---------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:

    st.title("📡 Telecom Support")

    st.caption("Powered by RAG + Groq")

    st.divider()

    st.markdown("### Sample Questions")

    for q in SAMPLE_QUESTIONS:

        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------

st.title("Customer Care Assistant")

st.caption(
    "Ask questions about mobile service, billing, roaming, SIM cards, network issues and more."
)

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ---------------------------------------------------------------------
# Get User Question
# ---------------------------------------------------------------------

question = st.chat_input("Describe your issue...")

if st.session_state.pending_question:

    question = st.session_state.pending_question

    st.session_state.pending_question = None

# ---------------------------------------------------------------------
# Handle Question
# ---------------------------------------------------------------------

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching Telecom Knowledge Base..."):

            result = retrieve_with_confidence(question)

            docs = result["docs"]
            confidence = result["confidence"]
            best_score = result["best_score"]

            # Debug information
            with st.expander("Retrieval Details", expanded=False):
                st.write(f"Confidence: **{confidence}**")
                st.write(f"Best Score: **{best_score:.4f}**")

                for doc in docs:

                    source = doc.metadata.get("source")

                    if source == "faq":
                        ref = f"FAQ #{doc.metadata.get('faq_id')}"

                    elif source == "ticket":
                        ref = f"Ticket {doc.metadata.get('ticket_id')}"

                    elif source == "guide":
                        ref = f"Guide Page {doc.metadata.get('page')}"

                    else:
                        ref = source

                    st.write(f"• {ref}")

            # Confidence check
            if confidence == "LOW":

                response = FALLBACK_RESPONSE

            else:

                response = generate_answer(
                    question=question,
                    docs=docs,
                )

                if confidence == "MEDIUM":
                    response = (
                        "_The following answer is based on the closest information available in our knowledge base._\n\n"
                        + response
                    )

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )