"""
RAG generation chain.

Responsibilities:
- Format retrieved documents
- Build the prompt
- Invoke the LLM
- Return the final answer

Retrieval and confidence checking happen in retriever.py.
"""
from dotenv import load_dotenv

load_dotenv()
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


SYSTEM_PROMPT = """
You are a professional Telecom Customer Support Assistant.

You answer ONLY using the supplied context.

Rules:

1. Never invent information.

2. If the context does not contain the answer,
   say you don't have enough information.

3. Every factual statement must come from the supplied context.

4. At the end of every answer include a section called:

Sources

listing all supporting documents.

Example:

Sources
• FAQ #7
• Support Ticket TK-001
• Telecom Guide (Page 5)

Do not cite sources that were not used.

Context:

{context}
"""


# ----------------------------------------------------------------------
# Format retrieved documents
# ----------------------------------------------------------------------

def format_docs(docs: List[Document]) -> str:
    """
    Convert retrieved documents into prompt context.
    """

    sections = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get("source", "").lower()

        if source == "faq":

            citation = f"FAQ #{doc.metadata.get('faq_id')}"

        elif source == "ticket":

            citation = (
                f"Support Ticket {doc.metadata.get('ticket_id')}"
            )

        elif source == "guide":

            citation = (
                f"Telecom Guide (Page {doc.metadata.get('page')})"
            )

        else:

            citation = "Unknown Source"

        sections.append(
f"""
DOCUMENT {i}

Source:
{citation}

Content:
{doc.page_content}
"""
        )

    return "\n\n-----------------------------\n\n".join(sections)


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Question:

{question}
""",
        ),
    ]
)


# ----------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_format="parsed",
    max_retries=2,
)


parser = StrOutputParser()


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def generate_answer(
    question: str,
    docs: List[Document],
) -> str:
    """
    Generate an answer from retrieved documents.

    Parameters
    ----------
    question
        User question.

    docs
        Retrieved documents.

    Returns
    -------
    str
        Final response.
    """

    context = format_docs(docs)

    messages = prompt.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    response = llm.invoke(messages)

    return parser.invoke(response)