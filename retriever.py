"""
Retriever for the Telecom RAG chatbot.

Searches FAQ, Tickets and Guide independently,
merges the results,
sorts them by similarity score,
and returns:

    docs
    best_similarity_score

Lower score = better match (Chroma distance).
"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)

# ---------------------------------------------------------
# Vector stores
# ---------------------------------------------------------

faq_store = Chroma(
    collection_name="faq",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

ticket_store = Chroma(
    collection_name="tickets",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

guide_store = Chroma(
    collection_name="guides",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve_with_confidence(
    query: str,
    k_per_store: int = 3,
    final_k: int = 5,
) -> Tuple[List[Document], float]:
    """
    Search all vector stores.

    Returns
    -------
    docs
        Top retrieved documents.

    best_score
        Lowest similarity distance.
        Lower = better.
    """

    results = []

    stores = [
        faq_store,
        ticket_store,
        guide_store,
    ]

    for store in stores:

        matches = store.similarity_search_with_score(
            query,
            k=k_per_store,
        )

        results.extend(matches)

    if not results:
        return [], float("inf")

    # ---------------------------------------------------------
    # Sort by similarity
    # ---------------------------------------------------------

    results.sort(key=lambda x: x[1])

    results = results[:final_k]

    docs = [doc for doc, _ in results]
    scores = [score for _, score in results]

    best_score = scores[0]

    if len(scores) > 1:
        score_gap = scores[1] - scores[0]
    else:
        score_gap = 0

    # ---------------------------------------------------------
    # Confidence classification
    # ---------------------------------------------------------

    if best_score <= 1.0:
        confidence = "HIGH"

    elif best_score <= 1.5 and score_gap > 0.15:
        confidence = "MEDIUM"

    else:
        confidence = "LOW"

    print("\n========== Retrieval ==========")

    for doc, score in results:

        source = doc.metadata.get("source", "Unknown")

        if source == "faq":
            ref = f"FAQ #{doc.metadata.get('faq_id')}"

        elif source == "ticket":
            ref = f"Ticket {doc.metadata.get('ticket_id')}"

        elif source == "guide":
            ref = f"Guide Page {doc.metadata.get('page')}"

        else:
            ref = source

        print(f"{ref:<25} score={score:.4f}")

    print("-------------------------------")
    print(f"Best score : {best_score:.4f}")
    print(f"Score gap  : {score_gap:.4f}")
    print(f"Confidence : {confidence}")
    print("===============================\n")

    return {
        "docs": docs,
        "best_score": best_score,
        "confidence": confidence,
    }