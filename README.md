# 📡 Telecom Customer Support AI Assistant

An AI-powered Telecom Customer Support Assistant that answers customer queries using **Retrieval-Augmented Generation (RAG)**.

Instead of relying on the LLM's internal knowledge, the assistant retrieves relevant information from multiple knowledge sources—including FAQs, resolved support tickets, and technical documentation—before generating an answer.

The assistant also includes **source citations**, **confidence-based retrieval**, and **safe fallback responses** to reduce hallucinations.

---

## Demo

> *(Add screenshots or GIFs here)*

### Chat Interface

![Chat Screenshot](images/chat.png)

### Source Citations

![Citations](images/citations.png)

### Confidence-Based Retrieval

![Confidence](images/confidence.png)

---

# Features

- Multi-source Retrieval-Augmented Generation (RAG)
- Semantic search using Chroma Vector Database
- FAQ retrieval
- Resolved support ticket retrieval
- Telecom PDF guide retrieval
- Source citations in every response
- Confidence-based retrieval scoring
- Automatic fallback when confidence is low
- Hallucination reduction
- Groq Qwen3-32B integration
- Interactive Streamlit chat interface

---

# Architecture

```text
                User
                  │
                  ▼
          Streamlit Interface
                  │
                  ▼
       retrieve_with_confidence()
                  │
      ┌───────────┼────────────┐
      ▼           ▼            ▼
   FAQ DB     Ticket DB    Guide DB
      │           │            │
      └───────────┼────────────┘
                  ▼
          Merge & Rank Results
                  │
          Confidence Evaluation
                  │
         ┌────────┴────────┐
         ▼                 ▼
  High Confidence     Low Confidence
         │                 │
         ▼                 ▼
     Generate Answer    Safe Fallback
         │
         ▼
     Response + Sources
```

---

# Knowledge Sources

The assistant retrieves information from three independent sources.

## FAQ Dataset

Contains structured customer questions and answers.

Examples:

- International Roaming
- Billing
- Wi-Fi Calling
- SIM Activation
- VoLTE
- Mobile Data

---

## Support Tickets

Contains previously resolved customer support tickets.

Examples:

- SIM Not Detected
- Call Drops
- Slow Internet
- VoLTE Compatibility
- Roaming Issues

---

## Telecom Technical Guide

A PDF technical reference containing:

- Mobile Networks
- APN Configuration
- VoLTE
- VoWiFi
- IMS
- Roaming
- LTE
- Network Troubleshooting

---

# Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| LLM | Groq (Qwen3-32B) |
| Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace MiniLM |
| UI | Streamlit |
| PDF Processing | PyPDF |
| Data Storage | SQLite |
| Environment | dotenv |

---

# Project Structure

```text
telecom-chatbot/
│
├── app.py
├── rag_chain.py
├── retriever.py
│
├── ingest_faq.py
├── ingest_pdf.py
├── ingest_tickets.py
│
├── chroma_store/
│
├── data/
│   ├── faq.csv
│   ├── telecom_guide.pdf
│   └── tickets.db
│
├── requirements.txt
└── README.md
```

---

# Retrieval Pipeline

The assistant performs the following steps:

1. Receive the user question.
2. Search the FAQ vector database.
3. Search the Support Ticket vector database.
4. Search the Telecom Guide vector database.
5. Merge retrieved documents.
6. Rank results by semantic similarity.
7. Evaluate retrieval confidence.
8. Generate an answer using Groq Qwen3-32B.
9. Return source citations.

---

# Confidence-Based Retrieval

The assistant evaluates retrieval quality before generating a response.

If confidence is low, the LLM is skipped entirely to reduce hallucinations.

Example fallback:

```
I couldn't find enough information in the Telecom knowledge base.

Please contact customer support at 611
or use the MyTelecom app.
```

---

# Source Citations

Each answer includes citations indicating where the information originated.

Example:

```
Sources

• FAQ #22

• Support Ticket TK-011

• Telecom Guide (Page 7)
```

---

# Example Questions

```
How do I activate international roaming?

How do I enable Wi-Fi Calling?

My SIM isn't detected.

Why is my mobile data slow?

My calls keep dropping.

I was charged for roaming even though I had a bundle.

What is VoLTE?

Explain APN settings.

How do I unlock my phone for another carrier?
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<your-username>/telecom-support-ai.git
```

Navigate into the project.

```bash
cd telecom-support-ai
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate the environment.

Windows

```bash
.venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file.

```env
GROQ_API_KEY=your_groq_api_key
```

---

# Build the Knowledge Base

Run the ingestion scripts.

```bash
python ingest_faq.py
```

```bash
python ingest_pdf.py
```

```bash
python ingest_tickets.py
```

---

# Run the Application

```bash
streamlit run app.py
```

---

# Future Improvements

- Conversation memory for follow-up questions
- Hybrid keyword + vector search
- Query rewriting
- Cross-encoder reranking
- Multi-turn conversational retrieval
- Streaming LLM responses
- Evaluation framework (RAGAS / DeepEval)
- Admin dashboard for knowledge management

---

# Key Learning Outcomes

This project demonstrates practical experience with:

- Retrieval-Augmented Generation (RAG)
- LangChain pipelines
- Vector databases
- Semantic search
- Prompt engineering
- Confidence-aware retrieval
- Hallucination mitigation
- Source attribution
- AI application development
- LLM integration with external knowledge

---

## License

This project is intended for educational and portfolio purposes.
