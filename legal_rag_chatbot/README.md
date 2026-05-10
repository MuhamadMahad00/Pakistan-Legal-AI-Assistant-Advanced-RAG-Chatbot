# ⚖️ Pakistan Legal AI Assistant — Advanced RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for a legal tech startup.
Answers questions strictly from loaded legal documents with proper citations. This version includes advanced features like Hybrid Search, Reranking, Document Summarization, and Multiple Collections support!

---

## 🗂️ Project Structure

```
legal_rag_chatbot/
│
├── documents/          ← PUT YOUR PDFs IN SUBFOLDERS HERE (e.g., documents/Contracts/)
├── vectorstore/        ← Auto-created FAISS indices
├── bm25_indices/       ← Auto-created BM25 indices
│
├── app.py              ← Streamlit frontend (UI)
├── rag_backend.py      ← All RAG logic (load, chunk, embed, hybrid search, rerank, generate)
├── config.py           ← All settings (model names, chunk size, thresholds)
├── requirements.txt    ← Python dependencies
├── .env                ← Your API keys (never share this)
└── README.md
```

---

## 📄 Step 1 — Organize Your Documents

This chatbot supports **Multiple Collections**. You must place your PDFs inside a subfolder within the `documents/` directory.

Example:
1. Create a folder: `documents/Contracts/`
2. Download these 4 official Pakistani legal PDFs and place them inside `documents/Contracts/`:

| # | Document | Link |
|---|----------|------|
| 1 | Employment Contract (SMEDA) | https://smeda.org/phocadownload/Commercial_Contracts/Employment/new/Model%20Employment%20Contract.pdf |
| 2 | Consultancy Services Contract (CPPA) | https://www.cppa.gov.pk/storage/uploads/downloads/vsaxEsHcSQ5yKm5xQCmsBctYRjBSU6zdKoMJmCPx.pdf |
| 3 | Standard Terms & Conditions (Establishment Division) | https://www.establishment.gov.pk/SiteImage/Misc/files/Standarad%20Terms%20and%20Conditions%20of%20Contract%20Appointment.pdf |
| 4 | SECP General Conditions of Contract | https://www.secp.gov.pk/wp-content/uploads/2016/05/SampleAgreement-Annex-D_20140525.pdf |

---

## 🔑 Step 2 — Get Your FREE Groq API Key

1. Go to → https://console.groq.com
2. Sign up (free)
3. Create an API key
4. Open the `.env` file and paste it:

```
GROQ_API_KEY=gsk_your_key_here
```

---

## 📦 Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Step 4 — Run the App

```bash
streamlit run app.py
```

Open your browser at → http://localhost:8501

---

## 🌟 Advanced Features (Bonus Challenges Implemented)

1. **Hybrid Search**: Combines **FAISS** (Semantic similarity) and **BM25** (Exact keyword matching) using LangChain's `EnsembleRetriever` to guarantee we don't miss highly specific legal terms.
2. **Reranking**: Fetches the top 10 chunks from Hybrid Search and passes them through a HuggingFace Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to strictly extract the absolute Top 3 most relevant chunks.
3. **Multiple Collections**: Dynamically reads subfolders in `documents/` allowing users to switch contexts (e.g., HR vs. Corporate) on the fly via the Streamlit sidebar.
4. **Document Summarization**: A sidebar button triggers a Map-Reduce LangChain pipeline that reads the entire selected document collection and generates a concise executive summary.

---

## 🔄 How The Pipeline Works

```
PDF Documents (Split by Collection)
     ↓
PyPDFDirectoryLoader + RecursiveCharacterTextSplitter 
     ↓
Embeddings (FAISS) + BM25 Keyword Extraction
     ↓
User Query → Hybrid Search (EnsembleRetriever) → Top 10 Chunks
     ↓
Cross-Encoder Reranker → Refines to Top 3 Chunks
     ↓
Groq LLaMA3-8B  (Generates grounded answer)
     ↓
Answer + Citations displayed in Streamlit UI
```

---

## 💬 Sample Questions to Test

- What are the termination clauses in the employment contract?
- What is the probationary period mentioned in the contract?
- What are the confidentiality obligations of the consultant?
- What happens in case of a dispute between parties?
- What are the payment terms in the consultancy agreement?

---

## ⚙️ Customize Settings (config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| GROQ_MODEL | llama3-8b-8192 | Change to mixtral-8x7b-32768 for better quality |
| CHUNK_SIZE | 1000 | Characters per chunk |
| RETRIEVER_K | 10 | Chunks retrieved initially by Hybrid search |
| TOP_K_RESULTS | 3 | Final chunks retained after Reranking |
| TEMPERATURE | 0.0 | 0 = factual, no hallucination |

---

## ❓ Troubleshooting

**"No PDFs found"** → Make sure your PDFs are inside a subfolder, e.g., `documents/Default/`
**"GROQ_API_KEY not found"** → Check your `.env` file has the key correctly
**Slow first run** → Normal! The embedding and reranker models download once (~150MB total)
