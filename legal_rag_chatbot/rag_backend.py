# ============================================================
#  rag_backend.py — Document ingestion, retrieval & generation
# ============================================================

import os
import pickle
from config import (
    GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, RERANKER_MODEL,
    DOCS_FOLDER, VECTORSTORE_DIR, BM25_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K, TOP_K_RESULTS,
    MAX_TOKENS, TEMPERATURE
)

from langchain_community.document_loaders  import PyPDFDirectoryLoader
from langchain_text_splitters              import RecursiveCharacterTextSplitter
from langchain_community.embeddings        import HuggingFaceEmbeddings
from langchain_community.vectorstores      import FAISS
from langchain_community.retrievers        import BM25Retriever
from langchain_classic.retrievers          import EnsembleRetriever
from langchain_core.retrievers             import BaseRetriever
from pydantic                              import Field, PrivateAttr
from langchain_groq                        import ChatGroq
from langchain_classic.chains              import ConversationalRetrievalChain
from langchain_classic.chains.summarize    import load_summarize_chain
from langchain_classic.memory              import ConversationBufferMemory
from langchain_core.prompts                import PromptTemplate

class CustomRerankingRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    reranker_model_name: str
    top_n: int = 3
    _model: any = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.reranker_model_name)
        
    def _get_relevant_documents(self, query: str, *, run_manager=None):
        docs = self.base_retriever.invoke(query)
        if not docs: return []
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self._model.predict(pairs)
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores[:self.top_n]]


# ─────────────────────────────────────────────────────────────
#  STEP 1 — Manage Collections and Load Documents
# ─────────────────────────────────────────────────────────────
def get_collections():
    """Get a list of all document collections (subfolders)."""
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
    collections = [d for d in os.listdir(DOCS_FOLDER) if os.path.isdir(os.path.join(DOCS_FOLDER, d))]
    
    # If no collections exist, create a 'Default' one
    if not collections:
        default_path = os.path.join(DOCS_FOLDER, "Default")
        os.makedirs(default_path)
        collections = ["Default"]
    return collections


def load_documents(collection_name):
    """Load every PDF inside the specific collection folder."""
    folder_path = os.path.join(DOCS_FOLDER, collection_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    return documents


# ─────────────────────────────────────────────────────────────
#  STEP 2 — Split documents into chunks
# ─────────────────────────────────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


# ─────────────────────────────────────────────────────────────
#  STEP 3 — Embeddings & Vector Store (with Hybrid Search DB)
# ─────────────────────────────────────────────────────────────
def get_embeddings():
    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name      = EMBEDDING_MODEL,
        model_kwargs    = {"device": "cpu"},
        encode_kwargs   = {"normalize_embeddings": True}
    )

def get_vectorstore_and_bm25(collection_name):
    """Load or build FAISS and BM25 indices for the selected collection."""
    embeddings = get_embeddings()
    vs_dir = os.path.join(VECTORSTORE_DIR, collection_name)
    bm25_path = os.path.join(BM25_DIR, f"{collection_name}.pkl")
    
    # If both indices exist, load them
    if os.path.exists(vs_dir) and os.path.exists(bm25_path):
        print(f"[INFO] Loading existing indices for {collection_name}...")
        vectorstore = FAISS.load_local(vs_dir, embeddings, allow_dangerous_deserialization=True)
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        return vectorstore, bm25_retriever

    # Otherwise, build them from scratch
    documents = load_documents(collection_name)
    if not documents:
        raise ValueError(
            f"No PDFs found in '{DOCS_FOLDER}/{collection_name}/'. "
            "Please add your PDF documents and rebuild."
        )

    chunks = split_documents(documents)
    print(f"[INFO] Building indices for {collection_name} ({len(chunks)} chunks)...")
    
    # 1. Build FAISS (Semantic)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vs_dir)
    
    # 2. Build BM25 (Keyword)
    if not os.path.exists(BM25_DIR):
        os.makedirs(BM25_DIR)
        
    bm25_retriever = BM25Retriever.from_documents(chunks)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    return vectorstore, bm25_retriever


# ─────────────────────────────────────────────────────────────
#  STEP 4 — Build RAG Prompt
# ─────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """
You are a Pakistani Legal AI Assistant for a legal tech startup.
Your job is to answer questions STRICTLY based on the provided document context.

RULES YOU MUST FOLLOW:
1. Answer ONLY from the context below. Do NOT use outside knowledge.
2. If the answer is not in the context, say: "I could not find this information in the provided documents."
3. Always cite the source at the end of your answer in this exact format:
   📄 Source: [document name] | Page: [page number]
4. Be precise, professional, and concise.

---
CONTEXT FROM DOCUMENTS:
{context}

---
CONVERSATION HISTORY:
{chat_history}

---
USER QUESTION:
{question}

---
YOUR ANSWER (with citation):
"""

def get_rag_prompt():
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=RAG_PROMPT_TEMPLATE
    )


# ─────────────────────────────────────────────────────────────
#  STEP 5 — Build the Advanced RAG chain (Hybrid + Reranking)
# ─────────────────────────────────────────────────────────────
def build_rag_chain(collection_name):
    """Build a conversational chain with Hybrid Search & Reranking."""
    
    vectorstore, bm25_retriever = get_vectorstore_and_bm25(collection_name)
    
    # FAISS Semantic Retriever
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    
    # BM25 Keyword Retriever
    bm25_retriever.k = RETRIEVER_K
    
    # 1. Hybrid Retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]  # 40% Keyword, 60% Semantic
    )
    
    # 2. Reranker
    print(f"[INFO] Loading reranker model: {RERANKER_MODEL}")
    compression_retriever = CustomRerankingRetriever(
        base_retriever=hybrid_retriever,
        reranker_model_name=RERANKER_MODEL,
        top_n=TOP_K_RESULTS
    )

    # Groq LLM
    llm = ChatGroq(
        api_key     = GROQ_API_KEY,
        model_name  = GROQ_MODEL,
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS
    )

    memory = ConversationBufferMemory(
        memory_key          = "chat_history",
        output_key          = "answer",
        return_messages     = True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm                  = llm,
        retriever            = compression_retriever,
        memory               = memory,
        combine_docs_chain_kwargs = {"prompt": get_rag_prompt()},
        return_source_documents   = True,
        verbose                   = False
    )

    return chain


# ─────────────────────────────────────────────────────────────
#  STEP 6 — Document Summarization (Bonus Feature)
# ─────────────────────────────────────────────────────────────
def summarize_collection(collection_name):
    """Generate a high-level summary of all documents in the collection."""
    documents = load_documents(collection_name)
    if not documents:
        return "No documents found in this collection to summarize."
        
    llm = ChatGroq(
        api_key     = GROQ_API_KEY,
        model_name  = GROQ_MODEL,
        temperature = 0.2, # Slight temperature for better flow
        max_tokens  = MAX_TOKENS
    )
    
    # Using 'map_reduce' safely handles long documents
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    print(f"[INFO] Summarizing {len(documents)} pages for {collection_name}...")
    
    try:
        summary_result = chain.invoke(documents)
        return summary_result.get("output_text", "Failed to extract summary text.")
    except Exception as e:
        return f"Error during summarization: {str(e)}\n\n(Note: MapReduce on large collections may hit Groq rate limits)"


# ─────────────────────────────────────────────────────────────
#  STEP 7 — Query the chain
# ─────────────────────────────────────────────────────────────
def ask_question(chain, user_question):
    result = chain.invoke({"question": user_question})
    answer = result.get("answer", "No answer generated.")

    citations = []
    seen = set()
    for doc in result.get("source_documents", []):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page   = doc.metadata.get("page", 0) + 1
        key    = f"{source}-{page}"
        if key not in seen:
            seen.add(key)
            citations.append({"source": source, "page": page})

    return answer, citations
