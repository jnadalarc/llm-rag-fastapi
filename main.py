# Script complet
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_handler import RAGManager
from llm_handler import LLMHandler # <-- Importem el nou handler

# --- Configuració Centralitzada ---
MODEL_PATH = os.getenv("MODEL_PATH", "/data/models/model_1.gguf") # <-- Posa el nom del teu model
DOCS_DIR = Path(os.getenv("DOCS_DIR", "/data/documents")).resolve()
RAG_DB_PATH = Path(os.getenv("RAG_DB_PATH", "/data/index/rag.db")).resolve()

SYSTEM_PROMPT = "You are a concise assistant. Use the provided RAG_SNIPPETS to answer the user's question. Answer in the same language as the user's question."

# --- Inicialització de l'Aplicació i els Handlers ---
app = FastAPI(title="API per a RAG amb LLM (Model Integrat)", version="1.1.0")

# Actualitzem la inicialització del handler del model
llm_handler = LLMHandler(model_path=MODEL_PATH)
rag_manager = RAGManager(db_path=RAG_DB_PATH, docs_path=DOCS_DIR)

@app.on_event("startup")
def startup_event():
    if not RAG_DB_PATH.exists() or RAG_DB_PATH.stat().st_size == 0:
        print("La base de dades RAG està buida o no existeix. Iniciant indexació inicial...")
        rag_manager.ingest()
    else:
        print("Base de dades RAG trobada. S'omet la indexació inicial.")

# --- Models de Dades (Pydantic) ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="La pregunta de l'usuari en català.")
    top_k: int = Field(4, description="Nombre de documents a recuperar.")

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

class ReindexResponse(BaseModel):
    status: str
    message: str
    chunks_processed: int

# --- Endpoints de l'API (Aquests no canvien) ---
@app.get("/health", summary="Comprovació de salut")
async def health_check():
    return {"status": "ok"}

@app.post("/reindex", summary="Força la re-indexació dels documents", response_model=ReindexResponse)
async def reindex_documents():
    try:
        result = await rag_manager.aingest()
        return ReindexResponse(status=result["status"], message=result["message"], chunks_processed=result["chunks"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durant la re-indexació: {e}")

@app.post("/query", summary="Processa una pregunta amb RAG", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        query_es = await llm_handler.translate(request.question, "Spanish")
        hits = await rag_manager.asearch(query_es, k=request.top_k)
        
        context = ""
        if hits:
            rag_snippets_es = "\n---\n".join([f"Document: {hit['path']}\nContent: {hit['content']}" for hit in hits])
            context = f"RAG_SNIPPETS (Spanish original):\n{rag_snippets_es}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (context + "\n\nUser Question (Catalan): " + request.question)}
        ]
        
        answer = await llm_handler.achat(messages)
        
        return QueryResponse(answer=answer, sources=hits)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperat: {str(e)}")