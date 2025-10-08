# Script complet
import sqlite3
from pathlib import Path
import asyncio

class RAGManager:
    """Gestiona la inicialització, indexació i cerca a la base de dades RAG amb SQLite FTS5."""
    
    def __init__(self, db_path: Path, docs_path: Path):
        self.db_path = db_path
        self.docs_path = docs_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(path, content)")
            conn.commit()

    def _chunk_text(self, text: str) -> list[str]:
        chunk_size, chunk_overlap = 1200, 150
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    def ingest(self) -> dict:
        """Escaneja, processa i indexa els documents. Aquesta és una operació bloquejant."""
        print(f"🚀 Iniciant procés d'indexació des de: {self.docs_path}")
        if not self.docs_path.exists():
            msg = "⚠️ El directori de documents no existeix. S'omet la indexació."
            print(msg)
            return {"status": "skipped", "reason": msg, "chunks": 0}

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM docs")
            
            total_chunks, files_processed = 0, 0
            files = [p for p in self.docs_path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".log"}]
            
            for p in files:
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    for chunk in self._chunk_text(text):
                        cur.execute("INSERT INTO docs(path, content) VALUES(?, ?)", (str(p.name), chunk))
                        total_chunks += 1
                    files_processed += 1
                except Exception as e:
                    print(f"❌ Error processant {p}: {e}")
            conn.commit()

        msg = f"✅ Indexació finalitzada! Fitxers: {files_processed}, Fragments: {total_chunks}"
        print(msg)
        return {"status": "success", "message": msg, "chunks": total_chunks}

    def search(self, query: str, k: int = 5) -> list:
        """Executa una cerca FTS5. Aquesta és una operació bloquejant."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            safe_query = f'"{query}"'
            cur.execute("SELECT path, content FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?", (safe_query, k))
            return [{"path": p, "content": c} for p, c in cur.fetchall()]

    async def aingest(self) -> dict:
        """Versió asíncrona de 'ingest'."""
        return await asyncio.to_thread(self.ingest)
        
    async def asearch(self, query: str, k: int = 5) -> list:
        """Versió asíncrona de 'search'."""
        return await asyncio.to_thread(self.search, query, k)