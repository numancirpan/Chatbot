"""
create_vector_db.py

data/chunks.json -> db/chroma_db/

ChromaDB repoda tutulmaz; yerelde chunks.json dosyasindan yeniden uretilir.
Mevcut veritabani doluysa varsayilan olarak atlanir. Bos veya bozuksa yeniden
olusturulur. Zorla yenilemek icin --rebuild kullanin.
"""

import json
import os
import shutil
import sqlite3
import sys
from typing import Dict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DB_DIR = os.path.join(ROOT_DIR, "db", "chroma_db")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
COLLECTION_NAME = "langchain"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.chatbot import enrich_chunk_metadata


def build_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": True},
    )


def chroma_counts() -> Dict[str, int]:
    sqlite_path = os.path.join(DB_DIR, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return {}

    try:
        with sqlite3.connect(sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM embeddings")
            embedding_count = int(cur.fetchone()[0])
        return {COLLECTION_NAME: embedding_count}
    except Exception as exc:
        print(f"ChromaDB okunamadi: {exc}")
        return {}


def total_chroma_count() -> int:
    return sum(chroma_counts().values())


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if isinstance(chunks, dict):
        chunks = [chunks]
    return [enrich_chunk_metadata(chunk) for chunk in chunks]


def reset_db_dir() -> None:
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)


def clear_collection_in_place(vector_store: Chroma) -> None:
    collection = vector_store._collection
    offset = 0
    batch_size = 500

    while True:
        result = collection.get(include=[], limit=batch_size, offset=offset)
        ids = result.get("ids", [])
        if not ids:
            break
        collection.delete(ids=ids)
        if len(ids) < batch_size:
            break
    try:
        # Best effort to flush deletions before re-adding new records.
        vector_store._client.persist()
    except Exception:
        pass


def build(rebuild: bool = False):
    chunks = load_chunks()
    print(f"{len(chunks)} chunk yuklendi")

    existing_counts = chroma_counts()
    existing_total = sum(existing_counts.values())
    if existing_total > 0 and not rebuild:
        print(f"ChromaDB zaten mevcut ({existing_total} kayit). Atlanıyor.")
        print(f"Koleksiyonlar: {existing_counts}")
        print("Yeniden olusturmak icin: python pipeline/create_vector_db.py --rebuild")
        return

    if existing_counts and existing_total == 0:
        print("ChromaDB koleksiyonu var ama kayit sayisi 0. Yeniden olusturulacak.")

    print("Embedding modeli yukleniyor...")
    embedding_fn = build_embedding_function()

    recreate_dir = True
    try:
        reset_db_dir()
    except PermissionError as exc:
        recreate_dir = False
        print(f"DB klasoru silinemedi ({exc}). Koleksiyon yerinde temizlenip yeniden doldurulacak.")

    print("ChromaDB olusturuluyor...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
        embedding_function=embedding_fn,
    )

    if rebuild and not recreate_dir:
        clear_collection_in_place(vector_store)

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["content"] for chunk in batch]
        metas = [
            {
                "source_url": chunk.get("source_url", ""),
                "kategori": chunk.get("kategori", ""),
                "chunk_tipi": chunk.get("chunk_tipi", ""),
                "cekim_tarihi": chunk.get("cekim_tarihi", ""),
                "madde_no": chunk.get("madde_no", ""),
                "program_scope": chunk.get("program_scope", ""),
                "topic": chunk.get("topic", ""),
                "source_title": chunk.get("source_title", ""),
                "years": chunk.get("years", ""),
                "chunk_id": chunk.get("chunk_id", ""),
            }
            for chunk in batch
        ]
        ids = [chunk["chunk_id"] for chunk in batch]
        vector_store.add_texts(texts=texts, metadatas=metas, ids=ids)
        print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)} eklendi...")

    final_count = int(vector_store._collection.count())
    print(f"ChromaDB hazir: {final_count} kayit -> {DB_DIR}")
    if final_count != len(chunks):
        print(f"UYARI: Chunk sayisi ({len(chunks)}) ile ChromaDB kayit sayisi ({final_count}) esit degil.")


if __name__ == "__main__":
    build(rebuild="--rebuild" in sys.argv)
