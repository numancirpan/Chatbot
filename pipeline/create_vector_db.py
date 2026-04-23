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
import sys
from typing import Dict

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DB_DIR = os.path.join(ROOT_DIR, "db", "chroma_db")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.chatbot import enrich_chunk_metadata


def chroma_counts() -> Dict[str, int]:
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        return {}
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        return {collection.name: collection.count() for collection in client.list_collections()}
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
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    reset_db_dir()

    print("ChromaDB olusturuluyor...")
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_fn,
    )

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
        vector_store.add_texts(texts=texts, metadatas=metas)
        print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)} eklendi...")

    final_count = total_chroma_count()
    print(f"ChromaDB hazir: {final_count} kayit -> {DB_DIR}")
    if final_count != len(chunks):
        print(f"UYARI: Chunk sayisi ({len(chunks)}) ile ChromaDB kayit sayisi ({final_count}) esit degil.")


if __name__ == "__main__":
    build(rebuild="--rebuild" in sys.argv)
