"""
write_vector_db_worker.py

ChromaDB yazimini olabildigince sade bir process'te yapar.
Amac, ana build scriptinden tamamen ayrik sekilde HNSW indeks dosyalarini
guvenilir bicimde olusturmaktir.
"""

import json
import os
import shutil
import sys
import time

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
COLLECTION_NAME = "langchain"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.chatbot import enrich_chunk_metadata


def hnsw_files_ready(db_dir: str) -> bool:
    expected = {"data_level0.bin", "header.bin", "length.bin", "link_lists.bin"}
    if not os.path.exists(db_dir):
        return False
    for entry in os.scandir(db_dir):
        if not entry.is_dir():
            continue
        present = {name for name in os.listdir(entry.path) if name in expected}
        if expected.issubset(present):
            return True
    return False


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Kullanim: python pipeline/write_vector_db_worker.py <hedef_db_dir>")

    target_dir = sys.argv[1]
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if isinstance(chunks, dict):
        chunks = [chunks]
    chunks = [enrich_chunk_metadata(chunk) for chunk in chunks]

    embedding_fn = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": True},
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=target_dir,
        embedding_function=embedding_fn,
    )

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vector_store.add_texts(
            texts=[chunk["content"] for chunk in batch],
            metadatas=[
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
            ],
            ids=[chunk["chunk_id"] for chunk in batch],
        )
        print(f"  {min(start + batch_size, len(chunks))}/{len(chunks)} eklendi...")

    for attempt in range(1, 11):
        if hnsw_files_ready(target_dir):
            print(f"HNSW indeks dosyalari hazir ({attempt}/10).")
            return
        print(f"HNSW indeks dosyalari bekleniyor ({attempt}/10)...")
        time.sleep(2)

    raise RuntimeError(f"HNSW indeks dosyalari olusmadi: {target_dir}")


if __name__ == "__main__":
    main()
