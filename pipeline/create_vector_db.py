"""
create_vector_db.py

data/chunks.json -> db/chroma_db/

ChromaDB repoda tutulmaz; yerelde chunks.json dosyasindan yeniden uretilir.
Mevcut veritabani doluysa varsayilan olarak atlanir. Bos veya bozuksa yeniden
olusturulur. Zorla yenilemek icin --rebuild kullanin.
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List

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
from core.vector_db_utils import sqlite_embedding_count, subprocess_vector_store_health


def load_chunks() -> List[Dict]:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if isinstance(chunks, dict):
        chunks = [chunks]
    return [enrich_chunk_metadata(chunk) for chunk in chunks]


def build_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": True},
    )


def reset_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def chroma_counts(db_dir: str = DB_DIR) -> Dict[str, int]:
    embedding_count = sqlite_embedding_count(db_dir)
    if embedding_count is None:
        return {}
    return {COLLECTION_NAME: embedding_count}


def verify_db(db_dir: str, expected_count: int) -> int:
    sqlite_count = sqlite_embedding_count(db_dir)

    if sqlite_count == expected_count:
        return int(sqlite_count)

    health = subprocess_vector_store_health(db_dir)

    raise RuntimeError(
        "ChromaDB dogrulamasi basarisiz: "
        f"dir={db_dir} | "
        f"count={health['count']} | "
        f"sqlite_count={health['sqlite_count']} | "
        f"queryable={health['queryable']} | "
        f"count_error={health['count_error']} | "
        f"probe_error={health['probe_error']}"
    )


def build(rebuild: bool = False) -> None:
    chunks = load_chunks()
    print(f"{len(chunks)} chunk yuklendi")

    existing_counts = chroma_counts()
    existing_total = sum(existing_counts.values())

    if existing_total > 0 and not rebuild:
        health = subprocess_vector_store_health(DB_DIR)
        is_fresh = health["count"] == len(chunks)
        if is_fresh and health["queryable"]:
            print(f"ChromaDB zaten guncel ({health['count']} kayit). Atlaniyor.")
            print(f"Koleksiyonlar: {health['collection_names'] or [COLLECTION_NAME]}")
            print("Yeniden olusturmak icin: python pipeline/create_vector_db.py --rebuild")
            return

        print("Mevcut ChromaDB kullanilabilir degil veya guncel degil. Yeniden olusturulacak.")
        print(
            "Detaylar: "
            f"sqlite_count={health['sqlite_count']} | "
            f"collection_count={health['count']} | "
            f"queryable={health['queryable']} | "
            f"count_error={health['count_error']} | "
            f"probe_error={health['probe_error']}"
        )

    print("Embedding modeli yukleniyor...")
    embedding_fn = build_embedding_function()

    reset_dir(DB_DIR)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
        embedding_function=embedding_fn,
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
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
        print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)} eklendi...")

    final_count = verify_db(DB_DIR, len(chunks))
    print(f"ChromaDB hazir: {final_count} kayit -> {DB_DIR}")
    print("ChromaDB sayim kontrolu basarili.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(rebuild=args.rebuild)
