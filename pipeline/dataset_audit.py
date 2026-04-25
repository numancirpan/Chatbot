"""
dataset_audit.py

knowledge_base.json ve chunks.json dosyalarini hizli kalite kontrolunden gecirir.
Amac, veri kalitesini tartisilabilir olmaktan cikarip olculebilir hale getirmektir.
"""

import argparse
import json
import os
import sys
from collections import Counter
from statistics import mean
from urllib.parse import urlparse

try:
    import chromadb
except ImportError:
    chromadb = None


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
KB_FILE = os.path.join(DATA_DIR, "knowledge_base.json")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.chatbot import enrich_chunk_metadata
from core.vector_db_utils import (
    resolve_vector_db_dir,
    sqlite_embedding_count,
    subprocess_vector_store_health,
)

DB_DIR = resolve_vector_db_dir(ROOT_DIR)

NOISE_MARKERS = [
    "anasayfa",
    "copyright",
    "iletisim",
    "organizasyon",
    "kalite komisyonu",
    "baskanligimiz hakkimizda",
]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower() or "unknown"
    except Exception:
        return "unknown"


def chunk_lengths(chunks):
    return [len(item.get("content", "")) for item in chunks]


def noise_score(text: str) -> int:
    lower = text.lower()
    return sum(marker in lower for marker in NOISE_MARKERS)


def suspicious_chunks(chunks, limit: int = 5):
    scored = []
    for item in chunks:
        content = item.get("content", "")
        score = noise_score(content)
        if len(content) < 100:
            score += 2
        if len(content) > 1200:
            score += 2
        if "|" in content:
            score += 1
        if score:
            scored.append((score, item))
    scored.sort(key=lambda row: row[0], reverse=True)
    return scored[:limit]


def print_counter(title: str, counter: Counter, limit: int = 10):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in counter.most_common(limit):
        print(f"{key or 'bos'}: {value}")


def chroma_counts(db_dir: str = DB_DIR):
    subprocess_health = subprocess_vector_store_health(db_dir)
    if subprocess_health.get("queryable"):
        count = subprocess_health.get("count")
        if isinstance(count, int):
            return {
                "langchain": count,
                "_source": subprocess_health.get("health_source", "fresh_process"),
            }
    if chromadb is None:
        sqlite_count = sqlite_embedding_count(db_dir)
        if sqlite_count is None:
            return None
        return {"langchain": sqlite_count, "_source": "sqlite"}
    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        return {}
    try:
        client = chromadb.PersistentClient(path=db_dir)
        counts = {collection.name: collection.count() for collection in client.list_collections()}
        if sum(counts.values()) == 0:
            sqlite_count = sqlite_embedding_count(db_dir)
            if sqlite_count is not None:
                return {"langchain": sqlite_count, "_source": "sqlite_zero_fallback"}
        return counts
    except Exception as exc:
        sqlite_count = sqlite_embedding_count(db_dir)
        if sqlite_count is not None:
            return {
                "langchain": sqlite_count,
                "_source": "sqlite_error_fallback",
                "_warning": str(exc),
            }
        return {"ERROR": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default=KB_FILE)
    parser.add_argument("--chunks", default=CHUNKS_FILE)
    args = parser.parse_args()

    kb = load_json(args.kb)
    chunks = load_json(args.chunks)
    enriched_chunks = [enrich_chunk_metadata(chunk) for chunk in chunks]

    lengths = chunk_lengths(enriched_chunks)

    print("DATASET AUDIT")
    print("=============")
    print(f"Ham kayit sayisi: {len(kb)}")
    print(f"Chunk sayisi: {len(chunks)}")
    print(f"Benzersiz kaynak sayisi: {len({item.get('source_url', '') for item in enriched_chunks})}")
    print(f"Aktif DB yolu: {DB_DIR}")

    counts = chroma_counts()
    if counts is None:
        print("ChromaDB kayit sayisi: kontrol edilemedi (chromadb kurulu degil)")
    elif "ERROR" in counts:
        print(f"ChromaDB kayit sayisi: kontrol hatasi ({counts['ERROR']})")
    else:
        count_values = [value for key, value in counts.items() if not key.startswith("_")]
        total_chroma = sum(count_values)
        print(f"ChromaDB kayit sayisi: {total_chroma}")
        if counts.get("_source"):
            print(f"Not: sayim kaynagi = {counts['_source']}")
        if counts.get("_warning"):
            print(f"Not: Chroma istemci uyarisi = {counts['_warning']}")
        if total_chroma != len(chunks):
            print(f"UYARI: ChromaDB ({total_chroma}) ve chunks.json ({len(chunks)}) sayilari esit degil.")

    if lengths:
        print(f"Ortalama chunk uzunlugu: {round(mean(lengths), 1)}")
        print(f"Min chunk uzunlugu: {min(lengths)}")
        print(f"Max chunk uzunlugu: {max(lengths)}")
        print(f"1200 ustu chunk: {sum(length > 1200 for length in lengths)}")
        print(f"100 alti chunk: {sum(length < 100 for length in lengths)}")

    print_counter(
        "Icerik tipleri",
        Counter(item.get("icerik_tipi", "bos") for item in kb)
    )
    print_counter(
        "Kaynak hostlari",
        Counter(host_of(item.get("url", "")) for item in kb)
    )
    print_counter(
        "Chunk kategorileri",
        Counter(item.get("kategori", "bos") for item in enriched_chunks)
    )
    print_counter(
        "Chunk konulari",
        Counter(item.get("topic", "bos") for item in enriched_chunks)
    )
    print_counter(
        "Chunk kapsam/birim tahmini",
        Counter(item.get("program_scope", "bos") for item in enriched_chunks)
    )

    print("\nSupheli chunk ornekleri")
    print("-----------------------")
    for score, item in suspicious_chunks(enriched_chunks):
        snippet = item.get("content", "").replace("\n", " ")[:220]
        print(f"Skor={score} | Kategori={item.get('kategori', 'bos')} | URL={item.get('source_url', '')}")
        print(f"  {snippet}")


if __name__ == "__main__":
    main()
