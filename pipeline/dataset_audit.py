"""
dataset_audit.py

knowledge_base.json ve chunks.json dosyalarini hizli kalite kontrolunden gecirir.
Amac, veri kalitesini tartisilabilir olmaktan cikarip olculebilir hale getirmektir.
"""

import argparse
import json
import os
from collections import Counter
from statistics import mean
from urllib.parse import urlparse


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
KB_FILE = os.path.join(DATA_DIR, "knowledge_base.json")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default=KB_FILE)
    parser.add_argument("--chunks", default=CHUNKS_FILE)
    args = parser.parse_args()

    kb = load_json(args.kb)
    chunks = load_json(args.chunks)

    lengths = chunk_lengths(chunks)

    print("DATASET AUDIT")
    print("=============")
    print(f"Ham kayit sayisi: {len(kb)}")
    print(f"Chunk sayisi: {len(chunks)}")
    print(f"Benzersiz kaynak sayisi: {len({item.get('source_url', '') for item in chunks})}")

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
        Counter(item.get("kategori", "bos") for item in chunks)
    )

    print("\nSupheli chunk ornekleri")
    print("-----------------------")
    for score, item in suspicious_chunks(chunks):
        snippet = item.get("content", "").replace("\n", " ")[:220]
        print(f"Skor={score} | Kategori={item.get('kategori', 'bos')} | URL={item.get('source_url', '')}")
        print(f"  {snippet}")


if __name__ == "__main__":
    main()
