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
import time
from typing import Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DB_PARENT_DIR = os.path.join(ROOT_DIR, "db")
DB_DIR = os.path.join(DB_PARENT_DIR, "chroma_store_live")
STAGING_DB_DIR = os.path.join(DB_PARENT_DIR, "chroma_store_staging")
BACKUP_DB_DIR = os.path.join(DB_PARENT_DIR, "chroma_store_backup")
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
    last_health = None

    for attempt in range(1, 7):
        last_health = subprocess_vector_store_health(db_dir)
        if last_health["queryable"] and last_health["count"] == expected_count:
            return int(last_health["count"])

        print(
            f"Dogrulama bekleniyor ({attempt}/6): "
            f"count={last_health['count']} | "
            f"sqlite_count={last_health['sqlite_count']} | "
            f"queryable={last_health['queryable']}"
        )
        time.sleep(3)

    raise RuntimeError(
        "ChromaDB dogrulamasi basarisiz: "
        f"dir={db_dir} | "
        f"count={last_health['count']} | "
        f"sqlite_count={last_health['sqlite_count']} | "
        f"queryable={last_health['queryable']} | "
        f"count_error={last_health['count_error']} | "
        f"probe_error={last_health['probe_error']}"
    )


def swap_staging_into_place() -> None:
    if os.path.exists(BACKUP_DB_DIR):
        shutil.rmtree(BACKUP_DB_DIR)
    if os.path.exists(DB_DIR):
        os.replace(DB_DIR, BACKUP_DB_DIR)
    os.replace(STAGING_DB_DIR, DB_DIR)
    if os.path.exists(BACKUP_DB_DIR):
        shutil.rmtree(BACKUP_DB_DIR)


def run_write_subprocess(target_dir: str) -> None:
    command = [
        sys.executable,
        os.path.join(ROOT_DIR, "pipeline", "write_vector_db_worker.py"),
        target_dir,
    ]
    import subprocess

    subprocess.run(command, check=True)


def run_verify_subprocess(target_dir: str, expected_count: int) -> None:
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--verify-dir",
        target_dir,
        "--expected-count",
        str(expected_count),
    ]
    import subprocess

    subprocess.run(command, check=True)


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

    print("Staging veritabani olusturuluyor...")
    run_write_subprocess(STAGING_DB_DIR)
    print("Staging veritabani ayri process'te dogrulaniyor...")
    run_verify_subprocess(STAGING_DB_DIR, len(chunks))

    print("Staging dogrulandi, canli veritabani ile yer degistiriliyor...")
    swap_staging_into_place()
    print("Canli veritabani son kez dogrulaniyor...")
    run_verify_subprocess(DB_DIR, len(chunks))

    print(f"ChromaDB hazir: {len(chunks)} kayit -> {DB_DIR}")
    print("ChromaDB saglik kontrolu basarili: sayim ve sorgu provasi gecti.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--verify-dir")
    parser.add_argument("--expected-count", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify_dir:
        verified_count = verify_db(args.verify_dir, args.expected_count)
        print(f"Dogrulama basarili: {verified_count} kayit -> {args.verify_dir}")
    else:
        build(rebuild=args.rebuild)
