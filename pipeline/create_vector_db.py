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
import subprocess
import sys
import time
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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


def build_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"local_files_only": True},
    )


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


<<<<<<< Updated upstream
def build(rebuild: bool = False):
=======
def run_write_subprocess(target_dir: str) -> None:
    worker_code = f"""
import json
import os
import shutil
import sys
import time

root = {ROOT_DIR!r}
target_dir = {target_dir!r}
chunks_file = {CHUNKS_FILE!r}
collection_name = {COLLECTION_NAME!r}
model_name = {EMBEDDING_MODEL_NAME!r}

sys.path.insert(0, root)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from core.chatbot import enrich_chunk_metadata

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir, exist_ok=True)

with open(chunks_file, encoding="utf-8") as f:
    chunks = json.load(f)
if isinstance(chunks, dict):
    chunks = [chunks]
chunks = [enrich_chunk_metadata(c) for c in chunks]

emb = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={{"local_files_only": True}})
store = Chroma(collection_name=collection_name, persist_directory=target_dir, embedding_function=emb)

batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    texts = [c["content"] for c in batch]
    metas = [{{
        "source_url": c.get("source_url", ""),
        "kategori": c.get("kategori", ""),
        "chunk_tipi": c.get("chunk_tipi", ""),
        "cekim_tarihi": c.get("cekim_tarihi", ""),
        "madde_no": c.get("madde_no", ""),
        "program_scope": c.get("program_scope", ""),
        "topic": c.get("topic", ""),
        "source_title": c.get("source_title", ""),
        "years": c.get("years", ""),
        "chunk_id": c.get("chunk_id", ""),
    }} for c in batch]
    ids = [c["chunk_id"] for c in batch]
    store.add_texts(texts=texts, metadatas=metas, ids=ids)
    print(f"  {{min(i + batch_size, len(chunks))}}/{{len(chunks)}} eklendi...")
print("Yazim tamamlandi, dogrulama ayri process'te yapilacak.")
"""
    command = [
        sys.executable,
        "-c",
        worker_code,
    ]
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
    subprocess.run(command, check=True)


def build(rebuild: bool = False) -> None:
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
    reset_db_dir()
=======
    print("Staging veritabani ayri process'te dogrulaniyor...")
    run_verify_subprocess(STAGING_DB_DIR, len(chunks))
>>>>>>> Stashed changes

    print("Staging dogrulandi, canli veritabani ile yer degistiriliyor...")
    swap_staging_into_place()

<<<<<<< Updated upstream
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
=======
    print("Canli veritabani son kez dogrulaniyor...")
    run_verify_subprocess(DB_DIR, len(chunks))

    print(f"ChromaDB hazir: {len(chunks)} kayit -> {DB_DIR}")
    print("ChromaDB saglik kontrolu basarili: sayim ve sorgu provasi gecti.")
>>>>>>> Stashed changes


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
