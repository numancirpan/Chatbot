from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple


def chroma_sqlite_path(db_dir: str) -> str:
    return os.path.join(db_dir, "chroma.sqlite3")


def candidate_vector_db_dirs(root_dir: str) -> List[str]:
    db_root = os.path.join(root_dir, "db")
    return [
        os.path.join(db_root, "chroma_store_live"),
        os.path.join(db_root, "chroma_db_candidate"),
        os.path.join(db_root, "chroma_db"),
    ]


def resolve_vector_db_dir(root_dir: str) -> str:
    for db_dir in candidate_vector_db_dirs(root_dir):
        health = subprocess_vector_store_health(db_dir)
        if health.get("queryable") and health.get("sqlite_count"):
            return db_dir
    for db_dir in candidate_vector_db_dirs(root_dir):
        if sqlite_embedding_count(db_dir):
            return db_dir
    return candidate_vector_db_dirs(root_dir)[0]


def sqlite_embedding_count(db_dir: str) -> Optional[int]:
    sqlite_path = chroma_sqlite_path(db_dir)
    if not os.path.exists(sqlite_path):
        return None
    try:
        with sqlite3.connect(sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM embeddings")
            return int(cur.fetchone()[0])
    except Exception:
        return None


def sqlite_collection_names(db_dir: str) -> List[str]:
    sqlite_path = chroma_sqlite_path(db_dir)
    if not os.path.exists(sqlite_path):
        return []
    try:
        with sqlite3.connect(sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM collections")
            return [str(row[0]) for row in cur.fetchall() if row and row[0]]
    except Exception:
        return []


def safe_vector_count(vector_store: Any) -> Tuple[Optional[int], Optional[str]]:
    try:
        return int(vector_store._collection.count()), None
    except Exception as exc:
        return None, str(exc)


def safe_similarity_probe(vector_store: Any, sample_query: str = "ogrenci") -> Tuple[bool, Optional[str]]:
    try:
        vector_store.similarity_search(sample_query, k=1)
        return True, None
    except Exception as exc:
        return False, str(exc)


def vector_store_health(vector_store: Any, db_dir: str, sample_query: str = "ogrenci") -> Dict[str, Any]:
    count, count_error = safe_vector_count(vector_store)
    queryable, probe_error = safe_similarity_probe(vector_store, sample_query=sample_query)
    sqlite_count = sqlite_embedding_count(db_dir)
    return {
        "count": count,
        "count_error": count_error,
        "queryable": queryable,
        "probe_error": probe_error,
        "sqlite_count": sqlite_count,
        "collection_names": sqlite_collection_names(db_dir),
    }


def _run_probe_subprocess(db_dir: str, mode: str, query: str = "ogrenci", k: int = 1) -> Dict[str, Any]:
    probe_code = """
import json
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.vector_db_utils import sqlite_collection_names, sqlite_embedding_count, vector_store_health

db_dir = sys.argv[1]
mode = sys.argv[2]
query = sys.argv[3]
k = int(sys.argv[4])
output_path = sys.argv[5]

emb = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True},
)
store = Chroma(
    collection_name="langchain",
    persist_directory=db_dir,
    embedding_function=emb,
)

if mode == "health":
    payload = vector_store_health(store, db_dir, sample_query=query)
else:
    docs = store.similarity_search(query, k=k)
    payload = {
        "results": [
            {
                "page_content": doc.page_content,
                "metadata": dict(doc.metadata),
            }
            for doc in docs
        ],
        "sqlite_count": sqlite_embedding_count(db_dir),
        "collection_names": sqlite_collection_names(db_dir),
    }

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)
"""
    output_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    output_handle.close()
    output_path = output_handle.name
    command = [sys.executable, "-c", probe_code, db_dir, mode, query, str(k), output_path]
    env = dict(os.environ)
    for key in ["KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK", "TORCHINDUCTOR_CACHE_DIR"]:
        env.pop(key, None)
    env["TOKENIZERS_PARALLELISM"] = "false"
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(stderr or f"Vector DB subprocess probe failed for {db_dir}")
        if not os.path.exists(output_path) or not os.path.getsize(output_path):
            raise RuntimeError(f"Vector DB subprocess probe returned empty output for {db_dir}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def subprocess_vector_store_health(db_dir: str, sample_query: str = "ogrenci") -> Dict[str, Any]:
    try:
        result = _run_probe_subprocess(db_dir, "health", query=sample_query, k=1)
        result["health_source"] = "fresh_process"
        return result
    except Exception as exc:
        return {
            "count": None,
            "count_error": str(exc),
            "queryable": False,
            "probe_error": str(exc),
            "sqlite_count": sqlite_embedding_count(db_dir),
            "collection_names": sqlite_collection_names(db_dir),
            "health_source": "fresh_process_error",
        }


def subprocess_similarity_search(db_dir: str, query: str, k: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        result = _run_probe_subprocess(db_dir, "search", query=query, k=k)
        return result.get("results", []), None
    except Exception as exc:
        return [], str(exc)
