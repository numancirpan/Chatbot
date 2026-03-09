"""
create_vector_db.py  (pipeline/create_vector_db.py)

chunks.json → ChromaDB (db/chroma_db/)

Eğer chroma_db zaten doluysa yeniden oluşturmaz, sadece bilgi verir.
Yeniden oluşturmak için --rebuild flag'i kullanın.
"""

import json
import os
import sys
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
DB_DIR     = os.path.join(ROOT_DIR, "db", "chroma_db")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")

os.makedirs(DB_DIR, exist_ok=True)


def build(rebuild: bool = False):
    # Mevcut DB varsa ve rebuild istenmiyorsa atla
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR) and not rebuild:
        client = chromadb.PersistentClient(path=DB_DIR)
        cols = client.list_collections()
        if cols:
            total = cols[0].count()
            print(f"✅ ChromaDB zaten mevcut ({total} kayıt). Atlanıyor.")
            print("   Yeniden oluşturmak için: python create_vector_db.py --rebuild")
            return

    print("🔄 Embedding modeli yükleniyor...")
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"📥 {len(chunks)} chunk yüklendi")

    # Eski DB'yi temizle
    import shutil
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR)

    print("🔄 ChromaDB oluşturuluyor...")
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_fn
    )

    # Toplu ekleme (batch) — büyük veri setleri için
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]
        metas = [{
            "source_url": c.get("source_url", ""),
            "kategori":   c.get("kategori", ""),
            "chunk_tipi": c.get("chunk_tipi", ""),
            "cekim_tarihi": c.get("cekim_tarihi", ""),
            "madde_no":   c.get("madde_no", ""),
        } for c in batch]
        vector_store.add_texts(texts=texts, metadatas=metas)
        print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)} eklendi...")

    print(f"✅ ChromaDB hazır → {DB_DIR}")


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    build(rebuild=rebuild)
