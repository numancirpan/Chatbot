"""Merge curated official supplemental records into data/knowledge_base.json.

The crawler is still the preferred source for large-scale collection. This script
keeps small, high-value official records in a separate JSON file so missing
coverage can be added without hiding facts inside prompt code.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
SUPPLEMENTAL_PATH = DATA_DIR / "supplemental_sources.json"


def content_hash(text: str) -> str:
    return hashlib.md5(" ".join(text.split()).encode("utf-8")).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalized_url(record: dict) -> str:
    return str(record.get("url", "")).strip().rstrip("/").lower()


def main() -> int:
    knowledge_base = load_json(KNOWLEDGE_BASE_PATH)
    supplemental = load_json(SUPPLEMENTAL_PATH)
    if isinstance(knowledge_base, dict):
        knowledge_base = [knowledge_base]

    existing_by_url = {normalized_url(record): index for index, record in enumerate(knowledge_base)}
    merged = 0
    updated = 0

    for source in supplemental:
        record = {
            "url": str(source["url"]).strip(),
            "kategori": str(source.get("kategori", "genel")).strip() or "genel",
            "icerik": str(source.get("icerik", "")).strip(),
            "icerik_tipi": str(source.get("icerik_tipi", "html")).strip() or "html",
            "source_title": str(source.get("source_title", "")).strip(),
            "cekim_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "durum": "basarili",
            "degisiklik": True,
        }
        record["icerik_hash"] = content_hash(record["icerik"])
        key = normalized_url(record)
        if key in existing_by_url:
            index = existing_by_url[key]
            current = knowledge_base[index]
            if current.get("icerik_hash") != record["icerik_hash"] or current.get("kategori") != record["kategori"]:
                knowledge_base[index] = {**current, **record}
                updated += 1
            continue
        knowledge_base.append(record)
        existing_by_url[key] = len(knowledge_base) - 1
        merged += 1

    write_json(KNOWLEDGE_BASE_PATH, knowledge_base)
    print(json.dumps({"added": merged, "updated": updated, "total": len(knowledge_base)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
