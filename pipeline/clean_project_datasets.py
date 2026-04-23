from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import normalize_text
from pipeline.evaluate_golden import case_turns

DATA_DIR = ROOT_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
GOLDEN_PATH = DATA_DIR / "golden_questions.json"
RETRIEVAL_PATH = DATA_DIR / "retrieval_finetune_data.json"
GENERATION_PATH = DATA_DIR / "generation_finetune_data.json"

TITLE_OVERRIDES = {
    "https://bm.mf.duzce.edu.tr/sayfa/4a82/staj": "Bilgisayar Mühendisliği - Staj",
    "https://bm.mf.duzce.edu.tr/sayfa/878b/stajlar-hakkinda-sikca-sorulan-sorular": "Bilgisayar Mühendisliği - Staj SSS",
    "https://mf.duzce.edu.tr/sayfa/8a10/ders-kayitlari-esnasinda-sikca-sorulan-sorular-ve-cevaplari": "Ders Kayıtları Esnasında Sıkça Sorulan Sorular ve Cevapları",
    "https://cdn.duzce.edu.tr/File/GetFile/264a1c72-d5aa-4885-961e-75e0ba94acd5": "Akademik Takvim",
    "https://cdn.duzce.edu.tr/File/GetFile/b67441bf-8724-4f5a-b340-a963f8354d6a": "Ders Muafiyeti ve İntibak Esasları",
    "https://cdn.duzce.edu.tr/File/GetFile/5c5910a2-7cf3-475c-86ff-83539c7b73da": "Kayıt Yenileme İşlemleri",
    "https://cdn.duzce.edu.tr/File/GetFile/a40f9bfd-0852-44d9-ade0-1c543f4fee6b": "Kayıt Yenileme İşlemleri",
    "https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943": "Kayıt Yenileme İşlemleri",
    "https://cdn.duzce.edu.tr/File/GetFile/0a10d86d-7915-4164-b60b-2e4d3ac59f19": "Yaz Okulu Uygulama Esasları",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def score_text(query: str, text: str) -> int:
    query_terms = {term for term in normalize_text(query).split() if len(term) > 2}
    text_terms = set(normalize_text(text).split())
    return len(query_terms & text_terms)


def best_chunk_for_url(chunks_by_url: Dict[str, List[Dict]], url: str, query: str, used_ids: set[str] | None = None) -> Dict | None:
    candidates = chunks_by_url.get(url, [])
    ranked = sorted(
        candidates,
        key=lambda chunk: (
            score_text(query, chunk.get("content", "")),
            len(chunk.get("content", "")),
        ),
        reverse=True,
    )
    for chunk in ranked:
        chunk_id = str(chunk.get("chunk_id", ""))
        if used_ids and chunk_id in used_ids:
            continue
        return chunk
    return None


def infer_title(url: str, fallback: str = "") -> str:
    return TITLE_OVERRIDES.get(url, fallback)


def canonical_source_url(record: Dict) -> str:
    suggested = str(record.get("suggested_source_url", "")).strip()
    source_url = str(record.get("source_url", "")).strip()
    return suggested or source_url


def cleanup_golden(golden: List[Dict]) -> List[Dict]:
    cleaned = []
    for record in golden:
        clone = dict(record)
        url = canonical_source_url(record)
        if url:
            clone["source_url"] = url
            clone["source_title"] = infer_title(url, str(clone.get("source_title", "")).strip())
        clone.pop("suggested_source_url", None)
        clone.pop("url_candidates", None)
        cleaned.append(clone)
    return cleaned


def cleanup_retrieval(retrieval: List[Dict], chunks_by_url: Dict[str, List[Dict]]) -> List[Dict]:
    cleaned = []
    for record in retrieval:
        query = str(record.get("query", "")).strip() or " ".join(case_turns(record))
        used_ids: set[str] = set()
        clone = {
            "id": record.get("id"),
            "topic": record.get("topic"),
            "query": query,
            "conversation": record.get("conversation", case_turns(record)),
        }

        positives = []
        for item in record.get("positive_chunks", [])[:1]:
            url = str(item.get("source_url", "")).strip()
            chunk = best_chunk_for_url(chunks_by_url, url, query, used_ids)
            if not chunk:
                continue
            used_ids.add(str(chunk.get("chunk_id", "")))
            positives.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_url": url,
                    "content": chunk.get("content", ""),
                }
            )
        clone["positive_chunks"] = positives

        negatives = []
        seen_urls = {item["source_url"] for item in positives}
        for item in record.get("hard_negative_chunks", []):
            url = str(item.get("source_url", "")).strip()
            if not url or url in seen_urls:
                continue
            chunk = best_chunk_for_url(chunks_by_url, url, query, used_ids)
            if not chunk:
                continue
            used_ids.add(str(chunk.get("chunk_id", "")))
            seen_urls.add(url)
            negatives.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_url": url,
                    "content": chunk.get("content", ""),
                }
            )
            if len(negatives) >= 3:
                break
        clone["hard_negative_chunks"] = negatives

        cleaned.append(clone)
    return cleaned


def normalize_source_item(source: Dict) -> Dict:
    clone = dict(source)
    url = str(clone.get("url", "")).strip()
    if url:
        clone["url"] = url
        if "title" in clone:
            clone["title"] = infer_title(url, str(clone.get("title", "")).strip())
        if "baslik" in clone:
            clone["baslik"] = infer_title(url, str(clone.get("baslik", "")).strip())
    return clone


def cleanup_generation(generation: List[Dict]) -> List[Dict]:
    cleaned = []
    for record in generation:
        clone = {
            "id": record.get("id"),
            "topic": record.get("topic"),
            "messages": record.get("messages", []),
            "assistant": record.get("assistant", ""),
            "sources": [normalize_source_item(source) for source in record.get("sources", [])],
            "expected_points": record.get("expected_points", record.get("expected_answer_terms", [])),
        }
        cleaned.append(clone)
    return cleaned


def build_chunks_by_url(chunks: Iterable[Dict]) -> Dict[str, List[Dict]]:
    mapping: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        url = str(chunk.get("source_url", "")).strip()
        if not url:
            continue
        mapping.setdefault(url, []).append(chunk)
    return mapping


def main() -> int:
    chunks = load_json(CHUNKS_PATH)
    chunks_by_url = build_chunks_by_url(chunks)

    golden = cleanup_golden(load_json(GOLDEN_PATH))
    retrieval = cleanup_retrieval(load_json(RETRIEVAL_PATH), chunks_by_url)
    generation = cleanup_generation(load_json(GENERATION_PATH))

    dump_json(GOLDEN_PATH, golden)
    dump_json(RETRIEVAL_PATH, retrieval)
    dump_json(GENERATION_PATH, generation)

    print(
        json.dumps(
            {
                "golden_records": len(golden),
                "retrieval_records": len(retrieval),
                "generation_records": len(generation),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
