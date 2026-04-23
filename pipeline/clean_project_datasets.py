from __future__ import annotations

import json
import re
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
    "https://cdn.duzce.edu.tr/File/GetFile/264a1c72-d5aa-4885-961e-75e0ba94acd5": "2024-2025 Eğitim-Öğretim Yılı Akademik Takvimi",
    "https://cdn.duzce.edu.tr/File/GetFile/b67441bf-8724-4f5a-b340-a963f8354d6a": "Ders Muafiyeti ve İntibak Esasları",
    "https://cdn.duzce.edu.tr/File/GetFile/5c5910a2-7cf3-475c-86ff-83539c7b73da": "Düzce Üniversitesi Ön Lisans Eğitim-Öğretim ve Sınav Yönetmeliği",
    "https://cdn.duzce.edu.tr/File/GetFile/a40f9bfd-0852-44d9-ade0-1c543f4fee6b": "Düzce Üniversitesi Yabancı Dil Hazırlık Sınıfı Eğitim-Öğretim ve Sınav Yönetmeliği",
    "https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943": "Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği",
    "https://cdn.duzce.edu.tr/File/GetFile/0a10d86d-7915-4164-b60b-2e4d3ac59f19": "Yaz Okulu Uygulama Esasları",
    "https://cdn.duzce.edu.tr/File/GetFile/1ed55517-ee0c-431a-a214-7a41d74eaa70": "2020-2021 Eğitim-Öğretim Yılı Akademik Takvimi",
}

RECORD_OVERRIDES = {
    "staj_001": {
        "source_url": "https://bm.mf.duzce.edu.tr/sayfa/4a82/staj",
        "source_title": "Bilgisayar Mühendisliği - Staj",
    },
    "ders_kaydi_001": {
        "source_url": "https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943",
        "source_title": "Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği",
    },
    "ders_kaydi_003": {
        "source_url": "https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943",
        "source_title": "Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği",
    },
    "add_drop_001": {
        "source_url": "https://cdn.duzce.edu.tr/File/GetFile/264a1c72-d5aa-4885-961e-75e0ba94acd5",
        "source_title": "2024-2025 Eğitim-Öğretim Yılı Akademik Takvimi",
    },
    "add_drop_002": {
        "source_url": "https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943",
        "source_title": "Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği",
    },
    "add_drop_003": {
        "source_url": "https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943",
        "source_title": "Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği",
    },
    "devamsizlik_001": {
        "source_url": "https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943",
        "source_title": "Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği",
    },
    "sinavlar_003": {
        "source_url": "https://cdn.duzce.edu.tr/File/GetFile/264a1c72-d5aa-4885-961e-75e0ba94acd5",
        "source_title": "2024-2025 Eğitim-Öğretim Yılı Akademik Takvimi",
    },
    "not_sistemi_ortalama_001": {
        "source_url": "https://cdn.duzce.edu.tr/File/GetFile/e61c395b-bd05-4300-b312-0753ab748943",
        "source_title": "Düzce Üniversitesi Lisans Eğitim-Öğretim ve Sınav Yönetmeliği",
    },
}

POSITIVE_CHUNK_OVERRIDES = {
    "staj_003": "7d103ce44ad2d6dd9d92b8a276dbad38",
    "add_drop_001": "d1775d9771e3b7d6eb135744db96de9a",
    "add_drop_002": "3a37be35d2398551bd67c01feddcd575",
    "add_drop_003": "3a37be35d2398551bd67c01feddcd575",
}

STOP_TERMS = {
    "ve",
    "veya",
    "ile",
    "icin",
    "için",
    "nasil",
    "nasıl",
    "nedir",
    "ne",
    "mi",
    "mı",
    "mu",
    "mü",
    "kadar",
    "olan",
    "olarak",
    "ilgili",
    "gerekli",
    "gerekir",
    "soru",
    "sorusu",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", normalize_text(text))
        if len(token) > 2 and token not in STOP_TERMS
    ]

def record_terms(record: Dict, fallback_query: str = "") -> List[str]:
    values: List[str] = []
    if record:
        values.append(str(record.get("query", "")))
        values.extend(str(point) for point in record.get("expected_points", []))
        values.extend(str(item) for item in record.get("query_variants", [])[:3])
        values.extend(str(item) for item in record.get("followups", [])[:2])
    if fallback_query:
        values.append(fallback_query)
    terms: List[str] = []
    for value in values:
        terms.extend(tokenize(value))
    return sorted(set(terms))


def score_text(terms: List[str], text: str) -> int:
    normalized = normalize_text(text)
    score = 0
    for term in terms:
        if term in normalized:
            score += max(1, min(len(term.split()) + len(term) // 8, 4))
    return score


def best_chunk(
    candidates: List[Dict],
    terms: List[str],
    used_ids: set[str] | None = None,
) -> Dict | None:
    ranked = sorted(
        candidates,
        key=lambda chunk: (
            score_text(terms, chunk.get("content", "")),
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
        override = RECORD_OVERRIDES.get(str(clone.get("id", "")).strip(), {})
        if override:
            clone.update(override)
        url = canonical_source_url(record)
        if override.get("source_url"):
            url = str(override["source_url"]).strip()
        if url:
            clone["source_url"] = url
            clone["source_title"] = str(override.get("source_title", "")).strip() or str(clone.get("source_title", "")).strip() or infer_title(url, "")
        clone.pop("suggested_source_url", None)
        clone.pop("url_candidates", None)
        cleaned.append(clone)
    return cleaned


def build_gold_map(golden: List[Dict]) -> Dict[str, Dict]:
    return {str(record.get("id", "")).strip(): record for record in golden}


def cleanup_retrieval(
    retrieval: List[Dict],
    chunks_by_url: Dict[str, List[Dict]],
    chunks_by_id: Dict[str, Dict],
    golden_by_id: Dict[str, Dict],
) -> List[Dict]:
    cleaned = []
    for record in retrieval:
        record_id = str(record.get("id", "")).strip()
        golden_record = golden_by_id.get(record_id, {})
        query = str(record.get("query", "")).strip() or str(golden_record.get("query", "")).strip() or " ".join(case_turns(record))
        terms = record_terms(golden_record, query)
        used_ids: set[str] = set()
        clone = {
            "id": record_id,
            "topic": record.get("topic") or golden_record.get("topic"),
            "query": query,
            "conversation": record.get("conversation", case_turns(record)),
        }

        positives = []
        positive_url = str(golden_record.get("source_url", "")).strip()
        if positive_url:
            override_chunk_id = POSITIVE_CHUNK_OVERRIDES.get(record_id, "")
            chunk = chunks_by_id.get(override_chunk_id) if override_chunk_id else None
            if chunk and str(chunk.get("source_url", "")).strip() != positive_url:
                chunk = None
            if not chunk:
                chunk = best_chunk(chunks_by_url.get(positive_url, []), terms, used_ids)
            if chunk:
                used_ids.add(str(chunk.get("chunk_id", "")))
                positives.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "source_url": positive_url,
                        "content": chunk.get("content", ""),
                    }
                )
        clone["positive_chunks"] = positives

        negatives = []
        ranked_negatives = []
        for url, url_chunks in chunks_by_url.items():
            if not url or url == positive_url:
                continue
            chunk = best_chunk(url_chunks, terms, used_ids)
            if not chunk:
                continue
            ranked_negatives.append((score_text(terms, chunk.get("content", "")), chunk, url))

        for _, chunk, url in sorted(ranked_negatives, key=lambda item: item[0], reverse=True):
            if str(chunk.get("chunk_id", "")) in used_ids:
                continue
            used_ids.add(str(chunk.get("chunk_id", "")))
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


def assistant_text(source_title: str, expected_points: List[str], follow_up: bool) -> str:
    points = ", ".join(point for point in expected_points if point) or "ilgili resmî hükümlerin"
    if follow_up:
        return (
            "Sayın öğrencimiz,\n"
            f"Takip sorunuz değerlendirilirken {source_title} kaynağındaki ilgili hükümler ile özellikle {points} dikkate alınmalıdır.\n\n"
            "Kesin ve güncel uygulama için ilgili madde veya resmî duyuru metninin incelenmesi önerilir."
        )

    return (
        "Sayın öğrencimiz,\n"
        f"Bu konuda esas alınması gereken bilgi {source_title} başlıklı resmî kaynaktır. Yanıt verilirken özellikle {points} dikkate alınmalıdır.\n\n"
        "Kesin ve güncel uygulama için ilgili resmî duyuru, yönerge veya yönetmelik maddesinin incelenmesi önerilir."
    )


def cleanup_generation(generation: List[Dict], golden_by_id: Dict[str, Dict]) -> List[Dict]:
    cleaned = []
    for record in generation:
        record_id = str(record.get("id", "")).strip()
        base_id = record_id.split("_dialog_")[0]
        golden_record = golden_by_id.get(base_id, {})
        source_url = str(golden_record.get("source_url", "")).strip()
        source_title = str(golden_record.get("source_title", "")).strip() or infer_title(source_url, "")
        expected_points = list(golden_record.get("expected_points", record.get("expected_points", record.get("expected_answer_terms", []))))
        messages = record.get("messages", [])
        follow_up = "_dialog_" in record_id or len([m for m in messages if m.get("role") == "user"]) > 1
        clone = {
            "id": record_id,
            "topic": record.get("topic") or golden_record.get("topic"),
            "messages": messages,
            "assistant": assistant_text(source_title, expected_points, follow_up),
            "sources": [{"title": source_title, "url": source_url}] if source_url else [normalize_source_item(source) for source in record.get("sources", [])],
            "expected_points": expected_points,
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


def build_chunks_by_id(chunks: Iterable[Dict]) -> Dict[str, Dict]:
    return {str(chunk.get("chunk_id", "")).strip(): chunk for chunk in chunks}


def main() -> int:
    chunks = load_json(CHUNKS_PATH)
    chunks_by_url = build_chunks_by_url(chunks)
    chunks_by_id = build_chunks_by_id(chunks)

    golden = cleanup_golden(load_json(GOLDEN_PATH))
    golden_by_id = build_gold_map(golden)
    retrieval = cleanup_retrieval(load_json(RETRIEVAL_PATH), chunks_by_url, chunks_by_id, golden_by_id)
    generation = cleanup_generation(load_json(GENERATION_PATH), golden_by_id)

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
