from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import enrich_chunk_metadata, infer_source_title, normalize_text, repair_text_encoding

CHUNKS_FILE = ROOT_DIR / "data" / "chunks.json"
RETRIEVAL_OUTPUT = ROOT_DIR / "data" / "retrieval_finetune_data.json"
GENERATION_OUTPUT = ROOT_DIR / "data" / "generation_finetune_data.json"

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
NOISE_MARKERS = [
    "baskanligimiz hakkimizda",
    "kalite komisyon",
    "organizasyon semasi",
    "tanitim videosu",
    "gorsel",
    "logo",
]
QUESTION_STOPWORDS = {
    "duzce",
    "universitesi",
    "ogrenci",
    "isleri",
    "daire",
    "baskanligi",
    "madde",
    "birinci",
    "ikinci",
    "ucuncu",
    "bolum",
    "genel",
    "esas",
    "hakkinda",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def clean_text(value: str) -> str:
    return repair_text_encoding(str(value)).strip()


def dedupe_by_key(items: Iterable[Dict], key_fields: List[str]) -> List[Dict]:
    seen = set()
    unique = []
    for item in items:
        key = tuple(str(item.get(field, "")).strip() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def normalize_url(url: str) -> str:
    return str(url).strip().lower().rstrip("/")


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def split_for_id(sample_id: str) -> str:
    bucket = int(stable_hash(sample_id)[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def split_sentences(text: str) -> List[str]:
    sentences = []
    text = " ".join(clean_text(text).split())
    for raw in SENTENCE_SPLIT_PATTERN.split(text):
        sentence = " ".join(raw.split()).strip(" -\t")
        if len(sentence) < 80:
            continue
        normalized = normalize_text(sentence)
        if any(marker in normalized for marker in NOISE_MARKERS):
            continue
        letters = [char for char in sentence if char.isalpha()]
        if letters:
            uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
            if uppercase_ratio > 0.65:
                continue
        sentences.append(sentence)
    return sentences


def important_terms(text: str) -> List[str]:
    terms = []
    for token in normalize_text(text).split():
        if len(token) < 4:
            continue
        if token in QUESTION_STOPWORDS:
            continue
        terms.append(token)
    return terms


def chunk_is_usable(chunk: Dict) -> bool:
    content = clean_text(chunk.get("content", ""))
    normalized = normalize_text(content)
    if len(content) < 120:
        return False
    if any(marker in normalized for marker in NOISE_MARKERS):
        return False
    if not chunk.get("source_url"):
        return False
    return bool(split_sentences(content))


def source_title(chunk: Dict) -> str:
    return clean_text(chunk.get("source_title") or infer_source_title(chunk) or "Resmi kaynak")


def query_focus_terms(chunk: Dict) -> List[str]:
    text = " ".join(
        [
            source_title(chunk),
            clean_text(chunk.get("kategori", "")),
            clean_text(chunk.get("topic", "")),
            clean_text(chunk.get("content", ""))[:500],
        ]
    )
    seen = set()
    terms = []
    for term in important_terms(text):
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= 6:
            break
    return terms


def question_templates(chunk: Dict) -> List[str]:
    title = source_title(chunk)
    content_norm = normalize_text(chunk.get("content", ""))
    terms = query_focus_terms(chunk)
    focus = " ".join(terms[:3]).strip()

    questions = [
        f"{title} hakkinda bilgi verir misiniz?",
        f"{title} icin resmi kurallar nelerdir?",
    ]

    if focus:
        questions.append(f"{focus} konusunda ne yapmam gerekiyor?")
    if "basvuru" in content_norm:
        questions.append(f"{title} basvurusu nasil yapilir?")
    if any(marker in content_norm for marker in ["sart", "kosul", "gerekli"]):
        questions.append(f"{title} sartlari nelerdir?")
    if any(marker in content_norm for marker in ["tarih", "takvim", "sure", "gun", "hafta"]):
        questions.append(f"{title} icin tarih veya sure bilgisi nedir?")
    if any(marker in content_norm for marker in ["belge", "evrak", "form", "dilekce"]):
        questions.append(f"{title} icin hangi belgeler gerekir?")

    cleaned = []
    seen = set()
    for question in questions:
        question = clean_text(question)
        key = normalize_text(question)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(question)
    return cleaned[:4]


def chunk_query_score(query: str, chunk: Dict) -> float:
    query_terms = set(important_terms(query))
    content_terms = chunk.get("_training_content_terms") or set(important_terms(chunk.get("content", "")))
    title_terms = chunk.get("_training_title_terms") or set(important_terms(source_title(chunk)))
    score = float(len(query_terms & content_terms))
    score += 2.0 * len(query_terms & title_terms)
    if chunk.get("topic") and normalize_text(chunk.get("topic", "")) in normalize_text(query):
        score += 4.0
    return score


def prepare_training_chunks(chunks: List[Dict], max_chunks: int) -> List[Dict]:
    usable = []
    per_source_counts: Dict[str, int] = {}
    for chunk in chunks:
        if not chunk_is_usable(chunk):
            continue
        url = normalize_url(chunk.get("source_url", ""))
        source_count = per_source_counts.get(url, 0)
        if source_count >= 4:
            continue
        prepared = dict(chunk)
        prepared["_training_content_terms"] = set(important_terms(prepared.get("content", "")))
        prepared["_training_title_terms"] = set(important_terms(source_title(prepared)))
        usable.append(prepared)
        per_source_counts[url] = source_count + 1
        if max_chunks and len(usable) >= max_chunks:
            break
    return usable


def find_negative_chunks(query: str, chunks: List[Dict], positives: List[Dict], limit: int = 4) -> List[Dict]:
    positive_ids = {chunk.get("chunk_id") for chunk in positives}
    positive_urls = {normalize_url(chunk.get("source_url", "")) for chunk in positives}
    scored = []

    for chunk in chunks:
        if chunk.get("chunk_id") in positive_ids:
            continue
        if normalize_url(chunk.get("source_url", "")) in positive_urls:
            continue
        score = chunk_query_score(query, chunk)
        if score <= 0:
            continue
        scored.append((score, chunk))

    negatives = []
    used_urls = set()
    for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True):
        url = normalize_url(chunk.get("source_url", ""))
        if url in used_urls:
            continue
        used_urls.add(url)
        negatives.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "source_url": chunk.get("source_url", ""),
                "source_title": source_title(chunk),
                "content": clean_text(chunk.get("content", "")),
            }
        )
        if len(negatives) >= limit:
            break
    return negatives


def evidence_sentences(chunk: Dict, limit: int = 3) -> List[str]:
    sentences = split_sentences(chunk.get("content", ""))
    if not sentences:
        return []
    return sentences[:limit]


def sample_id(prefix: str, chunk: Dict, query: str) -> str:
    base = chunk.get("chunk_id") or stable_hash(chunk.get("content", ""))[:16]
    suffix = stable_hash(query)[:10]
    return f"{prefix}_{base}_{suffix}"


def positive_payload(chunk: Dict) -> Dict:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "source_url": chunk.get("source_url", ""),
        "source_title": source_title(chunk),
        "content": clean_text(chunk.get("content", "")),
    }


def build_retrieval_samples(usable_chunks: List[Dict], max_questions_per_chunk: int = 3) -> List[Dict]:
    samples = []

    for chunk in usable_chunks:
        positives = [chunk]
        for query in question_templates(chunk)[:max_questions_per_chunk]:
            sid = sample_id("retrieval", chunk, query)
            samples.append(
                {
                    "id": sid,
                    "split": split_for_id(sid),
                    "topic": clean_text(chunk.get("topic", "")),
                    "query": query,
                    "positive_chunks": [positive_payload(item) for item in positives],
                    "hard_negative_chunks": find_negative_chunks(query, usable_chunks, positives),
                    "source": "chunks",
                }
            )

    return dedupe_by_key(samples, ["id"])


def assistant_text(chunk: Dict) -> str:
    title = source_title(chunk)
    sentences = evidence_sentences(chunk, limit=3)
    body = "\n".join(f"- {sentence}" for sentence in sentences)
    return clean_text(
        "\n".join(
            [
                "Sayin ogrencimiz,",
                f"{title} kaynaginda yer alan bilgiye gore:",
                "",
                body,
                "",
                "Kesin ve guncel uygulama icin ilgili resmi kaynak metni esas alinmalidir.",
            ]
        )
    )


def build_generation_samples(usable_chunks: List[Dict], max_questions_per_chunk: int = 2) -> List[Dict]:
    samples = []

    for chunk in usable_chunks:
        sentences = evidence_sentences(chunk)
        if not sentences:
            continue
        for query in question_templates(chunk)[:max_questions_per_chunk]:
            sid = sample_id("generation", chunk, query)
            samples.append(
                {
                    "id": sid,
                    "split": split_for_id(sid),
                    "topic": clean_text(chunk.get("topic", "")),
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Sen Duzce Universitesi icin resmi kaynak odakli, kisa ve dogru "
                                "cevap ureten bir asistansin. Belgelerde olmayan bilgiyi uydurmazsin."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    "assistant": assistant_text(chunk),
                    "sources": [
                        {
                            "title": source_title(chunk),
                            "url": chunk.get("source_url", ""),
                            "chunk_id": chunk.get("chunk_id"),
                        }
                    ],
                    "evidence_sentences": [clean_text(sentence) for sentence in sentences],
                    "source": "chunks",
                }
            )

    return dedupe_by_key(samples, ["id"])


def split_counts(samples: List[Dict]) -> Dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for sample in samples:
        split = sample.get("split", "train")
        counts[split] = counts.get(split, 0) + 1
    return counts


def summarize(chunks: List[Dict], usable_chunks: List[Dict], retrieval_samples: List[Dict], generation_samples: List[Dict]) -> Dict:
    return {
        "source": "chunks",
        "chunks_total": len(chunks),
        "chunks_usable": len(usable_chunks),
        "retrieval_samples": len(retrieval_samples),
        "retrieval_split_counts": split_counts(retrieval_samples),
        "generation_samples": len(generation_samples),
        "generation_split_counts": split_counts(generation_samples),
        "retrieval_output": str(RETRIEVAL_OUTPUT),
        "generation_output": str(GENERATION_OUTPUT),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build fine-tune datasets from official source chunks. "
            "Evaluation cases are intentionally not used for training."
        )
    )
    parser.add_argument("--chunks", default=str(CHUNKS_FILE))
    parser.add_argument("--max-chunks", type=int, default=1500)
    parser.add_argument("--max-retrieval-questions-per-chunk", type=int, default=3)
    parser.add_argument("--max-generation-questions-per-chunk", type=int, default=2)
    args = parser.parse_args()

    chunks = [enrich_chunk_metadata(chunk) for chunk in load_json(Path(args.chunks))]

    usable_chunks = prepare_training_chunks(chunks, args.max_chunks)

    retrieval_samples = build_retrieval_samples(usable_chunks, args.max_retrieval_questions_per_chunk)
    generation_samples = build_generation_samples(usable_chunks, args.max_generation_questions_per_chunk)

    dump_json(RETRIEVAL_OUTPUT, retrieval_samples)
    dump_json(GENERATION_OUTPUT, generation_samples)

    print(json.dumps(summarize(chunks, usable_chunks, retrieval_samples, generation_samples), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
