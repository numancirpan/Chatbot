from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import TOPIC_HINTS, normalize_text

CHUNKS_PATH = ROOT_DIR / "data" / "chunks.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def collect_query_texts(record: Dict) -> List[str]:
    texts: List[str] = []
    for key in ["query", "query_variants", "followups", "expected_points"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        elif isinstance(value, list):
            texts.extend(str(item).strip() for item in value if str(item).strip())

    for key in ["messages", "conversation"]:
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    content = str(item.get("content", "")).strip()
                    if content:
                        texts.append(content)
                elif isinstance(item, str) and item.strip():
                    texts.append(item.strip())

    assistant = record.get("assistant")
    if isinstance(assistant, str) and assistant.strip():
        texts.append(assistant.strip())
    return texts


def important_terms(texts: Iterable[str]) -> List[str]:
    bag = normalize_text(" ".join(texts))
    tokens = [token for token in bag.split() if len(token) > 2]
    return tokens


def fake_url(url: str) -> bool:
    normalized = normalize_text(url)
    return not url or "ornek edu tr" in normalized or "example" in normalized


def candidate_score(record: Dict, chunk: Dict, tokens: List[str]) -> float:
    score = 0.0
    normalized_content = normalize_text(chunk.get("content", ""))
    normalized_url = normalize_text(chunk.get("source_url", ""))
    content_tokens = set(normalized_content.split())
    topic = record.get("topic", "")

    score += len(set(tokens) & content_tokens) * 2.0

    if topic and topic in TOPIC_HINTS:
        hints = TOPIC_HINTS[topic]
        hint_hits = sum(1 for hint in hints if hint in normalized_content or hint in normalized_url)
        score += hint_hits * 3.0

    query = record.get("query", "")
    if query:
        normalized_query = normalize_text(query)
        if any(part in normalized_content for part in normalized_query.split() if len(part) > 3):
            score += 4.0

    title = normalize_text(record.get("source_title", ""))
    if title and title in normalized_content:
        score += 2.0

    if chunk.get("kategori") == "staj" and topic == "staj":
        score += 4.0
    if "akademik takvim" in normalized_content and topic in {"ders_kaydi", "akademik_takvim_duyurular", "yaz_okulu"}:
        score += 3.0
    if "transkript" in normalized_content and topic == "ogrenci_belgesi_transkript":
        score += 4.0
    if "askerlik" in normalized_content and topic == "askerlik_tecili":
        score += 4.0

    return score


def top_candidates(record: Dict, chunks: List[Dict], limit: int = 5) -> List[Dict]:
    texts = collect_query_texts(record)
    tokens = important_terms(texts)
    ranked: List[Tuple[float, Dict]] = []

    for chunk in chunks:
        score = candidate_score(record, chunk, tokens)
        if score <= 0:
            continue
        ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    seen_urls = set()
    candidates: List[Dict] = []
    for score, chunk in ranked:
        url = chunk.get("source_url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        candidates.append(
            {
                "score": round(score, 2),
                "source_url": url,
                "kategori": chunk.get("kategori", ""),
                "content_preview": chunk.get("content", "")[:280],
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def patch_golden(records: List[Dict], chunks: List[Dict]) -> List[Dict]:
    patched = []
    for record in records:
        clone = dict(record)
        candidates = top_candidates(record, chunks)
        clone["url_candidates"] = candidates
        if candidates:
            best_url = candidates[0]["source_url"]
            clone["suggested_source_url"] = best_url
            expected_source_terms = clone.get("expected_source_terms")
            if isinstance(expected_source_terms, list) and (
                not expected_source_terms or any(fake_url(term) for term in expected_source_terms)
            ):
                clone["expected_source_terms"] = [best_url]
        patched.append(clone)
    return patched


def patch_retrieval(records: List[Dict], chunks: List[Dict]) -> List[Dict]:
    patched = []
    for record in records:
        clone = dict(record)
        candidates = top_candidates(record, chunks)
        clone["url_candidates"] = candidates
        if candidates:
            best_url = candidates[0]["source_url"]
            for group_name in ["positive_chunks", "hard_negative_chunks"]:
                group = clone.get(group_name, [])
                if not isinstance(group, list):
                    continue
                if group_name == "positive_chunks":
                    for item in group:
                        if fake_url(str(item.get("source_url", ""))):
                            item["source_url"] = best_url
                else:
                    alt_urls = [candidate["source_url"] for candidate in candidates[1:]]
                    for index, item in enumerate(group):
                        if fake_url(str(item.get("source_url", ""))) and alt_urls:
                            item["source_url"] = alt_urls[min(index, len(alt_urls) - 1)]
        patched.append(clone)
    return patched


def patch_generation(records: List[Dict], chunks: List[Dict]) -> List[Dict]:
    patched = []
    for record in records:
        clone = dict(record)
        candidates = top_candidates(record, chunks)
        clone["url_candidates"] = candidates
        if candidates:
            best_url = candidates[0]["source_url"]
            sources = clone.get("sources", [])
            if isinstance(sources, list):
                for item in sources:
                    if fake_url(str(item.get("url", ""))):
                        item["url"] = best_url
        patched.append(clone)
    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--suffix", default="_url_suggestions")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Ayrica duzeltilmis veriyi giris dosyalarinin uzerine yazar.",
    )
    args = parser.parse_args()

    chunks = load_json(CHUNKS_PATH)
    golden_path = Path(args.golden)
    retrieval_path = Path(args.retrieval)
    generation_path = Path(args.generation)

    golden = load_json(golden_path)
    retrieval = load_json(retrieval_path)
    generation = load_json(generation_path)

    golden_out = golden_path.with_name(golden_path.stem + args.suffix + golden_path.suffix)
    retrieval_out = retrieval_path.with_name(retrieval_path.stem + args.suffix + retrieval_path.suffix)
    generation_out = generation_path.with_name(generation_path.stem + args.suffix + generation_path.suffix)

    patched_golden = patch_golden(golden, chunks)
    patched_retrieval = patch_retrieval(retrieval, chunks)
    patched_generation = patch_generation(generation, chunks)

    dump_json(golden_out, patched_golden)
    dump_json(retrieval_out, patched_retrieval)
    dump_json(generation_out, patched_generation)

    if args.apply:
        dump_json(golden_path, patched_golden)
        dump_json(retrieval_path, patched_retrieval)
        dump_json(generation_path, patched_generation)

    print(
        json.dumps(
            {
                "golden_output": str(golden_out),
                "retrieval_output": str(retrieval_out),
                "generation_output": str(generation_out),
                "applied_to_inputs": args.apply,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
