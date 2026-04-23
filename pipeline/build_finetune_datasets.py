from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import RAGChatbot, normalize_text
from pipeline.evaluate_golden import DEFAULT_GOLDEN_FILE, run_case

CHUNKS_FILE = ROOT_DIR / "data" / "chunks.json"
RETRIEVAL_OUTPUT = ROOT_DIR / "data" / "retrieval_finetune_data.json"
GENERATION_OUTPUT = ROOT_DIR / "data" / "generation_finetune_data.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_url(url: str) -> str:
    return url.strip().lower().rstrip("/")


def dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id") or f"{chunk.get('source_url', '')}:{chunk.get('content', '')[:80]}"
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(chunk)
    return unique


def find_positive_chunks(case: Dict, chunks: List[Dict]) -> List[Dict]:
    source_terms = [term.lower() for term in case.get("expected_source_terms", []) if term]
    if not source_terms:
        return []

    positives = []
    for chunk in chunks:
        url = normalize_url(chunk.get("source_url", ""))
        if any(term in url for term in source_terms):
            positives.append(chunk)
    return dedupe_chunks(positives)


def find_negative_chunks(case: Dict, chunks: List[Dict], positives: List[Dict], limit: int = 5) -> List[Dict]:
    positive_ids = {chunk.get("chunk_id") for chunk in positives}
    query_terms = set(normalize_text(" ".join(case.get("turns", []))).split())
    negatives = []

    for chunk in chunks:
        if chunk.get("chunk_id") in positive_ids:
            continue
        normalized_content = normalize_text(chunk.get("content", ""))
        content_terms = set(normalized_content.split())
        overlap = len(query_terms & content_terms)
        if overlap == 0:
            continue
        candidate = {
            "chunk_id": chunk.get("chunk_id"),
            "source_url": chunk.get("source_url", ""),
            "content": chunk.get("content", ""),
            "score": overlap,
        }
        negatives.append(candidate)

    ranked = sorted(negatives, key=lambda item: item["score"], reverse=True)
    trimmed = ranked[:limit]
    for item in trimmed:
        item.pop("score", None)
    return trimmed


def build_retrieval_samples(cases: List[Dict], chunks: List[Dict]) -> List[Dict]:
    samples = []
    for case in cases:
        positives = find_positive_chunks(case, chunks)
        if not positives:
            continue
        negatives = find_negative_chunks(case, chunks, positives)
        samples.append(
            {
                "id": case["id"],
                "query": case["turns"][-1],
                "conversation": case["turns"],
                "positive_chunks": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "source_url": chunk.get("source_url", ""),
                        "content": chunk.get("content", ""),
                    }
                    for chunk in positives
                ],
                "hard_negative_chunks": negatives,
            }
        )
    return samples


def build_generation_samples(cases: List[Dict], bot: RAGChatbot) -> List[Dict]:
    samples = []
    for case in cases:
        bot.clear_memory()
        result = None
        for turn in case["turns"]:
            result = bot.chat(turn)

        if result is None:
            continue

        evaluation = run_case(bot, case)
        if not evaluation["passed"]:
            continue

        samples.append(
            {
                "id": case["id"],
                "messages": [
                    {"role": "user", "content": turn}
                    for turn in case["turns"]
                ],
                "assistant": result["cevap"],
                "sources": result["kaynaklar"],
                "expected_answer_terms": case.get("expected_answer_terms", []),
            }
        )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN_FILE))
    parser.add_argument("--chunks", default=str(CHUNKS_FILE))
    args = parser.parse_args()

    cases = load_json(Path(args.golden))
    chunks = load_json(Path(args.chunks))

    retrieval_samples = build_retrieval_samples(cases, chunks)
    bot = RAGChatbot()
    generation_samples = build_generation_samples(cases, bot)

    with open(RETRIEVAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(retrieval_samples, f, ensure_ascii=False, indent=2)

    with open(GENERATION_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(generation_samples, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "retrieval_samples": len(retrieval_samples),
                "generation_samples": len(generation_samples),
                "retrieval_output": str(RETRIEVAL_OUTPUT),
                "generation_output": str(GENERATION_OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
