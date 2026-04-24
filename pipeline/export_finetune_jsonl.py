from __future__ import annotations

import json
from hashlib import md5
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RETRIEVAL_PATH = DATA_DIR / "retrieval_finetune_data.json"
GENERATION_PATH = DATA_DIR / "generation_finetune_data.json"
EXPORT_DIR = DATA_DIR / "exports"

import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import repair_text_encoding


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: str) -> str:
    return repair_text_encoding(str(value)).strip()


def dataset_split(record_id: str, validation_ratio: float = 0.15) -> str:
    bucket = int(md5(record_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "validation" if bucket < int(validation_ratio * 100) else "train"


def build_retrieval_triplets(records):
    rows = []
    for record in records:
        query = record.get("query", "")
        positives = record.get("positive_chunks", [])
        negatives = record.get("hard_negative_chunks", [])
        for positive in positives:
            for negative in negatives:
                rows.append(
                    {
                        "id": record.get("id"),
                        "split": dataset_split(str(record.get("id", ""))),
                        "query": query,
                        "positive": clean_text(positive.get("content", "")),
                        "negative": clean_text(negative.get("content", "")),
                        "positive_source_url": positive.get("source_url", ""),
                        "negative_source_url": negative.get("source_url", ""),
                        "topic": record.get("topic", ""),
                    }
                )
    return rows


def build_retrieval_pairs(records):
    rows = []
    for record in records:
        query = record.get("query", "")
        for positive in record.get("positive_chunks", []):
            rows.append(
                {
                    "id": record.get("id"),
                    "split": dataset_split(str(record.get("id", ""))),
                    "query": query,
                    "document": clean_text(positive.get("content", "")),
                    "label": 1,
                    "source_url": positive.get("source_url", ""),
                    "topic": record.get("topic", ""),
                }
            )
        for negative in record.get("hard_negative_chunks", []):
            rows.append(
                {
                    "id": record.get("id"),
                    "split": dataset_split(str(record.get("id", ""))),
                    "query": query,
                    "document": clean_text(negative.get("content", "")),
                    "label": 0,
                    "source_url": negative.get("source_url", ""),
                    "topic": record.get("topic", ""),
                }
            )
    return rows


def build_generation_messages(records):
    rows = []
    system_text = (
        "Sen Düzce Üniversitesi için resmi kaynak odaklı, kısa ve doğru cevap üreten bir asistansın. "
        "Belgelerde olmayan bilgiyi uydurmazsın."
    )
    for record in records:
        messages = [{"role": "system", "content": system_text}]
        messages.extend(
            {"role": msg.get("role", ""), "content": clean_text(msg.get("content", ""))}
            for msg in record.get("messages", [])
        )
        messages.append({"role": "assistant", "content": clean_text(record.get("assistant", ""))})
        rows.append(
            {
                "id": record.get("id"),
                "split": dataset_split(str(record.get("id", ""))),
                "topic": record.get("topic", ""),
                "messages": messages,
                "sources": record.get("sources", []),
                "expected_points": record.get("expected_points", []),
                "evidence_sentences": record.get("evidence_sentences", []),
            }
        )
    return rows


def build_generation_instruct(records):
    rows = []
    for record in records:
        user_turns = [msg.get("content", "") for msg in record.get("messages", []) if msg.get("role") == "user"]
        prompt = "\n".join(clean_text(turn) for turn in user_turns)
        rows.append(
            {
                "id": record.get("id"),
                "split": dataset_split(str(record.get("id", ""))),
                "instruction": "Soruyu resmi kaynak odaklı ve kurumsal bir dille yanıtla.",
                "input": prompt,
                "output": clean_text(record.get("assistant", "")),
                "topic": record.get("topic", ""),
                "sources": record.get("sources", []),
                "evidence_sentences": record.get("evidence_sentences", []),
            }
        )
    return rows


def main() -> int:
    retrieval = load_json(RETRIEVAL_PATH)
    generation = load_json(GENERATION_PATH)

    retrieval_triplets = build_retrieval_triplets(retrieval)
    retrieval_pairs = build_retrieval_pairs(retrieval)
    generation_messages = build_generation_messages(generation)
    generation_instruct = build_generation_instruct(generation)

    write_jsonl(EXPORT_DIR / "retrieval_triplets.jsonl", retrieval_triplets)
    write_jsonl(EXPORT_DIR / "retrieval_pairs.jsonl", retrieval_pairs)
    write_jsonl(EXPORT_DIR / "generation_messages.jsonl", generation_messages)
    write_jsonl(EXPORT_DIR / "generation_instruct.jsonl", generation_instruct)

    summary = {
        "retrieval_triplets": len(retrieval_triplets),
        "retrieval_pairs": len(retrieval_pairs),
        "generation_messages": len(generation_messages),
        "generation_instruct": len(generation_instruct),
        "export_dir": str(EXPORT_DIR),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
