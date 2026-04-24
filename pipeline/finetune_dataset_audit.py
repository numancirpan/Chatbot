from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RETRIEVAL_PATH = DATA_DIR / "retrieval_finetune_data.json"
GENERATION_PATH = DATA_DIR / "generation_finetune_data.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_counter(title: str, counter: Counter, limit: int = 20) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in counter.most_common(limit):
        print(f"{key or 'bos'}: {value}")


def contains_mojibake(value: str) -> bool:
    return any(marker in value for marker in ("Ã", "Ä", "Å", "�"))


def main() -> int:
    retrieval = load_json(RETRIEVAL_PATH)
    generation = load_json(GENERATION_PATH)

    retrieval_topics = Counter(str(item.get("topic", "")).strip() for item in retrieval)
    generation_topics = Counter(str(item.get("topic", "")).strip() for item in generation)
    positives = [len(item.get("positive_chunks", [])) for item in retrieval]
    negatives = [len(item.get("hard_negative_chunks", [])) for item in retrieval]
    turns = [len(item.get("conversation", [])) for item in retrieval]
    generation_turns = [
        len([message for message in item.get("messages", []) if message.get("role") == "user"])
        for item in generation
    ]
    evidence_counts = [len(item.get("evidence_sentences", [])) for item in generation]
    retrieval_unique_ids = len({item.get("id") for item in retrieval})
    generation_unique_ids = len({item.get("id") for item in generation})
    mojibake_generation = sum(
        1 for item in generation if contains_mojibake(item.get("assistant", "")) or any(
            contains_mojibake(sentence) for sentence in item.get("evidence_sentences", [])
        )
    )
    short_generation = sum(1 for item in generation if len(str(item.get("assistant", "")).strip()) < 80)

    print("FINETUNE DATASET AUDIT")
    print("======================")
    print(f"Retrieval sample sayisi: {len(retrieval)}")
    print(f"Generation sample sayisi: {len(generation)}")
    print(f"Retrieval benzersiz id: {retrieval_unique_ids}")
    print(f"Generation benzersiz id: {generation_unique_ids}")
    if positives:
        print(f"Ortalama positive chunk: {mean(positives):.2f}")
        print(f"Ortalama hard negative chunk: {mean(negatives):.2f}")
        print(f"Ortalama retrieval diyalog turu: {mean(turns):.2f}")
    if generation_turns:
        print(f"Ortalama generation diyalog turu: {mean(generation_turns):.2f}")
    if evidence_counts:
        print(f"Ortalama evidence sentence: {mean(evidence_counts):.2f}")
    print(f"Mojibake supheli generation kaydi: {mojibake_generation}")
    print(f"Cok kisa generation kaydi: {short_generation}")

    print_counter("Retrieval topic dagilimi", retrieval_topics)
    print_counter("Generation topic dagilimi", generation_topics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
