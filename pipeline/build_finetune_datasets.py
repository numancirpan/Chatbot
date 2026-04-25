from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import enrich_chunk_metadata, normalize_text, repair_text_encoding
from pipeline.evaluate_golden import DEFAULT_GOLDEN_FILE, case_expected_source_terms, case_turns

CHUNKS_FILE = ROOT_DIR / "data" / "chunks.json"
RETRIEVAL_OUTPUT = ROOT_DIR / "data" / "retrieval_finetune_data.json"
GENERATION_OUTPUT = ROOT_DIR / "data" / "generation_finetune_data.json"
NOISE_MARKERS = [
    "baskanligimiz hakkimizda",
    "kalite kalite komisyon",
    "bolumumuz bolumu tanitim videosu",
    "yonetim organizasyon semasi",
]
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
SENTENCE_NOISE_MARKERS = [
    "baskanligimiz hakkimizda",
    "kalite komisyon",
    "organizasyon semasi",
    "tanitim videosu",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_url(url: str) -> str:
    return url.strip().lower().rstrip("/")


def clean_text(value: str) -> str:
    return repair_text_encoding(str(value)).strip()


def split_sentences(text: str) -> List[str]:
    sentences = []
    for raw in SENTENCE_SPLIT_PATTERN.split(clean_text(text)):
        sentence = " ".join(raw.split()).strip(" -\t")
        if len(sentence) < 40:
            continue
        normalized = normalize_text(sentence)
        if any(marker in normalized for marker in SENTENCE_NOISE_MARKERS):
            continue
        sentences.append(sentence)
    return sentences


def fold_text(text: str) -> str:
    folded = clean_text(text).lower()
    translation_table = str.maketrans(
        {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
            "Ç": "c",
            "Ğ": "g",
            "İ": "i",
            "Ö": "o",
            "Ş": "s",
            "Ü": "u",
        }
    )
    folded = folded.replace("i̇", "i")
    folded = folded.translate(translation_table)
    return re.sub(r"\s+", " ", folded)


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


def build_chunks() -> List[Dict]:
    chunks = load_json(CHUNKS_FILE)
    return [enrich_chunk_metadata(chunk) for chunk in chunks]


def build_chunks_by_url(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    mapping: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        url = normalize_url(chunk.get("source_url", ""))
        if not url:
            continue
        mapping.setdefault(url, []).append(chunk)
    return mapping


def important_terms(text: str) -> List[str]:
    return [
        token
        for token in normalize_text(text).split()
        if len(token) >= 3 and token not in {"icin", "gore", "nasil", "nedir", "olan", "veya", "ile"}
    ]


def conversation_variants(case: Dict) -> List[List[str]]:
    query = clean_text(case.get("query", ""))
    query_variants = [clean_text(item) for item in case.get("query_variants", []) if clean_text(item)]
    followups = [clean_text(item) for item in case.get("followups", []) if clean_text(item)]

    conversations: List[List[str]] = []
    if query:
        conversations.append([query])
    conversations.extend([[variant] for variant in query_variants])

    seed_queries = [query] + query_variants[:2]
    for seed in [item for item in seed_queries if item]:
        for followup in followups:
            conversations.append([seed, followup])

    return dedupe_conversations(conversations)


def dedupe_conversations(conversations: List[List[str]]) -> List[List[str]]:
    seen = set()
    unique = []
    for conversation in conversations:
        cleaned = tuple(turn.strip() for turn in conversation if turn.strip())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(list(cleaned))
    return unique


def find_positive_chunks(case: Dict, chunks_by_url: Dict[str, List[Dict]]) -> List[Dict]:
    source_terms = case_expected_source_terms(case)
    primary_source_url = str(source_terms[0]).strip() if source_terms else ""
    target_text = " ".join(
        [
            clean_text(case.get("query", "")),
            " ".join(clean_text(item) for item in case.get("query_variants", [])[:3]),
            " ".join(clean_text(item) for item in case.get("expected_points", [])),
        ]
    ).strip()
    target_terms = set(important_terms(target_text))
    positives = []
    for source_term in case_expected_source_terms(case):
        for url, url_chunks in chunks_by_url.items():
            if source_term.lower() in url.lower():
                positives.extend(url_chunks)

    scored = []
    for chunk in dedupe_by_key(positives, ["chunk_id"]):
        content = clean_text(chunk.get("content", ""))
        normalized_content = normalize_text(content)
        if any(marker in normalized_content for marker in NOISE_MARKERS):
            continue
        score = float(len(target_terms & set(important_terms(content))))
        if chunk.get("topic") == str(case.get("topic", "")).strip():
            score += 4
        if primary_source_url and chunk.get("source_url", "") == primary_source_url:
            score += 2
        if len(content) < 120:
            score -= 3
        scored.append((score, chunk))

    ranked = [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)]
    return ranked[:2]


def sentence_score(sentence: str, query: str, expected_points_list: List[str], topic: str, case: Dict) -> float:
    normalized_sentence = normalize_text(sentence)
    sentence_terms = set(normalized_sentence.split())
    query_terms = set(important_terms(query))
    expected_terms = set(important_terms(" ".join(expected_points_list)))
    query_normalized = fold_text(query)
    score = 0.0

    score += 3.0 * len(query_terms & sentence_terms)
    score += 2.0 * len(expected_terms & sentence_terms)

    if topic == "staj" and "staj" in normalized_sentence:
        score += 4.0
    if topic == "yaz_okulu" and "yaz okulu" in normalized_sentence:
        score += 4.0
    if topic == "ders_kaydi" and any(marker in normalized_sentence for marker in ["kayit", "ders", "danisman"]):
        score += 4.0

    if "kac" in query_normalized and any(unit in normalized_sentence for unit in ["is gunu", "gun", "hafta", "akts", "kredi"]):
        score += 6.0
    if "ne zaman" in query_normalized and any(ch.isdigit() for ch in sentence):
        score += 6.0
    if any(marker in query_normalized for marker in ["nasil", "adim", "surec"]) and any(
        marker in normalized_sentence for marker in ["basvuru", "teslim", "onay", "yukle", "sec"]
    ):
        score += 5.0

    topic_label = fold_text(case.get("topic_label", ""))
    if topic_label and topic_label in fold_text(sentence):
        score += 2.0
    if len(sentence) > 260:
        score -= 2.0
    return score


def sentence_matches_focus(query: str, sentence: str) -> bool:
    normalized_query = fold_text(query)
    normalized_sentence = fold_text(sentence)
    sentence_tokens = set(re.findall(r"[a-z0-9]+", normalized_sentence))

    if "staj" in normalized_query and "kac" in normalized_query:
        return "is gunu" in normalized_sentence or any(token in sentence_tokens for token in ["gun", "hafta", "ay"])
    if "yaz okulu" in normalized_query and "ne zaman" in normalized_query:
        return bool(re.search(r"\b20\d{2}\b", sentence) or re.search(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b", sentence))
    if "yaz okulu" in normalized_query and "kac" in normalized_query:
        return "hafta" in sentence_tokens
    if "hangi belge" in normalized_query or "evrak" in normalized_query:
        return any(marker in sentence_tokens for marker in ["belge", "evrak", "form", "rapor", "dilekce"])
    if "ne zaman" in normalized_query:
        return any(ch.isdigit() for ch in sentence)
    if "kac" in normalized_query:
        return (
            "is gunu" in normalized_sentence
            or any(marker in sentence_tokens for marker in ["gun", "hafta", "ay", "akts", "kredi"])
        )
    if any(marker in normalized_query for marker in ["nasil", "adim", "surec"]):
        return any(marker in sentence_tokens for marker in ["basvuru", "teslim", "onay", "yukle", "sec", "islem"])
    return True


def select_evidence_sentences(case: Dict, conversation: List[str], positives: List[Dict], limit: int = 3) -> List[str]:
    query = conversation[-1]
    topic = str(case.get("topic", "")).strip()
    points = expected_points(case)
    ranked = []

    for chunk in positives:
        for sentence in split_sentences(chunk.get("content", "")):
            ranked.append((sentence_score(sentence, query, points, topic, case), sentence))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if ranked:
        best_score = ranked[0][0]
    else:
        best_score = 0.0

    selected = []
    seen_normalized = set()
    for score, sentence in ranked:
        normalized = normalize_text(sentence)
        if score <= 0 or normalized in seen_normalized:
            continue
        if best_score and score < max(4.0, best_score - 5.0):
            continue
        if not sentence_matches_focus(query, sentence):
            continue
        seen_normalized.add(normalized)
        selected.append(sentence)
        if len(selected) >= limit:
            break
    return selected


def chunk_query_score(query: str, chunk: Dict, target_topic: str = "") -> float:
    query_terms = set(important_terms(query))
    content = clean_text(chunk.get("content", ""))
    content_terms = set(important_terms(content))
    score = float(len(query_terms & content_terms))

    if chunk.get("topic") == target_topic and target_topic:
        score += 6
    if any(term in normalize_text(content) for term in query_terms):
        score += 2
    if chunk.get("program_scope") and chunk.get("program_scope") != "genel" and chunk.get("program_scope") in normalize_text(query):
        score += 4
    if chunk.get("kategori") == "staj" and "staj" in normalize_text(query):
        score += 4
    if chunk.get("kategori") == "akademik_takvim" and "yaz okulu" in normalize_text(query):
        score += 4
    return score


def find_negative_chunks(
    query: str,
    all_chunks: List[Dict],
    positives: List[Dict],
    target_topic: str = "",
    limit: int = 4,
) -> List[Dict]:
    positive_ids = {chunk.get("chunk_id") for chunk in positives}
    positive_urls = {normalize_url(chunk.get("source_url", "")) for chunk in positives}
    scored = []
    for chunk in all_chunks:
        if chunk.get("chunk_id") in positive_ids:
            continue
        if normalize_url(chunk.get("source_url", "")) in positive_urls:
            continue
        score = chunk_query_score(query, chunk, target_topic=target_topic)
        if score <= 0:
            continue
        scored.append((score, chunk))

    ranked = [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)]
    negatives = []
    used_urls = set()
    for chunk in ranked:
        url = normalize_url(chunk.get("source_url", ""))
        if url in used_urls:
            continue
        used_urls.add(url)
        negatives.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "source_url": chunk.get("source_url", ""),
                "content": clean_text(chunk.get("content", "")),
            }
        )
        if len(negatives) >= limit:
            break
    return negatives


def retrieval_sample_id(case_id: str, conversation: List[str]) -> str:
    suffix = "__".join(normalize_text(turn).replace(" ", "_")[:48] for turn in conversation)
    return f"{case_id}__{suffix}"


def build_retrieval_samples(cases: List[Dict], chunks: List[Dict]) -> List[Dict]:
    chunks_by_url = build_chunks_by_url(chunks)
    samples = []

    for case in cases:
        topic = str(case.get("topic", "")).strip()
        positives = find_positive_chunks(case, chunks_by_url)
        if not positives:
            continue

        for conversation in conversation_variants(case):
            query = conversation[-1]
            negatives = find_negative_chunks(query, chunks, positives, target_topic=topic)
            samples.append(
                {
                    "id": retrieval_sample_id(str(case.get("id", "")), conversation),
                    "base_id": case.get("id"),
                    "topic": topic,
                    "query": clean_text(query),
                    "conversation": [clean_text(turn) for turn in conversation],
                    "positive_chunks": [
                        {
                            "chunk_id": chunk.get("chunk_id"),
                            "source_url": chunk.get("source_url", ""),
                            "content": clean_text(chunk.get("content", "")),
                        }
                        for chunk in positives
                    ],
                    "hard_negative_chunks": negatives,
                }
            )

    return dedupe_by_key(samples, ["id"])


def source_title(case: Dict) -> str:
    return (
        clean_text(case.get("source_title", ""))
        or clean_text(case.get("topic_label", ""))
        or clean_text(case.get("topic", ""))
        or "resmi kaynak"
    )


def expected_points(case: Dict) -> List[str]:
    points = [clean_text(item) for item in case.get("expected_points", []) if clean_text(item)]
    if points:
        return points

    expected_terms = [clean_text(item) for item in case.get("expected_answer_terms", []) if clean_text(item)]
    return expected_terms or [clean_text(case.get("topic", "resmi bilgi"))]


def summarise_evidence_sentence(sentence: str) -> str:
    sentence = clean_text(sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.rstrip(" ;,")


def assistant_text(case: Dict, conversation: List[str], positives: List[Dict]) -> str:
    title = source_title(case)
    points = expected_points(case)
    is_followup = len(conversation) > 1
    answer_style = clean_text(case.get("answer_style", "kurumsal_kisa"))
    evidence_sentences = select_evidence_sentences(case, conversation, positives)

    if evidence_sentences:
        if answer_style == "kurumsal_maddeli":
            body_lines = [f"- {summarise_evidence_sentence(sentence)}" for sentence in evidence_sentences[:3]]
            closing = "İlgili uygulamada resmi kaynak metni esas alınmalıdır."
            if is_followup:
                closing = "Takip işlemlerinde aynı resmi kaynak ve ilgili birim duyuruları esas alınmalıdır."
            return "\n".join(
                [
                    "Sayın öğrencimiz,",
                    f"{title} kaynağında yer alan ilgili bilgiler aşağıdadır:",
                    "",
                    *body_lines,
                    "",
                    closing,
                ]
            )

        summary = " ".join(summarise_evidence_sentence(sentence) for sentence in evidence_sentences[:2])
        if is_followup:
            return (
                "Sayın öğrencimiz,\n"
                f"Takip sorunuz açısından {title} kaynağında yer alan bilgiye göre {summary}\n\n"
                "Kesin uygulamada resmi kaynak metni esas alınmalıdır."
            )
        return (
            "Sayın öğrencimiz,\n"
            f"{title} kaynağındaki bilgiye göre {summary}\n\n"
            "Kesin ve güncel uygulama için resmi kaynak metni esas alınmalıdır."
        )

    point_text = ", ".join(points[:4])
    if is_followup:
        return (
            "Sayın öğrencimiz,\n"
            f"Takip sorunuz değerlendirilirken {title} kaynağındaki ilgili hükümler esas alınmalıdır. "
            f"Yanıt verilirken özellikle {point_text} dikkate alınmalıdır.\n\n"
            "Kesin uygulama için ilgili resmi kaynak metni birlikte kontrol edilmelidir."
        )

    return (
        "Sayın öğrencimiz,\n"
        f"Bu konuda esas alınması gereken bilgi {title} başlıklı resmi kaynaktır. "
        f"Yanıt verilirken özellikle {point_text} dikkate alınmalıdır.\n\n"
        "Kesin ve güncel uygulama için ilgili resmi kaynak metni esas alınmalıdır."
    )


def build_generation_samples(cases: List[Dict], chunks: List[Dict]) -> List[Dict]:
    chunks_by_url = build_chunks_by_url(chunks)
    samples = []
    for case in cases:
        positives = find_positive_chunks(case, chunks_by_url)
        if not positives:
            continue
        for conversation in conversation_variants(case):
            evidence_sentences = select_evidence_sentences(case, conversation, positives)
            samples.append(
                {
                    "id": retrieval_sample_id(str(case.get("id", "")), conversation),
                    "base_id": case.get("id"),
                    "topic": case.get("topic"),
                    "messages": [{"role": "user", "content": clean_text(turn)} for turn in conversation],
                    "assistant": clean_text(assistant_text(case, conversation, positives)),
                    "sources": [
                        {
                            "title": clean_text(source_title(case)),
                            "url": str(case_expected_source_terms(case)[0]).strip() if case_expected_source_terms(case) else "",
                        }
                    ],
                    "expected_points": expected_points(case),
                    "evidence_sentences": [clean_text(sentence) for sentence in evidence_sentences],
                }
            )
    return dedupe_by_key(samples, ["id"])


def summarize(cases: List[Dict], retrieval_samples: List[Dict], generation_samples: List[Dict]) -> Dict:
    return {
        "golden_cases": len(cases),
        "retrieval_samples": len(retrieval_samples),
        "generation_samples": len(generation_samples),
        "topics": sorted({str(case.get("topic", "")).strip() for case in cases if str(case.get("topic", "")).strip()}),
        "retrieval_output": str(RETRIEVAL_OUTPUT),
        "generation_output": str(GENERATION_OUTPUT),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN_FILE))
    parser.add_argument("--chunks", default=str(CHUNKS_FILE))
    args = parser.parse_args()

    cases = load_json(Path(args.golden))
    chunks = [enrich_chunk_metadata(chunk) for chunk in load_json(Path(args.chunks))]

    retrieval_samples = build_retrieval_samples(cases, chunks)
    generation_samples = build_generation_samples(cases, chunks)

    dump_json(RETRIEVAL_OUTPUT, retrieval_samples)
    dump_json(GENERATION_OUTPUT, generation_samples)

    print(json.dumps(summarize(cases, retrieval_samples, generation_samples), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
