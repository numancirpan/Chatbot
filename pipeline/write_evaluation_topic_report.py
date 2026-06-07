from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import normalize_text

DEFAULT_EVALUATION_FILE = ROOT_DIR / "data" / "evaluation_cases.json"
DEFAULT_RESULTS_FILE = ROOT_DIR / "outputs" / "evaluation_cases_with_acceptable_latest.json"
DEFAULT_CHUNKS_FILE = ROOT_DIR / "data" / "chunks.json"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "outputs" / "evaluation_topic_failure_report.md"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_result(result: dict) -> str:
    answer = str(result.get("answer", ""))
    if not result.get("sources") or "bilgiye ulasilamadi" in normalize_text(answer):
        return "no_answer"
    return "wrong_source"


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "Yok"
    return "; ".join(
        f"{source.get('baslik', 'Kaynak')} ({source.get('url', '')})"
        for source in sources[:3]
    )


def source_text(chunk: dict) -> str:
    return " ".join(
        str(chunk.get(field, ""))
        for field in ("source_title", "source_url", "kategori", "content")
    )


def query_terms(query: str) -> list[str]:
    ignored = {
        "icin",
        "hangi",
        "nasil",
        "nereden",
        "nedir",
        "neye",
        "gore",
        "kac",
        "mi",
        "mu",
        "midir",
        "olur",
        "yapilir",
        "gerekiyor",
    }
    return [
        token
        for token in normalize_text(query).split()
        if len(token) >= 4 and token not in ignored
    ]


def chunk_evidence(row: dict, chunks: list[dict]) -> dict:
    expected_terms = [term for term in row["expected"] if term]
    normalized_expected = [normalize_text(term) for term in expected_terms]
    terms = query_terms(row["query"])
    best: dict | None = None

    for chunk in chunks:
        searchable = normalize_text(source_text(chunk))
        score = 0
        reasons: list[str] = []

        for raw_term, expected in zip(expected_terms, normalized_expected):
            if not expected:
                continue
            if raw_term.startswith("http") and raw_term in str(chunk.get("source_url", "")):
                score += 12
                reasons.append("expected_url")
            elif expected in searchable:
                score += 8
                reasons.append(f"expected_term:{raw_term[:60]}")

        matched_query_terms = [term for term in terms if term in searchable]
        score += min(len(matched_query_terms), 6)
        if matched_query_terms:
            reasons.append("query_terms:" + ",".join(matched_query_terms[:6]))

        if score and (best is None or score > best["score"]):
            best = {
                "score": score,
                "title": chunk.get("source_title", "Kaynak"),
                "url": chunk.get("source_url", ""),
                "reasons": reasons[:4],
                "preview": str(chunk.get("content", "")).replace("\n", " ")[:220],
            }

    if best is None or best["score"] < 4:
        return {"found": False}
    best["found"] = True
    return best


def recommendation(rows: list[dict]) -> str:
    class_counts = Counter(row["class"] for row in rows)
    if class_counts["no_answer"] >= class_counts["wrong_source"]:
        return (
            "No-answer vakalarinda chunks.json kanitini kontrol et. Kanit varsa esik, topic metadata "
            "veya skorlamayi guclendir; kanit yoksa crawler/kaynak listesi eksik."
        )
    return (
        "Retrieval yanlis resmi belgeyi one cikariyor. Bu topic icin _candidate_score icinde kaynak basligi, "
        "URL oruntusu ve ilgili anahtar terimlere pozitif agirlik; sik karisan topic'lere negatif agirlik eklenmeli."
    )


def build_rows(cases: dict, results: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for result in results:
        if result.get("passed"):
            continue
        case = cases[result["id"]]
        topic = case.get("topic") or result["id"].rsplit("_", 1)[0]
        grouped[topic].append(
            {
                "id": result["id"],
                "class": classify_result(result),
                "query": case.get("query", ""),
                "expected": case.get("acceptable_source_terms", []),
                "returned": result.get("sources", []),
                "answer": str(result.get("answer", "")).replace("\n", " ")[:260],
            }
        )
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", default=str(DEFAULT_EVALUATION_FILE))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_FILE))
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_FILE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE))
    args = parser.parse_args()

    cases = {case["id"]: case for case in load_json(Path(args.evaluation))}
    results = load_json(Path(args.results))["results"]
    chunks = load_json(Path(args.chunks))
    grouped = build_rows(cases, results)

    lines = [
        "# Evaluation Topic Failure Report",
        "",
        f"Total failed topics: {len(grouped)}",
        "",
    ]

    for topic, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        counts = Counter(row["class"] for row in rows)
        lines.extend(
            [
                f"## {topic}",
                "",
                f"- Fail count: {len(rows)}",
                f"- Classes: no_answer={counts['no_answer']}, wrong_source={counts['wrong_source']}",
                f"- Recommendation: {recommendation(rows)}",
                "",
            ]
        )

        for row in rows[:3]:
            evidence = chunk_evidence(row, chunks) if row["class"] == "no_answer" else None
            lines.extend(
                [
                    f"### {row['id']} ({row['class']})",
                    "",
                    f"- Query: {row['query']}",
                    f"- Expected acceptable_source_terms: {', '.join(row['expected'])}",
                    f"- Returned sources: {format_sources(row['returned'])}",
                    f"- Answer preview: {row['answer']}",
                ]
            )
            if evidence:
                if evidence["found"]:
                    lines.extend(
                        [
                            f"- Chunk check: FOUND score={evidence['score']} title={evidence['title']} url={evidence['url']}",
                            f"- Chunk match reasons: {', '.join(evidence['reasons'])}",
                            f"- Chunk preview: {evidence['preview']}",
                        ]
                    )
                else:
                    lines.append("- Chunk check: NOT_FOUND in chunks.json by expected/query terms")
            lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
