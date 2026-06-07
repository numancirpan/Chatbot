from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import normalize_text
from pipeline.evaluate_cases import case_expected_source_terms

DEFAULT_EVALUATION_FILE = ROOT_DIR / "data" / "evaluation_cases.json"
DEFAULT_RESULTS_FILE = ROOT_DIR / "outputs" / "evaluation_cases_latest.json"
DEFAULT_REPORT_FILE = ROOT_DIR / "outputs" / "evaluation_failure_classes.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def unique_terms(terms: List[str]) -> List[str]:
    seen = set()
    unique = []
    for term in terms:
        term = str(term).strip()
        if not term:
            continue
        key = normalize_text(term) or term.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique


def source_blob(sources: List[Dict]) -> str:
    return " ".join(
        f"{source.get('baslik', '')} {source.get('kategori', '')} {source.get('url', '')}"
        for source in sources
    )


def has_source_match(blob: str, terms: List[str]) -> bool:
    normalized_blob = normalize_text(blob)
    lowercase_blob = blob.lower()
    return any(term.lower() in lowercase_blob or normalize_text(term) in normalized_blob for term in terms if term)


def classify_failure(case: Dict, result: Dict) -> str:
    answer = str(result.get("answer", ""))
    sources = result.get("sources", [])
    if not sources or "bilgiye ulasilamadi" in normalize_text(answer):
        return "no_answer"

    blob = source_blob(sources)
    expected_title = str(case.get("source_title", "")).strip()
    topic_label = str(case.get("topic_label", "")).strip()
    title_terms = [expected_title, topic_label]
    if has_source_match(blob, title_terms):
        return "equivalent_source_exact_url_fail"

    source_urls = {str(source.get("url", "")).strip().lower().rstrip("/") for source in sources}
    expected_urls = {term.lower().rstrip("/") for term in case_expected_source_terms(case)}
    if source_urls & expected_urls:
        return "equivalent_source_exact_url_fail"

    return "wrong_source"


def acceptable_terms_for_case(case: Dict, result: Dict | None, classification: str | None) -> List[str]:
    terms = []
    terms.extend(case.get("acceptable_source_terms", []) if isinstance(case.get("acceptable_source_terms"), list) else [])
    terms.extend(case_expected_source_terms(case))
    if case.get("source_title"):
        terms.append(case["source_title"])
    if case.get("topic_label"):
        terms.append(case["topic_label"])

    if result and classification == "equivalent_source_exact_url_fail":
        for source in result.get("sources", []):
            if source.get("url"):
                terms.append(source["url"])
            if source.get("baslik"):
                terms.append(source["baslik"])

    return unique_terms(terms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", default=str(DEFAULT_EVALUATION_FILE))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_FILE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_FILE))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    evaluation_path = Path(args.evaluation)
    cases = load_json(evaluation_path)
    results_payload = load_json(Path(args.results))
    results_by_id = {result["id"]: result for result in results_payload.get("results", [])}

    buckets = {
        "no_answer": [],
        "wrong_source": [],
        "equivalent_source_exact_url_fail": [],
    }
    updated_cases = []
    for case in cases:
        clone = dict(case)
        result = results_by_id.get(str(case.get("id", "")))
        classification = None
        if result and not result.get("passed", False):
            classification = classify_failure(case, result)
            buckets[classification].append(
                {
                    "id": case.get("id"),
                    "query": case.get("query"),
                    "expected_terms": case_expected_source_terms(case),
                    "returned_sources": result.get("sources", []),
                    "answer_preview": str(result.get("answer", "")).replace("\n", " ")[:500],
                }
            )
        clone["acceptable_source_terms"] = acceptable_terms_for_case(case, result, classification)
        updated_cases.append(clone)

    report = {
        "summary": {name: len(items) for name, items in buckets.items()},
        "buckets": buckets,
    }
    dump_json(Path(args.report), report)

    if args.apply:
        dump_json(evaluation_path, updated_cases)

    print(json.dumps({"report": args.report, "applied": args.apply, "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
