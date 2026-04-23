"""
Evaluate the chatbot against a small golden question set.

The goal is not to "train" the model. This script protects us from regressions:
answers must contain expected facts, avoid forbidden claims, and cite the
expected source when a source is required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import RAGChatbot, normalize_text

DEFAULT_GOLDEN_FILE = ROOT_DIR / "data" / "golden_questions.json"


def normalized_contains(text: str, expected: str) -> bool:
    return normalize_text(expected) in normalize_text(text)


def case_turns(case: dict) -> list[str]:
    if isinstance(case.get("turns"), list) and case["turns"]:
        return [str(turn).strip() for turn in case["turns"] if str(turn).strip()]

    if isinstance(case.get("messages"), list):
        turns = [
            str(message.get("content", "")).strip()
            for message in case["messages"]
            if isinstance(message, dict) and message.get("role") == "user" and str(message.get("content", "")).strip()
        ]
        if turns:
            return turns

    turns = []
    query = str(case.get("query", "")).strip()
    if query:
        turns.append(query)

    followups = case.get("followups", [])
    if isinstance(followups, list):
        turns.extend(str(item).strip() for item in followups if str(item).strip())

    return turns


def case_expected_source_terms(case: dict) -> list[str]:
    terms = case.get("expected_source_terms")
    if isinstance(terms, list) and terms:
        return [str(term).strip() for term in terms if str(term).strip()]

    if str(case.get("source_url", "")).strip():
        return [str(case["source_url"]).strip()]

    if str(case.get("suggested_source_url", "")).strip():
        return [str(case["suggested_source_url"]).strip()]

    return []


def normalized_case(case: dict) -> dict:
    clone = dict(case)
    clone["turns"] = case_turns(case)
    clone["expected_source_terms"] = case_expected_source_terms(case)
    return clone


def run_case(bot: RAGChatbot, case: dict) -> dict:
    case = normalized_case(case)
    bot.clear_memory()
    result = None
    for turn in case["turns"]:
        result = bot.chat(turn)

    assert result is not None
    answer = result["cevap"]
    sources = result["kaynaklar"]
    source_blob = " ".join(
        f"{source.get('baslik', '')} {source.get('kategori', '')} {source.get('url', '')}"
        for source in sources
    )

    missing_answer_terms = [
        term for term in case.get("expected_answer_terms", []) if not normalized_contains(answer, term)
    ]
    forbidden_answer_terms = [
        term for term in case.get("forbidden_answer_terms", []) if normalized_contains(answer, term)
    ]
    missing_source_terms = [
        term for term in case.get("expected_source_terms", []) if term.lower() not in source_blob.lower()
    ]
    source_error = bool(case.get("expect_no_sources")) and bool(sources)

    passed = not any(
        [
            missing_answer_terms,
            forbidden_answer_terms,
            missing_source_terms,
            source_error,
        ]
    )

    return {
        "id": case["id"],
        "passed": passed,
        "missing_answer_terms": missing_answer_terms,
        "forbidden_answer_terms": forbidden_answer_terms,
        "missing_source_terms": missing_source_terms,
        "source_error": source_error,
        "answer": answer,
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN_FILE))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON results.")
    args = parser.parse_args()

    with open(args.golden, "r", encoding="utf-8") as f:
        cases = json.load(f)

    bot = RAGChatbot()
    results = [run_case(bot, case) for case in cases]
    passed_count = sum(1 for result in results if result["passed"])

    if args.json:
        print(json.dumps({"passed": passed_count, "total": len(results), "results": results}, ensure_ascii=False, indent=2))
    else:
        print("\nGOLDEN EVALUATION")
        print("=================")
        print(f"Passed: {passed_count}/{len(results)}")
        for result in results:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status}: {result['id']}")
            if not result["passed"]:
                print(f"  missing_answer_terms={result['missing_answer_terms']}")
                print(f"  forbidden_answer_terms={result['forbidden_answer_terms']}")
                print(f"  missing_source_terms={result['missing_source_terms']}")
                print(f"  source_error={result['source_error']}")
                print(f"  sources={result['sources']}")
                print(f"  answer={result['answer']}")

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
