import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT_DIR / "data" / "casual_regression_questions.json"
REPORT_DIR = ROOT_DIR / "data" / "reports"
OUT_JSON = REPORT_DIR / "casual_regression_latest.json"
OUT_MD = REPORT_DIR / "casual_regression_latest.md"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import RAGChatbot, normalize_text


def contains_any(text: str, terms: List[str]) -> bool:
    normalized = normalize_text(text)
    for term in terms:
        if term.endswith(":"):
            if term in text:
                return True
            continue
        if normalize_text(term) in normalized:
            return True
    return False


def run_case(bot: RAGChatbot, case: Dict) -> Dict:
    bot.clear_memory()
    result = bot.chat(case["query"])
    answer = result.get("cevap", "")
    sources = result.get("kaynaklar", [])

    missing_required = []
    if case.get("required_any") and not contains_any(answer, case["required_any"]):
        missing_required = case["required_any"]

    forbidden_hits = [term for term in case.get("forbidden", []) if contains_any(answer, [term])]
    source_error = False
    if case.get("expect_no_sources") and sources:
        source_error = True
    if case.get("expect_sources") and not sources:
        source_error = True

    passed = not missing_required and not forbidden_hits and not source_error
    return {
        "id": case["id"],
        "query": case["query"],
        "passed": passed,
        "missing_required": missing_required,
        "forbidden_hits": forbidden_hits,
        "source_error": source_error,
        "answer": answer,
        "sources": [source.get("baslik", "") for source in sources],
    }


def to_markdown(report: Dict) -> str:
    lines = [
        "# Casual Regression Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Passed: {report['summary']['passed']}/{report['summary']['total']}",
        "",
    ]
    for item in report["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        sources = ", ".join(item["sources"]) if item["sources"] else "-"
        answer = item["answer"].replace("\n", " ").strip()
        lines.extend(
            [
                f"## {status}: {item['id']}",
                f"- Q: {item['query']}",
                f"- Sources: {sources}",
                f"- Missing required: {item['missing_required']}",
                f"- Forbidden hits: {item['forbidden_hits']}",
                f"- Source error: {item['source_error']}",
                f"- A: {answer}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run casual/off-topic guardrail regression tests.")
    parser.add_argument("--no-report", action="store_true", help="Print JSON only; do not write reports.")
    args = parser.parse_args()

    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    bot = RAGChatbot()
    results = [run_case(bot, case) for case in data.get("cases", [])]
    passed = sum(1 for result in results if result["passed"])
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {"passed": passed, "total": len(results)},
        "results": results,
    }

    if not args.no_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        OUT_MD.write_text(to_markdown(report), encoding="utf-8")
        print(f"Wrote {OUT_JSON}")
        print(f"Wrote {OUT_MD}")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
