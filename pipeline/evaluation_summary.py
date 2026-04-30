import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT_DIR / "data" / "reports" / "manual_regression_latest.json"
EXPECTATIONS_PATH = ROOT_DIR / "data" / "regression_expectations.json"
OUT_JSON = ROOT_DIR / "data" / "reports" / "evaluation_summary_latest.json"
OUT_MD = ROOT_DIR / "data" / "reports" / "evaluation_summary_latest.md"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import normalize_text


def iter_answers(report: Dict) -> Iterable[Dict]:
    for item in report.get("single_questions", []):
        yield {
            "type": "single",
            "group": item.get("group", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "sources": item.get("sources", []),
            "flags": item.get("flags", []),
        }
    for flow in report.get("followup_flows", []):
        for item in flow.get("messages", []):
            yield {
                "type": "followup",
                "group": flow.get("name", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "sources": item.get("sources", []),
                "flags": item.get("flags", []),
            }


def percent(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(value * 100 / total, 2)


def is_source_applicable_answer(item: Dict) -> bool:
    normalized_answer = normalize_text(item.get("answer", ""))
    non_source_markers = [
        "bolum veya fakulteye gore degisebilmektedir",
        "bolum program belirterek",
        "belirttiginiz bolum veya program icin acik ve dogrudan bir resmi dayanak bulamadim",
        "bu konuda resmi belgelerde bilgiye ulasilamadim",
        "resmi belgelerde bilgiye ulasilamadi",
        "dogrudan resmi bir kaynak bulamadim",
        "dogrudan resmi kaynak bulamadim",
        "kesin cevap veremiyorum",
        "net cevap veremiyorum",
    ]
    return not any(marker in normalized_answer for marker in non_source_markers)


def load_expectations() -> List[Dict]:
    if not EXPECTATIONS_PATH.exists():
        return []
    data = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    return data.get("expectations", [])


def contains_all(haystack: str, terms: List[str]) -> bool:
    normalized_haystack = normalize_text(haystack)
    return all(normalize_text(term) in normalized_haystack for term in terms)


def contains_any(haystack: str, terms: List[str]) -> bool:
    normalized_haystack = normalize_text(haystack)
    return any(normalize_text(term) in normalized_haystack for term in terms)


def matching_expectations(question: str, expectations: List[Dict]) -> List[Dict]:
    normalized_question = normalize_text(question)
    matches = []
    for expectation in expectations:
        markers = expectation.get("question_contains", [])
        if not markers:
            continue
        if all(normalize_text(marker) in normalized_question for marker in markers):
            matches.append(expectation)
    return matches


def check_expectation(item: Dict, expectation: Dict) -> Dict:
    answer = item.get("answer", "")
    source_blob = " ".join(item.get("sources", []))
    missing_required = [
        term for term in expectation.get("required_all", []) if not contains_any(answer, [term])
    ]
    missing_any_groups = [
        group for group in expectation.get("required_any_groups", []) if not contains_any(answer, group)
    ]
    forbidden_hits = [
        term for term in expectation.get("forbidden", []) if contains_any(answer + " " + source_blob, [term])
    ]
    source_title_any = expectation.get("source_title_any", [])
    missing_source_title = bool(source_title_any) and not contains_any(source_blob, source_title_any)

    passed = not (missing_required or missing_any_groups or forbidden_hits or missing_source_title)
    return {
        "id": expectation.get("id", ""),
        "passed": passed,
        "group": item.get("group", ""),
        "question": item.get("question", ""),
        "missing_required": missing_required,
        "missing_any_groups": missing_any_groups,
        "forbidden_hits": forbidden_hits,
        "missing_source_title": missing_source_title,
        "sources": item.get("sources", []),
    }


def build_summary(report: Dict) -> Dict:
    answers = list(iter_answers(report))
    expectations = load_expectations()
    total = len(answers)
    source_applicable_answers = [item for item in answers if is_source_applicable_answer(item)]
    source_applicable_total = len(source_applicable_answers)
    with_source = sum(1 for item in answers if item["sources"])
    source_applicable_with_source = sum(1 for item in source_applicable_answers if item["sources"])
    clean = sum(1 for item in answers if not item["flags"])
    no_answer = sum(1 for item in answers if "no_direct_answer" in item["flags"])
    noisy = sum(1 for item in answers if "noisy_source" in item["flags"])
    redirect = sum(1 for item in answers if "student_affairs_redirect" in item["flags"])
    scope = sum(1 for item in answers if "scope_clarification" in item["flags"])

    by_group: Dict[str, Dict] = {}
    for item in answers:
        group = item["group"] or "Genel"
        stats = by_group.setdefault(group, {"total": 0, "clean": 0, "flag_counts": {}})
        stats["total"] += 1
        if not item["flags"]:
            stats["clean"] += 1
        for flag in item["flags"]:
            stats["flag_counts"][flag] = stats["flag_counts"].get(flag, 0) + 1

    for stats in by_group.values():
        stats["clean_rate"] = percent(stats["clean"], stats["total"])

    high_risk = [
        item
        for item in answers
        if any(flag in item["flags"] for flag in ["no_direct_answer", "noisy_source"])
    ]
    expectation_checks = []
    for item in answers:
        for expectation in matching_expectations(item["question"], expectations):
            expectation_checks.append(check_expectation(item, expectation))
    expectation_total = len(expectation_checks)
    expectation_passed = sum(1 for item in expectation_checks if item["passed"])
    expectation_failures = [item for item in expectation_checks if not item["passed"]]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_report": str(REPORT_PATH),
        "total_answers": total,
        "metrics": {
            "clean_answer_rate": percent(clean, total),
            "source_coverage_rate": percent(with_source, total),
            "answerable_source_coverage_rate": percent(source_applicable_with_source, source_applicable_total),
            "no_direct_answer_rate": percent(no_answer, total),
            "noisy_source_rate": percent(noisy, total),
            "student_affairs_redirect_rate": percent(redirect, total),
            "scope_clarification_rate": percent(scope, total),
            "expectation_pass_rate": percent(expectation_passed, expectation_total),
        },
        "expectation_checks": {
            "total": expectation_total,
            "passed": expectation_passed,
            "failed": expectation_total - expectation_passed,
            "failures": expectation_failures,
        },
        "by_group": by_group,
        "high_risk_items": high_risk,
    }


def to_markdown(summary: Dict) -> str:
    lines: List[str] = [
        "# RAG Evaluation Summary",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Total answers: {summary['total_answers']}",
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary["metrics"].items():
        lines.append(f"- {key}: {value}%")

    lines.extend(["", "## Group Quality", ""])
    for group, stats in summary["by_group"].items():
        flags = json.dumps(stats["flag_counts"], ensure_ascii=False)
        lines.append(f"- {group}: {stats['clean_rate']}% clean ({stats['clean']}/{stats['total']}), flags={flags}")

    lines.extend(["", "## High Risk Items", ""])
    if not summary["high_risk_items"]:
        lines.append("- No high risk items.")
    for item in summary["high_risk_items"]:
        flags = ", ".join(item["flags"])
        sources = ", ".join(item["sources"]) if item["sources"] else "-"
        lines.append(f"- [{flags}] {item['group']} / {item['question']} | sources: {sources}")

    checks = summary.get("expectation_checks", {})
    lines.extend(["", "## Factual Expectation Checks", ""])
    lines.append(f"- Passed: {checks.get('passed', 0)}/{checks.get('total', 0)}")
    failures = checks.get("failures", [])
    if not failures:
        lines.append("- No expectation failures.")
    for item in failures:
        problems = []
        if item.get("missing_required"):
            problems.append(f"missing_required={item['missing_required']}")
        if item.get("missing_any_groups"):
            problems.append(f"missing_any_groups={item['missing_any_groups']}")
        if item.get("forbidden_hits"):
            problems.append(f"forbidden_hits={item['forbidden_hits']}")
        if item.get("missing_source_title"):
            problems.append("missing_source_title=True")
        problem_text = "; ".join(problems)
        sources = ", ".join(item.get("sources", [])) if item.get("sources") else "-"
        lines.append(f"- [{item.get('id', '')}] {item.get('group', '')} / {item.get('question', '')} | {problem_text} | sources: {sources}")

    return "\n".join(lines) + "\n"


def main() -> int:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    summary = build_summary(report)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(to_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
