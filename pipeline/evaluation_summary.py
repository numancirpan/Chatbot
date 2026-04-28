import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT_DIR / "data" / "reports" / "manual_regression_latest.json"
OUT_JSON = ROOT_DIR / "data" / "reports" / "evaluation_summary_latest.json"
OUT_MD = ROOT_DIR / "data" / "reports" / "evaluation_summary_latest.md"


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


def build_summary(report: Dict) -> Dict:
    answers = list(iter_answers(report))
    total = len(answers)
    with_source = sum(1 for item in answers if item["sources"])
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

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_report": str(REPORT_PATH),
        "total_answers": total,
        "metrics": {
            "clean_answer_rate": percent(clean, total),
            "source_coverage_rate": percent(with_source, total),
            "no_direct_answer_rate": percent(no_answer, total),
            "noisy_source_rate": percent(noisy, total),
            "student_affairs_redirect_rate": percent(redirect, total),
            "scope_clarification_rate": percent(scope, total),
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
