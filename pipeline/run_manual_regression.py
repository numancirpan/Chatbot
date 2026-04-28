import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = ROOT_DIR / "data" / "manual_regression_questions.json"
REPORT_DIR = ROOT_DIR / "data" / "reports"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import RAGChatbot, normalize_text


RISK_MARKERS = {
    "scope_clarification": [
        "bolum veya fakulteye gore degisebilmektedir",
        "bolum/program belirterek",
    ],
    "no_direct_answer": [
        "dogrudan resmi bir kaynak bulamadim",
        "resmi belgelerde bilgiye ulasilamadi",
        "kesin bir cevap veremiyorum",
    ],
    "student_affairs_redirect": [
        "iletisime gec",
    ],
    "noisy_source": [
        "t.c.",
        "fakulte bolum",
        "yonetim dekanlik",
        "ogrenci panolari",
        "isyeri egitimi",
        "teknoloji fakultesi",
        "ucretli staj icin",
    ],
}


def load_questions() -> Dict:
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def source_titles(result: Dict) -> List[str]:
    return [source.get("baslik", "") for source in result.get("kaynaklar", [])]


def should_expect_scope_clarification(question: str) -> bool:
    normalized = normalize_text(question)
    program_markers = [
        "bilgisayar muhendisligi",
        "muhendislik fakultesi",
        "egitim fakultesi",
        "eczacilik fakultesi",
        "orman fakultesi",
        "isletme fakultesi",
        "bolumunde",
        "bolumu",
        "programi",
    ]
    scope_sensitive_markers = ["staj", "isyeri egitimi", "uygulamali egitim"]
    return any(marker in normalized for marker in scope_sensitive_markers) and not any(
        marker in normalized for marker in program_markers
    )


def classify_result(question: str, answer: str, titles: List[str]) -> List[str]:
    normalized_answer = normalize_text(answer)
    normalized_titles = [normalize_text(title) for title in titles]
    flags = []

    for flag, markers in RISK_MARKERS.items():
        if flag == "student_affairs_redirect" and "yurutuldugu belirtilmektedir" in normalized_answer:
            continue
        if flag == "noisy_source":
            haystack = " ".join(normalized_titles)
        else:
            haystack = normalized_answer
        if any(marker in haystack for marker in markers):
            flags.append(flag)

    if should_expect_scope_clarification(question):
        flags = [flag for flag in flags if flag != "scope_clarification"]

    return flags


def run_single_questions(bot: RAGChatbot, groups: List[Dict], limit: int = 0) -> List[Dict]:
    results = []
    count = 0
    for group in groups:
        for question in group.get("questions", []):
            if limit and count >= limit:
                return results
            bot.clear_memory()
            result = bot.chat(question)
            answer = result.get("cevap", "")
            titles = source_titles(result)
            results.append(
                {
                    "group": group.get("name", ""),
                    "question": question,
                    "answer": answer,
                    "sources": titles,
                    "flags": classify_result(question, answer, titles),
                }
            )
            count += 1
    return results


def run_followup_flows(bot: RAGChatbot, flows: List[Dict], limit: int = 0) -> List[Dict]:
    results = []
    for index, flow in enumerate(flows):
        if limit and index >= limit:
            return results
        bot.clear_memory()
        messages = []
        for question in flow.get("messages", []):
            result = bot.chat(question)
            answer = result.get("cevap", "")
            titles = source_titles(result)
            messages.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": titles,
                    "flags": classify_result(question, answer, titles),
                }
            )
        results.append({"name": flow.get("name", ""), "messages": messages})
    return results


def summarize(single_results: List[Dict], flow_results: List[Dict]) -> Dict:
    flag_counts = {}
    total = 0

    for result in single_results:
        total += 1
        for flag in result["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    for flow in flow_results:
        for message in flow["messages"]:
            total += 1
            for flag in message["flags"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return {"total_answers": total, "flag_counts": flag_counts}


def to_markdown(report: Dict) -> str:
    lines = [
        "# Manual Regression Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Total answers: {report['summary']['total_answers']}",
        f"- Flag counts: {json.dumps(report['summary']['flag_counts'], ensure_ascii=False)}",
        "",
        "## Single Questions",
        "",
    ]

    for item in report["single_questions"]:
        flags = ", ".join(item["flags"]) if item["flags"] else "ok"
        sources = ", ".join(item["sources"]) if item["sources"] else "-"
        answer = item["answer"].replace("\n", " ").strip()
        lines.extend(
            [
                f"### {item['group']} / {item['question']}",
                f"- Flags: {flags}",
                f"- Sources: {sources}",
                f"- Answer: {answer}",
                "",
            ]
        )

    lines.extend(["## Follow-up Flows", ""])
    for flow in report["followup_flows"]:
        lines.append(f"### {flow['name']}")
        for message in flow["messages"]:
            flags = ", ".join(message["flags"]) if message["flags"] else "ok"
            sources = ", ".join(message["sources"]) if message["sources"] else "-"
            answer = message["answer"].replace("\n", " ").strip()
            lines.extend(
                [
                    f"- Q: {message['question']}",
                    f"  Flags: {flags}",
                    f"  Sources: {sources}",
                    f"  A: {answer}",
                ]
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual RAG regression questions.")
    parser.add_argument("--limit", type=int, default=0, help="Limit single-question count.")
    parser.add_argument("--flow-limit", type=int, default=0, help="Limit follow-up flow count.")
    parser.add_argument("--no-report", action="store_true", help="Print JSON only; do not write reports.")
    args = parser.parse_args()

    data = load_questions()
    bot = RAGChatbot()
    single_results = run_single_questions(bot, data.get("groups", []), limit=args.limit)
    flow_results = run_followup_flows(bot, data.get("followup_flows", []), limit=args.flow_limit)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summarize(single_results, flow_results),
        "single_questions": single_results,
        "followup_flows": flow_results,
    }

    if not args.no_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        json_path = REPORT_DIR / "manual_regression_latest.json"
        markdown_path = REPORT_DIR / "manual_regression_latest.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(to_markdown(report), encoding="utf-8")
        print(f"Wrote {json_path}")
        print(f"Wrote {markdown_path}")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
