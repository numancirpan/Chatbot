import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT_DIR / "data" / "reports" / "manual_regression_latest.json"
REPORT_DIR = ROOT_DIR / "data" / "reports"
OUT_JSON = REPORT_DIR / "source_quality_latest.json"
OUT_MD = REPORT_DIR / "source_quality_latest.md"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import infer_query_topic, normalize_text  # noqa: E402


NOISY_SOURCE_MARKERS = [
    "t.c.",
    "bitis",
    "fakulte bolum",
    "yonetim dekanlik",
    "ogrenci panolari",
    "11 mayis",
    "9 kasim",
    "17 haziran",
]

TOPIC_SOURCE_MARKERS = {
    "staj": ["staj", "bilgisayar muhendisligi", "muhendislik ve teknoloji fakulteleri"],
    "yaz_okulu": ["yaz okulu", "akademik takvim"],
    "ders_kaydi": ["ders kayit", "kayit yenileme", "akademik takvim"],
    "add_drop": ["ekle", "ders kayit", "akademik takvim"],
    "devamsizlik": ["devam", "sinav yonetmeligi", "lisans egitim"],
    "sinavlar": ["sinav", "tek", "cift", "mazeret", "staj sss", "ogrenci isleri sikca", "final program"],
    "mezuniyet": ["mezuniyet", "diploma", "lisans egitim", "stajlar hakkinda", "transkript"],
    "ogrenci_belgesi_transkript": ["ogrenci belgesi", "transkript", "ogrenci isleri"],
    "cap_yandal": ["cap", "cift anadal", "yandal"],
    "yatay_gecis": ["yatay gecis"],
    "disiplin": ["disiplin", "yonetmelik", "2547"],
    "not_sistemi": ["not", "sinav yonetmeligi", "gano", "resmi gazete"],
    "harc_ucret": ["harc", "ucret", "katki payi", "kayit yenileme"],
    "askerlik_tecili": ["askerlik", "ogrenci isleri", "hizmet standartlari"],
}

PROGRAM_SPECIFIC_ALLOWED_MARKERS = [
    "bilgisayar muhendisligi",
    "muhendislik fakultesi",
    "muhendislik ve teknoloji fakulteleri",
    "yaz okulu yonergesi",
    "akademik takvim",
    "lisans egitim",
    "diploma",
    "ogrenci isleri",
]

NON_SOURCE_ANSWER_MARKERS = [
    "bolum veya fakulteye gore degisebilmektedir",
    "bolum/program belirterek",
    "dogrudan resmi bir kaynak bulamadim",
    "resmi belgelerde bilgiye ulasilamadi",
    "kesin cevap veremiyorum",
    "net cevap veremiyorum",
]


def iter_answers(report: Dict) -> Iterable[Dict]:
    for item in report.get("single_questions", []):
        yield {
            "group": item.get("group", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "sources": item.get("sources", []),
        }
    for flow in report.get("followup_flows", []):
        for item in flow.get("messages", []):
            yield {
                "group": flow.get("name", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "sources": item.get("sources", []),
            }


def is_answerable(item: Dict) -> bool:
    normalized_answer = normalize_text(item.get("answer", ""))
    return not any(marker in normalized_answer for marker in NON_SOURCE_ANSWER_MARKERS)


def contains_any(text: str, markers: List[str]) -> bool:
    normalized = normalize_text(text)
    return any(marker in normalized for marker in markers)


def is_program_specific(question: str) -> bool:
    normalized = normalize_text(question)
    return any(
        marker in normalized
        for marker in [
            "bilgisayar muhendisligi",
            "muhendislik fakultesi",
            "bolumunde",
            "bolumu",
            "programi",
        ]
    )


def audit_item(item: Dict) -> Dict:
    question = item["question"]
    topic = infer_query_topic(question)
    sources = item.get("sources", [])
    normalized_sources = [normalize_text(source) for source in sources]
    source_text = " ".join(normalized_sources)

    errors: List[str] = []
    if not sources and is_answerable(item):
        errors.append("answerable_without_source")

    noisy_hits = [source for source in sources if contains_any(source, NOISY_SOURCE_MARKERS)]
    if noisy_hits:
        errors.append("noisy_source_title")

    expected_markers = TOPIC_SOURCE_MARKERS.get(topic, [])
    if sources and expected_markers and not any(contains_any(source, expected_markers) for source in sources):
        errors.append(f"topic_source_mismatch:{topic}")

    if (
        is_program_specific(question)
        and sources
        and topic in {"staj", "yaz_okulu", "mezuniyet"}
        and not any(
        contains_any(source, PROGRAM_SPECIFIC_ALLOWED_MARKERS) for source in sources
        )
    ):
        errors.append("program_specific_source_weak")

    return {
        "group": item["group"],
        "question": question,
        "topic": topic,
        "sources": sources,
        "errors": errors,
        "passed": not errors,
        "source_count": len(sources),
        "source_text": source_text,
    }


def to_markdown(report: Dict) -> str:
    lines = [
        "# Source Quality Audit",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Passed: {report['summary']['passed']}/{report['summary']['total']}",
        f"- Checked answerable items: {report['summary']['checked_answerable']}",
        "",
    ]
    for item in report["items"]:
        if item["passed"]:
            continue
        sources = ", ".join(item["sources"]) if item["sources"] else "-"
        errors = ", ".join(item["errors"]) if item["errors"] else "-"
        lines.extend(
            [
                f"## FAIL: {item['group']} / {item['question']}",
                f"- Topic: {item['topic']}",
                f"- Errors: {errors}",
                f"- Sources: {sources}",
                "",
            ]
        )
    if all(item["passed"] for item in report["items"]):
        lines.append("No source quality issues found.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    if not REPORT_PATH.exists():
        raise SystemExit(f"Manual regression report not found: {REPORT_PATH}")

    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    checked_items = [item for item in iter_answers(data) if is_answerable(item)]
    results = [audit_item(item) for item in checked_items]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total": len(results),
            "passed": sum(1 for item in results if item["passed"]),
            "checked_answerable": len(results),
        },
        "items": results,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(to_markdown(report), encoding="utf-8")
    print(f"Source quality: {report['summary']['passed']}/{report['summary']['total']} answerable items")
    if report["summary"]["passed"] != report["summary"]["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
