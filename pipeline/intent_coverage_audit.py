import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT_DIR / "data" / "reports"
OUT_JSON = REPORT_DIR / "intent_coverage_latest.json"
OUT_MD = REPORT_DIR / "intent_coverage_latest.md"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import (  # noqa: E402
    build_casual_response,
    build_intent_query_expansions,
    infer_query_topic,
    intent_candidate_markers,
    normalize_text,
)


INTENT_CASES: List[Dict] = [
    {
        "id": "staj_scope_duration",
        "topic": "staj",
        "sample_queries": [
            "Staj kaç gün?",
            "Bilgisayar mühendisliği zorunlu staj kaç gün?",
        ],
        "required_expansion_any": ["25 is gunu", "zorunlu staj", "staj suresi"],
        "required_marker_any": ["25 is gunu", "bm399"],
    },
    {
        "id": "staj_late_upload",
        "topic": "staj",
        "sample_queries": [
            "Stajımı zamanında teslim etmezsem bilgisayar mühendisliği için süreç nasıl işler?",
            "Staj raporu SBS'ye ne zaman yüklenmeli?",
        ],
        "required_expansion_any": ["staj raporu", "sbs", "30 gun", "gec teslim"],
        "required_marker_any": ["staj raporu", "sbs", "30 gun"],
    },
    {
        "id": "staj_insurance",
        "topic": "staj",
        "sample_queries": [
            "Bilgisayar mühendisliği için staj sigortasını kim yapıyor?",
            "Zorunlu staj dışında gönüllü staj yaparsam sigortamı okul yapar mı?",
        ],
        "required_expansion_any": ["sigorta", "zorunlu staj", "gonullu staj"],
        "required_marker_any": ["sigorta", "zorunlu staj"],
    },
    {
        "id": "registration_calendar",
        "topic": "ders_kaydi",
        "sample_queries": [
            "2025-2026 bahar döneminde ders kaydımı ne zaman yapmalıyım ve ekle-sil haftası hangi tarihlerde?",
            "Ders kaydı ne zaman başlıyor?",
        ],
        "required_expansion_any": ["ders kaydi", "kayit yenileme", "akademik takvim", "ekle sil"],
        "required_marker_any": ["ders kaydi", "kayit yenileme", "ekle sil"],
    },
    {
        "id": "registration_advisor_approval",
        "topic": "ders_kaydi",
        "sample_queries": [
            "Ders kaydımı yaptım ama danışman onayı verilmeden ekle-sil haftası bitti. Kaydım geçerli sayılır mı?",
            "Danışman onayı olmadan ders kaydım tamamlanır mı?",
        ],
        "required_expansion_any": ["danisman onayi", "ders kaydi", "kayit yenileme"],
        "required_marker_any": ["danisman onay", "ders kaydi"],
    },
    {
        "id": "makeup_exam_deadline",
        "topic": "sinavlar",
        "sample_queries": [
            "Mazeret sınavına başvururken raporumu 3 iş günü içinde veremedim; yine de sınava girme hakkım olur mu?",
            "Mazeret sınavı için kaç gün içinde başvuru yapılmalı?",
        ],
        "required_expansion_any": ["mazeret sinavi", "3 is gunu", "rapor"],
        "required_marker_any": ["mazeret sinavi", "3 gun"],
    },
    {
        "id": "attendance_reported_absence",
        "topic": "devamsizlik",
        "sample_queries": [
            "İlk kez aldığım bir uygulamalı derste devam zorunluluğu kaçtır ve raporlu olduğum günler devamsızlıktan sayılır mı?",
            "Uygulamalı derslerde devam zorunluluğu kaçtır?",
        ],
        "required_expansion_any": ["devam zorunlulugu", "yuzde 70", "yuzde 80", "raporlu"],
        "required_marker_any": ["devam zorunlulugu", "raporlu", "80"],
    },
    {
        "id": "summer_school_external_equivalence",
        "topic": "yaz_okulu",
        "sample_queries": [
            "Yaz okulunda başka üniversiteden aldığım dersi saydırmak istiyorum; ders içerikleri benziyor ama AKTS farklıysa ne olur?",
            "Başka üniversiteden alacağım yaz okulu dersinin sayılması için hangi şartlar gerekir?",
        ],
        "required_expansion_any": ["baska universite", "esdeger", "akts", "bolum baskanligi"],
        "required_marker_any": ["baska universite", "esdeger"],
    },
    {
        "id": "graduation_staj_missing",
        "topic": "mezuniyet",
        "sample_queries": [
            "Mezuniyet için tüm derslerimi geçtim ama zorunlu stajım eksik. Geçici mezuniyet belgesi alabilir miyim?",
            "Bilgisayar mühendisliği öğrencisi mezuniyet için staj eksikse diplomasını alabilir mi?",
        ],
        "required_expansion_any": ["mezuniyet", "diploma", "staj", "akademik yukumluluk"],
        "required_marker_any": ["mezuniyet", "diploma", "akademik yukumluluk"],
    },
    {
        "id": "single_double_exam_staj",
        "topic": "sinavlar",
        "sample_queries": [
            "Bilgisayar mühendisliği öğrencisiyim; staj dersine daha önce hiç kayıtlanmadıysam tek/çift ders sınavına girebilir miyim?",
            "Tek ders sınavına girmek için stajımı tamamlamış olmam gerekir mi?",
        ],
        "required_expansion_any": ["tek cift", "staj dersi", "yz"],
        "required_marker_any": ["tek", "cift", "yz"],
    },
    {
        "id": "documents_transcript",
        "topic": "ogrenci_belgesi_transkript",
        "sample_queries": [
            "Öğrenci belgesi veya transkript almam gerekirse önce nereden almalıyım, e-Devlet belgesi yeterli olur mu?",
            "Transkript nereden alınır?",
        ],
        "required_expansion_any": ["ogrenci belgesi", "transkript", "e devlet"],
        "required_marker_any": ["ogrenci belgesi", "transkript"],
    },
    {
        "id": "cap_yandal_conditions",
        "topic": "cap_yandal",
        "sample_queries": [
            "Çift anadal başvurusu yapmak için genel not ortalamam kaç olmalı, ayrıca başarı sıralaması şartı var mı?",
            "ÇAP başvurusu için ortalama şartı nedir?",
        ],
        "required_expansion_any": ["cift anadal", "cap", "not ortalamasi", "yuzde 20"],
        "required_marker_any": ["cift anadal", "yuzde 20"],
    },
    {
        "id": "yatay_gecis_calendar",
        "topic": "yatay_gecis",
        "sample_queries": [
            "Yatay geçiş başvuruları ne zaman yapılır?",
            "Kurum içi yatay geçiş için takvimi nereden takip etmeliyim?",
        ],
        "required_expansion_any": ["yatay gecis", "basvuru", "takvim"],
        "required_marker_any": ["yatay gecis", "basvuru"],
    },
    {
        "id": "discipline_scholarship",
        "topic": "disiplin",
        "sample_queries": [
            "Disiplin cezası alan öğrenci bursunu kaybeder mi?",
            "Disiplin işlemleriyle ilgili mevzuata nereden ulaşabilirim?",
        ],
        "required_expansion_any": ["disiplin", "burs", "yonetmelik"],
        "required_marker_any": ["disiplin", "burs"],
    },
    {
        "id": "casual_greeting",
        "topic": "casual",
        "sample_queries": [
            "Merhaba nasılsın?",
            "Teşekkür ederim.",
        ],
        "expect_casual": True,
    },
    {
        "id": "casual_off_topic",
        "topic": "casual",
        "sample_queries": [
            "Bugün hava nasıl?",
            "Python nedir?",
        ],
        "expect_casual": True,
    },
]


def contains_any(values: List[str], terms: List[str]) -> bool:
    normalized_values = [normalize_text(value) for value in values]
    return any(normalize_text(term) in value for term in terms for value in normalized_values)


def marker_terms(marker_groups: List[List[str]]) -> List[str]:
    terms: List[str] = []
    for group in marker_groups:
        terms.extend(group)
    return terms


def audit_query(case: Dict, query: str) -> Dict:
    topic = infer_query_topic(query)
    expansions = build_intent_query_expansions(query)
    markers = intent_candidate_markers(query)
    casual = build_casual_response(query)

    errors: List[str] = []
    if case.get("expect_casual"):
        if not casual:
            errors.append("casual_response_missing")
    else:
        if casual:
            errors.append("academic_query_marked_casual")
        if topic != case["topic"]:
            errors.append(f"topic_mismatch:{topic}")
        if case.get("required_expansion_any") and not contains_any(expansions, case["required_expansion_any"]):
            errors.append("required_expansion_missing")
        if case.get("required_marker_any") and not contains_any(marker_terms(markers), case["required_marker_any"]):
            errors.append("required_marker_missing")

    return {
        "query": query,
        "topic": topic,
        "expansion_count": len(expansions),
        "marker_group_count": len(markers),
        "casual_response": bool(casual),
        "errors": errors,
        "passed": not errors,
        "expansions": expansions,
        "marker_groups": markers,
    }


def audit_case(case: Dict) -> Dict:
    results = [audit_query(case, query) for query in case["sample_queries"]]
    return {
        "id": case["id"],
        "expected_topic": case["topic"],
        "passed": all(item["passed"] for item in results),
        "results": results,
    }


def to_markdown(report: Dict) -> str:
    lines = [
        "# Intent Coverage Audit",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Passed cases: {report['summary']['passed_cases']}/{report['summary']['total_cases']}",
        f"- Passed queries: {report['summary']['passed_queries']}/{report['summary']['total_queries']}",
        "",
    ]

    for case in report["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        lines.extend([f"## {status}: {case['id']}", f"- Expected topic: {case['expected_topic']}"])
        for item in case["results"]:
            query_status = "PASS" if item["passed"] else "FAIL"
            errors = ", ".join(item["errors"]) if item["errors"] else "-"
            lines.extend(
                [
                    f"- {query_status}: {item['query']}",
                    f"  Topic: {item['topic']} | Expansions: {item['expansion_count']} | Marker groups: {item['marker_group_count']} | Casual: {item['casual_response']} | Errors: {errors}",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [audit_case(case) for case in INTENT_CASES]
    total_queries = sum(len(case["results"]) for case in cases)
    passed_queries = sum(1 for case in cases for item in case["results"] if item["passed"])
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_cases": len(cases),
            "passed_cases": sum(1 for case in cases if case["passed"]),
            "total_queries": total_queries,
            "passed_queries": passed_queries,
        },
        "cases": cases,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(to_markdown(report), encoding="utf-8")
    print(
        f"Intent coverage: {report['summary']['passed_cases']}/{report['summary']['total_cases']} cases, "
        f"{passed_queries}/{total_queries} queries"
    )
    if passed_queries != total_queries:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
