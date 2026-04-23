"""
Small regression checks for the RAG chatbot.

This is intentionally lightweight: it checks the failure cases we have fixed
recently, especially source grounding, scope clarification, and follow-up
questions. It does not replace a full RAGAS/DeepEval evaluation suite.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import RAGChatbot, normalize_text


@dataclass
class Case:
    name: str
    queries: list[str]
    must_contain: list[str] = field(default_factory=list)
    source_contains: list[str] = field(default_factory=list)
    expect_no_sources: bool = False


CASES = [
    Case(
        name="discipline scholarship uncertainty",
        queries=["Disiplin cezasi alan ogrenci bursunu kaybeder mi"],
        must_contain=["kesin cevap veremiyorum"],
        expect_no_sources=True,
    ),
    Case(
        name="program-specific internship duration",
        queries=["Bilgisayar muhendisligi staj kac gun suruyor"],
        must_contain=["25 is gunu"],
        source_contains=["bm.mf.duzce.edu.tr/sayfa/4a82"],
    ),
    Case(
        name="internship report follow-up",
        queries=[
            "Stajini zamaninda teslim etmeyen ogrenci icin surec nasil isler",
            "Bilgisayar muhendisligi",
        ],
        must_contain=["sbs", "yaklasik 30 gun"],
        source_contains=["bm.mf.duzce.edu.tr/sayfa/878b"],
    ),
    Case(
        name="summer school duration",
        queries=["Yaz okulu kac hafta suruyor"],
        must_contain=["5 hafta", "7 hafta"],
        source_contains=["a44682bb-4437-4908-aaf0-c6020e1fc991"],
    ),
    Case(
        name="summer school start",
        queries=["Bilgisayar muhendisligi 2026 yili icin yaz okulu ne zaman basliyor"],
        must_contain=["6 07 2026"],
        source_contains=["a44682bb-4437-4908-aaf0-c6020e1fc991"],
    ),
]


def main() -> int:
    bot = RAGChatbot()
    failures = []

    for case in CASES:
        bot.clear_memory()
        result = None
        for query in case.queries:
            result = bot.chat(query)

        assert result is not None
        answer = normalize_text(result["cevap"])
        sources = result["kaynaklar"]
        source_blob = " ".join(source.get("url", "") for source in sources)

        missing_answer_terms = [
            expected for expected in case.must_contain if normalize_text(expected) not in answer
        ]
        missing_sources = [
            expected for expected in case.source_contains if expected.lower() not in source_blob.lower()
        ]
        source_error = case.expect_no_sources and bool(sources)

        if missing_answer_terms or missing_sources or source_error:
            failures.append(
                {
                    "case": case.name,
                    "missing_answer_terms": missing_answer_terms,
                    "missing_sources": missing_sources,
                    "sources": sources,
                    "answer": result["cevap"],
                }
            )
            print(f"FAIL: {case.name}")
        else:
            print(f"PASS: {case.name}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure['case']}")
            print(f"  missing_answer_terms={failure['missing_answer_terms']}")
            print(f"  missing_sources={failure['missing_sources']}")
            print(f"  sources={failure['sources']}")
            print(f"  answer={failure['answer']}")
        return 1

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
