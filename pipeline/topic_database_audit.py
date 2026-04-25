from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.chatbot import RAGChatbot, enrich_chunk_metadata, infer_query_topic, normalize_text

CHUNKS_PATH = ROOT_DIR / "data" / "chunks.json"

AUDIT_QUERIES = [
    "Bilgisayar mühendisliği yaz okulunda devam zorunluluğu var mı?",
    "Bilgisayar mühendisliği bölümünde yaz okulunda başka üniversiteden ders alabilir miyim?",
    "Bilgisayar mühendisliği öğrencisi mezuniyet için staj eksikse diplomasını alabilir mi?",
    "Bilgisayar mühendisliğinde mezun olabilmek için staj dışında hangi temel şartlar sağlanmalı?",
    "Bilgisayar mühendisliği için staj sigortasını kim yapıyor?",
    "Stajımı zamanında teslim etmezsem bilgisayar mühendisliği için süreç nasıl işler?",
]


def load_chunks():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return [enrich_chunk_metadata(chunk) for chunk in chunks]


def short_text(text: str, limit: int = 260) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        shortened = text
    else:
        shortened = text[: limit - 3] + "..."
    return shortened.encode("cp1254", errors="replace").decode("cp1254")


def bucket_chunks(chunks, query: str):
    normalized_query = normalize_text(query)
    topic = infer_query_topic(query)
    groups = {
        "all_hits": [],
        "topic_hits": [],
        "scope_hits": [],
        "focus_hits": [],
    }

    for chunk in chunks:
        content = chunk.get("content", "")
        normalized_content = normalize_text(content)
        if not any(token in normalized_content for token in normalized_query.split() if len(token) >= 4):
            continue

        groups["all_hits"].append(chunk)
        if chunk.get("topic") == topic:
            groups["topic_hits"].append(chunk)
        if chunk.get("program_scope") == "bilgisayar_muhendisligi":
            groups["scope_hits"].append(chunk)
        if topic == "yaz_okulu" and any(marker in normalized_content for marker in ["devam", "universite", "esdeger"]):
            groups["focus_hits"].append(chunk)
        elif topic == "mezuniyet" and any(
            marker in normalized_content for marker in ["mezun", "mezuniyet", "diploma", "basarili", "tamamlamis"]
        ):
            groups["focus_hits"].append(chunk)
        elif topic == "staj" and any(
            marker in normalized_content for marker in ["sigorta", "rapor", "teslim", "sbs", "takip eden yariyil"]
        ):
            groups["focus_hits"].append(chunk)

    return topic, groups


def print_chunk_section(title: str, items, limit: int = 5):
    print(f"\n{title}")
    print("-" * len(title))
    if not items:
        print("yok")
        return
    for index, item in enumerate(items[:limit], start=1):
        print(f"{index}. title={item.get('source_title')} | topic={item.get('topic')} | scope={item.get('program_scope')}")
        print(f"   url={item.get('source_url')}")
        print(f"   text={short_text(item.get('content', ''))}")


def print_counter(title: str, items, field: str, limit: int = 8):
    print(f"\n{title}")
    print("-" * len(title))
    if not items:
        print("yok")
        return
    counter = Counter(item.get(field, "") for item in items)
    for key, value in counter.most_common(limit):
        print(f"{key or 'bos'}: {value}")


def run_query_audit(bot: RAGChatbot, chunks, query: str):
    topic, groups = bucket_chunks(chunks, query)
    search_query = bot._build_search_query(query)
    results = bot.hybrid_search(search_query, k=7)
    direct_answer = bot._extract_direct_answer(search_query, results)

    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print(f"resolved_topic={topic}")
    print(f"search_query={search_query}")
    print(f"retrieval_titles={[item.get('source_title') for item in results[:7]]}")
    print(f"retrieval_topics={[item.get('topic') for item in results[:7]]}")
    print(f"retrieval_scopes={[item.get('program_scope') for item in results[:7]]}")
    print(f"direct_answer={short_text(direct_answer or 'yok', 500)}")

    for key in ["all_hits", "topic_hits", "scope_hits", "focus_hits"]:
        print(f"{key}_count={len(groups[key])}")

    print_counter("Top source titles in topic hits", groups["topic_hits"], "source_title")
    print_counter("Top source titles in focus hits", groups["focus_hits"], "source_title")
    print_chunk_section("Focus hit examples", groups["focus_hits"])
    print_chunk_section("Hybrid retrieval top results", results, limit=7)


def main() -> int:
    chunks = load_chunks()
    bot = RAGChatbot()

    print("TOPIC DATABASE AUDIT")
    print("====================")
    print(f"chunk_count={len(chunks)}")

    source_title_counter = Counter(chunk.get("source_title", "") for chunk in chunks)
    ambiguous_titles = [
        title for title, count in source_title_counter.items() if title in {"Öğrenci İşleri", "Akademik Takvim", "Merkezi Mevzuat", "Fakulte Bolum"}
    ]
    print(f"ambiguous_titles={ambiguous_titles}")

    for query in AUDIT_QUERIES:
        run_query_audit(bot, chunks, query)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
