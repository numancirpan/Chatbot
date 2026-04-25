import json
import hashlib
import os
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from core.vector_db_utils import (
    candidate_vector_db_dirs,
    sqlite_embedding_count,
    sqlite_collection_names,
    subprocess_similarity_search,
)


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
PREFERRED_MODELS = ["llama3", "llama3:8b", "qwen2.5:7b"]
NO_ANSWER_TEXT = (
    "Bu konuda resmi belgelerde bilgiye ulaşılamadım. "
    "Lütfen Öğrenci İşleri birimi ile iletişime geçiniz."
)
ASSISTANT_IDENTITY = (
    "Düzce Üniversitesi Öğrenci İşleri Daire Başkanlığı için geliştirilen, "
    "akademik ve idari sorulara hızlı, güvenilir, tutarlı ve kurumsal biçimde yanıt veren dijital asistansın."
)
ASSISTANT_GOALS = [
    "öğrencilerin resmi bilgiye hızlı ve doğru erişmesini sağlamak",
    "öğrenci işleri personelinin tekrar eden soru yükünü azaltmak",
    "kurumsal dil yapısına uygun, kaynak dayanaklı yanıt üretmek",
    "takip sorularında konuşma bağlamını koruyarak tutarlı diyalog yürütmek",
]
ASSISTANT_PERSONALITY = [
    "profesyonel",
    "nazik",
    "empatik",
    "yardımsever",
    "kurumsal iletişim diline uygun",
]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(ROOT_DIR, "data", "chunks.json")
KNOWLEDGE_BASE_FILE = os.path.join(ROOT_DIR, "data", "knowledge_base.json")
LOCAL_VECTOR_EMBEDDINGS_FILE = os.path.join(ROOT_DIR, "data", "local_vector_index.npy")
LOCAL_VECTOR_META_FILE = os.path.join(ROOT_DIR, "data", "local_vector_index_meta.json")
DB_DIR = next(
    (path for path in candidate_vector_db_dirs(ROOT_DIR) if sqlite_embedding_count(path)),
    os.path.join(ROOT_DIR, "db", "chroma_store_live"),
)
MAX_MEMORY_TURNS = 5
COLLECTION_NAME = "langchain"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_PROGRAM_SCOPE = ""
GENERAL_SCOPE = "genel"
OTHER_SCOPE = "diger_birim"
PROGRAM_SCOPE_HINTS = {
    "bilgisayar_muhendisligi": [
        "bm.mf.duzce.edu.tr",
        "bilgisayar muhendisligi",
        "bm399",
        "bm499",
    ],
    "orman_muhendisligi": [
        "orman muhendisligi",
    ],
    "orman_endustri_muhendisligi": [
        "orman endustri muhendisligi",
    ],
    "peyzaj_mimarligi": [
        "peyzaj mimarligi",
    ],
    "agac_isleri_endustri_muhendisligi": [
        "agac isleri endustri muhendisligi",
    ],
    "insaat_muhendisligi": [
        "insaat muhendisligi",
        "santiye staji",
        "buro staji",
    ],
    "mimarlik": [
        "mimarlik",
    ],
    "isletme": [
        "isletme",
    ],
    "meslek_yuksekokulu": [
        "meslek yuksekokulu",
        "on lisans",
        "onlisans",
    ],
}
PROGRAM_SCOPE_LABELS = {
    "bilgisayar_muhendisligi": "Bilgisayar Mühendisliği",
    "orman_muhendisligi": "Orman Mühendisliği",
    "orman_endustri_muhendisligi": "Orman Endüstri Mühendisliği",
    "peyzaj_mimarligi": "Peyzaj Mimarlığı",
    "agac_isleri_endustri_muhendisligi": "Ağaç İşleri Endüstri Mühendisliği",
    "insaat_muhendisligi": "İnşaat Mühendisliği",
    "mimarlik": "Mimarlık",
    "isletme": "İşletme",
    "meslek_yuksekokulu": "Meslek Yüksekokulu",
}
OTHER_UNIT_HINTS = [
    "orman muhendisligi",
    "orman endustri muhendisligi",
    "yaban hayati",
    "peyzaj mimarligi",
    "agac isleri endustri muhendisligi",
    "insaat",
    "mimarlik",
    "isletme",
    "teknik egitim fakultesi",
    "meslek yuksekokulu",
    "on lisans",
    "onlisans",
    "santiye",
    "buro staji",
]
TOPIC_HINTS = {
    "staj": ["staj", "sbs", "bm399", "bm499", "staj rapor", "staj defter"],
    "ders_kaydi": ["ders kaydi", "kayit yenile", "ders sec", "obs", "kayitlarini yenilemek"],
    "add_drop": ["add drop", "ekle sil", "ders ekle", "ders birak", "ders sil"],
    "devamsizlik": ["devamsizlik", "devam zorunlulugu", "yoklama"],
    "sinavlar": ["vize", "final", "but", "mazeret", "sinav"],
    "not_sistemi": ["ortalama", "gano", "agno", "not sistemi", "harf notu", "akts"],
    "mezuniyet": ["mezuniyet", "mezun", "diploma", "mezun durumda"],
    "cap_yandal": ["cift anadal", "cap", "yandal"],
    "yatay_gecis": ["yatay gecis", "kurumlararasi gecis", "merkezi yatay gecis"],
    "harc_ucret": ["harc", "katki payi", "ogrenim ucreti", "ucret"],
    "burs": ["burs", "bursu", "bursunu"],
    "askerlik_tecili": ["askerlik", "tecil", "askerlik tecili"],
    "ogrenci_belgesi_transkript": ["ogrenci belgesi", "transkript", "not dokumu"],
    "disiplin": ["disiplin", "uzaklastirma", "kinama"],
    "akademik_takvim_duyurular": ["akademik takvim", "duyuru", "onemli basvuru", "derslerin baslamasi"],
    "yaz_okulu": ["yaz okulu", "yaz okulu kayit", "yaz okulunun"],
    "muafiyet_intibak": ["muafiyet", "intibak", "esdegerlik", "ders saydirma"],
}
TOPIC_LABELS = {
    "staj": "Staj",
    "ders_kaydi": "Ders Kaydi / Kayit Yenileme",
    "add_drop": "Add-Drop",
    "devamsizlik": "Devamsizlik",
    "sinavlar": "Sinavlar",
    "not_sistemi": "Not Sistemi / Ortalama",
    "mezuniyet": "Mezuniyet",
    "cap_yandal": "CAP / Yandal",
    "yatay_gecis": "Yatay Gecis",
    "harc_ucret": "Harc / Ucret",
    "burs": "Burs",
    "ogrenci_belgesi_transkript": "Ogrenci Belgesi / Transkript",
    "askerlik_tecili": "Askerlik Tecili",
    "disiplin": "Disiplin Islemleri",
    "akademik_takvim_duyurular": "Akademik Takvim / Duyurular",
    "yaz_okulu": "Yaz Okulu",
    "muafiyet_intibak": "Muafiyet / Intibak",
}
SOURCE_STOPWORDS = {
    "sayin",
    "ogrencimiz",
    "gore",
    "icin",
    "olan",
    "olarak",
    "ancak",
    "veya",
    "dair",
    "ilgili",
    "resmi",
    "belge",
    "belgelerde",
    "kaynak",
    "kaynakta",
    "bilgi",
    "bulunmaktadir",
    "belirtilmektedir",
    "lütfen",
    "lutfen",
    "birimi",
    "iletisim",
    "geciniz",
}
GENERIC_SOURCE_TITLES = {
    "ogrenci isleri",
    "merkezi mevzuat",
    "fakulte bolum",
    "fakulte bolum sayfasi",
    "akademik takvim",
    "genel",
}
SCOPE_CLARIFICATION_TEXT = (
    "Bu bilgi bölüm veya fakülteye göre değişebilmektedir. "
    "Lütfen bölüm/program belirterek tekrar sorunuz."
)
FALLBACK_PATTERNS = [
    "Bu konuda resmi belgelerde bilgiye ulaşılamadım. Lütfen Öğrenci İşleri birimi ile iletişime geçiniz.",
    "Bu konuda resmi belgelerde bilgiye ulasilamadim. Lutfen Ogrenci Isleri birimi ile iletisime geciniz.",
]
NUMERIC_UNIT_PATTERN = re.compile(
    r"\b\d+(?:\s*-\s*\d+)?\s*(?:iş günü|işgünü|gün|hafta|ay|akts|kredi|yarıyıl)\b",
    re.IGNORECASE,
)
WORKDAY_RANGE_PATTERN = re.compile(r"\b(\d+)\s*-\s*(\d+)\s*(?:iş günü|işgünü)\b", re.IGNORECASE)
WORKDAY_NUMBER_PATTERN = re.compile(r"\b(\d+)\s*(?:\([^)]*\)\s*)?(?:iş günü|işgünü)\b", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b")
WEEK_PATTERN = re.compile(r"\b(\d+)\s*hafta\b", re.IGNORECASE)
WEEK_ENDING_PATTERN = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})\s+Yaz Okulu\s+(\d+)\s*hafta", re.IGNORECASE)
DATE_RANGE_PATTERN = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})\s*/\s*(\d{1,2}\.\d{1,2}\.\d{4})")
SUMMER_SCHOOL_START_PATTERN = re.compile(
    r"YAZ OKULU\s*[\r\n ].{0,30}?(\d{1,2}\.\d{1,2}\.\d{4})",
    re.IGNORECASE,
)
SUMMER_SCHOOL_EXPLICIT_START_PATTERN = re.compile(
    r"YAZ\s+OKULU.{0,80}?Başlangıç\s+(\d{1,2}\.\d{1,2}\.\d{4})",
    re.IGNORECASE | re.DOTALL,
)
SUMMER_SCHOOL_RANGE_PATTERN = re.compile(
    r"(\d{1,2}\.\d{1,2}\.\d{4})\s*/\s*(\d{1,2}\.\d{1,2}\.\d{4}).{0,80}?YAZ OKULU",
    re.IGNORECASE | re.DOTALL,
)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
SOURCE_REF_PATTERN = re.compile(r"\[Kaynak\s+\d+\]")
SEMESTER_PAIR_PATTERN = re.compile(r"\b(\d+)\s+ve\s+(\d+)\s+yariyillarda\b", re.IGNORECASE)
SUMMER_AFTER_PATTERN = re.compile(
    r"\b(\d+)\s+yariyil\w*\s+ve\s+(\d+)\s+yariyil\w*\s+izleyen\s+yaz",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    translation_table = str.maketrans(
        {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "İ": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
            "Ç": "c",
            "Ğ": "g",
            "Ö": "o",
            "Ş": "s",
            "Ü": "u",
        }
    )
    text = text.translate(translation_table).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def repair_text_encoding(text: str) -> str:
    repaired = text
    suspicious_markers = ("Ã", "Ä", "Å", "Â")
    for _ in range(3):
        if not any(marker in repaired for marker in suspicious_markers):
            break
        try:
            repaired = repaired.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
    return repaired


def is_short_factual_query(query: str) -> bool:
    normalized = normalize_text(query)
    factual_markers = [
        "kac",
        "ne kadar",
        "hangi tarihte",
        "ne zaman",
        "kac gun",
        "kac hafta",
        "kac ay",
        "kac akts",
        "kac kredi",
        "suresi",
        "sure",
    ]
    return len(tokenize(query)) <= 10 and any(marker in normalized for marker in factual_markers)


def asks_staj_timing(query: str) -> bool:
    normalized = normalize_text(query)
    return "staj" in normalized and any(
        marker in normalized
        for marker in ["hangi donem", "hangi donemlerde", "hangi yariyil", "ne zaman yap", "ne zaman yapmam"]
    )


def asks_staj_course_registration(query: str) -> bool:
    normalized = normalize_text(query)
    return "staj" in normalized and any(
        marker in normalized for marker in ["saydir", "staj 1", "staj1", "obs", "takip eden yariyil"]
    )


def asks_staj_missed_period(query: str) -> bool:
    normalized = normalize_text(query)
    return "staj" in normalized and any(
        marker in normalized
        for marker in ["yapamazsam", "yapamaz isem", "doneminde", "ertelersem", "yapamazsak", "ne olur"]
    )


def asks_staj_duration(query: str) -> bool:
    normalized = normalize_text(query)
    if "staj" not in normalized:
        return False
    return any(
        re.search(pattern, normalized)
        for pattern in [
            r"\bkac gun\b",
            r"\bkac is gunu\b",
            r"\bsure\b",
            r"\bsuresi\b",
            r"\bne kadar\b",
        ]
    )


def asks_staj_count(query: str) -> bool:
    normalized = normalize_text(query)
    if "staj" not in normalized:
        return False
    count_markers = [
        "kac kere",
        "kac staj",
        "kac tane staj",
        "kac zorunlu staj",
        "kac kez",
        "staj sayisi",
        "staj yapmaliyim",
        "staj yapmali",
        "staj i",
        "staj ii",
        "bm399",
        "bm499",
    ]
    return any(marker in normalized for marker in count_markers)


def asks_staj_report_submission(query: str) -> bool:
    normalized = normalize_text(query)
    if "staj" not in normalized:
        return False
    report_markers = [
        "rapor",
        "defter",
        "dosya",
        "sbs",
        "teslim",
        "yukle",
        "degerlendiril",
    ]
    process_markers = [
        "zamaninda",
        "gec",
        "surec",
        "nasil",
        "ne olur",
        "ne zaman",
        "teslim etmeyen",
        "teslim etmez",
        "yuklemez",
        "yuklemedim",
    ]
    return any(marker in normalized for marker in report_markers) and any(
        marker in normalized for marker in process_markers
    )


def asks_disciplinary_scholarship_loss(query: str) -> bool:
    normalized = normalize_text(query)
    return "burs" in normalized and any(marker in normalized for marker in ["disiplin", "ceza", "uzaklastirma"])


def asks_makeup_exam_with_missing_internship(query: str) -> bool:
    normalized = normalize_text(query)
    exam_markers = [
        "tek cift",
        "tek ders",
        "cift ders",
        "tek cift sinav",
        "tek cift ders sinav",
    ]
    internship_markers = [
        "staj",
        "yz",
        "yetersiz",
        "kalan",
        "tek dersim",
        "bir dersim",
        "kayitlanmamis",
        "yapmamis",
        "yapamamis",
    ]
    return any(marker in normalized for marker in exam_markers) and any(
        marker in normalized for marker in internship_markers
    )


def asks_yaz_okulu_duration(query: str) -> bool:
    normalized = normalize_text(query)
    return "yaz okulu" in normalized and any(
        marker in normalized for marker in ["kac hafta", "ne kadar sur", "sure", "suresi"]
    )


def asks_yaz_okulu_start(query: str) -> bool:
    normalized = normalize_text(query)
    return "yaz okulu" in normalized and any(
        marker in normalized for marker in ["ne zaman", "hangi tarihte", "baslangic", "basliyor", "baslar"]
    )


def asks_yaz_staji_schedule(query: str) -> bool:
    normalized = normalize_text(query)
    return "yaz staji" in normalized and any(
        marker in normalized for marker in ["ne zaman", "hangi tarihte", "baslangic", "basliyor", "donem"]
    )


def asks_yaz_okulu_attendance(query: str) -> bool:
    normalized = normalize_text(query)
    return "yaz okulu" in normalized and any(
        marker in normalized for marker in ["devam", "devamsizlik", "devam zorunlulugu", "yoklama"]
    )


def asks_staj_insurance(query: str) -> bool:
    normalized = normalize_text(query)
    return "staj" in normalized and "sigorta" in normalized


def asks_external_summer_school_course(query: str) -> bool:
    normalized = normalize_text(query)
    if "yaz okulu" not in normalized:
        return False
    if re.search(r"\b\w*niversit\w*\b", normalized):
        return True
    return any(
        marker in normalized
        for marker in [
            "baska universite",
            "baska universit",
            "diger universite",
            "universiteden",
            "niversiteden",
            "universitelerden",
            "universitemiz disinda",
            "misafir ogrenci",
            "disaridan ders",
        ]
    )


def asks_graduation_with_incomplete_internship(query: str) -> bool:
    normalized = normalize_text(query)
    graduation_markers = ["mezun", "mezuniyet", "diploma"]
    internship_markers = ["staj", "stajini", "stajim", "staj eksik", "staj eksikse", "stajini tamamlamamis"]
    incomplete_markers = ["eksik", "tamamlamamis", "tamamlanmamis", "yapmamis", "kalmis", "almadan", "bitirmeden"]
    return (
        any(marker in normalized for marker in graduation_markers)
        and any(marker in normalized for marker in internship_markers)
        and any(marker in normalized for marker in incomplete_markers)
    )


def asks_graduation_requirements(query: str) -> bool:
    normalized = normalize_text(query)
    if not any(marker in normalized for marker in ["mezun", "mezuniyet", "diploma"]):
        return False
    return any(
        marker in normalized
        for marker in ["sart", "sartlar", "kosul", "kosullar", "gerek", "gerekiyor", "temel"]
    )


def asks_post_upload_graduation(query: str) -> bool:
    normalized = normalize_text(query)
    if "staj" not in normalized and "rapor" not in normalized:
        return False
    if "mezun" not in normalized:
        return False
    return any(
        marker in normalized
        for marker in ["yukledikten sonra", "yukledim", "raporu yukle", "mail", "e posta", "degerlendirme talebi"]
    )


def asks_period_count(query: str) -> bool:
    normalized = normalize_text(query)
    return any(marker in normalized for marker in ["kac donem", "donem var", "kac tane donem"])


def extract_years(text: str) -> List[int]:
    return [int(year) for year in re.findall(r"\b20\d{2}\b", text)]


def get_confirmed_day_count(query: str) -> Optional[int]:
    normalized = normalize_text(query)
    if "staj" not in normalized or "mu" not in normalized:
        return None
    match = re.search(r"\b(\d+)\s*(?:is gunu|gun)\s*mu\b", normalized)
    if not match:
        return None
    return int(match.group(1))


def is_program_specific_query(query: str) -> bool:
    normalized = normalize_text(query)
    if infer_query_scope(query):
        return True
    program_specific_markers = [
        "staj",
        "obs",
        "bm399",
        "bm499",
        "staj1",
        "staj 1",
        "staj2",
        "staj 2",
    ]
    return any(marker in normalized for marker in program_specific_markers)


def infer_chunk_scope(chunk: Dict) -> str:
    normalized_content = normalize_text(chunk.get("content", ""))
    normalized_url = chunk.get("source_url", "").lower()

    for scope_name, hints in PROGRAM_SCOPE_HINTS.items():
        if any(hint in normalized_content or hint in normalized_url for hint in hints):
            return scope_name

    if any(hint in normalized_content or hint in normalized_url for hint in OTHER_UNIT_HINTS):
        return OTHER_SCOPE

    return GENERAL_SCOPE


def mentions_other_unit(query: str) -> bool:
    normalized = normalize_text(query)
    return any(hint in normalized for hint in OTHER_UNIT_HINTS)


def infer_query_scope(query: str) -> str:
    normalized = normalize_text(query)
    for scope_name, hints in PROGRAM_SCOPE_HINTS.items():
        if any(hint in normalized for hint in hints):
            return scope_name
    return ""


def infer_topic(chunk: Dict) -> str:
    normalized_content = normalize_text(chunk.get("content", ""))
    normalized_url = chunk.get("source_url", "").lower()
    kategori = normalize_text(chunk.get("kategori", ""))
    source_title = normalize_text(chunk.get("source_title", ""))
    haystack = f"{normalized_content} {normalized_url} {kategori} {source_title}"

    if "bm.mf.duzce.edu.tr/sayfa/878b" in normalized_url:
        return "staj"
    if "bm.mf.duzce.edu.tr/sayfa/17ac" in normalized_url:
        return "yaz_okulu"
    if "mf.duzce.edu.tr/sayfa/967a" in normalized_url:
        return "staj"

    specific_categories = {
        "staj",
        "yaz_okulu",
        "cap_yandal",
        "yatay_gecis",
        "muafiyet_intibak",
        "mezuniyet",
        "harc_ucret",
        "disiplin",
        "burs",
        "askerlik_tecili",
        "ders_kaydi",
        "add_drop",
        "devamsizlik",
        "sinavlar",
        "not_sistemi",
        "ogrenci_belgesi_transkript",
    }
    if kategori in specific_categories:
        return kategori

    priority_topics = [
        "cap_yandal",
        "yatay_gecis",
        "muafiyet_intibak",
        "yaz_okulu",
        "staj",
        "harc_ucret",
        "burs",
        "askerlik_tecili",
        "disiplin",
        "ogrenci_belgesi_transkript",
        "ders_kaydi",
        "add_drop",
        "devamsizlik",
        "sinavlar",
        "not_sistemi",
        "mezuniyet",
        "akademik_takvim_duyurular",
    ]

    for topic in priority_topics:
        hints = TOPIC_HINTS.get(topic, [])
        if any(hint in haystack for hint in hints):
            if topic == "cap_yandal" and any(marker in haystack for marker in ["mezuniyet", "diploma", "gecici mezuniyet"]):
                continue
            return topic

    if kategori:
        return kategori
    return "genel"


def infer_query_topic(query: str) -> str:
    normalized = normalize_text(query)

    if asks_post_upload_graduation(query):
        return "staj"
    if asks_graduation_with_incomplete_internship(query) or asks_graduation_requirements(query):
        return "mezuniyet"
    if asks_yaz_okulu_attendance(query) or "yaz okulu" in normalized:
        return "yaz_okulu"
    if (
        asks_staj_timing(query)
        or asks_staj_course_registration(query)
        or asks_staj_missed_period(query)
        or asks_staj_duration(query)
        or asks_staj_count(query)
        or asks_staj_report_submission(query)
        or asks_makeup_exam_with_missing_internship(query)
        or asks_staj_insurance(query)
        or "staj" in normalized
    ):
        return "staj"
    if "devamsiz" in normalized or "devam zorunlulugu" in normalized or "yoklama" in normalized:
        return "devamsizlik"
    if "sinav" in normalized or "butunleme" in normalized or "mazeret" in normalized:
        return "sinavlar"
    if "muafiyet" in normalized or "intibak" in normalized:
        return "muafiyet_intibak"
    if "harc" in normalized or "ucret" in normalized or "katki payi" in normalized:
        return "harc_ucret"
    if "burs" in normalized:
        return "burs"
    if "yatay gecis" in normalized:
        return "yatay_gecis"
    if "cap" in normalized or "cift anadal" in normalized or "yandal" in normalized:
        return "cap_yandal"

    return infer_topic({"content": query, "source_url": "", "kategori": ""})


def _title_from_url_slug(url: str) -> str:
    parts = [part for part in url.split("/") if part]
    for part in reversed(parts):
        if re.fullmatch(r"[0-9a-f]{3,}", part.lower()):
            continue
        candidate = part.replace("-", " ").replace("_", " ").strip()
        if len(normalize_text(candidate)) >= 8:
            return candidate.title()
    return ""


def _title_from_content_lines(content: str) -> str:
    lines = []
    for raw_line in str(content).splitlines():
        line = " ".join(raw_line.split()).strip(" -:|")
        if not line:
            continue
        normalized = normalize_text(line)
        if len(normalized) < 6 or len(line) > 140:
            continue
        lines.append(line)
        if len(lines) >= 8:
            break

    if not lines:
        return ""

    joined = normalize_text(" ".join(lines[:4]))
    if "yaz okulu yonergesi" in joined:
        return "Yaz Okulu Yonergesi"
    if "yaz okulu egitimi" in joined:
        return "Yaz Okulu Egitimi"
    if "stajlar hakkinda sikca sorulan sorular" in joined:
        return "Bilgisayar Muhendisligi - Staj SSS"
    if "bilgisayar muhendisligi" in joined and "staj" in joined:
        return "Bilgisayar Muhendisligi - Staj"
    if "yaz staji" in joined:
        return "Muhendislik Fakultesi - Yaz Staji"
    if "cift anadal" in joined or "yandal" in joined:
        return "CAP ve Yandal"
    if "diploma" in joined or "mezuniyet" in joined:
        return "Diploma ve Mezuniyet Belgeleri Yonergesi"
    if "akademik takvim" in joined:
        return "Akademik Takvim"

    return lines[0]


def _is_graduation_whitelisted_source(result: Dict) -> bool:
    source_title = normalize_text(str(result.get("source_title", "")))
    source_url = str(result.get("source_url", "")).lower()
    content = normalize_text(str(result.get("content", ""))[:1200])

    if any(marker in source_url for marker in ["cift-anadal", "yandal-program", "cap"]):
        return False
    if any(
        marker in source_title
        for marker in [
            "cap",
            "yandal",
            "akreditasyon",
            "yabanci uyruklu",
            "ek ",
            "ek-",
            "uluslararasi",
        ]
    ):
        return False
    if any(
        marker in source_title
        for marker in [
            "mezuniyet",
            "diploma",
            "lisans egitim",
            "sinav yonetmeligi",
            "stajlar hakkinda sikca sorulan sorular",
            "bilgisayar muhendisligi staj sss",
        ]
    ):
        return True
    if any(
        marker in content
        for marker in [
            "mezun olmaya hak kazanir",
            "butun calismalari tamamlamis",
            "gecici mezuniyet belgesi",
            "diploma defteri",
            "stajini tamamlamayan ogrenci",
        ]
    ):
        return True
    return False


def infer_source_title(chunk: Dict) -> str:
    explicit_title = str(chunk.get("source_title", "")).strip()
    if explicit_title and normalize_text(explicit_title) not in GENERIC_SOURCE_TITLES:
        return explicit_title

    url = chunk.get("source_url", "")
    kategori = chunk.get("kategori", "Genel")
    content = str(chunk.get("content", ""))
    normalized_url = url.lower()
    if any(marker in normalized_url for marker in ["cift-anadal", "yandal-program", "/yandal"]):
        return "ÇAP ve Yandal"
    if "bm.mf.duzce.edu.tr/sayfa/878b" in normalized_url:
        return "Bilgisayar Mühendisliği - Staj SSS"
    if "bm.mf.duzce.edu.tr/sayfa/4a82" in normalized_url:
        return "Bilgisayar Mühendisliği - Staj"
    if "bm.mf.duzce.edu.tr/sayfa/17ac" in normalized_url:
        return "Bilgisayar Mühendisliği - Yaz Okulu"
    if "mf.duzce.edu.tr/sayfa/967a" in normalized_url:
        return "Mühendislik Fakültesi - Yaz Stajı"
    normalized_content = normalize_text(content[:800])
    if "yaz okulu yonergesi" in normalized_content:
        return "Yaz Okulu Yönergesi"
    if "diploma diploma defteri gecici mezuniyet belgesi" in normalized_content:
        return "Diploma ve Mezuniyet Belgeleri Yönergesi"
    if "lisans egitim ogretim ve sinav yonetmeligi" in normalized_content:
        return "Lisans Eğitim-Öğretim ve Sınav Yönetmeliği"
    if "cift anadal" in normalized_content or "yandal" in normalized_content:
        return "ÇAP ve Yandal"
    if "yatay gecis" in normalized_content:
        return "Yatay Geçiş"
    if "muafiyet" in normalized_content and "intibak" in normalized_content:
        return "Muafiyet ve İntibak Esasları"
    if "akademik-takvim" in normalized_url or chunk.get("kategori") == "akademik_takvim":
        return "Akademik Takvim"
    extracted_title = _title_from_content_lines(content)
    if extracted_title:
        return extracted_title
    slug_title = _title_from_url_slug(url)
    if slug_title:
        return slug_title
    if "ogrenciisleri.duzce.edu.tr" in normalized_url:
        return "Öğrenci İşleri"
    if kategori == "fakulte_bolum":
        return "Fakülte/Bölüm Sayfası"
    if kategori == "merkezi_mevzuat":
        return "Merkezi Mevzuat"
    return kategori.replace("_", " ").title()


def extract_metadata_years(text: str) -> str:
    years = sorted(set(re.findall(r"\b20\d{2}(?:\s*-\s*20\d{2})?\b", text)))
    return ",".join(year.replace(" ", "") for year in years[:5])


def enrich_chunk_metadata(chunk: Dict) -> Dict:
    enriched = dict(chunk)
    enriched["program_scope"] = enriched.get("program_scope") or infer_chunk_scope(enriched)
    enriched["topic"] = enriched.get("topic") or infer_topic(enriched)
    enriched["source_title"] = enriched.get("source_title") or infer_source_title(enriched)
    enriched["years"] = enriched.get("years") or extract_metadata_years(
        f"{enriched.get('content', '')} {enriched.get('source_url', '')}"
    )
    if "chunk_id" not in enriched:
        content = enriched.get("content", "")
        source_url = enriched.get("source_url", "")
        enriched["chunk_id"] = hashlib.md5(f"{source_url}\n{content}".encode("utf-8")).hexdigest()
    return enriched


def is_scope_clarification_query(query: str) -> bool:
    normalized = normalize_text(query)
    if not infer_query_scope(query):
        return False
    if len(tokenize(query)) > 8:
        return False
    topic_markers = [
        "staj",
        "yaz okulu",
        "ders",
        "sinav",
        "harc",
        "akts",
        "belge",
        "basvuru",
        "kac",
        "ne zaman",
        "hangi",
        "nasil",
        "nedir",
        "olur mu",
        "girebilir",
    ]
    return not any(marker in normalized for marker in topic_markers)


def is_follow_up_query(query: str) -> bool:
    normalized = normalize_text(query)
    follow_up_markers = [
        "peki",
        "tamam",
        "o zaman",
        "bu durumda",
        "buna gore",
        "bunun icin",
        "bunlar",
        "bunlardan",
        "bu ders",
        "bu staj",
        "bu belge",
        "o ders",
        "o staj",
        "o belge",
        "ya",
    ]
    if any(normalized.startswith(marker) for marker in follow_up_markers):
        return True
    if len(tokenize(query)) <= 5 and any(marker in normalized for marker in ["kac kere", "hangi donemde", "ne zaman", "nasil"]):
        return True
    return False


def build_query_variants(query: str) -> List[str]:
    normalized = normalize_text(query)
    variants = [query]

    if "staj" in normalized:
        variants.extend(["staj süresi", "staj iş günü", "staj kaç iş günü"])
        if any(marker in normalized for marker in ["kac gun", "sure", "suresi", "ne kadar"]):
            variants.extend(
                [
                    "staj süresi iş günü",
                    "staj süresi kaç iş günü",
                    "staj iş günü süresi",
                ]
            )
        if any(marker in normalized for marker in ["hangi donem", "hangi donemlerde", "hangi yariyil", "ne zaman"]):
            variants.extend(
                [
                    "staj hangi dönemde yapılır",
                    "staj hangi yarıyılda yapılır",
                    "staj yaz dönemi",
                    "staj 4. ve 6. yarıyıl",
                    "staj 5. ve 7. yarıyıl",
                ]
            )
        if any(marker in normalized for marker in ["saydir", "obs", "staj 1", "staj1", "takip eden yariyil"]):
            variants.extend(
                [
                    "staj takip eden yarıyılda obs",
                    "staj yaz döneminde yapıp sonraki yarıyılda ders alma",
                    "bm399 yaz stajı obs",
                ]
            )
        if asks_staj_missed_period(query):
            variants.extend(
                [
                    "staj döneminde yapamazsam ne olur",
                    "staj takip eden akademik yılların staj dönemlerinde yapılır",
                    "ilk staj döneminde staj yapma hakkı kazanamayan 6. yarıyıldan sonra",
                ]
            )

        if asks_staj_count(query):
            variants.extend(
                [
                    "zorunlu staj sayisi",
                    "kac zorunlu staj var",
                    "staj i staj ii",
                    "bm399 bm499",
                    "5 ve 7 yariyillarda 25 is gunu staj",
                ]
            )
        if asks_staj_report_submission(query):
            variants.extend(
                [
                    "staj raporu teslimi nasil ve ne zaman olmali",
                    "staj raporunuzu sbs ye yuklemeniz gerekmektedir",
                    "sistemde yuklemek icin son bir tarih bulunmamaktadir",
                    "guz donemi basladiktan yaklasik 30 gun sonra",
                    "staj defterleri staj bitim tarihinden itibaren en gec 1 ay icinde teslim",
                    "duzeltme yapmasi istenen ogrenci en cok 1 ay icinde duzeltme yapmakla yukumludur",
                ]
            )

        if asks_makeup_exam_with_missing_internship(query):
            variants.extend(
                [
                    "tek cift ders sinavi staj",
                    "tek dersi ve staji kalan ogrenci",
                    "staj dersine hic kayitlanmamis ise tek cift sinavina girme hakki yoktur",
                    "staj dersini alip yz notu almis ise tek cift sinavina basvurabilir",
                    "tek dersi ve bahar yariyilindan staji kalan ogrenci tek cift sinavina sadece dersten girebilir",
                ]
            )

    if asks_disciplinary_scholarship_loss(query):
        variants.extend(
            [
                "disiplin cezasi burs kesilmesi",
                "disiplin cezasi burs kaybi",
                "burs disiplin cezasi",
                "uzaklastirma cezasi burs",
            ]
        )

    if "yaz okulu" in normalized:
        variants.extend(
            [
                "yaz okulu akademik takvim",
                "yaz okulu baslangic",
                "yaz okulu kayitlari",
                "2025-2026 akademik takvim yaz okulu",
            ]
        )
        if asks_yaz_okulu_duration(query):
            variants.extend(
                [
                    "yaz okulu kac hafta",
                    "yaz okulu 5 hafta 7 hafta",
                    "yaz okulu suresi hafta",
                ]
            )
        if asks_yaz_okulu_start(query):
            variants.extend(
                [
                    "YAZ OKULU Baslangic",
                    "yaz okulu ne zaman basliyor akademik takvim",
                    "2025-2026 yaz okulu baslangic tarihi",
                ]
            )

    if asks_yaz_staji_schedule(query):
        variants.extend(
            [
                "2025-2026 yaz staji donemleri",
                "yaz staji donemleri",
                "muhendislik fakultesi yaz staji tarihleri",
            ]
        )

    unique_variants = []
    seen = set()
    for variant in variants:
        normalized_variant = normalize_text(variant)
        if not normalized_variant or normalized_variant in seen:
            continue
        seen.add(normalized_variant)
        unique_variants.append(variant)
    return unique_variants


class BM25Search:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(chunk["content"]) for chunk in chunks])

    def search(self, query: str, k: int = 5) -> List[Dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        return [self.chunks[i] for i in scores.argsort()[-k:][::-1]]


class Reranker:
    def __init__(self):
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            self.model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
                local_files_only=True,
            )

    def rerank(self, query: str, chunks: List[Dict], k: int = 5) -> List[Dict]:
        if not chunks:
            return []
        self._ensure_model()
        scores = self.model.predict([[query, chunk["content"]] for chunk in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
        return [chunk for chunk, _ in ranked[:k]]


class LocalVectorIndex:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.model = None
        self.embeddings = None
        self.chunk_ids = [chunk.get("chunk_id", "") for chunk in chunks]
        self._load_or_build()

    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)

    def _load_or_build(self):
        if os.path.exists(LOCAL_VECTOR_EMBEDDINGS_FILE) and os.path.exists(LOCAL_VECTOR_META_FILE):
            try:
                with open(LOCAL_VECTOR_META_FILE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("chunk_ids") == self.chunk_ids:
                    self.embeddings = np.load(LOCAL_VECTOR_EMBEDDINGS_FILE)
                    return
            except Exception:
                pass
        self._build()

    def _build(self):
        self._ensure_model()
        self.embeddings = self.model.encode(
            [chunk.get("content", "") for chunk in self.chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        np.save(LOCAL_VECTOR_EMBEDDINGS_FILE, self.embeddings)
        with open(LOCAL_VECTOR_META_FILE, "w", encoding="utf-8") as f:
            json.dump({"chunk_ids": self.chunk_ids}, f, ensure_ascii=False)

    def search(self, query: str, k: int) -> List[Dict]:
        if self.embeddings is None or not len(self.chunks):
            return []
        self._ensure_model()
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.chunks[int(index)] for index in top_indices]


class RAGChatbot:
    def __init__(self, program_scope: str = DEFAULT_PROGRAM_SCOPE):
        self.program_scope = program_scope
        self.model_name = self._resolve_model_name()
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)
        if isinstance(self.chunks, dict):
            self.chunks = [self.chunks]
        if isinstance(self.knowledge_base, dict):
            self.knowledge_base = [self.knowledge_base]
        self.chunks = [enrich_chunk_metadata(chunk) for chunk in self.chunks]
        self.raw_records = []
        for record in self.knowledge_base:
            content = record.get("icerik", "")
            if not content:
                continue
            mapped = {
                "content": content,
                "source_url": record.get("url", ""),
                "kategori": record.get("kategori", ""),
                "icerik_tipi": record.get("icerik_tipi", ""),
            }
            self.raw_records.append(enrich_chunk_metadata(mapped))

        self.bm25_search = BM25Search(self.chunks)
        self.local_vector_index = LocalVectorIndex(self.chunks)
        self.vector_store = None
        self.vector_db_health = self._vector_store_health()
        self.vector_count = self._vector_store_count()
        self.reranker = Reranker()
        self.message_history = InMemoryChatMessageHistory()
        self.conversation_state = {
            "program_scope": self.program_scope or "",
            "topic": "",
        }
        self.last_answer_context: List[Dict] = []

        self._ollama_kontrol()
        print(f"{len(self.chunks)} chunk yuklendi")
        if self.vector_count:
            print(f"ChromaDB hazir ({self.vector_count} kayit)")
            print(f"Aktif DB yolu: {DB_DIR}")
            if self.vector_db_health.get("health_source"):
                print(f"DB saglik kontrolu: {self.vector_db_health['health_source']}")
            if self.vector_db_health.get("sqlite_count") not in (None, self.vector_count):
                print(
                    "UYARI: ChromaDB istemci sayimi ve sqlite sayimi farkli. "
                    "DB sagligi tekrar kontrol edilmeli."
                )
        else:
            print("UYARI: ChromaDB bos veya okunamiyor. Arama BM25 uzerinden devam edecek.")
            print(f"Aktif DB yolu: {DB_DIR}")
            print("DB'yi yenilemek icin: python pipeline/create_vector_db.py --rebuild")
        print("BM25 + Reranker + ChromaDB + sohbet hafizasi hazir")

    def _resolve_program_scope(self, query: str) -> str:
        return (
            infer_query_scope(query)
            or self.program_scope
            or self.conversation_state.get("program_scope", "")
        )

    def _resolve_topic(self, query: str) -> str:
        query_topic = infer_query_topic(query)
        if query_topic != "genel":
            return query_topic
        return self.conversation_state.get("topic", "")

    def _scope_label(self, scope: str) -> str:
        return PROGRAM_SCOPE_LABELS.get(scope, scope.replace("_", " ").title()).strip()

    def _topic_label(self, topic: str) -> str:
        return TOPIC_LABELS.get(topic, topic.replace("_", " ")).strip()

    def _dominant_scope(self, context: List[Dict]) -> str:
        scopes = [
            chunk.get("program_scope", infer_chunk_scope(chunk))
            for chunk in context
            if chunk.get("program_scope", infer_chunk_scope(chunk)) not in {GENERAL_SCOPE, OTHER_SCOPE, ""}
        ]
        if not scopes:
            return ""
        return Counter(scopes).most_common(1)[0][0]

    def _dominant_topic(self, context: List[Dict]) -> str:
        topics = [
            chunk.get("topic", infer_topic(chunk))
            for chunk in context
            if chunk.get("topic", infer_topic(chunk)) not in {"", "genel"}
        ]
        if not topics:
            return ""
        return Counter(topics).most_common(1)[0][0]

    def _update_conversation_state(
        self,
        query: str,
        answer: str = "",
        context: Optional[List[Dict]] = None,
    ) -> None:
        inferred_scope = infer_query_scope(query)
        inferred_topic = infer_query_topic(query)

        context = context or []
        dominant_scope = self._dominant_scope(context)
        dominant_topic = self._dominant_topic(context)

        next_scope = inferred_scope or dominant_scope or self.conversation_state.get("program_scope", "")
        next_topic = (
            inferred_topic
            if inferred_topic != "genel"
            else dominant_topic or self.conversation_state.get("topic", "")
        )

        if next_scope:
            self.conversation_state["program_scope"] = next_scope
        if next_topic:
            self.conversation_state["topic"] = next_topic

    def _should_carry_context(self, query: str) -> bool:
        if is_scope_clarification_query(query) or is_follow_up_query(query):
            return True

        query_topic = infer_query_topic(query)
        previous_topic = self.conversation_state.get("topic", "")
        if query_topic != "genel":
            return query_topic == previous_topic and bool(previous_topic)
        return False

    def _resolve_model_name(self) -> str:
        try:
            response = requests.get(OLLAMA_TAGS_URL, timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            available = {item.get("name", "") for item in models}
            for preferred in PREFERRED_MODELS:
                if preferred in available:
                    return preferred
        except Exception:
            pass
        return PREFERRED_MODELS[-1]

    def _ollama_kontrol(self):
        try:
            requests.get("http://localhost:11434", timeout=3)
            print(f"Ollama calisiyor (model: {self.model_name})")
        except requests.exceptions.ConnectionError:
            print("Ollama bulunamadi! 'ollama serve' komutunu calistirin.")

    def _vector_store_health(self) -> Dict:
        sqlite_count = sqlite_embedding_count(DB_DIR)
        return {
            "count": sqlite_count,
            "count_error": None,
            "queryable": bool(sqlite_count),
            "probe_error": None,
            "sqlite_count": sqlite_count,
            "collection_names": sqlite_collection_names(DB_DIR),
            "health_source": "sqlite_metadata",
        }

    def _vector_store_count(self) -> int:
        health = self.vector_db_health
        queryable = bool(health.get("queryable"))
        count = health.get("count")
        sqlite_count = health.get("sqlite_count")

        if queryable and isinstance(count, int):
            return count
        if queryable and isinstance(sqlite_count, int):
            return sqlite_count
        if health.get("health_source") in {"fresh_process", "sqlite_metadata"} and isinstance(sqlite_count, int) and sqlite_count > 0:
            return sqlite_count
        if isinstance(count, int) and isinstance(sqlite_count, int) and count == sqlite_count:
            return count
        return 0

    def _vector_search_available(self) -> bool:
        count = self.vector_count
        return count > 0

    def _refresh_vector_db_health(self) -> None:
        self.vector_db_health = self._vector_store_health()
        self.vector_count = self._vector_store_count()

    def _safe_similarity_search(self, query: str, k: int) -> List:
        if not self._vector_search_available():
            return []
        local_results = self.local_vector_index.search(query, k)
        if local_results:
            self.vector_db_health["search_source"] = "local_index"
            return [
                Document(
                    page_content=item.get("content", ""),
                    metadata={
                        "source_url": item.get("source_url", ""),
                        "kategori": item.get("kategori", ""),
                        "program_scope": item.get("program_scope", ""),
                        "topic": item.get("topic", ""),
                        "source_title": item.get("source_title", ""),
                        "years": item.get("years", ""),
                        "chunk_id": item.get("chunk_id", ""),
                    },
                )
                for item in local_results
            ]
        subprocess_results, subprocess_error = subprocess_similarity_search(DB_DIR, query, k)
        if subprocess_results:
            self.vector_db_health["queryable"] = True
            self.vector_db_health["search_source"] = "fresh_process"
            return [
                Document(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("metadata", {}),
                )
                for item in subprocess_results
            ]

        self.vector_db_health["queryable"] = False
        self.vector_db_health["search_source"] = "disabled_after_error"
        self.vector_db_health["fallback_error"] = subprocess_error
        self.vector_count = 0
        print("UYARI: ChromaDB subprocess sorgusu basarisiz, vector arama devre disi.")
        if subprocess_error:
            print(f"DB fallback hatasi: {subprocess_error}")
        return []

    def _specialized_candidates(self, query: str) -> List[Dict]:
        normalized_query = normalize_text(query)
        candidates = []

        if "yaz okulu" in normalized_query:
            for chunk in self.chunks + self.raw_records:
                content = chunk.get("content", "")
                normalized_content = normalize_text(content)
                kategori = chunk.get("kategori", "")
                if asks_yaz_okulu_duration(query):
                    if "yaz okulu" in normalized_content and ("hafta" in normalized_content or kategori == "akademik_takvim"):
                        candidates.append(chunk)
                elif asks_yaz_okulu_start(query):
                    if (
                        "yaz okulu" in normalized_content
                        and (DATE_PATTERN.search(content) or "baslangic" in normalized_content or kategori == "akademik_takvim")
                    ):
                        candidates.append(chunk)
                elif "yaz okulu" in normalized_content:
                    candidates.append(chunk)

        if asks_yaz_staji_schedule(query):
            for chunk in self.chunks + self.raw_records:
                content = chunk.get("content", "")
                normalized_content = normalize_text(content)
                if any(
                    marker in normalized_content
                    for marker in ["yaz staji", "staj donemleri", "yaz okulu sonrasi staj donemi"]
                ):
                    candidates.append(chunk)
                elif DATE_RANGE_PATTERN.search(content) and "staj" in normalized_content:
                    candidates.append(chunk)

        if asks_staj_count(query):
            for chunk in self.chunks + self.raw_records:
                normalized_content = normalize_text(chunk.get("content", ""))
                if "staj" not in normalized_content:
                    continue
                if any(
                    marker in normalized_content
                    for marker in [
                        "bm399",
                        "bm499",
                        "staj i",
                        "staj ii",
                        "5 ve 7 yariyillarda",
                        "5. ve 7. yariyillarda",
                        "iki staj",
                        "25 is gunu staj yapma zorunlulugu",
                    ]
                ):
                    candidates.append(chunk)

        if asks_staj_report_submission(query):
            for chunk in self.chunks + self.raw_records:
                normalized_content = normalize_text(chunk.get("content", ""))
                if "staj" not in normalized_content:
                    continue
                if any(
                    marker in normalized_content
                    for marker in [
                        "staj raporunuzu yazdiktan sonra",
                        "sistemde yuklemek icin son bir tarih bulunmamaktadir",
                        "yaklasik 30 gun sonrasina kadar yukleyebilirsiniz",
                        "staj raporu icin okula imzalatmam gerekiyor mu",
                        "staj bitim tarihinden itibaren en gec 1 bir ay icinde",
                        "duzeltme yapmasi istenen ogrenci en cok 1 bir ay icinde",
                    ]
                ):
                    candidates.append(chunk)

        if asks_makeup_exam_with_missing_internship(query):
            for chunk in self.chunks + self.raw_records:
                normalized_content = normalize_text(chunk.get("content", ""))
                if "tek cift" not in normalized_content:
                    continue
                if any(
                    marker in normalized_content
                    for marker in [
                        "staj dersine hic kayitlanmamis",
                        "staj dersini alip yz notu almis",
                        "tek dersi ve staji kalan ogrenci",
                        "tek dersi ve bahar yariyilindan staji kalan ogrenci",
                    ]
                ):
                    candidates.append(chunk)

        if asks_disciplinary_scholarship_loss(query):
            for chunk in self.chunks + self.raw_records:
                normalized_content = normalize_text(chunk.get("content", ""))
                if "burs" in normalized_content and any(
                    marker in normalized_content for marker in ["disiplin", "ceza", "uzaklastirma"]
                ):
                    candidates.append(chunk)

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            fingerprint = hashlib.md5(candidate.get("content", "").encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique_candidates.append(candidate)

        return sorted(unique_candidates, key=lambda item: self._candidate_score(query, item), reverse=True)[:20]

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        candidate_k = max(k * 4, 12)
        bm25_results: List[Dict] = []
        vector_results: List[Dict] = []
        specialized_candidates = self._specialized_candidates(query)

        for variant in build_query_variants(query):
            bm25_results.extend(self.bm25_search.search(variant, k=candidate_k))
            if self._vector_search_available():
                vector_docs = self._safe_similarity_search(variant, k=candidate_k)
                vector_results.extend(
                    [
                        enrich_chunk_metadata(
                            {
                                "content": doc.page_content,
                                "source_url": doc.metadata.get("source_url", ""),
                                "kategori": doc.metadata.get("kategori", ""),
                                "program_scope": doc.metadata.get("program_scope", ""),
                                "topic": doc.metadata.get("topic", ""),
                                "source_title": doc.metadata.get("source_title", ""),
                                "years": doc.metadata.get("years", ""),
                                "chunk_id": doc.metadata.get("chunk_id", ""),
                            }
                        )
                        for doc in vector_docs
                    ]
                )

        bm25_results.extend(specialized_candidates)

        seen, unique = set(), []
        for result in bm25_results + vector_results:
            content = result.get("content", "").strip()
            if not content:
                continue
            fingerprint = hashlib.md5(content.encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(result)

        unique = self._filter_candidates_by_scope(query, unique)
        unique = self._filter_candidates_by_topic(query, unique)
        scored = sorted(unique, key=lambda item: self._candidate_score(query, item), reverse=True)
        top_candidates = scored[: max(candidate_k, 16)]

        if asks_makeup_exam_with_missing_internship(query) or asks_staj_report_submission(query):
            prioritized = []
            prioritized_seen = set()
            for candidate in specialized_candidates + top_candidates:
                content = candidate.get("content", "").strip()
                if not content:
                    continue
                fingerprint = hashlib.md5(content.encode("utf-8")).hexdigest()
                if fingerprint in prioritized_seen:
                    continue
                prioritized_seen.add(fingerprint)
                prioritized.append(candidate)
            return prioritized[: max(k, 10)]

        if (
            is_short_factual_query(query)
            or asks_staj_timing(query)
            or asks_staj_course_registration(query)
            or asks_staj_count(query)
            or asks_staj_missed_period(query)
            or asks_staj_report_submission(query)
            or asks_yaz_okulu_duration(query)
            or asks_yaz_okulu_start(query)
            or asks_yaz_staji_schedule(query)
        ):
            return top_candidates[:k]
        return self.reranker.rerank(query, top_candidates, k=k)

    def _filter_candidates_by_scope(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not is_program_specific_query(query):
            return candidates

        effective_scope = self._resolve_program_scope(query)
        if not effective_scope:
            return candidates

        scoped_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("program_scope", GENERAL_SCOPE) in {effective_scope, GENERAL_SCOPE}
        ]
        if not scoped_candidates:
            return candidates

        has_program_specific = any(
            candidate.get("program_scope", GENERAL_SCOPE) == effective_scope for candidate in scoped_candidates
        )
        if has_program_specific:
            return scoped_candidates

        general_only = [
            candidate for candidate in scoped_candidates if candidate.get("program_scope", GENERAL_SCOPE) == GENERAL_SCOPE
        ]
        return general_only or scoped_candidates

    def _filter_candidates_by_topic(self, query: str, candidates: List[Dict]) -> List[Dict]:
        effective_topic = self._resolve_topic(query)
        if not effective_topic:
            return candidates

        topical_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("topic", infer_topic(candidate)) in {effective_topic, "genel", ""}
        ]
        if not topical_candidates:
            return candidates

        has_specific_topic = any(
            candidate.get("topic", infer_topic(candidate)) == effective_topic for candidate in topical_candidates
        )
        if has_specific_topic:
            return topical_candidates
        return candidates

    def _candidate_score(self, query: str, candidate: Dict) -> float:
        normalized_query = normalize_text(query)
        query_tokens = set(tokenize(query))
        normalized_content = normalize_text(candidate.get("content", ""))
        content_tokens = set(normalized_content.split())
        source_url = candidate.get("source_url", "").lower()
        kategori = normalize_text(candidate.get("kategori", ""))
        candidate_scope = candidate.get("program_scope", GENERAL_SCOPE)
        effective_scope = self._resolve_program_scope(query)
        candidate_topic = candidate.get("topic", infer_topic(candidate))
        effective_topic = self._resolve_topic(query)
        requested_years = extract_years(query)

        score = float(len(query_tokens & content_tokens))

        if "staj" in query_tokens and "staj" in content_tokens:
            score += 5
        if "staj" in query_tokens and kategori == "staj":
            score += 6
        if "staj" in normalized_query and "/staj" in source_url:
            score += 2
        if is_program_specific_query(query):
            if effective_scope and candidate_scope == effective_scope:
                score += 14
            elif candidate_scope == GENERAL_SCOPE:
                score += 3
            else:
                score -= 20
        if effective_topic:
            if candidate_topic == effective_topic:
                score += 12
            elif candidate_topic not in {"", "genel"}:
                score -= 10
        if asks_graduation_with_incomplete_internship(query) or asks_graduation_requirements(query):
            if _is_graduation_whitelisted_source(candidate):
                score += 20
            else:
                score -= 24
            if candidate_topic == "cap_yandal":
                score -= 28
            if any(marker in normalize_text(candidate.get("source_title", "")) for marker in ["cap", "yandal"]):
                score -= 24
        if asks_external_summer_school_course(query) or asks_yaz_okulu_attendance(query):
            if candidate_topic == "staj":
                score -= 18
        if asks_post_upload_graduation(query):
            if "mezun durumundaysaniz" in normalized_content and "staj komisyonuna" in normalized_content:
                score += 26
            if "mail" in normalized_content or "e posta" in normalized_content:
                score += 14
        if asks_staj_report_submission(query) and candidate_topic in {"cap_yandal", "mezuniyet"}:
            score -= 22
        if asks_staj_timing(query):
            if "yariyil" in normalized_content:
                score += 6
            if "bm399" in normalized_content or "bm499" in normalized_content:
                score += 8
            if "yaz" in normalized_content:
                score += 4
        if asks_staj_count(query):
            if "bm399" in normalized_content or "bm499" in normalized_content:
                score += 12
            if "staj i" in normalized_content or "staj ii" in normalized_content:
                score += 10
            if "5 yariyil" in normalized_content or "7 yariyil" in normalized_content:
                score += 8
            if "25 is gunu" in normalized_content:
                score += 8
            if "tek cift" in normalized_content or "rapor" in normalized_content:
                score -= 8
        if asks_staj_course_registration(query):
            if "obs" in normalized_content:
                score += 8
            if "takip eden yariyilda" in normalized_content:
                score += 10
            if "tekrar almaniza gerek yoktur" in normalized_content:
                score += 4
        if asks_staj_missed_period(query):
            if "takip eden akademik yillar" in normalized_content:
                score += 12
            if "kesintisiz 40 is gunu" in normalized_content:
                score += 8
            if "stajini erteleyen" in normalized_content or "staj yapma hakki kazanamayan" in normalized_content:
                score += 8
            if "rapor" in normalized_content or "sbs" in normalized_content:
                score -= 6
        if asks_staj_report_submission(query):
            if "staj rapor" in normalized_content or "staj defter" in normalized_content:
                score += 10
            if "sbs" in normalized_content or "teslim" in normalized_content or "yukle" in normalized_content:
                score += 8
            if "sistemde yuklemek icin son bir tarih bulunmamaktadir" in normalized_content:
                score += 20
            if "yaklasik 30 gun" in normalized_content:
                score += 18
            if "staj bitim tarihinden itibaren en gec 1 bir ay" in normalized_content:
                score += 14
            if "duzeltme yapmasi istenen ogrenci" in normalized_content:
                score += 10
            if "is gunu" in normalized_content and not any(marker in normalized_content for marker in ["rapor", "defter", "teslim", "yukle"]):
                score -= 12
        if asks_makeup_exam_with_missing_internship(query):
            if "tek cift" in normalized_content:
                score += 14
            if "tek dersi ve staji kalan ogrenci" in normalized_content:
                score += 16
            if "tek dersi ve bahar yariyilindan staji kalan ogrenci" in normalized_content:
                score += 16
            if "staj dersine hic kayitlanmamis" in normalized_content:
                score += 18
            if "yz" in normalized_content or "yetersiz" in normalized_content:
                score += 10
            if "sadece dersten girebilir" in normalized_content:
                score += 12
        if asks_disciplinary_scholarship_loss(query):
            if "burs" in normalized_content:
                score += 8
            if "disiplin" in normalized_content or "ceza" in normalized_content:
                score += 8
            if any(marker in normalized_content for marker in ["kayb", "kesil", "iptal", "devam"]):
                score += 10

        if is_short_factual_query(query):
            if NUMERIC_UNIT_PATTERN.search(candidate.get("content", "")):
                score += 8
            if "is gunu" in normalized_content or "isgunu" in normalized_content:
                score += 8
            if "sure" in normalized_content or "suresi" in normalized_content:
                score += 4
            if "zorunlulugu" in normalized_content:
                score += 6
            if "arasında" in candidate.get("content", "") or "arasinda" in normalized_content:
                score += 4
            if "kac gun" in normalized_query and ("gün" in candidate.get("content", "") or "iş günü" in candidate.get("content", "")):
                score += 4
            if any(
                marker in normalized_content
                for marker in ["birlestir", "uzat", "maksimum", "mezun durumunda", "degerlendirilmesi", "rapor"]
            ):
                score -= 10

        if "yaz okulu" in normalized_query:
            if "yaz okulu" in normalized_content:
                score += 12
            if kategori in {"akademik_takvim", "yaz_okulu"}:
                score += 10
            if "akademik takvim" in normalized_content or "takvim" in normalized_content:
                score += 6
            if "hafta" in normalized_content:
                score += 4
            if "yaz okulunda ogretim suresi" in normalized_content:
                score += 10
            if asks_yaz_okulu_duration(query) and WEEK_PATTERN.search(candidate.get("content", "")):
                score += 12
            if asks_yaz_okulu_duration(query) and "bes en fazla yedi hafta" in normalized_content:
                score += 16
            if asks_yaz_okulu_start(query):
                if DATE_PATTERN.search(candidate.get("content", "")):
                    score += 10
                if "baslangic" in normalized_content:
                    score += 10
                if chunk := candidate.get("content", ""):
                    if SUMMER_SCHOOL_START_PATTERN.search(chunk):
                        score += 18
            if "yaz staji" in normalized_content and "yaz okulu" not in normalized_content:
                score -= 12
        if asks_yaz_okulu_attendance(query):
            if "yaz okulu" in normalized_content and "devam" in normalized_content:
                score += 22
            elif "yaz okulu" in normalized_content:
                score -= 8
            if "devam zorunlulugu aranmaz" in normalized_content:
                score += 24
            if "devam kosulunu yerine getirilen bir dersin tekrari" in normalized_content:
                score += 18
        if asks_external_summer_school_course(query):
            if any(
                marker in normalized_content
                for marker in ["diger universitelerin", "diger universitelerden", "universitemiz disinda", "esdeger"]
            ):
                score += 24
            if "bolum anabilim dali baskanliklarinin uygun gormesi" in normalized_content:
                score += 16
        if asks_staj_insurance(query):
            if "sigorta" in normalized_content:
                score += 24
            if "okul tarafindan sigorta yapilmiyor" in normalized_content or "okulun sigorta yapmasi" in normalized_content:
                score += 18
        if asks_graduation_with_incomplete_internship(query):
            if any(
                marker in normalized_content
                for marker in ["stajini tamamlamayan ogrenci", "mezun olmaya hak kazanir", "butun calismalari tamamlamis"]
            ):
                score += 26
            if any(
                marker in normalized_content
                for marker in ["staj yapacaginiz firmada", "elektrik elektronik muhendisi", "muhendisi olmasi gerekir"]
            ):
                score -= 30
        if asks_graduation_requirements(query):
            if any(
                marker in normalized_content
                for marker in ["mezun olmaya hak kazanir", "butun calismalari tamamlamis", "diploma", "mezuniyet"]
            ):
                score += 20
            if any(
                marker in normalized_content
                for marker in ["staj yapacaginiz firmada", "muhendisi olmasi gerekir", "sbs"]
            ):
                score -= 24

        if asks_yaz_staji_schedule(query):
            if "yaz staji" in normalized_content:
                score += 12
            if "staj donemleri" in normalized_content or "staj tarihleri" in normalized_content:
                score += 12
            if DATE_PATTERN.search(candidate.get("content", "")):
                score += 8
            if DATE_RANGE_PATTERN.search(candidate.get("content", "")):
                score += 12
            if "yaz okulu sonrasi staj donemi" in normalized_content:
                score += 8
            if kategori == "staj":
                score += 6

        if requested_years and any(str(year) in candidate.get("content", "") for year in requested_years):
            score += 6

        source_title = normalize_text(candidate.get("source_title") or infer_source_title(candidate))
        if any(marker in source_title for marker in ["fakulte bolum", "bolum", "ogrenci isleri"]):
            score -= 5

        return score

    def _memory_as_text(self, query: Optional[str] = None) -> str:
        if query and not self._should_carry_context(query):
            return "Yok"

        messages = self.message_history.messages[-MAX_MEMORY_TURNS * 2 :]
        if not messages:
            return "Yok"

        lines = []
        for message in messages:
            if isinstance(message, HumanMessage):
                prefix = "Öğrenci"
            else:
                prefix = "Asistan"
            lines.append(f"{prefix}: {message.content}")
        return "\n".join(lines)

    def _last_user_query(self) -> Optional[str]:
        for message in reversed(self.message_history.messages):
            if isinstance(message, HumanMessage) and message.content.strip():
                return message.content.strip()
        return None

    def _build_search_query(self, query: str) -> str:
        previous_user_query = self._last_user_query()
        if not self._should_carry_context(query):
            return query

        scope = self._resolve_program_scope(query)
        topic = self._resolve_topic(query)
        additions = []
        normalized_query = normalize_text(query)

        if is_program_specific_query(query) and scope and not infer_query_scope(query):
            additions.append(self._scope_label(scope))
        topic_label = self._topic_label(topic)
        if topic and topic != "genel" and normalize_text(topic_label) not in normalized_query:
            additions.append(topic_label)

        rewritten_query = query
        if is_scope_clarification_query(query):
            if previous_user_query:
                additions.append(previous_user_query)
            rewritten_query = query.strip()

        if additions:
            rewritten_query = f"{rewritten_query.strip()} {' '.join(additions)}".strip()
        return rewritten_query

    def _save_to_memory(self, query: str, answer: str) -> None:
        self.message_history.add_message(HumanMessage(content=query))
        self.message_history.add_message(AIMessage(content=answer))
        max_messages = MAX_MEMORY_TURNS * 2
        if len(self.message_history.messages) > max_messages:
            self.message_history.messages = self.message_history.messages[-max_messages:]
        self._update_conversation_state(query, answer, self.last_answer_context)

    def clear_memory(self) -> None:
        self.message_history.clear()
        self.conversation_state = {
            "program_scope": self.program_scope or "",
            "topic": "",
        }

    def _cleanup_response(self, text: str) -> str:
        text = repair_text_encoding(text)
        normalized_text = normalize_text(text)
        fallback_normalized = [normalize_text(pattern) for pattern in FALLBACK_PATTERNS]
        has_other_content = normalized_text not in fallback_normalized

        if has_other_content:
            for pattern in FALLBACK_PATTERNS:
                if pattern in text:
                    text = text.split(pattern, 1)[0].strip()

        cleaned_lines = []
        fallback_present = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized_line = normalize_text(line)
            if normalized_line in fallback_normalized:
                fallback_present = True
                continue
            if line.startswith("Saygılarımla") or line.startswith("Saygilarimla"):
                continue
            if line.startswith("Öğrenci İşleri") or line.startswith("Ogrenci Isleri"):
                continue
            if line.startswith("[") and "duzce universitesi" in normalized_line:
                continue
            if normalized_line.startswith("bu bilgiler resmi belgeler bolumunden alinmistir"):
                continue
            if normalized_line.startswith("lutfen her zaman en guncel ve resmi kaynaklardan bilgi almayi unutmayin"):
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()
        if cleaned:
            return cleaned
        if fallback_present:
            return NO_ANSWER_TEXT
        return text.strip()

    def _finalize_answer(self, text: str) -> str:
        cleaned = self._cleanup_response(text)
        cleaned = repair_text_encoding(cleaned)
        return cleaned

    def _answer_should_show_sources(self, answer: str) -> bool:
        normalized_answer = normalize_text(answer)
        no_source_markers = [
            normalize_text(NO_ANSWER_TEXT),
            normalize_text(SCOPE_CLARIFICATION_TEXT),
            "resmi belgelerde acik bir bilgi bulunmamaktadir",
            "resmi belgelerde bilgiye ulasilamadi",
            "resmi belgelerde bilgiye ulasilamadim",
            "dogrudan resmi bir kaynak bulamadim",
            "net cevap veremiyorum",
            "kesin cevap veremiyorum",
        ]
        return not any(marker and marker in normalized_answer for marker in no_source_markers)

    def _source_title(self, result: Dict) -> str:
        url = result.get("source_url", "")
        kategori = result.get("kategori", "Genel")
        normalized_url = url.lower()
        if "bm.mf.duzce.edu.tr/sayfa/878b" in normalized_url:
            return "Bilgisayar Mühendisliği - Staj SSS"
        if "bm.mf.duzce.edu.tr/sayfa/4a82" in normalized_url:
            return "Bilgisayar Mühendisliği - Staj"
        if "bm.mf.duzce.edu.tr/sayfa/17ac" in normalized_url:
            return "Bilgisayar Mühendisliği - Yaz Okulu"
        if "mf.duzce.edu.tr/sayfa/967a" in normalized_url:
            return "Mühendislik Fakültesi - Yaz Stajı"
        if "akademik-takvim" in normalized_url or result.get("kategori") == "akademik_takvim":
            return "Akademik Takvim"
        if "ogrenciisleri.duzce.edu.tr" in normalized_url:
            return "Öğrenci İşleri"
        return kategori.replace("_", " ").title()

    def _important_terms(self, text: str) -> set:
        return {
            token
            for token in tokenize(text)
            if len(token) >= 3 and token not in SOURCE_STOPWORDS and not token.isdigit()
        }

    def _source_support_score(self, query: str, answer: str, result: Dict) -> float:
        content = result.get("content", "")
        normalized_content = normalize_text(content)
        content_terms = set(normalized_content.split())
        query_terms = self._important_terms(query)
        answer_terms = self._important_terms(answer)
        score = 0.0

        score += 3.0 * len(query_terms & content_terms)
        score += 2.0 * len(answer_terms & content_terms)

        for value in re.findall(r"\b\d+(?:[./-]\d+)*(?:\s*-\s*\d+(?:[./-]\d+)*)?\b", answer):
            if value and value in content:
                score += 8.0

        effective_scope = self._resolve_program_scope(query)
        candidate_scope = result.get("program_scope", infer_chunk_scope(result))
        if effective_scope and candidate_scope == effective_scope:
            score += 10.0
        elif candidate_scope == GENERAL_SCOPE:
            score += 2.0

        if asks_staj_report_submission(query) and any(
            marker in normalized_content
            for marker in [
                "staj raporunuzu",
                "sbs ye yukle",
                "yaklasik 30 gun",
                "staj defter",
                "duzeltme yapmasi istenen ogrenci",
            ]
        ):
            score += 25.0
        if asks_disciplinary_scholarship_loss(query) and "burs" in normalized_content:
            score += 8.0
        if asks_yaz_okulu_start(query) and "yaz okulu" in normalized_content and DATE_PATTERN.search(content):
            score += 18.0
        if asks_yaz_staji_schedule(query) and "staj" in normalized_content and DATE_RANGE_PATTERN.search(content):
            score += 18.0
        if asks_staj_duration(query) and "staj" in normalized_content and "is gunu" in normalized_content:
            score += 18.0
        if asks_yaz_okulu_attendance(query):
            if "yaz okulu" in normalized_content and "devam" in normalized_content:
                score += 24.0
            elif "yaz okulu" in normalized_content:
                score -= 10.0
        if asks_external_summer_school_course(query):
            if any(
                marker in normalized_content
                for marker in ["diger universitelerden", "diger universite", "universitemiz disinda", "esdeger"]
            ):
                score += 24.0
        if asks_post_upload_graduation(query):
            if "mezun durumundaysaniz" in normalized_content and "staj komisyonuna" in normalized_content:
                score += 26.0
            if "mail" in normalized_content or "e posta" in normalized_content:
                score += 14.0
        if asks_staj_insurance(query):
            if "sigorta" in normalized_content:
                score += 24.0
        if asks_graduation_with_incomplete_internship(query) or asks_graduation_requirements(query):
            if _is_graduation_whitelisted_source(result):
                score += 20.0
            else:
                score -= 24.0
            if any(
                marker in normalized_content
                for marker in ["mezun olmaya hak kazanir", "butun calismalari tamamlamis", "diploma", "mezuniyet"]
            ):
                score += 24.0
            if any(
                marker in normalized_content
                for marker in ["staj yapacaginiz firmada", "elektrik elektronik muhendisi", "muhendisi olmasi gerekir"]
            ):
                score -= 24.0

        source_title = normalize_text(result.get("source_title") or self._source_title(result))
        if any(marker in source_title for marker in ["fakulte bolum", "bolum", "ogrenci isleri"]):
            score -= 6.0
        if asks_yaz_okulu_attendance(query) and "merkezi mevzuat" in source_title:
            score -= 8.0
        if (asks_graduation_with_incomplete_internship(query) or asks_graduation_requirements(query)) and any(
            marker in source_title for marker in ["cap", "yandal"]
        ):
            score -= 30.0

        return score

    def _format_sources(self, results: List[Dict], answer: str, query: str = "", limit: int = 3) -> List[Dict]:
        if not self._answer_should_show_sources(answer):
            return []
        best_by_url = {}
        query = query or self._last_user_query() or ""
        for result in results:
            url = result.get("source_url", "").strip()
            if not url:
                continue
            source_key = url.lower().rstrip("/")
            score = self._source_support_score(query, answer, result)
            current = best_by_url.get(source_key)
            if current and current["score"] >= score:
                continue
            normalized_result = dict(result)
            normalized_result["source_url"] = url
            best_by_url[source_key] = {"result": normalized_result, "score": score}

        ranked = sorted(best_by_url.values(), key=lambda item: item["score"], reverse=True)
        answer_dates = DATE_PATTERN.findall(answer)
        has_date_support = bool(answer_dates) and any(
            any(date in item["result"].get("content", "") for date in answer_dates) for item in ranked
        )
        effective_scope = self._resolve_program_scope(query)
        scoped_source_count = sum(
            1
            for item in ranked
            if effective_scope and item["result"].get("program_scope", infer_chunk_scope(item["result"])) == effective_scope
        )
        answer_mentions_general_rule = "genel" in normalize_text(answer)
        sources = []
        for item in ranked:
            if item["score"] <= 0:
                continue
            result = item["result"]
            if self._is_noisy_source_for_query(query, result):
                continue
            if has_date_support and not any(date in result.get("content", "") for date in answer_dates):
                continue
            candidate_scope = result.get("program_scope", infer_chunk_scope(result))
            if (
                effective_scope
                and scoped_source_count >= 2
                and candidate_scope != effective_scope
                and not answer_mentions_general_rule
            ):
                continue
            sources.append(
                {
                    "kategori": result.get("kategori", "Genel"),
                    "baslik": result.get("source_title") or self._source_title(result),
                    "url": result.get("source_url", ""),
                }
            )
            if len(sources) >= limit:
                break
        return sources

    def _attach_source_summary(self, answer: str, sources: List[Dict]) -> str:
        if not sources:
            return answer
        if "Dayanak:" in answer:
            return answer

        refs = []
        for index, source in enumerate(sources, start=1):
            title = source.get("baslik", "Kaynak")
            refs.append(f"[Kaynak {index}] {title}")
        return f"{answer}\n\nDayanak: " + "; ".join(refs)

    def _is_noisy_source_for_query(self, query: str, result: Dict) -> bool:
        title = normalize_text(result.get("source_title") or self._source_title(result))
        url = (result.get("source_url") or "").lower()

        if any(
            marker in title
            for marker in [
                "yonetim dekanlik",
                "ogrenci panolari",
                "ucretli staj",
                "komisyonu tarafindan ogrenci panolari",
                "yaz okulu egitimi",
            ]
        ):
            return True
        if asks_external_summer_school_course(query) or asks_yaz_okulu_attendance(query):
            if "cap ve yandal" in title:
                return True
            if "yaz okulu egitimi" in title:
                return True
        if asks_staj_insurance(query) or asks_staj_report_submission(query) or asks_post_upload_graduation(query):
            if any(marker in title for marker in ["yonetim dekanlik", "ucretli staj", "ogrenci panolari"]):
                return True
        if asks_graduation_with_incomplete_internship(query) or asks_graduation_requirements(query):
            if not _is_graduation_whitelisted_source(result) and "akademik takvim" not in title:
                return True
        if "/duyuru/" in url and not asks_yaz_staji_schedule(query):
            return True
        return False

    def _extract_staj_timing_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_staj_timing(query):
            return None

        has_course_semesters = False
        has_summer_timing = False
        has_preparation_semesters = False

        for chunk in context:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "staj" not in normalized_content:
                continue
            if "bm399" in normalized_content and "bm499" in normalized_content:
                has_course_semesters = True
            if (
                "4 yariyil" in normalized_content
                and "6 yariyil" in normalized_content
                and any(marker in normalized_content for marker in ["izleyen yaz", "yaz tatilinde", "yaz aylarinda"])
            ) or "bahar yariyilinin bitimi ve guz yariyilinin baslangici arasinda" in normalized_content:
                has_summer_timing = True
            if "yaz doneminde stajini yapan" in normalized_content or "yaz staji" in normalized_content:
                has_summer_timing = True
            if (
                "4 yariyilin bahar doneminde" in normalized_content
                and "6 yariyilin bahar doneminde" in normalized_content
                and "staj yeri aramaya" in normalized_content
            ):
                has_preparation_semesters = True

        if has_summer_timing and has_course_semesters and has_preparation_semesters:
            return (
                "Sayın öğrencimiz,\n"
                "Bilgisayar Mühendisliği için zorunlu stajlar yaz döneminde yapılır. "
                "Kaynakta staj yeri aramaya en geç 4. yarıyılın ve 6. yarıyılın Bahar döneminde başlanabileceği belirtilmektedir. "
                "Staj dersleri ise takip eden 5. ve 7. yarıyıllarda BM399 ve BM499 olarak alınır."
            )

        if has_summer_timing and has_course_semesters:
            return (
                "Sayın öğrencimiz,\n"
                "Zorunlu stajlar 4. ve 6. yarıyılları izleyen yaz dönemlerinde yapılır. "
                "Staj dersleri ise takip eden 5. ve 7. yarıyıllarda BM399 ve BM499 olarak alınır."
            )

        if has_summer_timing:
            return (
                "Sayın öğrencimiz,\n"
                "Zorunlu stajlar 4. ve 6. yarıyılları izleyen yaz dönemlerinde yapılır."
            )

        if has_course_semesters:
            return (
                "Sayın öğrencimiz,\n"
                "Kaynağa göre staj dersleri 5. ve 7. yarıyıllarda BM399 ve BM499 olarak yürütülmektedir."
            )

        return None

    def _extract_staj_count_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_staj_count(query):
            return None

        has_first_internship = False
        has_second_internship = False
        has_both_courses = False
        internship_days = None

        for chunk in context:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "staj" not in normalized_content:
                continue

            if "bm399" in normalized_content or "staj i" in normalized_content or "staj 1" in normalized_content:
                has_first_internship = True
            if "bm499" in normalized_content or "staj ii" in normalized_content or "staj 2" in normalized_content:
                has_second_internship = True
            if "bm399" in normalized_content and "bm499" in normalized_content:
                has_both_courses = True

            day_match = WORKDAY_NUMBER_PATTERN.search(content)
            if day_match and internship_days is None:
                internship_days = int(day_match.group(1))

        if has_first_internship and has_second_internship:
            has_both_courses = True

        if not has_both_courses:
            return None

        scope = self._resolve_program_scope(query)
        scope_label = self._scope_label(scope) if scope else "ilgili bölüm"
        if internship_days:
            return (
                "Sayın öğrencimiz,\n"
                f"{scope_label} bölümünde 2 zorunlu staj bulunmaktadır. "
                f"Bunlar Staj I (BM399) ve Staj II'dir (BM499). "
                f"Her biri {internship_days} iş günüdür."
            )

        return (
            "Sayın öğrencimiz,\n"
            f"{scope_label} bölümünde 2 zorunlu staj bulunmaktadır. "
            "Bunlar Staj I (BM399) ve Staj II'dir (BM499)."
        )

    def _extract_staj_course_registration_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_staj_course_registration(query):
            return None

        following_term_content = None
        no_retake_content = None

        for chunk in context:
            normalized_content = normalize_text(chunk.get("content", ""))
            if "staj" not in normalized_content:
                continue
            if "takip eden yariyilda" in normalized_content and "obs" in normalized_content:
                following_term_content = normalized_content
            if "tekrar almaniza gerek yoktur" in normalized_content or "bir kere daha almaniza gerek yoktur" in normalized_content:
                no_retake_content = normalized_content

        if not following_term_content:
            return None

        answer = (
            "Sayın öğrencimiz,\n"
            "Evet. Kaynağa göre yaz döneminde yapılan staj için stajı takip eden yarıyılda ilgili staj dersinin OBS'de alınması gerekir."
        )
        if "2. sınıf" in query and "3. sınıf" in query:
            answer += " Bu nedenle 2. sınıfın yazında yapılan Staj I, 3. sınıfın güz döneminde ilgili staj dersi alınarak saydırılabilir."
        if no_retake_content:
            answer += " Dersi daha önce OBS'de aldıysanız tekrar almanız gerekmez."
        return answer

    def _extract_staj_missed_period_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_staj_missed_period(query):
            return None

        can_take_later = False
        can_merge_after_sixth = False

        for chunk in context:
            normalized_content = normalize_text(chunk.get("content", ""))
            if "staj" not in normalized_content:
                continue
            if "takip eden akademik yillarin staj donemlerinde" in normalized_content:
                can_take_later = True
            if (
                "ilk staj doneminde staj yapma hakki kazanamayan" in normalized_content
                or "stajini erteleyen" in normalized_content
            ) and "6 yariyildan sonra" in normalized_content and "kesintisiz 40 is gunu" in normalized_content:
                can_merge_after_sixth = True

        if can_take_later and can_merge_after_sixth:
            return (
                "Sayın öğrencimiz,\n"
                "Döneminde staj yapamazsanız stajınızı takip eden akademik yılların staj dönemlerinde yapabilirsiniz. "
                "Ayrıca ilgili yönergede, ilk staj dönemini yapamayan veya erteleyen öğrencilerin yeterlilik şartlarını sağladıklarında 6. yarıyıldan sonra kesintisiz 40 iş günü staj yapabileceği de belirtilmektedir."
            )

        if can_take_later:
            return (
                "Sayın öğrencimiz,\n"
                "Döneminde staj yapamazsanız stajınızı takip eden akademik yılların staj dönemlerinde yapabilirsiniz."
            )

        if can_merge_after_sixth:
            return (
                "Sayın öğrencimiz,\n"
                "Döneminde staj yapamadığınız durumda stajınızı sonraki uygun staj döneminde tamamlamanız gerekir. "
                "Ayrıca ilgili yönergede, ilk staj dönemini yapamayan veya stajını erteleyen öğrencilerin yeterlilik şartlarını sağladıklarında 6. yarıyıldan sonra kesintisiz 40 iş günü staj yapabileceği de belirtilmektedir."
            )

        return None

    def _extract_makeup_exam_with_missing_internship_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_makeup_exam_with_missing_internship(query):
            return None

        has_never_enrolled_rule = False
        has_yz_exception = False
        has_course_only_rule = False

        for chunk in context:
            normalized_content = normalize_text(chunk.get("content", ""))
            if "tek cift" not in normalized_content:
                continue
            if "staj dersine hic kayitlanmamis" in normalized_content:
                has_never_enrolled_rule = True
            if "staj dersini alip yz notu almis ise" in normalized_content or " yz " in f" {normalized_content} ":
                has_yz_exception = True
            if "tek dersi ve bahar yariyilindan staji kalan ogrenci tek cift sinavina sadece dersten girebilir" in normalized_content:
                has_course_only_rule = True

        if not any([has_never_enrolled_rule, has_yz_exception, has_course_only_rule]):
            return None

        parts = []
        if has_never_enrolled_rule:
            parts.append("Staj dersine hic kayitlanmadiysaniz tek/cift ders sinavina girme hakkiniz yoktur.")
        if has_yz_exception:
            parts.append("Ancak staj dersini daha once alip YZ notu aldiysaniz tek/cift ders sinavina basvurabilirsiniz.")
        if has_course_only_rule:
            parts.append("Tek dersi ve bahar yariyilindan staji kalan ogrenci tek/cift sinavina sadece dersten girebilir.")

        return "Sayın öğrencimiz,\n" + " ".join(parts)

    def _extract_staj_report_submission_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_staj_report_submission(query):
            return None

        has_bm_submission_rule = False
        has_bm_no_fixed_deadline = False
        has_bm_approx_30_days = False
        has_bm_graduation_note = False
        has_general_one_month_rule = False
        has_general_correction_rule = False

        for chunk in context:
            normalized_content = normalize_text(chunk.get("content", ""))
            if "staj" not in normalized_content:
                continue
            if "staj raporunuzu yazdiktan sonra" in normalized_content and "sbs ye yuklemeniz gerekmektedir" in normalized_content:
                has_bm_submission_rule = True
            if "sistemde yuklemek icin son bir tarih bulunmamaktadir" in normalized_content:
                has_bm_no_fixed_deadline = True
            if "yaklasik 30 gun sonrasina kadar yukleyebilirsiniz" in normalized_content or "yaklasik 30 gun sonra" in normalized_content:
                has_bm_approx_30_days = True
            if "mezun durumundaysaniz" in normalized_content and "staj komisyonuna mail" in normalized_content:
                has_bm_graduation_note = True
            if "staj bitim tarihinden itibaren en gec 1 bir ay icinde" in normalized_content:
                has_general_one_month_rule = True
            if "duzeltme yapmasi istenen ogrenci" in normalized_content and "aksi takdirde staj reddedilmis sayilir" in normalized_content:
                has_general_correction_rule = True

        if has_bm_submission_rule or has_bm_no_fixed_deadline or has_bm_approx_30_days:
            parts = [
                "Bilgisayar Mühendisliği Staj SSS kaynağına göre staj raporu, imza ve kaşe işlemleri tamamlandıktan sonra taranarak SBS'ye yüklenir."
            ]
            if has_bm_no_fixed_deadline:
                parts.append("Aynı kaynakta, sistem yüklemesi için sabit bir son tarih bulunmadığı belirtilmektedir.")
            if has_bm_approx_30_days:
                parts.append("Yaz stajı için raporun, yeni güz dönemi başladıktan sonra yaklaşık 30 gün içinde yüklenebileceği ve değerlendirmenin staj komisyonu toplandıktan sonra yapılacağı ifade edilmektedir.")
            if has_bm_graduation_note:
                parts.append("Mezun durumundaysanız ve stajlar dışında dersiniz yoksa, raporu yükledikten sonra bölüm staj komisyonuna e-posta ile değerlendirme talebi iletmeniz gerektiği belirtilmiştir.")
            if has_general_correction_rule:
                parts.append("Genel staj yönergesinde, komisyon düzeltme isterse bu düzeltmenin en fazla 1 ay içinde yapılması gerektiği; aksi durumda stajın reddedilmiş sayılacağı belirtilmektedir.")
            parts.append("Kaynaklarda geç teslim için otomatik burs kesintisi veya ayrı bir disiplin cezası şeklinde açık bir yaptırım yer almamaktadır.")
            return "Sayın öğrencimiz,\n" + " ".join(parts)

        if has_general_one_month_rule or has_general_correction_rule:
            parts = []
            if has_general_one_month_rule:
                parts.append("Genel staj yönergesine göre staj defterleri staj bitim tarihinden itibaren en geç 1 ay içinde ilgili Bölüm Başkanlığına teslim edilmelidir; bu sürenin uzatılması komisyon kararına bağlıdır.")
            if has_general_correction_rule:
                parts.append("Komisyon düzeltme isterse öğrenci en çok 1 ay içinde düzeltmeyi yapmakla yükümlüdür; aksi halde staj reddedilmiş sayılır.")
            return "Sayın öğrencimiz,\n" + " ".join(parts)

        return None

    def _extract_disciplinary_scholarship_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_disciplinary_scholarship_loss(query):
            return None

        has_direct_rule = False
        for chunk in context:
            normalized_content = normalize_text(chunk.get("content", ""))
            if not ("burs" in normalized_content and any(marker in normalized_content for marker in ["disiplin", "ceza", "uzaklastirma"])):
                continue
            if any(marker in normalized_content for marker in ["bursunu kaybeder", "burs kesilir", "bursu kesilir", "burs iptal", "burs devam"]):
                has_direct_rule = True
                break

        if has_direct_rule:
            return None

        return (
            "Sayın öğrencimiz,\n"
            "Resmi belgelerde disiplin cezası alan öğrencinin bursunu kaybedip kaybetmeyeceğine dair açık bir hüküm bulamadım. "
            "Bu nedenle kesin cevap veremiyorum; burs türüne göre ilgili birimden doğrulama alınmalıdır."
        )

    def _extract_staj_duration_confirmation_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        confirmed_days = get_confirmed_day_count(query)
        if confirmed_days is None:
            return None

        best_match = None
        best_score = float("-inf")
        has_min_20_rule = False

        for chunk in context:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "staj" not in normalized_content:
                continue

            if "20 is gununden az olmamak uzere" in normalized_content or "bir staj donemi icin staj suresi kesintisiz en az 20 is gunudur" in normalized_content:
                has_min_20_rule = True

            for sentence in SENTENCE_SPLIT_PATTERN.split(content):
                sentence = sentence.strip(" -\t")
                if not sentence:
                    continue

                normalized_sentence = normalize_text(sentence)
                if "staj" not in normalized_sentence:
                    continue

                range_match = WORKDAY_RANGE_PATTERN.search(sentence)
                number_match = WORKDAY_NUMBER_PATTERN.search(sentence)
                if not range_match and not number_match:
                    continue

                sentence_score = 0.0
                if chunk.get("kategori") == "staj":
                    sentence_score += 6
                if "bilgisayar muhendisligi" in normalized_sentence:
                    sentence_score += 12
                if "zorunlulugu" in normalized_sentence or "staj suresi" in normalized_sentence:
                    sentence_score += 10
                if any(
                    marker in normalized_sentence
                    for marker in ["birlestir", "uzat", "maksimum", "mezun", "degerlendirilmesi", "sigorta", "rapor"]
                ):
                    sentence_score -= 12

                if sentence_score > best_score:
                    best_score = sentence_score
                    best_match = {
                        "normalized": normalized_sentence,
                        "range_match": range_match,
                        "number_match": number_match,
                    }

        if not best_match:
            return None

        if best_match["range_match"]:
            start = int(best_match["range_match"].group(1))
            end = int(best_match["range_match"].group(2))
            if start <= confirmed_days <= end:
                return (
                    "Sayın öğrencimiz,\n"
                    f"Kesin olarak {confirmed_days} iş günü denilemez. Kaynağa göre staj süresi ilgili akademik birimin yönergesine bağlı olarak {start}-{end} iş günü arasındadır."
                )
            return (
                "Sayın öğrencimiz,\n"
                f"Hayır. Kaynağa göre staj süresi ilgili akademik birimin yönergesine bağlı olarak {start}-{end} iş günü arasındadır."
            )

        actual_days = int(best_match["number_match"].group(1))
        if actual_days == confirmed_days:
            answer = f"Sayın öğrencimiz,\nEvet. Staj süresi {actual_days} iş günüdür."
        else:
            answer = f"Sayın öğrencimiz,\nHayır. Staj süresi {actual_days} iş günüdür."

        if actual_days == 25 and has_min_20_rule:
            answer += " Ancak bazı merkezi mevzuat metinlerinde bir staj döneminin en az 20 iş günü olabileceği de belirtilmektedir."
        return answer

    def _extract_yaz_okulu_attendance_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_yaz_okulu_attendance(query):
            return None

        exception_sentence = ""
        general_sentence = ""
        for chunk in context + self.chunks + self.raw_records:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "yaz okulu" not in normalized_content or "devam" not in normalized_content:
                continue
            for sentence in SENTENCE_SPLIT_PATTERN.split(content):
                normalized_sentence = normalize_text(sentence)
                if "yaz okulu" not in normalized_sentence or "devam" not in normalized_sentence:
                    continue
                if "devam zorunlulugu aranmaz" in normalized_sentence:
                    exception_sentence = sentence.strip()
                elif any(
                    marker in normalized_sentence
                    for marker in ["basari durumlari devam", "devam ara sinavlar", "devam kosulu"]
                ):
                    general_sentence = sentence.strip()
            if exception_sentence and general_sentence:
                break

        if not exception_sentence and not general_sentence:
            return None

        parts = []
        if general_sentence:
            parts.append(
                "Yaz Okulu Uygulama Esaslarina gore yaz okulunda basari degerlendirmesinde devam durumu dikkate alinir."
            )
        if exception_sentence:
            parts.append(
                "Ancak universite disinda yaz okulunda ders alip esdeger ders icin devam kosulunu sagladigini belgeleyen ogrenciler icin devam zorunlulugu aranmaz."
            )
        return "Sayin ogrencimiz,\n" + " ".join(parts)

    def _extract_staj_insurance_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_staj_insurance(query):
            return None

        yes_sentence = ""
        no_sentence = ""
        for chunk in context + self.chunks + self.raw_records:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "sigorta" not in normalized_content or "staj" not in normalized_content:
                continue
            for sentence in SENTENCE_SPLIT_PATTERN.split(content):
                normalized_sentence = normalize_text(sentence)
                if "sigorta" not in normalized_sentence:
                    continue
                if "okul tarafindan sigorta yapilmiyor" in normalized_sentence:
                    no_sentence = sentence.strip()
                elif "okulun sigorta yapmasi" in normalized_sentence or "okul sigortami yapacagi icin" in normalized_sentence:
                    yes_sentence = sentence.strip()
            if yes_sentence and no_sentence:
                break

        if yes_sentence:
            answer = (
                "Sayin ogrencimiz,\n"
                "Bilgisayar Muhendisligi Staj SSS kaynagina gore zorunlu stajlarda sigorta islemleri okul tarafindan yurutulur. "
                "Basvuru sirasinda okulun sigorta islemini yapabilmesi icin gerekli beyanin dogru sekilde verilmesi gerekir."
            )
            if no_sentence:
                answer += (
                    " Buna karsilik zorunlu stajlardan ayri yapilan gonullu staj, staj uzatma veya ilan edilen donemler disindaki stajlarda okul tarafindan sigorta yapilmadigi belirtilmektedir."
                )
            return answer
        if no_sentence:
            return (
                "Sayin ogrencimiz,\n"
                "Bilgisayar Muhendisligi Staj SSS kaynagina gore zorunlu staj kapsami disindaki gonullu veya farkli tarihlerde yapilan stajlarda okul tarafindan sigorta yapilmamaktadir."
            )
        return None

    def _extract_post_upload_graduation_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_post_upload_graduation(query):
            return None

        found_rule = False
        for chunk in context + self.chunks + self.raw_records:
            normalized_content = normalize_text(chunk.get("content", ""))
            if "mezun durumundaysaniz" in normalized_content and (
                "staj komisyonuna mail" in normalized_content or "e posta" in normalized_content
            ):
                found_rule = True
                break

        if not found_rule:
            return None

        return (
            "Sayin ogrencimiz,\n"
            "Mezun durumundaysaniz ve stajlar disinda dersiniz yoksa, staj raporunuzu sisteme yukledikten sonra "
            "bolum staj komisyonuna e-posta ile degerlendirme talebinizi iletmeniz gerekir."
        )

    def _extract_external_summer_school_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_external_summer_school_course(query):
            return None

        permission_sentence = ""
        condition_sentence = ""
        for chunk in context + self.chunks + self.raw_records:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "yaz okulu" not in normalized_content:
                continue
            if (
                not permission_sentence
                and any(marker in normalized_content for marker in ["diger universitelerde", "diger universite", "universitemiz disinda"])
                and "ders alabilir" in normalized_content
            ):
                permission_sentence = "Yaz okulunda baska universitelerden ders alinabilir."
            if (
                not condition_sentence
                and any(marker in normalized_content for marker in ["esdeger", "uygun gormesi", "yuzde sekseninin uyumlu"])
            ):
                condition_sentence = "Bu derslerin sayilabilmesi icin esdegerlik kosullarinin saglanmasi ve bolum onayinin alinmasi gerekir."
            for sentence in SENTENCE_SPLIT_PATTERN.split(content):
                normalized_sentence = normalize_text(sentence)
                if (
                    "universite" in normalized_sentence
                    and any(marker in normalized_sentence for marker in ["diger", "universitemiz disinda", "baska"])
                ):
                    permission_sentence = sentence.strip()
                if any(
                    marker in normalized_sentence
                    for marker in ["bolum anabilim dali baskanliklarinin uygun gormesi", "esdeger", "notlari kabul edilmez"]
                ):
                    condition_sentence = sentence.strip()
            if permission_sentence and condition_sentence:
                break

        if not permission_sentence:
            return None

        answer = (
            "Sayin ogrencimiz,\n"
            "Yaz Okulu Uygulama Esaslarina gore baska universitelerden yaz okulu dersi alinabilmesi mumkundur."
        )
        if condition_sentence:
            answer += " Ancak dersin esdeger sayilabilmesi icin bolum baskanliginin uygun gormesi ve ilgili esdegerlik sartlarinin saglanmasi gerekir."
        return answer

    def _extract_graduation_with_internship_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_graduation_with_incomplete_internship(query):
            return None

        support_sentence = ""
        for chunk in context + self.chunks + self.raw_records:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "mezun" not in normalized_content and "diploma" not in normalized_content:
                continue
            for sentence in SENTENCE_SPLIT_PATTERN.split(content):
                normalized_sentence = normalize_text(sentence)
                if any(
                    marker in normalized_sentence
                    for marker in ["butun calismalari tamamlamis olan ogrenci mezun olmaya hak kazanir", "stajini tamamlamayan ogrenci"]
                ):
                    support_sentence = sentence.strip()
                    break
            if support_sentence:
                break

        if not support_sentence:
            return None

        return (
            "Sayin ogrencimiz,\n"
            "Hayir. Mezuniyet icin yalnizca derslerin degil, ilgili mevzuatta yer alan tum akademik yukumluluklerin de tamamlanmis olmasi gerekir. "
            "Bu nedenle staj yukumlulugu eksikse diploma islemleri tamamlanmis sayilmaz."
        )

    def _extract_graduation_requirements_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_graduation_requirements(query):
            return None

        requirement_sentences = []
        for chunk in context + self.chunks + self.raw_records:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if not any(marker in normalized_content for marker in ["mezun", "diploma"]):
                continue
            for sentence in SENTENCE_SPLIT_PATTERN.split(content):
                normalized_sentence = normalize_text(sentence)
                if any(
                    marker in normalized_sentence
                    for marker in ["mezun olmaya hak kazanir", "butun calismalari tamamlamis", "basarili olmus"]
                ):
                    cleaned = sentence.strip()
                    if cleaned and cleaned not in requirement_sentences:
                        requirement_sentences.append(cleaned)
                if len(requirement_sentences) >= 2:
                    break
            if len(requirement_sentences) >= 2:
                break

        if not requirement_sentences:
            return None

        return (
            "Sayin ogrencimiz,\n"
            "Mezuniyet icin temel kosul, ogretim programinda yer alan derslerin ve ilgili mevzuatta ongorulen diger akademik yukumluluklerin tamamlanmis olmasidir. "
            "Baska bir ifadeyle, mezuniyet icin yalnizca dersleri gecmek degil, programin tum resmi sartlarini yerine getirmek gerekir."
        )

    def _extract_yaz_okulu_duration_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_yaz_okulu_duration(query):
            return None

        durations_preview = set()
        ending_preview = {}
        for chunk in context + self.raw_records:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "yaz okulu" not in normalized_content:
                continue
            for week in WEEK_PATTERN.findall(content):
                durations_preview.add(int(week))
            if "bes" in normalized_content and "hafta" in normalized_content:
                durations_preview.add(5)
            if "yedi" in normalized_content and "hafta" in normalized_content:
                durations_preview.add(7)
            for end_date, week in WEEK_ENDING_PATTERN.findall(content):
                durations_preview.add(int(week))
                ending_preview[int(week)] = end_date

        ordered_preview = sorted(durations_preview)
        if ordered_preview == [5, 7]:
            answer = "Sayın öğrencimiz,\nAkademik takvime göre yaz okulu 5 hafta veya 7 hafta olarak uygulanabilmektedir."
            if 5 in ending_preview and 7 in ending_preview:
                answer += f" 5 haftalık yaz okulu {ending_preview[5]}, 7 haftalık yaz okulu ise {ending_preview[7]} tarihinde sona ermektedir."
            return answer
        if len(ordered_preview) == 1:
            return f"Sayın öğrencimiz,\nAkademik takvime göre yaz okulu {ordered_preview[0]} hafta sürmektedir."

        durations = set()
        ending_dates = {}

        for chunk in context + self.raw_records:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "yaz okulu" not in normalized_content:
                continue

            for week in WEEK_PATTERN.findall(content):
                durations.add(int(week))

            if "bes" in normalized_content and "hafta" in normalized_content:
                durations.add(5)
            if "yedi" in normalized_content and "hafta" in normalized_content:
                durations.add(7)

            for end_date, week in WEEK_ENDING_PATTERN.findall(content):
                durations.add(int(week))
                ending_dates[int(week)] = end_date

        if not durations:
            return None

        ordered = sorted(durations)
        if ordered == [5, 7]:
            answer = (
                "Sayın öğrencimiz,\n"
                "Akademik takvime göre yaz okulu 5 hafta veya 7 hafta olarak uygulanabilmektedir."
            )
            if 5 in ending_dates and 7 in ending_dates:
                answer += (
                    f" 5 haftalık yaz okulu {ending_dates[5]}, "
                    f"7 haftalık yaz okulu ise {ending_dates[7]} tarihinde sona ermektedir."
                )
            return answer

        if len(ordered) == 1:
            return f"Sayın öğrencimiz,\nAkademik takvime göre yaz okulu {ordered[0]} hafta sürmektedir."

        joined = " ve ".join(f"{week} hafta" for week in ordered)
        return f"Sayın öğrencimiz,\nAkademik takvime göre yaz okulu {joined} olarak uygulanabilmektedir."

    def _extract_yaz_okulu_start_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_yaz_okulu_start(query):
            return None

        requested_years = extract_years(query)
        best_match = None
        best_score = float("-inf")

        normalized_query = normalize_text(query)

        for chunk in self.raw_records:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if requested_years and not any(str(year) in content for year in requested_years):
                continue
            explicit_start_match = SUMMER_SCHOOL_EXPLICIT_START_PATTERN.search(content)
            if chunk.get("kategori") == "akademik_takvim" and "yaz okulu" in normalized_content and explicit_start_match:
                return (
                    "Sayın öğrencimiz,\n"
                    f"Akademik takvimde yaz okulunun başlangıcı {explicit_start_match.group(1)} olarak görünmektedir."
                )
            if "muhendislik" in normalized_query and "yaz okulu final haftasi dahil" in normalized_content:
                range_match = DATE_RANGE_PATTERN.search(content)
                if range_match:
                    return (
                        "Sayın öğrencimiz,\n"
                        f"Mühendislik Fakültesi takvimine göre yaz okulu (final haftası dahil) "
                        f"{range_match.group(1)} tarihinde başlayıp {range_match.group(2)} tarihinde sona ermektedir."
                    )

        if "muhendislik" in normalized_query:
            for chunk in self.raw_records:
                content = chunk.get("content", "")
                normalized_content = normalize_text(content)
                if "yaz okulu final haftasi dahil" not in normalized_content:
                    continue
                range_match = DATE_RANGE_PATTERN.search(content)
                if range_match:
                    return (
                        "Sayın öğrencimiz,\n"
                        f"Mühendislik Fakültesi takvimine göre yaz okulu (final haftası dahil) "
                        f"{range_match.group(1)} tarihinde başlayıp {range_match.group(2)} tarihinde sona ermektedir."
                    )

        for chunk in self.raw_records:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            explicit_start_match = SUMMER_SCHOOL_EXPLICIT_START_PATTERN.search(content)
            if chunk.get("kategori") == "akademik_takvim" and "yaz okulu" in normalized_content and explicit_start_match:
                return (
                    "Sayın öğrencimiz,\n"
                    f"Akademik takvimde yaz okulunun başlangıcı {explicit_start_match.group(1)} olarak görünmektedir."
                )

        for chunk in self.raw_records + context:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "yaz okulu" not in normalized_content:
                continue

            score = 0.0
            if chunk.get("kategori") == "akademik_takvim":
                score += 12
            if "muhendislik" in normalized_query and "mf.duzce.edu.tr" in chunk.get("source_url", ""):
                score += 14
            if "baslangic" in normalized_content:
                score += 10
            if requested_years and any(str(year) in content for year in requested_years):
                score += 8

            start_match = SUMMER_SCHOOL_EXPLICIT_START_PATTERN.search(content) or SUMMER_SCHOOL_START_PATTERN.search(content)
            range_match = SUMMER_SCHOOL_RANGE_PATTERN.search(content) or DATE_RANGE_PATTERN.search(content)
            if start_match:
                score += 16
            elif range_match:
                score += 10

            if score > best_score:
                best_score = score
                best_match = {
                    "start": start_match.group(1) if start_match else None,
                    "range_start": range_match.group(1) if range_match else None,
                    "range_end": range_match.group(2) if range_match else None,
                }

        if not best_match:
            return None

        if best_match["range_start"] and best_match["range_end"] and "muhendislik" in normalized_query:
            return (
                "Sayın öğrencimiz,\n"
                f"Mühendislik Fakültesi takvimine göre yaz okulu (final haftası dahil) "
                f"{best_match['range_start']} tarihinde başlayıp {best_match['range_end']} tarihinde sona ermektedir."
            )

        if best_match["start"]:
            return (
                "Sayın öğrencimiz,\n"
                f"Akademik takvimde yaz okulunun başlangıcı {best_match['start']} olarak görünmektedir."
            )

        if best_match["range_start"] and best_match["range_end"]:
            return (
                "Sayın öğrencimiz,\n"
                f"Kaynağa göre yaz okulu {best_match['range_start']} tarihinde başlayıp "
                f"{best_match['range_end']} tarihinde sona ermektedir."
            )

        return None

    def _extract_yaz_staji_schedule_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not asks_yaz_staji_schedule(query):
            return None

        requested_years = extract_years(query)

        for chunk in self.raw_records:
            content = chunk.get("content", "")
            if not content:
                continue
            if requested_years and not any(str(year) in content for year in requested_years):
                continue

            normalized_content = normalize_text(content)
            if "yaz okulu sonrasi staj donemi" not in normalized_content:
                continue

            period_numbers = sorted({int(value) for value in re.findall(r"\b(\d+)\.\s*D", content)})
            if not period_numbers:
                continue

            period_count = len(period_numbers)
            range_match = DATE_RANGE_PATTERN.search(content)
            if asks_period_count(query):
                answer = (
                    "Sayın öğrencimiz,\n"
                    f"Mühendislik Fakültesi takviminde yaz okulu sonrası staj için {period_count} dönem görünmektedir."
                )
                if range_match:
                    answer += (
                        f" Aynı kayıtta yaz okulu (final haftası dahil) "
                        f"{range_match.group(1)} - {range_match.group(2)} aralığında gösterilmektedir."
                    )
                answer += " Ancak kayıt düz metne dönüştüğü için her staj döneminin tek tek başlangıç tarihi güvenli biçimde ayrışamıyor."
                return answer

            if range_match:
                return (
                    "Sayın öğrencimiz,\n"
                    "Mühendislik Fakültesi takviminde yaz stajı dönemleri 'Yaz Okulu Sonrası Staj Dönemi' olarak ayrı gösterilmektedir. "
                    f"Aynı kayıtta yaz okulu (final haftası dahil) {range_match.group(1)} - {range_match.group(2)} aralığında yer almaktadır. "
                    "Düz metne dönüşen kayıtta staj dönemlerinin tek tek başlangıç tarihi net ayrışmadığı için ilk staj döneminin kesin başlangıç gününü güvenle söyleyemiyorum."
                )

        best_match = None
        best_score = float("-inf")

        for chunk in self.raw_records + context:
            content = chunk.get("content", "")
            normalized_content = normalize_text(content)
            if "yaz staji" not in normalized_content and "staj donem" not in normalized_content:
                continue

            score = 0.0
            if DATE_PATTERN.search(content):
                score += 10
            if "staj donemleri" in normalized_content:
                score += 12
            if "yaz okulu sonrasi staj donemi" in normalized_content:
                score += 10
            if chunk.get("kategori") in {"staj", "fakulte_bolum"}:
                score += 6

            range_match = DATE_RANGE_PATTERN.search(content) or SUMMER_SCHOOL_RANGE_PATTERN.search(content)
            date_matches = DATE_PATTERN.findall(content)
            if range_match:
                score += 8

            if score > best_score:
                best_score = score
                best_match = {
                    "range_start": range_match.group(1) if range_match else None,
                    "range_end": range_match.group(2) if range_match else None,
                    "dates": date_matches[:4],
                }

        if not best_match:
            return None

        if best_match["range_start"] and best_match["range_end"]:
            return (
                "Sayın öğrencimiz,\n"
                f"Kaynağa göre ilgili yaz dönemi {best_match['range_start']} - {best_match['range_end']} aralığında planlanmıştır."
            )

        if len(best_match["dates"]) >= 2:
            return (
                "Sayın öğrencimiz,\n"
                f"Kaynağa göre ilgili yaz stajı/staj dönemi için öne çıkan tarihler {best_match['dates'][0]} ve {best_match['dates'][1]} olarak görünmektedir."
            )

        return None

    def _extract_direct_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        working_query = query
        if is_scope_clarification_query(query):
            previous_user_query = self._last_user_query()
            if previous_user_query:
                working_query = f"{previous_user_query} {query}".strip()

        direct_answer = self._extract_post_upload_graduation_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_graduation_with_internship_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_graduation_requirements_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_yaz_okulu_attendance_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_external_summer_school_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_insurance_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_yaz_okulu_duration_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_yaz_okulu_start_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_yaz_staji_schedule_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_disciplinary_scholarship_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_makeup_exam_with_missing_internship_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_report_submission_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_missed_period_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_duration_confirmation_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_count_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_timing_answer(working_query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_course_registration_answer(working_query, context)
        if direct_answer:
            return direct_answer

        if not is_short_factual_query(working_query):
            return None

        normalized_query = normalize_text(working_query)
        wants_staj_duration = asks_staj_duration(working_query)
        if not wants_staj_duration:
            return None

        best_match = None
        best_score = float("-inf")

        for chunk in context:
            content = chunk.get("content", "")
            for sentence in SENTENCE_SPLIT_PATTERN.split(content):
                sentence = sentence.strip(" -\t")
                if not sentence:
                    continue

                normalized_sentence = normalize_text(sentence)
                if "staj" not in normalized_sentence:
                    continue

                range_match = WORKDAY_RANGE_PATTERN.search(sentence)
                number_match = WORKDAY_NUMBER_PATTERN.search(sentence)
                if not range_match and not number_match:
                    continue

                sentence_score = 0.0
                if chunk.get("kategori") == "staj":
                    sentence_score += 6
                if "bilgisayar muhendisligi" in normalized_sentence:
                    sentence_score += 12
                if "zorunlulugu" in normalized_sentence:
                    sentence_score += 10
                if "staj suresi" in normalized_sentence or "arasinda" in normalized_sentence:
                    sentence_score += 8
                if any(
                    marker in normalized_sentence
                    for marker in ["birlestir", "uzat", "maksimum", "mezun", "degerlendirilmesi", "sigorta", "rapor"]
                ):
                    sentence_score -= 12
                if "50 is gunu" in normalized_sentence:
                    sentence_score -= 8

                if sentence_score > best_score:
                    best_score = sentence_score
                    best_match = {
                        "sentence": sentence,
                        "normalized": normalized_sentence,
                        "range_match": range_match,
                        "number_match": number_match,
                    }

        if not best_match:
            return None

        if "bilgisayar muhendisligi" in best_match["normalized"] and best_match["number_match"]:
            return (
                "Sayın öğrencimiz,\n"
                f"Bilgisayar Mühendisliği öğrencileri için staj süresi {best_match['number_match'].group(1)} iş günüdür."
            )

        if best_match["range_match"]:
            return (
                "Sayın öğrencimiz,\n"
                f"Staj süresi ilgili akademik birimin yönergesine göre {best_match['range_match'].group(1)}-{best_match['range_match'].group(2)} iş günü arasındadır."
            )

        if best_match["number_match"]:
            return f"Sayın öğrencimiz,\nStaj süresi {best_match['number_match'].group(1)} iş günüdür."

        return None

    def _should_require_program_scope(self, query: str) -> bool:
        return is_program_specific_query(query)

    def _has_program_specific_context(self, context: List[Dict]) -> bool:
        target_scope = self.program_scope
        return bool(target_scope) and any(
            chunk.get("program_scope", GENERAL_SCOPE) == target_scope for chunk in context
        )

    def _scope_guard_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        if not self._should_require_program_scope(query):
            return None
        effective_scope = self._resolve_program_scope(query)
        if effective_scope and any(
            chunk.get("program_scope", GENERAL_SCOPE) == effective_scope for chunk in context
        ):
            return None
        if effective_scope and asks_graduation_requirements(query):
            if any(_is_graduation_whitelisted_source(chunk) for chunk in context):
                return None
        if effective_scope:
            return (
                "Sayın öğrencimiz,\n"
                "Belirttiğiniz bölüm veya program için açık ve doğrudan bir resmi dayanak bulamadım. "
                "Yanlış yönlendirmemek için kesin bir cevap veremiyorum."
            )

        specific_scopes = {
            chunk.get("program_scope", GENERAL_SCOPE)
            for chunk in context
            if chunk.get("program_scope", GENERAL_SCOPE) not in {GENERAL_SCOPE, OTHER_SCOPE, ""}
        }
        if specific_scopes:
            return f"Sayın öğrencimiz,\n{SCOPE_CLARIFICATION_TEXT}"
        return f"Sayın öğrencimiz,\n{SCOPE_CLARIFICATION_TEXT}"

    def _evidence_focus_type(self, query: str) -> str:
        normalized = normalize_text(query)
        if asks_yaz_okulu_attendance(query):
            return "attendance"
        if asks_staj_insurance(query):
            return "insurance"
        if asks_external_summer_school_course(query):
            return "external_university"
        if asks_graduation_with_incomplete_internship(query) or asks_graduation_requirements(query):
            return "graduation_requirement"
        if asks_yaz_okulu_start(query) or asks_yaz_staji_schedule(query):
            return "date"
        if asks_yaz_okulu_duration(query):
            return "week_duration"
        if asks_staj_duration(query):
            return "workday_duration"
        if asks_staj_report_submission(query):
            return "submission_process"
        if asks_makeup_exam_with_missing_internship(query):
            return "tek_cift_rule"
        if asks_period_count(query):
            return "period_count"
        if any(marker in normalized for marker in ["hangi belge", "hangi belgeler", "evrak", "form", "dilekce"]):
            return "document"
        if any(marker in normalized for marker in ["ne zaman", "hangi tarihte", "son gun", "son tarih"]):
            return "date"
        if any(marker in normalized for marker in ["kac", "ne kadar", "suresi", "sure", "akts", "kredi"]):
            return "numeric"
        if any(marker in normalized for marker in ["nasil", "surec", "adim", "basvuru", "itiraz", "teslim"]):
            return "process"
        return ""

    def _snippet_matches_focus(self, query: str, snippet: str) -> bool:
        focus_type = self._evidence_focus_type(query)
        if not focus_type:
            return True

        normalized_snippet = normalize_text(snippet)
        snippet_tokens = set(normalized_snippet.split())

        if focus_type == "date":
            return bool(DATE_PATTERN.search(snippet) or re.search(r"\b20\d{2}\b", snippet))
        if focus_type == "week_duration":
            return "hafta" in snippet_tokens or WEEK_PATTERN.search(snippet) is not None
        if focus_type == "workday_duration":
            return "is gunu" in normalized_snippet or WORKDAY_NUMBER_PATTERN.search(snippet) is not None
        if focus_type == "submission_process":
            return any(
                marker in normalized_snippet
                for marker in ["rapor", "defter", "sbs", "teslim", "yukle", "yaklasik 30 gun"]
            )
        if focus_type == "attendance":
            return "devam" in snippet_tokens or "devamsizlik" in snippet_tokens or "yoklama" in snippet_tokens
        if focus_type == "insurance":
            return "sigorta" in snippet_tokens
        if focus_type == "external_university":
            return any(
                marker in normalized_snippet
                for marker in ["diger universite", "universitemiz disinda", "esdeger", "farkli fakulte", "misafir ogrenci"]
            )
        if focus_type == "graduation_requirement":
            return any(
                marker in normalized_snippet
                for marker in [
                    "mezun",
                    "mezuniyet",
                    "diploma",
                    "basarili",
                    "butun calismalari tamamlamis",
                    "mezun olmaya hak kazanir",
                ]
            )
        if focus_type == "tek_cift_rule":
            return any(marker in normalized_snippet for marker in ["tek cift", "tek ders", "cift ders"])
        if focus_type == "period_count":
            return any(marker in snippet_tokens for marker in ["donem", "yariyil"]) or any(
                marker in normalized_snippet for marker in ["donem", "yariyil"]
            )
        if focus_type == "document":
            return any(
                marker in snippet_tokens
                for marker in ["belge", "belgeler", "evrak", "form", "dilekce", "transkript", "rapor", "defter"]
            )
        if focus_type == "numeric":
            return bool(NUMERIC_UNIT_PATTERN.search(snippet)) or any(
                marker in snippet_tokens for marker in ["gun", "hafta", "ay", "kredi", "akts", "yariyil", "donem"]
            )
        if focus_type == "process":
            return any(
                marker in snippet_tokens
                for marker in ["basvuru", "teslim", "onay", "yukle", "islem", "itiraz", "duyuru", "obs", "sbs"]
            )
        return True

    def _evidence_score(self, query: str, snippet: str, source: Dict) -> float:
        normalized_snippet = normalize_text(snippet)
        snippet_terms = set(normalized_snippet.split())
        query_terms = self._important_terms(query)
        score = 0.0

        score += 4.0 * len(query_terms & snippet_terms)
        if len(query_terms) >= 3 and len(query_terms & snippet_terms) >= 2:
            score += 6.0

        effective_scope = self._resolve_program_scope(query)
        candidate_scope = source.get("program_scope", infer_chunk_scope(source))
        if effective_scope and candidate_scope == effective_scope:
            score += 10.0
        elif effective_scope and candidate_scope not in {GENERAL_SCOPE, ""}:
            score -= 12.0

        query_topic = infer_query_topic(query)
        if query_topic != "genel" and source.get("topic") == query_topic:
            score += 8.0

        requested_years = extract_years(query)
        if requested_years and any(str(year) in snippet for year in requested_years):
            score += 10.0

        if self._snippet_matches_focus(query, snippet):
            score += 12.0
        else:
            score -= 10.0

        if is_short_factual_query(query):
            if NUMERIC_UNIT_PATTERN.search(snippet):
                score += 8.0
            if DATE_PATTERN.search(snippet):
                score += 6.0

        if asks_yaz_okulu_start(query) and "yaz okulu" in normalized_snippet and DATE_PATTERN.search(snippet):
            score += 18.0
        if asks_yaz_okulu_duration(query) and "yaz okulu" in normalized_snippet and "hafta" in normalized_snippet:
            score += 18.0
        if asks_staj_duration(query) and "staj" in normalized_snippet and "is gunu" in normalized_snippet:
            score += 18.0
        if asks_staj_report_submission(query) and any(
            marker in normalized_snippet for marker in ["sbs", "teslim", "yukle", "yaklasik 30 gun"]
        ):
            score += 16.0
        if asks_makeup_exam_with_missing_internship(query) and "tek cift" in normalized_snippet:
            score += 18.0

        if normalized_snippet.startswith("baskanligimiz hakkimizda") or "kalite komisyon" in normalized_snippet:
            score -= 20.0

        return score

    def _select_evidence_context(self, query: str, context: List[Dict], limit: int = 6) -> List[Dict]:
        candidates = []
        seen = set()

        for source in context:
            content = source.get("content", "")
            if not content.strip():
                continue

            snippets = []
            if len(content) <= 700:
                snippets.append(content.strip())
            else:
                for sentence in SENTENCE_SPLIT_PATTERN.split(content):
                    sentence = sentence.strip(" -\t")
                    if len(sentence) >= 60:
                        snippets.append(sentence)

            for snippet in snippets:
                fingerprint = hashlib.md5(
                    f"{source.get('source_url', '')}\n{snippet}".encode("utf-8")
                ).hexdigest()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                score = self._evidence_score(query, snippet, source)
                if score <= 0:
                    continue
                evidence = dict(source)
                evidence["content"] = snippet
                evidence["evidence_score"] = score
                evidence["focus_match"] = self._snippet_matches_focus(query, snippet)
                evidence["source_title"] = source.get("source_title") or infer_source_title(source)
                candidates.append(evidence)

        if not candidates:
            return []

        ranked = sorted(candidates, key=lambda item: item["evidence_score"], reverse=True)
        top_score = ranked[0]["evidence_score"]
        if top_score < 6:
            return []

        if any(item.get("focus_match") for item in ranked):
            ranked = [item for item in ranked if item.get("focus_match")]
            if not ranked:
                return []
            top_score = ranked[0]["evidence_score"]

        threshold = max(5.0, top_score * 0.35)
        selected = [item for item in ranked if item["evidence_score"] >= threshold]
        return selected[:limit]

    def _format_evidence_text(self, evidence_context: List[Dict]) -> str:
        parts = []
        for index, item in enumerate(evidence_context, start=1):
            title = item.get("source_title") or infer_source_title(item)
            url = item.get("source_url", "")
            parts.append(f"[Kaynak {index}] {title}\nURL: {url}\n{item.get('content', '')}")
        return "\n\n---\n\n".join(parts)

    def generate_response(self, query: str, context: List[Dict]) -> str:
        self.last_answer_context = []
        if not context:
            return f"Sayın öğrencimiz,\n{NO_ANSWER_TEXT}"

        scope_guard = self._scope_guard_answer(query, context)
        if scope_guard:
            return scope_guard

        direct_answer = self._extract_direct_answer(query, context)
        if direct_answer:
            self.last_answer_context = context
            return direct_answer

        evidence_context = self._select_evidence_context(query, context)
        if not evidence_context:
            return f"Sayın öğrencimiz,\n{NO_ANSWER_TEXT}"
            return f"SayÄ±n Ã¶ÄŸrencimiz,\n{NO_ANSWER_TEXT}"

        self.last_answer_context = evidence_context
        context_text = self._format_evidence_text(evidence_context)
        memory_text = self._memory_as_text(query)

        goals_text = "\n".join(f"- {goal}" for goal in ASSISTANT_GOALS)
        personality_text = ", ".join(ASSISTANT_PERSONALITY)

        prompt = f"""Sen Düzce Üniversitesi Öğrenci İşleri Daire Başkanlığı'nın resmi Türkçe yapay zeka asistanısın.

Kimlik:
{ASSISTANT_IDENTITY}

Temel hedeflerin:
{goals_text}

İletişim kişiliğin:
{personality_text}

ZORUNLU KURALLAR:
1. Yalnızca Türkçe yaz.
2. Yalnızca aşağıdaki "Resmi Belgeler" bölümündeki bilgileri kullan.
3. Belgede olmayan hiçbir bilgiyi uydurma, tahmin etme veya ekleme.
4. Eğer soru belgelerde geçmiyorsa sadece şunu yaz: "{NO_ANSWER_TEXT}"
5. Soruda sayısal veya kısa olgusal bilgi isteniyorsa cevabı ilk cümlede doğrudan ver.
6. Cevap bulunuyorsa asla "{NO_ANSWER_TEXT}" cümlesini ekleme.
7. "Saygılarımla" gibi kapanış ifadeleri ekleme.
8. "Sohbet Geçmişi" bölümünü sadece bağlamı anlamak için kullan; bilgi kaynağı olarak kullanma.
9. Dayandığın her ana iddia için mümkün olduğunda köşeli parantez içinde kaynak etiketi kullan: [Kaynak 1], [Kaynak 2].
10. İç muhakemeni veya adım adım düşünceni açıklama; sadece sonuç ve kısa gerekçe ver.
11. Üslubun profesyonel, nazik ve empatik olsun; ancak resmi kurum dili dışına çıkma.
12. Öğrenciyi yönlendirirken kısa, açık ve güven veren bir dil kullan.

Sohbet Geçmişi:
{memory_text}

Kanitlar:
{context_text}

Öğrenci Sorusu: {query}

Cevap (Türkçe, "Sayın öğrencimiz," ile başla):"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "repeat_penalty": 1.2,
                "stop": ["Öğrenci Sorusu:", "Resmi Belgeler:"],
            },
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        answer = response.json()["response"].strip()

        english_markers = [
            "According",
            "Please",
            "In your",
            "Note that",
            "However",
            "I would",
            "You can",
            "You cannot",
        ]
        filtered_lines = [
            line
            for line in answer.splitlines()
            if not any(marker in line for marker in english_markers)
        ]
        cleaned = "\n".join(filtered_lines).strip()
        cleaned = self._cleanup_response(cleaned)

        if len(cleaned) < 20:
            cleaned = NO_ANSWER_TEXT

        return cleaned if cleaned.startswith("Sayın") else f"Sayın öğrencimiz,\n{cleaned}"

    def chat(self, query: str) -> Dict:
        search_query = self._build_search_query(query)
        results = self.hybrid_search(search_query, k=7)
        answer = self._finalize_answer(self.generate_response(search_query, results))
        source_context = self.last_answer_context or results
        sources = self._format_sources(source_context, answer, search_query)
        answer = self._attach_source_summary(answer, sources)
        self._save_to_memory(query, answer)
        return {
            "query": query,
            "cevap": answer,
            "kaynaklar": sources,
        }


if __name__ == "__main__":
    bot = RAGChatbot()
    result = bot.chat("Çift Anadal başvuru şartları nelerdir?")
    print(result["cevap"])
