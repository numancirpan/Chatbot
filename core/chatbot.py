import json
import hashlib
import os
import re
import unicodedata
from typing import Dict, List, Optional

import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_ADI = "qwen2.5:7b"
NO_ANSWER_TEXT = (
    "Bu konuda resmi belgelerde bilgiye ulaşılamadım. "
    "Lütfen Öğrenci İşleri birimi ile iletişime geçiniz."
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(ROOT_DIR, "data", "chunks.json")
KNOWLEDGE_BASE_FILE = os.path.join(ROOT_DIR, "data", "knowledge_base.json")
DB_DIR = os.path.join(ROOT_DIR, "db", "chroma_db")
MAX_MEMORY_TURNS = 5
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
    "yaz_okulu": ["yaz okulu", "yaz okulu kayit", "yaz okulunun"],
    "akademik_takvim": ["akademik takvim", "ders kayit", "derslerin baslamasi", "final"],
    "cap_yandal": ["cift anadal", "cap", "yandal"],
    "tek_cift_sinav": ["tek cift", "tek ders", "cift ders"],
    "burs": ["burs", "bursu", "bursunu"],
    "disiplin": ["disiplin", "uzaklastirma", "kinama"],
    "harc": ["harc", "katki payi", "ogrenim ucreti"],
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
    haystack = f"{normalized_content} {normalized_url} {kategori}"

    for topic, hints in TOPIC_HINTS.items():
        if any(hint in haystack for hint in hints):
            return topic

    if kategori:
        return kategori
    return "genel"


def infer_source_title(chunk: Dict) -> str:
    url = chunk.get("source_url", "")
    kategori = chunk.get("kategori", "Genel")
    normalized_url = url.lower()
    if "bm.mf.duzce.edu.tr/sayfa/878b" in normalized_url:
        return "Bilgisayar Mühendisliği - Staj SSS"
    if "bm.mf.duzce.edu.tr/sayfa/4a82" in normalized_url:
        return "Bilgisayar Mühendisliği - Staj"
    if "bm.mf.duzce.edu.tr/sayfa/17ac" in normalized_url:
        return "Bilgisayar Mühendisliği - Yaz Okulu"
    if "mf.duzce.edu.tr/sayfa/967a" in normalized_url:
        return "Mühendislik Fakültesi - Yaz Stajı"
    if "akademik-takvim" in normalized_url or chunk.get("kategori") == "akademik_takvim":
        return "Akademik Takvim"
    if "ogrenciisleri.duzce.edu.tr" in normalized_url:
        return "Öğrenci İşleri"
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
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    def rerank(self, query: str, chunks: List[Dict], k: int = 5) -> List[Dict]:
        if not chunks:
            return []
        scores = self.model.predict([[query, chunk["content"]] for chunk in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
        return [chunk for chunk, _ in ranked[:k]]


class RAGChatbot:
    def __init__(self, program_scope: str = DEFAULT_PROGRAM_SCOPE):
        self.program_scope = program_scope
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
        self.reranker = Reranker()
        self.vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        )
        self.vector_count = self._vector_store_count()
        self.memory: List[Dict[str, str]] = []
        self.last_answer_context: List[Dict] = []

        self._ollama_kontrol()
        print(f"{len(self.chunks)} chunk yuklendi")
        if self.vector_count:
            print(f"ChromaDB hazir ({self.vector_count} kayit)")
        else:
            print("UYARI: ChromaDB bos veya okunamiyor. Arama BM25 uzerinden devam edecek.")
            print("DB'yi yenilemek icin: python pipeline/create_vector_db.py --rebuild")
        print("BM25 + Reranker + ChromaDB + sohbet hafizasi hazir")

    def _resolve_program_scope(self, query: str) -> str:
        return self.program_scope or infer_query_scope(query)

    def _ollama_kontrol(self):
        try:
            requests.get("http://localhost:11434", timeout=3)
            print("Ollama calisiyor")
        except requests.exceptions.ConnectionError:
            print("Ollama bulunamadi! 'ollama serve' komutunu calistirin.")

    def _vector_store_count(self) -> int:
        try:
            return int(self.vector_store._collection.count())
        except Exception:
            return 0

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
            if self.vector_count > 0:
                try:
                    vector_docs = self.vector_store.similarity_search(variant, k=candidate_k)
                except Exception:
                    vector_docs = []
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

    def _candidate_score(self, query: str, candidate: Dict) -> float:
        normalized_query = normalize_text(query)
        query_tokens = set(tokenize(query))
        normalized_content = normalize_text(candidate.get("content", ""))
        content_tokens = set(normalized_content.split())
        source_url = candidate.get("source_url", "").lower()
        kategori = normalize_text(candidate.get("kategori", ""))
        candidate_scope = candidate.get("program_scope", GENERAL_SCOPE)
        effective_scope = self._resolve_program_scope(query)
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
        if asks_staj_timing(query):
            if "yariyil" in normalized_content:
                score += 6
            if "bm399" in normalized_content or "bm499" in normalized_content:
                score += 8
            if "yaz" in normalized_content:
                score += 4
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

        return score

    def _memory_as_text(self) -> str:
        if not self.memory:
            return "Yok"

        lines = []
        for message in self.memory[-MAX_MEMORY_TURNS * 2 :]:
            prefix = "Öğrenci" if message["role"] == "user" else "Asistan"
            lines.append(f"{prefix}: {message['content']}")
        return "\n".join(lines)

    def _last_user_query(self) -> Optional[str]:
        for message in reversed(self.memory):
            if message.get("role") == "user" and message.get("content", "").strip():
                return message["content"].strip()
        return None

    def _build_search_query(self, query: str) -> str:
        previous_user_query = self._last_user_query()
        if not previous_user_query:
            return query
        if is_scope_clarification_query(query):
            return f"{query.strip()} {previous_user_query}"
        return query

    def _save_to_memory(self, query: str, answer: str) -> None:
        self.memory.append({"role": "user", "content": query})
        self.memory.append({"role": "assistant", "content": answer})
        max_messages = MAX_MEMORY_TURNS * 2
        if len(self.memory) > max_messages:
            self.memory = self.memory[-max_messages:]

    def clear_memory(self) -> None:
        self.memory.clear()

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
                "Bilgisayar Mühendisliği SSS kaynağına göre staj raporu imza/kaşe onayından sonra taranıp SBS'ye yüklenir."
            ]
            if has_bm_no_fixed_deadline:
                parts.append("Aynı kaynakta sistemde yüklemek için sabit bir son tarih bulunmadığı belirtilmektedir.")
            if has_bm_approx_30_days:
                parts.append("Yaz stajı için raporun yeni güz dönemi başlangıcından itibaren yaklaşık 30 gün sonrasına kadar yüklenebileceği; değerlendirmenin staj komisyonu toplandıktan sonra yapılacağı yazmaktadır.")
            if has_bm_graduation_note:
                parts.append("Mezun durumundaysanız ve stajlar dışında dersiniz yoksa raporu yükledikten sonra bölüm staj komisyonuna e-posta ile değerlendirme talep etmeniz gerektiği belirtilmiştir.")
            if has_general_correction_rule:
                parts.append("Genel staj yönergesinde, komisyon düzeltme isterse istenen düzeltmenin en çok 1 ay içinde yapılması gerektiği; aksi durumda stajın reddedilmiş sayılacağı yer almaktadır.")
            parts.append("Kaynaklarda geç teslim için otomatik burs/ceza gibi ayrı bir yaptırım açıkça belirtilmiyor.")
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
        direct_answer = self._extract_yaz_okulu_duration_answer(query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_yaz_okulu_start_answer(query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_yaz_staji_schedule_answer(query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_disciplinary_scholarship_answer(query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_makeup_exam_with_missing_internship_answer(query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_report_submission_answer(query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_missed_period_answer(query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_duration_confirmation_answer(query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_timing_answer(query, context)
        if direct_answer:
            return direct_answer

        direct_answer = self._extract_staj_course_registration_answer(query, context)
        if direct_answer:
            return direct_answer

        if not is_short_factual_query(query):
            return None

        normalized_query = normalize_text(query)
        wants_staj_duration = asks_staj_duration(query)
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
        if effective_scope:
            return (
                "Sayın öğrencimiz,\n"
                "Belirttiğiniz bölüm veya program için doğrudan resmi bir kaynak bulamadım. "
                "Yanlış yönlendirmemek için net cevap veremiyorum."
            )

        specific_scopes = {
            chunk.get("program_scope", GENERAL_SCOPE)
            for chunk in context
            if chunk.get("program_scope", GENERAL_SCOPE) not in {GENERAL_SCOPE, OTHER_SCOPE, ""}
        }
        if specific_scopes:
            return f"Sayın öğrencimiz,\n{SCOPE_CLARIFICATION_TEXT}"
        return f"Sayın öğrencimiz,\n{SCOPE_CLARIFICATION_TEXT}"

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

        query_topic = infer_topic({"content": query, "source_url": "", "kategori": ""})
        if query_topic != "genel" and source.get("topic") == query_topic:
            score += 8.0

        requested_years = extract_years(query)
        if requested_years and any(str(year) in snippet for year in requested_years):
            score += 10.0

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
                evidence["source_title"] = source.get("source_title") or infer_source_title(source)
                candidates.append(evidence)

        if not candidates:
            return []

        ranked = sorted(candidates, key=lambda item: item["evidence_score"], reverse=True)
        top_score = ranked[0]["evidence_score"]
        if top_score < 6:
            return []

        threshold = max(5.0, top_score * 0.35)
        selected = [item for item in ranked if item["evidence_score"] >= threshold]
        return selected[:limit]

    def _format_evidence_text(self, evidence_context: List[Dict]) -> str:
        parts = []
        for index, item in enumerate(evidence_context, start=1):
            title = item.get("source_title") or infer_source_title(item)
            url = item.get("source_url", "")
            parts.append(f"[Kanit {index}] {title}\nURL: {url}\n{item.get('content', '')}")
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
        memory_text = self._memory_as_text()

        prompt = f"""Sen Düzce Üniversitesi Öğrenci İşleri Daire Başkanlığı'nın resmi Türkçe yapay zeka asistanısın.

ZORUNLU KURALLAR:
1. Yalnızca Türkçe yaz.
2. Yalnızca aşağıdaki "Resmi Belgeler" bölümündeki bilgileri kullan.
3. Belgede olmayan hiçbir bilgiyi uydurma, tahmin etme veya ekleme.
4. Eğer soru belgelerde geçmiyorsa sadece şunu yaz: "{NO_ANSWER_TEXT}"
5. Soruda sayısal veya kısa olgusal bilgi isteniyorsa cevabı ilk cümlede doğrudan ver.
6. Cevap bulunuyorsa asla "{NO_ANSWER_TEXT}" cümlesini ekleme.
7. "Saygılarımla" gibi kapanış ifadeleri ekleme.
8. "Sohbet Geçmişi" bölümünü sadece bağlamı anlamak için kullan; bilgi kaynağı olarak kullanma.

Sohbet Geçmişi:
{memory_text}

Kanitlar:
{context_text}

Öğrenci Sorusu: {query}

Cevap (Türkçe, "Sayın öğrencimiz," ile başla):"""

        payload = {
            "model": MODEL_ADI,
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
