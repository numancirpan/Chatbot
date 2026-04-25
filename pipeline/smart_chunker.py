"""
smart_chunker.py

knowledge_base.json -> chunks.json

- HTML/PDF iceriginden menu ve yan panel gurultusunu temizler
- Belge basligini icerikten cikarmaya calisir
- Yonetmelik/Yonerge metinlerini madde bazli boler
- Supheli ve anlamsiz chunk'lari olusmadan eler
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, List
from urllib.parse import unquote, urlparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "knowledge_base.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "chunks.json")

MIN_CHUNK = 120
MAX_CHUNK = 1200
CHUNK_OVERLAP = 120
MAX_TITLE_LENGTH = 140

LINE_NOISE_PATTERNS = [
    r"^anasayfa$",
    r"^baskanligimiz hakkimizda$",
    r"^yonetim organizasyon semasi$",
    r"^kalite$",
    r"^kalite komisyon u(y|i)eleri(miz)?$",
    r"^birim ic degerlendirme raporlari$",
    r"^sayisal bilgiler.*$",
    r"^stratejik amac ve hedefler$",
    r"^personel$",
    r"^akademik personel$",
    r"^idari personel$",
    r"^bolum sekreterligi$",
    r"^bolum baskaninin mesaji$",
    r"^laboratuvarlar$",
    r"^misyon ve vizyon$",
    r"^tarihce$",
    r"^iletisim$",
    r"^formlar$",
    r"^akademik birim formlari$",
    r"^ogrenci formlari$",
    r"^ogretim programi degisikligi formlari$",
    r"^yonetim$",
    r"^dekanlik$",
    r"^fakulte kurulu kararlari$",
    r"^yonetim kurulu kararlari$",
    r"^kurumsal gorev tanimlari.*$",
    r"^is akis surecleri$",
    r"^hassas gorevler$",
    r"^faaliyet raporu$",
    r"^hizmet standartlari.*$",
    r"^ic kontrol$",
    r"^ogrenci temsilcilikleri konseyi$",
    r"^aday muhendis programi$",
    r"^acik kaynak kodlu yazilimlar$",
    r"^obs$",
    r"^unite101$",
    r"^sss$",
    r"^oneri istek sikayet$",
    r"^ogrenci isleri hakk.*$",
    r"^sinavlarda uygulanacak kurallar$",
    r"^duzce universitesi$",
]

BLOCK_NOISE_PATTERNS = [
    r"baskanligimiz hakkimizda.*?iletisim",
    r"kurumsal gorev tanimlari.*?hizmet standartlari",
    r"yonetim dekanlik.*?bolumler",
]

TOPIC_KEYWORDS = {
    "yaz_okulu": ["yaz okulu", "yaz okulunda", "yaz okulu yonergesi", "yaz okulu egitimi"],
    "staj": ["staj", "sbs", "staj yonergesi", "stajlar hakkinda sikca sorulan sorular", "yaz staji"],
    "cap_yandal": ["cift anadal", "cap", "yandal"],
    "mezuniyet": ["mezuniyet", "diploma", "gecici mezuniyet belgesi"],
    "muafiyet_intibak": ["muafiyet", "intibak", "ders saydirma"],
    "yatay_gecis": ["yatay gecis", "kurum ici", "kurumlararasi"],
    "harc_ucret": ["harc", "ogrenim ucreti", "katki payi"],
}
SKIP_URL_FRAGMENTS = [
    "/sayfa/b149/mevzuat",
    "/duyurular",
    "/sayfa/bd90/pasaport-slemleri",
]


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
    text = re.sub(r"[^a-z0-9\s./:-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_noise_line(line: str) -> bool:
    normalized = normalize_text(line)
    if not normalized:
        return True
    if len(normalized) <= 2:
        return True
    if re.fullmatch(r"(pzt|sal|car|per|cum|cmt|paz)", normalized):
        return True
    if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", normalized):
        return True
    if re.fullmatch(r"\d+", normalized):
        return True
    if "copyright" in normalized:
        return True
    for pattern in LINE_NOISE_PATTERNS:
        if re.fullmatch(pattern, normalized):
            return True
    return False


def split_line_fragments(raw_line: str) -> List[str]:
    parts = re.split(r"\s{2,}|[•\u2022]+|\t+", raw_line)
    fragments = []
    for part in parts:
        piece = " ".join(part.split()).strip(" -:|")
        if piece:
            fragments.append(piece)
    return fragments or [" ".join(raw_line.split()).strip()]


def slug_candidates(url: str) -> List[str]:
    path = unquote(urlparse(url).path or "")
    bits = [bit for bit in path.split("/") if bit and bit != "sayfa"]
    candidates: List[str] = []
    for bit in bits:
        if re.fullmatch(r"[0-9a-f]{3,}", bit.lower()):
            continue
        phrase = bit.replace("-", " ").replace("_", " ").strip()
        if len(normalize_text(phrase)) >= 8:
            candidates.append(phrase)
    return candidates


def trim_to_relevant_start(text: str, url: str) -> str:
    normalized_text = normalize_text(text)
    best_index = None

    url_lower = url.lower()
    special_markers = []
    if "bm.mf.duzce.edu.tr/sayfa/878b" in url_lower:
        special_markers = [
            "ulusal staj programi kapsaminda",
            "1 ogrenci staj dersine daha onceden",
            "staj dersi ders planinda yer alan donemden",
            "stajini tamamlamayan ogrenci mezuniyet",
            "sbs den doldurup cikarttigimiz belgeyi",
        ]
    elif "bm.mf.duzce.edu.tr/sayfa/17ac" in url_lower:
        special_markers = [
            "yaz ogretiminde esdeger ders degerlendirmesi",
            "bolumumuz tarafindan duyurulan",
        ]
    elif "mf.duzce.edu.tr/sayfa/967a" in url_lower:
        special_markers = [
            "yaz staji donemleri icin tiklayiniz",
            "devlet katkisi formlari",
        ]

    for marker in special_markers:
        idx = normalized_text.find(marker)
        if idx >= 300:
            best_index = idx
            break

    for candidate in slug_candidates(url):
        normalized_candidate = normalize_text(candidate)
        if len(normalized_candidate) < 8:
            continue
        for match in re.finditer(re.escape(normalized_candidate), normalized_text):
            if match.start() >= 600:
                best_index = match.start()
                break
        if best_index is not None:
            break

    if best_index is None:
        for marker in [
            "2026 2027 egitim ogretim yili",
            "2025 2026 egitim ogretim yili",
            "staj raporu",
            "staj kabul",
            "ulusal staj programi kapsaminda",
            "1 ogrenci staj dersine",
            "staj dersi ders planinda",
            "yaz okulu acan akademik birim sayisi",
            "devlet katkisi formlari",
            "mezun olmaya hak kazanir",
            "ogrencinin staj bitis tarihini takiben",
        ]:
            idx = normalized_text.find(marker)
            if idx >= 600:
                best_index = idx
                break

    if best_index is None:
        return text

    approx_ratio = best_index / max(len(normalized_text), 1)
    raw_start = int(len(text) * approx_ratio)
    return text[max(0, raw_start - 120):].strip()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    kept_lines = []
    for raw_line in text.splitlines():
        for line in split_line_fragments(raw_line):
            if is_noise_line(line):
                continue
            kept_lines.append(line)

    cleaned = "\n".join(kept_lines)
    for pattern in BLOCK_NOISE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def extract_title(text: str, url: str, kategori: str, icerik_tipi: str) -> str:
    lines = [" ".join(line.split()).strip() for line in text.splitlines()]
    candidates = []
    for line in lines[:25]:
        if not line or is_noise_line(line):
            continue
        normalized = normalize_text(line)
        if len(normalized) < 4:
            continue
        if len(line) > MAX_TITLE_LENGTH:
            continue
        candidates.append(line)

    url_titles = slug_candidates(url)
    if "/duyuru/" in url and url_titles:
        return url_titles[0].title()
    if not candidates:
        if url_titles:
            return url_titles[0].title()
        return kategori.replace("_", " ").title()

    joined = " ".join(candidates[:4] + url_titles[:2])
    normalized_joined = normalize_text(joined)
    normalized_url_titles = normalize_text(" ".join(url_titles))

    if any(marker in normalized_url_titles for marker in ["cift anadal", "yandal"]):
        return "CAP ve Yandal"
    if "yaz okulu" in normalized_url_titles and "bilgisayar muhendisligi" in normalized_joined:
        return "Bilgisayar Muhendisligi - Yaz Okulu"
    if "yaz okulu yonergesi" in normalized_joined:
        return "Yaz Okulu Yonergesi"
    if "yaz okulu egitimi" in normalized_joined:
        return "Yaz Okulu Egitimi"
    if "stajlar hakkinda sikca sorulan sorular" in normalized_joined:
        return "Bilgisayar Muhendisligi - Staj SSS"
    if "bilgisayar muhendisligi" in normalized_joined and "staj" in normalized_joined:
        return "Bilgisayar Muhendisligi - Staj"
    if "yaz staji" in normalized_joined:
        return "Muhendislik Fakultesi - Yaz Staji"
    if "gecici mezuniyet belgesi" in normalized_joined or "diploma" in normalized_joined:
        return "Diploma ve Mezuniyet Belgeleri Yonergesi"
    if "cift anadal" in normalized_joined or "yandal" in normalized_joined:
        return "CAP ve Yandal"
    if "akademik takvim" in normalized_joined:
        return "Akademik Takvim"

    first = candidates[0]
    if icerik_tipi == "html" and len(first.split()) <= 2 and len(candidates) > 1:
        return candidates[1]
    return first


def category_from_content(url: str, mevcut: str, title: str, text: str) -> str:
    ul = url.lower()
    normalized_text_content = normalize_text(f"{title} {text[:1200]} {ul}")

    if "bm.mf.duzce.edu.tr/sayfa/878b" in ul:
        return "staj"
    if "bm.mf.duzce.edu.tr/sayfa/17ac" in ul:
        return "yaz_okulu"
    if "mf.duzce.edu.tr/sayfa/967a" in ul:
        return "staj"
    if mevcut == "duyuru" or "/duyuru/" in ul:
        return "duyuru"

    for kategori, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in normalized_text_content for keyword in keywords):
            return kategori

    if any(k in ul for k in ["yonetmelik", "yonerge", "mevzuat"]):
        return "yonetmelik"
    if any(k in ul for k in ["staj", "mesleki-egitim"]):
        return "staj"
    if any(k in ul for k in ["yaz-okulu", "yaz_okulu"]):
        return "yaz_okulu"
    if any(k in ul for k in ["duyuru", "haber"]):
        return "duyuru"
    if "takvim" in ul:
        return "akademik_takvim"
    if mevcut not in {"", "genel", "belgeler", "merkezi_mevzuat", "fakulte_bolum"}:
        return mevcut
    if "pdf" in ul or "getfile" in ul:
        return "belge_pdf"
    return mevcut or "genel"


def looks_like_calendar_grid(text: str) -> bool:
    normalized = normalize_text(text)
    weekday_hits = sum(day in normalized for day in ["pzt", "sal", "car", "per", "cum", "cmt", "paz"])
    period_hits = len(re.findall(r"\b\d+\. donem\b", normalized))
    date_hits = len(re.findall(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b", text))
    return weekday_hits >= 5 and (period_hits >= 3 or date_hits >= 10)


def noise_score(text: str, source_title: str = "") -> int:
    normalized = normalize_text(text)
    score = 0
    if looks_like_calendar_grid(text):
        score += 8
    if "|" in text:
        score += 2
    if len(re.findall(r"\b(pzt|sal|car|per|cum|cmt|paz)\b", normalized)) >= 5:
        score += 4
    if any(
        marker in normalized
        for marker in ["baskanligimiz hakkimizda", "yonetim organizasyon", "kalite komisyon", "tanitim videosu"]
    ):
        score += 6
    if any(
        marker in normalized
        for marker in ["akademik personel", "idari personel", "bolum baskaninin mesaji", "personel"]
    ):
        score += 5
    if source_title and normalize_text(source_title) in {"fakulte bolum", "ogrenci isleri"} and len(normalized.split()) < 40:
        score += 3
    return score


def should_skip_chunk(text: str, meta: Dict) -> bool:
    normalized = normalize_text(text)
    if len(text.strip()) < MIN_CHUNK:
        return True
    if noise_score(text, meta.get("source_title", "")) >= 7:
        return True
    if normalized.count("bolum") >= 6 and normalized.count("muhendisligi") >= 4 and len(normalized.split()) < 120:
        return True
    if normalized.startswith("pzt sal car per cum"):
        return True
    return False


def split_sentences(text: str, max_length: int) -> List[str]:
    if len(text) <= max_length:
        return [text]
    parts, buffer = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if buffer and len(buffer) + len(sentence) + 1 > max_length:
            parts.append(buffer.strip())
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip() if buffer else sentence
    if buffer:
        parts.append(buffer.strip())
    safe_parts = []
    for part in parts:
        if len(part) > max_length:
            safe_parts.extend(hard_split(part, max_length))
        else:
            safe_parts.append(part)
    return [part for part in safe_parts if len(part) >= MIN_CHUNK]


def hard_split(text: str, max_length: int) -> List[str]:
    if len(text) <= max_length:
        return [text]
    words = text.split()
    if not words:
        return []

    parts = []
    start = 0
    while start < len(words):
        length = 0
        end = start
        while end < len(words):
            extra = len(words[end]) + (1 if length else 0)
            if length + extra > max_length and end > start:
                break
            length += extra
            end += 1
        part = " ".join(words[start:end]).strip()
        if len(part) >= MIN_CHUNK:
            parts.append(part)
        if end >= len(words):
            break
        overlap = 0
        next_start = end
        while next_start > start:
            overlap += len(words[next_start - 1]) + 1
            if overlap >= CHUNK_OVERLAP:
                break
            next_start -= 1
        start = max(next_start, start + 1)
    return parts


def regulation_chunks(text: str, meta: Dict) -> List[Dict]:
    chunks = []
    for section in re.compile(r"(?=(?:MADDE|Madde)\s+\d+)", re.M).split(text):
        section = section.strip()
        if len(section) < MIN_CHUNK:
            continue
        match = re.match(r"(?:MADDE|Madde)\s+(\d+)", section)
        for sub in split_sentences(section, MAX_CHUNK):
            chunk = {
                "content": sub,
                **meta,
                "madde_no": match.group(1) if match else "",
                "chunk_tipi": "yonetmelik_maddesi",
            }
            if not should_skip_chunk(sub, meta):
                chunks.append(chunk)
    return chunks or paragraph_chunks(text, meta)


def paragraph_chunks(text: str, meta: Dict) -> List[Dict]:
    chunks = []
    buffer = ""
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > MAX_CHUNK:
            for sub in split_sentences(paragraph, MAX_CHUNK) or hard_split(paragraph, MAX_CHUNK):
                if not should_skip_chunk(sub, meta):
                    chunks.append({"content": sub, **meta, "chunk_tipi": "paragraf"})
            continue
        if buffer and len(buffer) + len(paragraph) > MAX_CHUNK:
            if len(buffer) >= MIN_CHUNK and not should_skip_chunk(buffer, meta):
                chunks.append({"content": buffer, **meta, "chunk_tipi": "paragraf"})
            buffer = paragraph
        else:
            buffer = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
    if len(buffer) >= MIN_CHUNK and not should_skip_chunk(buffer, meta):
        if len(buffer) > MAX_CHUNK:
            for sub in split_sentences(buffer, MAX_CHUNK) or hard_split(buffer, MAX_CHUNK):
                if not should_skip_chunk(sub, meta):
                    chunks.append({"content": sub, **meta, "chunk_tipi": "paragraf"})
        else:
            chunks.append({"content": buffer, **meta, "chunk_tipi": "paragraf"})
    return chunks


def announcement_chunks(text: str, meta: Dict) -> List[Dict]:
    chunks = []
    for paragraph in split_sentences(text, MAX_CHUNK):
        if not should_skip_chunk(paragraph, meta):
            chunks.append({"content": paragraph, **meta, "chunk_tipi": "duyuru"})
    return chunks


def process_entry(entry: Dict) -> List[Dict]:
    raw = entry.get("icerik", "").strip()
    url = entry.get("url", "")
    content_type = entry.get("icerik_tipi", "html")
    fetched_at = entry.get("cekim_tarihi", "")
    raw_category = entry.get("kategori", "genel")

    if any(fragment in url.lower() for fragment in SKIP_URL_FRAGMENTS):
        return []

    if not raw or len(raw) < MIN_CHUNK:
        return []

    cleaned = trim_to_relevant_start(clean_text(raw), url)
    if not cleaned or len(cleaned) < MIN_CHUNK:
        return []

    source_title = extract_title(cleaned, url, raw_category, content_type)
    category = category_from_content(url, raw_category, source_title, cleaned)

    meta = {
        "source_url": url,
        "kategori": category,
        "source_title": source_title,
        "icerik_tipi": content_type,
        "cekim_tarihi": fetched_at,
    }

    if re.search(r"(?:MADDE|Madde)\s+\d+", cleaned):
        return regulation_chunks(cleaned, meta)
    if category == "duyuru" and len(cleaned) < MAX_CHUNK * 2:
        return announcement_chunks(cleaned, meta)
    return paragraph_chunks(cleaned, meta)


def dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    cleaned = []
    for chunk in chunks:
        normalized_source = chunk.get("source_url", "").strip().rstrip("/").lower()
        normalized_content = " ".join(chunk["content"].split())
        fingerprint = hashlib.md5(f"{normalized_source}\n{normalized_content}".encode("utf-8")).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        chunk["chunk_id"] = fingerprint
        chunk["source_hash"] = hashlib.md5(normalized_source.encode("utf-8")).hexdigest()
        cleaned.append(chunk)
    return cleaned


if __name__ == "__main__":
    print("=" * 55)
    print("AKILLI CHUNK'LAMA")
    print("=" * 55)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"{len(records)} kayit yuklendi")

    all_chunks: List[Dict] = []
    category_counter: Dict[str, int] = {}
    for record in records:
        produced = process_entry(record)
        all_chunks.extend(produced)
        for chunk in produced:
            category_counter[chunk["kategori"]] = category_counter.get(chunk["kategori"], 0) + 1

    before = len(all_chunks)
    all_chunks = dedupe_chunks(all_chunks)
    print(f"{before - len(all_chunks)} tekrar kaldirildi")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"{len(all_chunks)} chunk -> {OUTPUT_FILE}")
    print("\nKategori dagilimi:")
    for key, value in sorted(category_counter.items(), key=lambda item: -item[1]):
        print(f"   {key:<25} {value}")
