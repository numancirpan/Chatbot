"""
smart_chunker.py  (pipeline/smart_chunker.py)

knowledge_base.json → chunks.json

- Navigasyon menüsü gürültüsünü temizler
- Yönetmelik/Yönerge metinlerini madde madde böler
- HTML sayfalarını paragraf sınırında keser
- Çok kısa ve tekrar eden chunk'ları atar
"""

import json
import re
import hashlib
import os
from typing import List, Dict

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
INPUT_FILE  = os.path.join(DATA_DIR, "knowledge_base.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "chunks.json")

MIN_CHUNK = 100
MAX_CHUNK = 1200
CHUNK_OVERLAP = 120

NAV_PATTERNS = [
    r"Düzce Üniversitesi \| Öğrenci İşleri Daire Başkanlığı \|[^\n]*",
    r"Başkanlığımız Hakkımızda Yönetim Organizasyon.*?İletişim",
    r"Anasayfa >\s*\S[^\n]*",
    r"© Copyright.*?2024",
]


# ── Temizleme ────────────────────────────────────────────────────────────────

def temizle(metin: str) -> str:
    for p in NAV_PATTERNS:
        metin = re.sub(p, " ", metin, flags=re.DOTALL | re.IGNORECASE)
    metin = re.sub(r"\s{3,}", "\n\n", metin)
    metin = re.sub(r" {2,}", " ", metin)
    return metin.strip()


def icerik_baslangici(metin: str) -> str:
    m = re.search(r"\n([A-ZÇĞİÖŞÜ\s]{4,})\n", metin)
    if m and m.start() > 50:
        return metin[m.start():].strip()
    return metin


# ── Bölme yardımcıları ───────────────────────────────────────────────────────

def cumle_bol(metin: str, max_uzunluk: int) -> List[str]:
    if len(metin) <= max_uzunluk:
        return [metin]
    parcalar, tampon = [], ""
    for cumle in re.split(r"(?<=[.!?])\s+", metin):
        if tampon and len(tampon) + len(cumle) > max_uzunluk:
            parcalar.append(tampon.strip())
            tampon = cumle
        else:
            tampon = (tampon + " " + cumle).strip() if tampon else cumle
    if tampon:
        parcalar.append(tampon.strip())
    guvenli = []
    for parca in parcalar:
        if len(parca) > max_uzunluk:
            guvenli.extend(sert_bol(parca, max_uzunluk))
        else:
            guvenli.append(parca)
    return [p for p in guvenli if len(p) >= MIN_CHUNK]


def sert_bol(metin: str, max_uzunluk: int) -> List[str]:
    if len(metin) <= max_uzunluk:
        return [metin]

    kelimeler = metin.split()
    if not kelimeler:
        return []

    parcalar = []
    baslangic = 0
    while baslangic < len(kelimeler):
        uzunluk = 0
        bitis = baslangic
        while bitis < len(kelimeler):
            eklenecek = len(kelimeler[bitis]) + (1 if uzunluk else 0)
            if uzunluk + eklenecek > max_uzunluk and bitis > baslangic:
                break
            uzunluk += eklenecek
            bitis += 1

        parca = " ".join(kelimeler[baslangic:bitis]).strip()
        if len(parca) >= MIN_CHUNK:
            parcalar.append(parca)

        if bitis >= len(kelimeler):
            break

        overlap_uzunlugu = 0
        yeni_baslangic = bitis
        while yeni_baslangic > baslangic:
            overlap_uzunlugu += len(kelimeler[yeni_baslangic - 1]) + 1
            if overlap_uzunlugu >= CHUNK_OVERLAP:
                break
            yeni_baslangic -= 1
        baslangic = max(yeni_baslangic, baslangic + 1)
    return parcalar


# ── Chunk stratejileri ───────────────────────────────────────────────────────

def madde_bazli(metin: str, meta: Dict) -> List[Dict]:
    chunks = []
    for parca in re.compile(r"(?=(?:MADDE|Madde)\s+\d+)", re.M).split(metin):
        parca = parca.strip()
        if len(parca) < MIN_CHUNK:
            continue
        m = re.match(r"(?:MADDE|Madde)\s+(\d+)", parca)
        for alt in cumle_bol(parca, MAX_CHUNK):
            chunks.append({
                "content": alt,
                **meta,
                "madde_no": m.group(1) if m else "",
                "chunk_tipi": "yonetmelik_maddesi"
            })
    return chunks or paragraf_bazli(metin, meta)


def paragraf_bazli(metin: str, meta: Dict) -> List[Dict]:
    chunks, tampon = [], ""
    for p in re.split(r"\n{2,}", metin):
        p = p.strip()
        if not p:
            continue
        if len(p) > MAX_CHUNK:
            for alt in cumle_bol(p, MAX_CHUNK) or sert_bol(p, MAX_CHUNK):
                chunks.append({"content": alt, **meta, "chunk_tipi": "paragraf"})
            continue
        if tampon and len(tampon) + len(p) > MAX_CHUNK:
            if len(tampon) >= MIN_CHUNK:
                chunks.append({"content": tampon, **meta, "chunk_tipi": "paragraf"})
            tampon = p
        else:
            tampon = (tampon + "\n\n" + p).strip() if tampon else p
    if len(tampon) >= MIN_CHUNK:
        if len(tampon) > MAX_CHUNK:
            for alt in cumle_bol(tampon, MAX_CHUNK) or sert_bol(tampon, MAX_CHUNK):
                chunks.append({"content": alt, **meta, "chunk_tipi": "paragraf"})
        else:
            chunks.append({"content": tampon, **meta, "chunk_tipi": "paragraf"})
    return chunks


def duyuru_bazli(metin: str, meta: Dict) -> List[Dict]:
    chunks = []
    for p in cumle_bol(metin, MAX_CHUNK):
        chunks.append({"content": p, **meta, "chunk_tipi": "duyuru"})
    return chunks


# ── Kategori tespiti ─────────────────────────────────────────────────────────

def kategori_tespit(url: str, mevcut: str) -> str:
    ul = url.lower()
    if any(k in ul for k in ["yonetmelik", "yonerge", "mevzuat"]): return "yonetmelik"
    if any(k in ul for k in ["staj", "mesleki-egitim"]):            return "staj"
    if any(k in ul for k in ["yaz-okulu", "yaz_okulu"]):            return "yaz_okulu"
    if any(k in ul for k in ["duyuru", "haber"]):                   return "duyuru"
    if any(k in ul for k in ["takvim"]):                            return "akademik_takvim"
    if any(k in ul for k in ["cap", "yandal", "cift-anadal"]):      return "cap_yandal"
    if mevcut not in {"", "genel", "belgeler"}:                      return mevcut
    if "pdf" in ul or "getfile" in ul:                               return "belge_pdf"
    return mevcut


# ── Ana işlem ────────────────────────────────────────────────────────────────

def isle(entry: Dict) -> List[Dict]:
    ham    = entry.get("icerik", "").strip()
    url    = entry.get("url", "")
    ctype  = entry.get("icerik_tipi", "html")
    tarih  = entry.get("cekim_tarihi", "")
    ham_kat = entry.get("kategori", "genel")

    if not ham or len(ham) < MIN_CHUNK:
        return []

    temiz = temizle(ham)
    if ctype == "html":
        temiz = icerik_baslangici(temiz)
    if not temiz or len(temiz) < MIN_CHUNK:
        return []

    kat = kategori_tespit(url, ham_kat)
    meta = {"source_url": url, "kategori": kat,
            "icerik_tipi": ctype, "cekim_tarihi": tarih}

    if re.search(r"(?:MADDE|Madde)\s+\d+", temiz):
        return madde_bazli(temiz, meta)
    elif kat == "duyuru" and len(temiz) < MAX_CHUNK * 2:
        return duyuru_bazli(temiz, meta)
    else:
        return paragraf_bazli(temiz, meta)


def tekrar_kaldir(chunks: List[Dict]) -> List[Dict]:
    goruldu, temiz = set(), []
    for c in chunks:
        h = hashlib.md5(f"{c.get('source_url', '')}\n{c['content']}".encode("utf-8")).hexdigest()
        if h not in goruldu:
            goruldu.add(h)
            c["chunk_id"] = h
            c["source_hash"] = hashlib.md5(c.get("source_url", "").encode("utf-8")).hexdigest()
            temiz.append(c)
    return temiz


# ── Çalıştır ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("AKILLI CHUNK'LAMA")
    print("=" * 55)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        kayitlar = json.load(f)
    print(f"{len(kayitlar)} kayit yuklendi")

    tum_chunks, sayac = [], {}
    for kayit in kayitlar:
        for c in isle(kayit):
            tum_chunks.append(c)
            sayac[c["kategori"]] = sayac.get(c["kategori"], 0) + 1

    once = len(tum_chunks)
    tum_chunks = tekrar_kaldir(tum_chunks)
    print(f"{once - len(tum_chunks)} tekrar kaldirildi")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(tum_chunks, f, ensure_ascii=False, indent=2)

    print(f"{len(tum_chunks)} chunk -> {OUTPUT_FILE}")
    print("\nKategori dagilimi:")
    for k, v in sorted(sayac.items(), key=lambda x: -x[1]):
        print(f"   {k:<25} {v}")
