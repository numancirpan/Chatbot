"""
veri_kalite_test.py  (pipeline/veri_kalite_test.py)

chunks.json içindeki chunk'ları Ollama ile kalite puanlaması yapar.
Kullanım: python veri_kalite_test.py [kaç_chunk_test_edilsin, default=5]
"""

import json
import sys
import os
import requests

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(ROOT_DIR, "data", "chunks.json")
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_ADI   = "qwen2.5:7b"


def kalite_degerlendir(n: int = 5):
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"Toplam {len(chunks)} chunk. İlk {n} tanesi test ediliyor...\n")

    for i, item in enumerate(chunks[:n], 1):
        metin   = item.get('content', '')[:1000]
        url     = item.get('source_url', 'Bilinmiyor')
        kat     = item.get('kategori', 'Bilinmiyor')
        tip     = item.get('chunk_tipi', 'Bilinmiyor')

        prompt = f"""Sen bir veri mühendisisin. Aşağıdaki metin bir RAG sisteminin veritabanına eklenecek.
Bu metnin bilgi değerini 1-10 arasında puanla.

Kriterler:
- Sadece menü/buton isimleri içeriyorsa: 1-4
- Kısmi veya kopuk bilgi içeriyorsa: 5-7
- Öğrenci sorusuna doğrudan cevap olabilecek net bilgi içeriyorsa: 8-10

Metin: "{metin}"

Sadece şu formatta cevap ver:
PUAN: [sayı]
SEBEP: [tek Türkçe cümle]"""

        try:
            print(f"[{i}] Kategori: {kat} | Tip: {tip}")
            print(f"     URL: {url[:60]}")
            r = requests.post(OLLAMA_URL, json={
                "model": MODEL_ADI, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.1}
            }, timeout=60)
            print(f"     {r.json()['response'].strip()}\n" + "-"*50)
        except Exception as e:
            print(f"     Hata: {e}\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    kalite_degerlendir(n)
