# TEZ4 — Düzce Üniversitesi Öğrenci İşleri RAG Chatbot

## 📁 Klasör Yapısı

```
TEZ4/
├── api.py                  ← FastAPI servis katmanı
├── config.json              ← Taranacak URL'ler
├── chatbot_interface.py     ← Streamlit arayüzü (ANA UYGULAMA)
│
├── data/                    ← Tüm veri dosyaları (otomatik oluşur)
│   ├── knowledge_base.json  ← Ham crawler çıktısı
│   ├── chunks.json          ← İşlenmiş chunk'lar
│   ├── retrieval_finetune_data.json   ← Retrieval eğitim verisi
│   └── generation_finetune_data.json  ← Generation eğitim verisi
│
├── db/
│   └── chroma_db/           ← ChromaDB (otomatik oluşur)
│
├── core/
│   └── chatbot.py           ← RAG motoru
│
├── pipeline/                ← Veri hazırlama (1 kez çalıştırılır)
│   ├── crawler.py           ← Siteyi tara → data/knowledge_base.json
│   ├── smart_chunker.py     ← Temizle & böl → data/chunks.json
│   ├── create_vector_db.py  ← ChromaDB oluştur → db/chroma_db/
│   ├── build_finetune_datasets.py  ← Fine-tune JSON'larını üret
│   └── veri_kalite_test.py  ← Chunk kalitesini puanla
│
└── logs/
    └── crawler_log.txt      ← Crawler logları
```

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimleri Yükle
```bash
pip install streamlit langchain-chroma langchain-huggingface
pip install sentence-transformers rank-bm25 chromadb
pip install requests beautifulsoup4 PyMuPDF python-docx fastapi uvicorn
```

### 2. Ollama'yı Başlat (ayrı terminal)
```bash
ollama serve
ollama pull qwen2.5:7b
```

### 3. Veri Pipeline'ını Çalıştır (sadece ilk kez)
```bash
# Siteyi tara
python pipeline/crawler.py

# Akıllı chunk'lama yap
python pipeline/smart_chunker.py

# ChromaDB oluştur
python pipeline/create_vector_db.py

# Fine-tune veri setlerini üret
python pipeline/build_finetune_datasets.py
```

Not: `db/chroma_db/` Git'e eklenmez. Yeni makinede, DB silindiyse veya ChromaDB kayıt sayısı
`0` görünüyorsa aşağıdaki komutla yerelde yeniden üretin:

```bash
python pipeline/create_vector_db.py --rebuild
```

### 4. Uygulamayı Başlat
```bash
streamlit run chatbot_interface.py
```

Alternatif olarak HTTP API servisini başlatmak için:

```bash
uvicorn api:app --reload
```

### 5. Dataset Audit Çalıştır
```bash
python pipeline/dataset_audit.py
```

Bu komut ham kayıt, chunk ve ChromaDB kayıt sayılarını birlikte gösterir. ChromaDB sayısı
`chunks.json` sayısından farklıysa vektör veritabanını yeniden oluşturun.

### 6. Golden Evaluation Çalıştır
```bash
python pipeline/evaluate_golden.py
```

`data/golden_questions.json` gerçek kullanıcı sorularından oluşan küçük bir regresyon setidir.
Bu test modeli eğitmez; cevapların beklenen bilgi, yasaklı ifade ve kaynak şartlarını sağlayıp
sağlamadığını kontrol eder. Hızlı kontrol için:

```bash
python pipeline/rag_smoke_test.py
```

### 7. Disaridan Hazirlanan Fine-Tune JSON URL'lerini Duzelt

Masaustunde veya baska bir klasorde hazirladiginiz `golden_questions.json`,
`retrieval_finetune_data.json` ve `generation_finetune_data.json` dosyalarindaki
ornek/fake URL'leri, projedeki `data/chunks.json` ile eslestirerek duzeltebilirsiniz:

```bash
python pipeline/suggest_dataset_urls.py ^
  --golden "C:\Users\Esra Kılıç\Desktop\Tübitak\golden_questions.json" ^
  --retrieval "C:\Users\Esra Kılıç\Desktop\Tübitak\retrieval_finetune_data.json" ^
  --generation "C:\Users\Esra Kılıç\Desktop\Tübitak\generation_finetune_data.json"
```

Bu komut orijinal dosyalari bozmaz; yanlarina `_url_suggestions.json` kopyalari uretir.
Eger dogrudan kendi dosyalarinizin ustune yazmak isterseniz `--apply` ekleyin:

```bash
python pipeline/suggest_dataset_urls.py ^
  --golden "C:\Users\Esra Kılıç\Desktop\Tübitak\golden_questions.json" ^
  --retrieval "C:\Users\Esra Kılıç\Desktop\Tübitak\retrieval_finetune_data.json" ^
  --generation "C:\Users\Esra Kılıç\Desktop\Tübitak\generation_finetune_data.json" ^
  --apply
```

Not: Script, her kayda `url_candidates` alanini ekler. Otomatik secim faydali bir baslangictir
ama nihai veri setinden once bu URL'leri gozle kontrol etmeniz onerilir.

## Cevaplama Mimarisi

Sistem model ağırlıklarına mevzuat ezberletmez. Bilgi `data/chunks.json` ve ChromaDB/BM25
arama katmanından gelir. Chatbot cevap üretmeden önce ilgili kaynaklardan kanıt cümlelerini
seçer; LLM yalnızca bu kanıtları resmi ve okunabilir Türkçe cevaba dönüştürmek için kullanılır.
Kanıt bulunamazsa cevap vermek yerine resmi belgelerde bilgiye ulaşılamadığını belirtmelidir.

## 🔄 Veriyi Güncellemek

```bash
python pipeline/crawler.py          # Yeni veriyi çek
python pipeline/smart_chunker.py    # Yeniden chunk'la
python pipeline/create_vector_db.py --rebuild  # DB'yi yenile
```

Kaynak kurallari `source_rules.json` dosyasindan yonetilir.
Bu dosyada yalnizca acikca gurultulu veya alakasiz URL oruntulerini dislayin;
veriyi gereksiz yere daraltmayin.

## 🧪 Chunk Kalitesini Test Et

```bash
# İlk 10 chunk'ı puanla
python pipeline/veri_kalite_test.py 10
```
