# TEZ4 — Düzce Üniversitesi Öğrenci İşleri RAG Chatbot

## 📁 Klasör Yapısı

```
TEZ4/
├── config.json              ← Taranacak URL'ler
├── chatbot_interface.py     ← Streamlit arayüzü (ANA UYGULAMA)
│
├── data/                    ← Tüm veri dosyaları (otomatik oluşur)
│   ├── knowledge_base.json  ← Ham crawler çıktısı
│   └── chunks.json          ← İşlenmiş chunk'lar
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
pip install requests beautifulsoup4 PyMuPDF python-docx
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
```

### 4. Uygulamayı Başlat
```bash
streamlit run chatbot_interface.py
```

## 🔄 Veriyi Güncellemek

```bash
python pipeline/crawler.py          # Yeni veriyi çek
python pipeline/smart_chunker.py    # Yeniden chunk'la
python pipeline/create_vector_db.py --rebuild  # DB'yi yenile
```

## 🧪 Chunk Kalitesini Test Et

```bash
# İlk 10 chunk'ı puanla
python pipeline/veri_kalite_test.py 10
```
