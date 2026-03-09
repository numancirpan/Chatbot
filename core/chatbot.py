import json
import requests
import os
from typing import List, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_ADI  = "qwen2.5:7b"

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(ROOT_DIR, "data", "chunks.json")
DB_DIR      = os.path.join(ROOT_DIR, "db", "chroma_db")


class BM25Search:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([c['content'].split() for c in chunks])

    def search(self, query: str, k: int = 5) -> List[Dict]:
        scores = self.bm25.get_scores(query.split())
        return [self.chunks[i] for i in scores.argsort()[-k:][::-1]]


class Reranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

    def rerank(self, query: str, chunks: List[Dict], k: int = 5) -> List[Dict]:
        if not chunks:
            return []
        scores = self.model.predict([[query, c['content']] for c in chunks])
        return [c for c, _ in sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:k]]


class RAGChatbot:
    def __init__(self):
        # Chunks
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        if isinstance(self.chunks, dict):
            self.chunks = [self.chunks]

        self.bm25_search = BM25Search(self.chunks)
        self.reranker    = Reranker()

        self.vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

        self._ollama_kontrol()
        print(f"✅ {len(self.chunks)} chunk yüklendi")
        print("✅ BM25 + Reranker + ChromaDB hazır")

    def _ollama_kontrol(self):
        try:
            requests.get("http://localhost:11434", timeout=3)
            print("✅ Ollama çalışıyor")
        except requests.exceptions.ConnectionError:
            print("⚠️  Ollama bulunamadı! 'ollama serve' komutunu çalıştırın.")

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        bm25_results = self.bm25_search.search(query, k=k * 2)

        vector_docs = self.vector_store.similarity_search(query, k=k * 2)
        vector_results = [{
            'content':    doc.page_content,
            'source_url': doc.metadata.get('source_url', ''),
            'kategori':   doc.metadata.get('kategori', '')
        } for doc in vector_docs]

        seen, unique = set(), []
        for r in bm25_results + vector_results:
            url = r.get('source_url', '')
            if url not in seen or url == '':
                seen.add(url)
                unique.append(r)

        return self.reranker.rerank(query, unique, k=k)

    def generate_response(self, query: str, context: List[Dict]) -> str:
        context_text = "\n\n---\n\n".join(c['content'] for c in context)

        prompt = f"""Sen Düzce Üniversitesi Öğrenci İşleri Daire Başkanlığı'nın resmi Türkçe yapay zeka asistanısın.

ZORUNLU KURALLAR:
1. YALNIZCA Türkçe yaz. İngilizce tek kelime bile kullanma.
2. YALNIZCA aşağıdaki "Resmi Belgeler" bölümündeki bilgileri kullan.
3. Belgede olmayan hiçbir bilgiyi uydurma, tahmin etme veya ekleme.
4. Eğer soru belgelerde geçmiyorsa SADECE şunu yaz: "Bu konuda resmi belgelerde bilgiye ulaşamadım. Lütfen Öğrenci İşleri birimi ile iletişime geçiniz."
5. Cevabın sonunda aynı bilgiyi tekrar etme.

Resmi Belgeler:
{context_text}

Öğrenci Sorusu: {query}

Cevap (Türkçe, "Sayın öğrencimiz," ile başla):"""

        payload = {
            "model": MODEL_ADI,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "repeat_penalty": 1.3,
                "stop": ["Öğrenci Sorusu:", "Resmi Belgeler:"]
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        cevap = response.json()['response'].strip()

        # İngilizce satır sızarsa temizle
        ingilizce = ["According", "Please", "In your", "Note that",
                     "However", "I would", "You can", "You cannot"]
        satirlar = [s for s in cevap.split('\n')
                    if not any(i in s for i in ingilizce)]
        temiz = '\n'.join(satirlar).strip()

        if len(temiz) < 20:
            temiz = "Bu konuda resmi belgelerde bilgiye ulaşamadım. Lütfen Öğrenci İşleri birimi ile iletişime geçiniz."

        return temiz if temiz.startswith("Sayın") else "Sayın öğrencimiz,\n" + temiz

    def chat(self, query: str) -> Dict:
        results = self.hybrid_search(query, k=3)
        cevap   = self.generate_response(query, results)
        return {
            'query': query,
            'cevap': cevap,
            'kaynaklar': [{
                'kategori': r.get('kategori', 'Genel'),
                'url':      r.get('source_url', '')
            } for r in results]
        }


if __name__ == "__main__":
    bot = RAGChatbot()
    sonuc = bot.chat("Çift Anadal başvuru şartları nelerdir?")
    print(sonuc['cevap'])
