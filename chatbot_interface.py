import sys
import os
import streamlit as st
import requests

# core/ klasörünü Python path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
from chatbot import RAGChatbot

st.set_page_config(
    page_title="Düzce Üni Asistan",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource(show_spinner=False)
def sistemi_yukle():
    return RAGChatbot()


if "chatbot_hazir" not in st.session_state:
    with st.spinner("⏳ Yapay Zeka Modelleri Yükleniyor... (Sonraki açılışlarda bu adım atlanır)"):
        chatbot = sistemi_yukle()
    st.session_state.chatbot_hazir = True
else:
    chatbot = sistemi_yukle()

# ── Yan panel ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/tr/thumb/9/9e/D%C3%BCzce_%C3%9Cniversitesi_logosu.svg/1200px-D%C3%BCzce_%C3%9Cniversitesi_logosu.svg.png",
        width=150
    )
    st.title("TÜBİTAK 2209-A Projesi")
    st.markdown("### ⚙️ Sistem Mimarisi")
    st.markdown("- **Dil Modeli:** Qwen2.5:7b (Ollama)")
    st.markdown("- **Vektör Veritabanı:** ChromaDB (Persistent)")
    st.markdown("- **Arama:** Hibrit (BM25 + Vektör)")
    st.markdown("- **Sıralama:** Cross-Encoder Reranker")
    st.divider()
    st.caption("Düzce Üniversitesi Bilgisayar Mühendisliği")

    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

# ── Ana ekran ─────────────────────────────────────────────────────────────────
st.title("🎓 Öğrenci İşleri Yapay Zeka Asistanı")
st.markdown("*Staj, ders kaydı, sınavlar veya mevzuat hakkında soru sorabilirsiniz.*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🏛️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Örn: Yaz okulu kayıtları ne zaman başlıyor?"):
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Veritabanı taranıyor..."):
            try:
                sonuc = chatbot.chat(prompt)
                cevap = sonuc['cevap']

                kaynaklar_md = "\n\n---\n**📚 Yararlanılan Kaynaklar:**\n"
                eklenen = set()
                for k in sonuc['kaynaklar']:
                    url, kat = k['url'], k['kategori']
                    if url and url not in eklenen:
                        eklenen.add(url)
                        kaynaklar_md += (f"- 🔗 [{kat}]({url})\n"
                                         if url.startswith("http") else f"- 📄 {kat}\n")

                final = cevap + kaynaklar_md if eklenen else cevap
                st.markdown(final)
                st.session_state.messages.append({"role": "assistant", "content": final})

            except requests.exceptions.ConnectionError:
                st.error("❌ Ollama'ya bağlanılamadı. Terminalde 'ollama serve' çalıştırın.")
            except Exception as e:
                st.error(f"Sistem Hatası: {str(e)}")
