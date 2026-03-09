import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Set
import urllib3
import hashlib
from urllib.parse import urljoin
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT_DIR, "data")
LOGS_DIR  = os.path.join(ROOT_DIR, "logs")
CONFIG_FILE  = os.path.join(ROOT_DIR, "config.json")
OUTPUT_FILE  = os.path.join(DATA_DIR, "knowledge_base.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'crawler_log.txt'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class AdvancedUniversityCrawler:
    def __init__(self):
        self.config = self._load_config()
        self.results = []
        self.failed_urls = []
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.previous_data = self._load_previous_data()
        self.base_urls = self._extract_base_urls()

    def _load_config(self) -> Dict:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_previous_data(self) -> Dict:
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                return {e['url']: e for e in json.load(f) if 'url' in e}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _extract_base_urls(self) -> List[str]:
        seen = []
        for urls in self.config.values():
            for url in urls:
                base = url.split('/sayfa/')[0].split('/tr/Sayfa/')[0]
                if base not in seen:
                    seen.append(base)
        return seen

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _fetch(self, url: str) -> str:
        try:
            r = self.session.get(url, timeout=15, verify=False, allow_redirects=True)
            r.raise_for_status()
            return r.text
        except Exception as e:
            logging.warning(f"Fetch hatası: {url} - {str(e)[:50]}")
            return ""

    def _links(self, html: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for tag in soup.find_all('a', href=True):
            href = tag['href']
            full = href if href.startswith('http') else urljoin(base_url, href)
            fl = full.lower()
            if ('cdn.duzce.edu.tr' in fl or 'getfile' in fl or
                    fl.endswith(('.pdf', '.doc', '.docx')) or
                    any(b in full for b in self.base_urls) or
                    'duzce.edu.tr' in fl):
                links.append(full)
        return list(set(links))

    def _html_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        return ' '.join(soup.get_text(separator=' ', strip=True).split())

    def _pdf_text(self, url: str) -> str:
        if not PDF_SUPPORT:
            return ""
        try:
            r = self.session.get(url, timeout=30, verify=False)
            r.raise_for_status()
            doc = fitz.open(stream=r.content, filetype="pdf")
            return "".join(p.get_text() for p in doc)[:15000]
        except Exception as e:
            logging.warning(f"PDF hatası: {url} - {str(e)[:50]}")
            return ""

    def _docx_text(self, url: str) -> str:
        if not DOCX_SUPPORT:
            return ""
        try:
            import io
            r = self.session.get(url, timeout=30, verify=False)
            r.raise_for_status()
            doc = Document(io.BytesIO(r.content))
            return "\n".join(p.text for p in doc.paragraphs)[:15000]
        except Exception as e:
            logging.warning(f"DOCX hatası: {url} - {str(e)[:50]}")
            return ""

    def process_url(self, url: str, category: str):
        if url in self.visited_urls:
            return None
        self.visited_urls.add(url)
        logging.info(f"İşleniyor: {url}")

        ct = ""
        try:
            ct = self.session.head(url, timeout=10, verify=False, allow_redirects=True)\
                     .headers.get('Content-Type', '').lower()
        except Exception:
            pass

        ul = url.lower()
        if 'application/pdf' in ct or ul.endswith('.pdf'):
            text, ctype = self._pdf_text(url), "pdf"
        elif 'word' in ct or ul.endswith(('.doc', '.docx')):
            text, ctype = self._docx_text(url), "docx"
        else:
            html = self._fetch(url)
            if not html:
                self.failed_urls.append(url)
                return None
            text, ctype = self._html_text(html), "html"

        if not text or len(text) < 20:
            self.failed_urls.append(url)
            return None

        h = self._hash(text)
        prev = self.previous_data.get(url)
        return {
            "url": url,
            "kategori": category,
            "icerik": text[:15000],
            "icerik_hash": h,
            "cekim_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "durum": "basarili",
            "degisiklik": prev.get('icerik_hash') != h if prev else False,
            "icerik_tipi": ctype
        }

    def _crawl(self, url: str, category: str, depth: int, max_depth: int):
        if depth > max_depth:
            return
        result = self.process_url(url, category)
        if result:
            self.results.append(result)
        if depth < max_depth and (not result or result.get('icerik_tipi') == 'html'):
            html = self._fetch(url)
            if html:
                for link in self._links(html, url):
                    if link not in self.visited_urls:
                        cat = "belgeler" if link.lower().endswith('.pdf') else category
                        time.sleep(0.5)
                        self._crawl(link, cat, depth + 1, max_depth)

    def crawl_all(self, max_depth: int = 3):
        logging.info(f"🚀 Crawler başlatıldı. Derinlik: {max_depth}")
        for category, urls in self.config.items():
            for url in urls:
                self._crawl(url, category, 0, max_depth)

    def save(self):
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logging.info(f"✅ {len(self.results)} kayıt → {OUTPUT_FILE}")

    def stats(self) -> Dict:
        return {
            "toplam": len(self.results),
            "basarisiz": len(self.failed_urls),
            "pdf": len([r for r in self.results if r.get('icerik_tipi') == 'pdf']),
            "degisen": len([r for r in self.results if r.get('degisiklik')]),
        }


if __name__ == "__main__":
    c = AdvancedUniversityCrawler()
    c.crawl_all(max_depth=3)
    c.save()
    print(c.stats())
