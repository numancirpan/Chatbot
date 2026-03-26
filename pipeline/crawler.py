import requests
from bs4 import BeautifulSoup
import argparse
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Set
import urllib3
import hashlib
import re
from urllib.parse import urljoin, urlparse, urldefrag
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
RULES_FILE   = os.path.join(ROOT_DIR, "source_rules.json")
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
        self.rules = self._load_rules()
        self.results = []
        self.failed_urls = []
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.previous_data = self._load_previous_data()
        self.base_urls = self._extract_base_urls()
        self.seed_urls = self._extract_seed_urls()
        self.seed_hosts = self._extract_seed_hosts()
        self.allowed_document_hosts = {
            "cdn.duzce.edu.tr",
            "panel.duzce.edu.tr",
            "w3.api.duzce.edu.tr",
        }
        self.focus_keywords = [
            "ogrenci-form", "ogrenci-bilgi", "ogrenci-degisim",
            "ogrenci-eposta", "staj", "yaz-okulu", "yaz_okulu",
            "mevzuat", "yonetmelik", "yonerge", "duyuru", "takvim",
            "cap", "cift-anadal", "cift_anadal", "yandal", "yatay",
            "muafiyet", "intibak", "ders-kay", "sinav", "kayit",
            "mezun", "diploma", "harc", "sss", "sikca-sorulan",
            "sikca_sorulan", "form", "pasaport", "onemli-basvuru",
        ]
        self.exclude_keywords = [
            "kalite", "komisyon", "toplanti", "organizasyon",
            "stratejik-plan", "faaliyet-raporu", "hakkimizda",
            "personel", "akreditasyon", "lab", "laboratuvar",
            "gallery", "galeri", "news-detail", "haber-detay",
            "yonetim", "gorev", "iletisim", "faydali-baglanti",
            "ic-kontrol", "degerlendirme-raporu", "derslik-kapasiteleri",
            "yks-sonuc-raporlari", "hassas-gorev",
        ]

    def _load_config(self) -> Dict:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_rules(self) -> Dict:
        try:
            with open(RULES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "exclude_exact_urls": [],
                "exclude_url_contains": [],
                "exclude_host_path_contains": {}
            }

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

    def _extract_seed_urls(self) -> Set[str]:
        seed_urls: Set[str] = set()
        for urls in self.config.values():
            for url in urls:
                seed_urls.add(url.rstrip('/'))
        return seed_urls

    def _extract_seed_hosts(self) -> Set[str]:
        hosts: Set[str] = set()
        for url in self.seed_urls:
            host = urlparse(url).netloc.lower()
            if host:
                hosts.add(host)
        return hosts

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _is_excluded_by_rules(self, url: str, path_text: str) -> bool:
        normalized_url = url.rstrip('/').lower()
        if normalized_url in {
            item.rstrip('/').lower()
            for item in self.rules.get("exclude_exact_urls", [])
        }:
            return True

        if any(token.lower() in normalized_url for token in self.rules.get("exclude_url_contains", [])):
            return True

        host = urlparse(url).netloc.lower()
        host_rules = self.rules.get("exclude_host_path_contains", {}).get(host, [])
        if any(token.lower() in path_text for token in host_rules):
            return True

        return False

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
            full = urldefrag(full)[0].rstrip('/')
            if self._is_allowed_link(full, base_url):
                links.append(full)
        return list(set(links))

    def _is_allowed_link(self, url: str, source_url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False

        normalized_url = url.rstrip('/')
        host = parsed.netloc.lower()
        lower_url = normalized_url.lower()
        searchable_text = f"{parsed.path}?{parsed.query}".lower()
        if normalized_url in self.seed_urls:
            return True

        if self._is_excluded_by_rules(normalized_url, searchable_text):
            return False

        if any(keyword in searchable_text for keyword in self.exclude_keywords):
            return False

        is_document = lower_url.endswith(('.pdf', '.doc', '.docx')) or 'getfile' in lower_url
        if is_document:
            return host in self.seed_hosts or host in self.allowed_document_hosts

        if host not in self.seed_hosts and host not in self.allowed_document_hosts:
            return False

        return any(keyword in searchable_text for keyword in self.focus_keywords)

    def _html_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'form']):
            tag.decompose()
        root = (
            soup.find('main')
            or soup.find('article')
            or soup.find(id=re.compile(r'(content|main|article)', re.I))
            or soup.body
            or soup
        )

        for br in root.find_all('br'):
            br.replace_with('\n')

        blocks = []
        for tag in root.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'tr']):
            text = ' '.join(tag.get_text(separator=' ', strip=True).split())
            if len(text) >= 20:
                blocks.append(text)

        if blocks:
            return '\n\n'.join(blocks)

        return '\n\n'.join(
            line.strip()
            for line in root.get_text(separator='\n', strip=True).splitlines()
            if line.strip()
        )

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
        url = urldefrag(url)[0].rstrip('/')
        searchable_text = f"{urlparse(url).path}?{urlparse(url).query}".lower()
        if self._is_excluded_by_rules(url, searchable_text):
            return None
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
                        time.sleep(0.2)
                        self._crawl(link, category, depth + 1, max_depth)

    def crawl_all(self, max_depth: int = 2):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=2)
    args = parser.parse_args()

    c = AdvancedUniversityCrawler()
    c.crawl_all(max_depth=args.max_depth)
    c.save()
    print(c.stats())
