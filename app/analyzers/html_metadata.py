import re
import time
from urllib.parse import urljoin
import httpx
from bs4 import BeautifulSoup

from app.analyzers.base import BaseAnalyzer
from app.schemas import AnalysisResult, Verdict

class HtmlMetadataAnalyzer(BaseAnalyzer):
    RTA_VALUES = {"adult", "rta-5042-1996-1400-1577-rta", "mature", "restricted"}
    EXPLICIT_KEYWORDS = {
        "porn", "xxx", "sex video", "free porn", "hd porn", "nude", "naked",
        "camgirl", "live sex", "hentai", "erotic", "nsfw", "18+ only",
        "explicit content", "adult video", "blowjob", "milf", "hardcore"
    }

    def __init__(self, timeout_sec: float = 5.0):
        self._name = "tier2_html_metadata"
        self.timeout_sec = timeout_sec
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SentinelClassifier/1.0"
        }

    @property
    def name(self) -> str:
        return self._name

    def analyze(self, url: str) -> AnalysisResult:
        start_time = time.perf_counter()
        normalized_url = url if "://" in url else f"https://{url}"

        try:
            with httpx.Client(timeout=self.timeout_sec, follow_redirects=True, headers=self.headers) as client:
                response = client.get(normalized_url)
        except Exception as exc:
            return self._build_result(start_time, Verdict.UNCERTAIN, 0.50, {"error": str(exc), "candidate_images": []})

        # 1. Check HTTP Response Header 'Rating: RTA-5042-1996-1400-1577-RTA'
        header_rating = response.headers.get("Rating", "").strip().lower()
        if header_rating in self.RTA_VALUES:
            return self._build_result(start_time, Verdict.NSFW, 1.0, {"reason": "http_rta_header", "candidate_images": []})

        soup = BeautifulSoup(response.text, "html.parser")

        # 2. Check <meta name="rating" content="adult"> or RTA label
        for meta in soup.find_all("meta"):
            meta_name = (meta.get("name") or meta.get("http-equiv") or "").strip().lower()
            meta_content = (meta.get("content") or "").strip().lower()
            if meta_name == "rating" and meta_content in self.RTA_VALUES:
                return self._build_result(start_time, Verdict.NSFW, 1.0, {"reason": f"meta_rating_{meta_content}", "candidate_images": []})

        # 3. Extract Title, Meta Description, and H1-H2 text
        title_text = soup.title.string if soup.title and soup.title.string else ""
        desc_meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        desc_text = desc_meta.get("content", "") if desc_meta else ""
        headings = " ".join(h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2"])[:10])

        corpus = f"{title_text} {desc_text} {headings}".lower()
        matched_keywords = [kw for kw in self.EXPLICIT_KEYWORDS if kw in corpus]

        # 4. Extract candidate image URLs for Tier 3 fallback
        candidate_images = self._extract_candidate_images(soup, str(response.url))

        if len(matched_keywords) >= 3:
            prob = min(0.80 + 0.05 * len(matched_keywords), 0.97)
            return self._build_result(
                start_time, Verdict.NSFW, prob,
                {"reason": "high_explicit_keyword_density", "matches": matched_keywords, "candidate_images": candidate_images}
            )
        elif len(matched_keywords) in (1, 2):
            return self._build_result(
                start_time, Verdict.UNCERTAIN, 0.55,
                {"reason": "ambiguous_text_signals", "matches": matched_keywords, "candidate_images": candidate_images}
            )

        # If page has minimal text (< 100 chars) but holds images, mark UNCERTAIN so Vision inspects it
        if len(corpus.strip()) < 100 and candidate_images:
            return self._build_result(
                start_time, Verdict.UNCERTAIN, 0.45,
                {"reason": "low_text_image_heavy_page", "candidate_images": candidate_images}
            )

        return self._build_result(
            start_time, Verdict.SAFE, 0.08,
            {"reason": "clean_metadata_and_text", "candidate_images": candidate_images}
        )

    def _extract_candidate_images(self, soup: BeautifulSoup, base_url: str, max_images: int = 3) -> list[str]:
        urls = []
        # Priority 1: OpenGraph thumbnail (og:image)
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            urls.append(urljoin(base_url, og_img["content"]))

        # Priority 2: Content <img> tags (skip tiny icons and SVGs)
        for img in soup.find_all("img", src=True):
            src = img["src"].strip()
            if src.startswith("data:") or src.lower().endswith(".svg"):
                continue
            width = img.get("width")
            if width and width.isdigit() and int(width) < 150:
                continue
            full_url = urljoin(base_url, src)
            if full_url not in urls:
                urls.append(full_url)
            if len(urls) >= max_images:
                break
        return urls

    def _build_result(self, start_time: float, verdict: Verdict, prob: float, meta: dict) -> AnalysisResult:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return AnalysisResult(
            analyzer_name=self.name,
            verdict=verdict,
            nsfw_probability=round(prob, 4),
            latency_ms=latency_ms,
            metadata=meta
        )