import re
import time
from urllib.parse import urljoin
import httpx
from bs4 import BeautifulSoup

from app.analyzers.base import BaseAnalyzer
from app.schemas import AnalysisResult, NsfwSubcategory, Verdict
from app.utils.taxonomy import classify_text_taxonomy


class HtmlMetadataAnalyzer(BaseAnalyzer):
    RTA_VALUES = {"adult", "rta-5042-1996-1400-1577-rta", "mature", "restricted"}

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
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AnalysisResult(
                analyzer_name=self.name,
                verdict=Verdict.UNCERTAIN,
                nsfw_probability=0.50,
                latency_ms=latency_ms,
                metadata={"error": str(exc), "candidate_images": []}
            )

        # 1. Check HTTP Response Header 'Rating: RTA-5042-1996-1400-1577-RTA'
        header_rating = response.headers.get("Rating", "").strip().lower()
        if header_rating in self.RTA_VALUES:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AnalysisResult(
                analyzer_name=self.name,
                verdict=Verdict.NSFW,
                nsfw_probability=1.0,
                primary_category=NsfwSubcategory.HARDCORE_EXPLICIT,
                subcategories=[NsfwSubcategory.HARDCORE_EXPLICIT],
                latency_ms=latency_ms,
                metadata={"reason": "http_rta_header", "candidate_images": []}
            )

        soup = BeautifulSoup(response.text, "html.parser")

        # 2. Check <meta name="rating" content="adult"> or RTA label
        for meta in soup.find_all("meta"):
            meta_name = (meta.get("name") or meta.get("http-equiv") or "").strip().lower()
            meta_content = (meta.get("content") or "").strip().lower()
            if meta_name == "rating" and meta_content in self.RTA_VALUES:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return AnalysisResult(
                    analyzer_name=self.name,
                    verdict=Verdict.NSFW,
                    nsfw_probability=1.0,
                    primary_category=NsfwSubcategory.HARDCORE_EXPLICIT,
                    subcategories=[NsfwSubcategory.HARDCORE_EXPLICIT],
                    latency_ms=latency_ms,
                    metadata={"reason": f"meta_rating_{meta_content}", "candidate_images": []}
                )

        # 3. Extract Title, Meta Description, H1-H3, and Body Snippet
        title_text = soup.title.string if soup.title and soup.title.string else ""
        desc_meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        desc_text = desc_meta.get("content", "") if desc_meta else ""
        headings = " ".join(h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])[:15])
        body_snippet = soup.get_text(" ", strip=True)[:4000]

        corpus = f"{title_text} {desc_text} {headings} {body_snippet}"
        primary_cat, subcats, cat_scores, matched_keywords = classify_text_taxonomy(corpus)

        # 4. Extract candidate image URLs for Tier 3 Vision fallback
        candidate_images = self._extract_candidate_images(soup, str(response.url))
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Stage 1 (Is it NSFW?) + Stage 2 (Which specific category?)
        if len(matched_keywords) >= 2:
            prob = min(0.80 + 0.04 * len(matched_keywords), 0.98)
            return AnalysisResult(
                analyzer_name=self.name,
                verdict=Verdict.NSFW,
                nsfw_probability=round(prob, 4),
                primary_category=primary_cat,
                subcategories=subcats,
                category_scores=cat_scores,
                latency_ms=latency_ms,
                metadata={
                    "reason": "taxonomy_keyword_density",
                    "matches": matched_keywords,
                    "candidate_images": candidate_images
                }
            )
        elif len(matched_keywords) == 1:
            return AnalysisResult(
                analyzer_name=self.name,
                verdict=Verdict.UNCERTAIN,
                nsfw_probability=0.60,
                primary_category=primary_cat,
                subcategories=subcats,
                category_scores=cat_scores,
                latency_ms=latency_ms,
                metadata={
                    "reason": "single_taxonomy_match_needs_vision",
                    "matches": matched_keywords,
                    "candidate_images": candidate_images
                }
            )

        # Preserve trigger for Playwright (Tier 2B) when page is an empty JS shell (< 100 chars)
        if len(corpus.strip()) < 100:
            return AnalysisResult(
                analyzer_name=self.name,
                verdict=Verdict.UNCERTAIN,
                nsfw_probability=0.45,
                latency_ms=latency_ms,
                metadata={"reason": "low_text_image_heavy_page", "candidate_images": candidate_images}
            )

        # If page has images, return UNCERTAIN so Tier 3 Vision inspects them
        if candidate_images:
            return AnalysisResult(
                analyzer_name=self.name,
                verdict=Verdict.UNCERTAIN,
                nsfw_probability=0.35,
                latency_ms=latency_ms,
                metadata={"reason": "no_text_hits_verify_images", "candidate_images": candidate_images}
            )

        return AnalysisResult(
            analyzer_name=self.name,
            verdict=Verdict.SAFE,
            nsfw_probability=0.05,
            latency_ms=latency_ms,
            metadata={"reason": "clean_text_and_no_images", "candidate_images": []}
        )

    def _extract_candidate_images(self, soup: BeautifulSoup, base_url: str, max_images: int = 3) -> list[str]:
        urls = []
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            urls.append(urljoin(base_url, og_img["content"]))

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