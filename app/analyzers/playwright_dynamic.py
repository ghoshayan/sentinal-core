import time
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

from app.analyzers.base import BaseAnalyzer
from app.analyzers.html_metadata import HtmlMetadataAnalyzer
from app.schemas import AnalysisResult, Verdict, NsfwSubcategory
from app.utils.taxonomy import classify_text_taxonomy

class PlaywrightDynamicAnalyzer(BaseAnalyzer):
    """
    Heavyweight Headless Browser Analyzer.
    Only invoked by the Orchestrator when fast httpx scraping hits a blank JS SPA or bot challenge.
    """
    def __init__(self, timeout_ms: int = 8000):
        self._name = "tier2b_playwright_dynamic"
        self.timeout_ms = timeout_ms
        # Reuse the HTML parsing and keyword/image extraction logic from HtmlMetadataAnalyzer
        self._html_helper = HtmlMetadataAnalyzer()

    @property
    def name(self) -> str:
        return self._name

    def analyze(self, url: str) -> AnalysisResult:
        start_time = time.perf_counter()
        normalized_url = url if "://" in url else f"https://{url}"

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
                    viewport={"width": 1280, "height": 800}
                )
                page = context.new_page()

                # Speed optimization: Abort heavy media, fonts, and CSS downloads; we only need JS + DOM
                page.route(
                    "**/*.{mp4,webm,avi,mov,woff,woff2,ttf,css}",
                    lambda route: route.abort()
                )

                page.goto(normalized_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                # Brief wait for SPA frameworks (React/Next/Vue) to hydrate the DOM
                page.wait_for_timeout(800)

                rendered_html = page.content()
                final_url = page.url
                browser.close()
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AnalysisResult(
                analyzer_name=self.name,
                verdict=Verdict.UNCERTAIN,
                nsfw_probability=0.50,
                latency_ms=latency_ms,
                metadata={"error": str(exc), "candidate_images": []}
            )

        # Parse the fully JavaScript-rendered DOM
        soup = BeautifulSoup(rendered_html, "html.parser")
        title_text = soup.title.string if soup.title and soup.title.string else ""
        body_text = soup.get_text(" ", strip=True)[:5000]

        primary_cat, subcats, cat_scores, matched_keywords = classify_text_taxonomy(
            f"{title_text} {body_text}"
        )
        candidate_images = self._html_helper._extract_candidate_images(soup, final_url)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if len(matched_keywords) >= 2:
            prob = min(0.80 + 0.04 * len(matched_keywords), 0.98)
            verdict = Verdict.NSFW
        elif len(matched_keywords) == 1 or candidate_images:
            prob = 0.55
            verdict = Verdict.UNCERTAIN
        else:
            prob = 0.10
            verdict = Verdict.SAFE

        return AnalysisResult(
            analyzer_name=self.name,
            verdict=verdict,
            nsfw_probability=round(prob, 4),
            primary_category=primary_cat,
            subcategories=subcats,
            category_scores=cat_scores,
            latency_ms=latency_ms,
            metadata={
                "title": title_text,
                "matched_keywords": matched_keywords,
                "candidate_images": candidate_images
            }
        )