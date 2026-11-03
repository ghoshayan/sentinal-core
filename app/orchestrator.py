import io
import time
import httpx
from PIL import Image

from app.analyzers.url_lexical import UrlLexicalAnalyzer
from app.analyzers.html_metadata import HtmlMetadataAnalyzer
from app.analyzers.playwright_dynamic import PlaywrightDynamicAnalyzer
from app.analyzers.vision_onnx import OfflineVisionAnalyzer
from app.schemas import AnalysisResult, PipelineReport, Verdict

class CascadeOrchestrator:
    def __init__(
        self,
        url_analyzer: UrlLexicalAnalyzer,
        html_analyzer: HtmlMetadataAnalyzer,
        vision_analyzer: OfflineVisionAnalyzer,
        playwright_analyzer: PlaywrightDynamicAnalyzer | None = None
    ):
        self.url_analyzer = url_analyzer
        self.html_analyzer = html_analyzer
        self.vision_analyzer = vision_analyzer
        self.playwright_analyzer = playwright_analyzer

    def classify_website(self, url: str) -> PipelineReport:
        pipeline_start = time.perf_counter()
        steps: list[AnalysisResult] = []

        # --- TIER 1: URL & Domain Blocklist ---
        t1_res = self.url_analyzer.analyze(url)
        steps.append(t1_res)
        if t1_res.verdict == Verdict.NSFW:
            return self._finalize(url, t1_res, steps, pipeline_start)

        # --- TIER 2A: Fast Static HTML Metadata & Text (httpx) ---
        t2_res = self.html_analyzer.analyze(url)
        steps.append(t2_res)

        # --- TIER 2B: Dynamic JS Fallback (Playwright) ---
        # Trigger Playwright ONLY if httpx hit an unrendered JS SPA or anti-bot error
        reason = (t2_res.metadata or {}).get("reason", "")
        has_error = "error" in (t2_res.metadata or {})

        if self.playwright_analyzer and (has_error or reason == "low_text_image_heavy_page"):
            t2_res = self.playwright_analyzer.analyze(url)
            steps.append(t2_res)

        # Short-circuit if Tier 2A or Tier 2B reached a definitive SAFE or NSFW verdict
        if t2_res.verdict in (Verdict.NSFW, Verdict.SAFE):
            return self._finalize(url, t2_res, steps, pipeline_start)

        # --- TIER 3: Visual DOM Fallback (Only runs when Tier 2 is UNCERTAIN) ---
        candidate_images = (t2_res.metadata or {}).get("candidate_images", [])
        highest_vision_res: AnalysisResult | None = None

        for img_url in candidate_images:
            img_res = self._analyze_remote_image(img_url)
            if img_res:
                steps.append(img_res)
                if highest_vision_res is None or img_res.nsfw_probability > highest_vision_res.nsfw_probability:
                    highest_vision_res = img_res
                # Early exit on the first high-confidence NSFW image found
                if img_res.verdict == Verdict.NSFW:
                    return self._finalize(url, img_res, steps, pipeline_start)

        final_step = highest_vision_res or t2_res
        return self._finalize(url, final_step, steps, pipeline_start)

    def _analyze_remote_image(self, img_url: str) -> AnalysisResult | None:
        try:
            with httpx.Client(timeout=4.0, follow_redirects=True) as client:
                resp = client.get(img_url)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content))
                result = self.vision_analyzer.analyze(image)
                if result.metadata is not None:
                    result.metadata["image_url"] = img_url
                return result
        except Exception:
            return None

    def _finalize(
        self,
        url: str,
        deciding_result: AnalysisResult,
        steps: list[AnalysisResult],
        start_time: float
    ) -> PipelineReport:
        total_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return PipelineReport(
            target=url,
            final_verdict=deciding_result.verdict,
            confidence=deciding_result.nsfw_probability,
            triggered_by=deciding_result.analyzer_name,
            total_latency_ms=total_ms,
            steps=steps
        )