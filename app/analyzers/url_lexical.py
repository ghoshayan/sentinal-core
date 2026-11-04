import re
import time
from pathlib import Path
from urllib.parse import urlparse

from app.analyzers.base import BaseAnalyzer
from app.schemas import AnalysisResult, Verdict, NsfwSubcategory
from app.utils.taxonomy import classify_text_taxonomy

class UrlLexicalAnalyzer(BaseAnalyzer):
    ADULT_TLDS = {".xxx", ".porn", ".adult", ".sex", ".cam", ".tube"}
    HIGH_RISK_TOKENS = {
        "pornhub", "xvideos", "xnxx", "xhamster", "redtube", "youporn",
        "onlyfans", "chaturbate", "stripchat", "spankbang", "eporner",
        "nsfw", "hentai", "rule34", "jav", "brazzers", "nude", "xxx"
    }

    def __init__(self, blocklist_path: str = "data/nsfw_domains.txt"):
        self._name = "tier1_url_lexical"
        self.blocked_domains = set()
        self._load_blocklist(Path(blocklist_path))

    @property
    def name(self) -> str:
        return self._name

    def _load_blocklist(self, path: Path) -> None:
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip().lower()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # Handle both '0.0.0.0 domain.com' hosts format and plain domain lists
                domain = parts[1] if len(parts) >= 2 and parts[0] in ("0.0.0.0", "127.0.0.1") else parts[0]
                if domain not in ("localhost", "0.0.0.0"):
                    self.blocked_domains.add(domain)

    def analyze(self, url: str) -> AnalysisResult:
        start_time = time.perf_counter()
        normalized_url = url if "://" in url else f"https://{url}"
        parsed = urlparse(normalized_url.lower())
        hostname = (parsed.hostname or "").removeprefix("www.")

        # 1. Exact or parent-domain match in offline blocklist
        if hostname in self.blocked_domains or f"www.{hostname}" in self.blocked_domains:
            return self._build_result(start_time, Verdict.NSFW, 0.99, {"reason": "domain_blocklist_match", "domain": hostname})

        # 2. Check explicit Top-Level Domains (.xxx, .porn, .adult)
        for tld in self.ADULT_TLDS:
            if hostname.endswith(tld):
                return self._build_result(start_time, Verdict.NSFW, 0.98, {"reason": f"explicit_tld_{tld}", "domain": hostname})

        # 3. Lexical token scan across domain and path
        primary_cat, subcats, cat_scores, matched_tokens = classify_text_taxonomy(f"{hostname} {parsed.path}")
        if matched_tokens:
            prob = min(0.85 + (0.04 * len(matched_tokens)), 0.98)
            return AnalysisResult(
                analyzer_name=self.name,
                verdict=Verdict.NSFW,
                nsfw_probability=round(prob, 4),
                primary_category=primary_cat,
                subcategories=subcats,
                category_scores=cat_scores,
                latency_ms=round((time.perf_counter() - start_time) * 1000, 3),
                metadata={"reason": "lexical_taxonomy_match", "matches": matched_tokens}
            )

        # Unknown URL: pass to Tier 2 (HTML/Metadata)
        return self._build_result(start_time, Verdict.UNCERTAIN, 0.15, {"domain": hostname})

    def _build_result(self, start_time: float, verdict: Verdict, prob: float, meta: dict) -> AnalysisResult:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        return AnalysisResult(
            analyzer_name=self.name,
            verdict=verdict,
            nsfw_probability=round(prob, 4),
            latency_ms=latency_ms,
            metadata=meta
        )