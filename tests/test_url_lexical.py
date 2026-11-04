from pathlib import Path
from app.analyzers.url_lexical import UrlLexicalAnalyzer
from app.schemas import Verdict


def test_url_lexical_adult_tld():
    analyzer = UrlLexicalAnalyzer(blocklist_path="nonexistent.txt")
    res = analyzer.analyze("https://random-cam-site.xxx")
    assert res.verdict == Verdict.NSFW
    assert res.nsfw_probability >= 0.98


def test_url_lexical_high_risk_token():
    analyzer = UrlLexicalAnalyzer(blocklist_path="nonexistent.txt")
    res = analyzer.analyze("https://example.com/gallery/hentai")
    assert res.verdict == Verdict.NSFW
    assert res.nsfw_probability >= 0.85


def test_url_lexical_clean_url():
    analyzer = UrlLexicalAnalyzer(blocklist_path="nonexistent.txt")
    res = analyzer.analyze("https://news.ycombinator.com")
    assert res.verdict == Verdict.UNCERTAIN

