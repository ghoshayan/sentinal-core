from pathlib import Path
from PIL import Image
from app.core.contracts import AnalysisContext, AnalysisResult, BaseAnalyzer, Verdict


def test_analysis_context_blackboard():
    ctx = AnalysisContext(url="https://example.com")
    assert ctx.url == "https://example.com"
    assert ctx.extracted_text == ""
    assert ctx.candidate_image_urls == []
    assert ctx.needs_dynamic_render is False

    # Simulate Blackboard state evolution
    ctx.page_title = "Example Title"
    ctx.extracted_text = "Some extracted body content"
    ctx.candidate_image_urls.append("https://example.com/thumb.jpg")
    ctx.needs_dynamic_render = True

    assert ctx.page_title == "Example Title"
    assert len(ctx.candidate_image_urls) == 1
    assert ctx.needs_dynamic_render is True


def test_analysis_result_tier_and_short_circuit():
    res = AnalysisResult(
        analyzer_name="tier1_test",
        tier=10,
        verdict=Verdict.NSFW,
        nsfw_probability=0.99,
        latency_ms=1.5,
        short_circuit=True,
        metadata={"match": "domain_blocklist"},
    )
    assert res.tier == 10
    assert res.short_circuit is True
    assert res.verdict == Verdict.NSFW
