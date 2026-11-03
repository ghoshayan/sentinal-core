from unittest.mock import MagicMock
from app.orchestrator import CascadeOrchestrator
from app.schemas import AnalysisResult, PipelineReport, Verdict


def test_cascade_short_circuit_tier1():
    url_mock = MagicMock()
    url_mock.name = "tier1_url_lexical"
    url_mock.analyze.return_value = AnalysisResult(
        analyzer_name="tier1_url_lexical",
        verdict=Verdict.NSFW,
        nsfw_probability=0.99,
        latency_ms=1.2,
        metadata={"reason": "explicit_tld"},
    )
    html_mock = MagicMock()
    vision_mock = MagicMock()

    orchestrator = CascadeOrchestrator(
        url_analyzer=url_mock,
        html_analyzer=html_mock,
        vision_analyzer=vision_mock,
    )

    report = orchestrator.classify_website("https://example.xxx")
    assert isinstance(report, PipelineReport)
    assert report.final_verdict == Verdict.NSFW
    assert report.triggered_by == "tier1_url_lexical"
    assert report.confidence == 0.99
    assert len(report.steps) == 1
    # Ensure html and vision were not invoked (short-circuited)
    html_mock.analyze.assert_not_called()
    vision_mock.analyze.assert_not_called()


def test_cascade_tier2_safe():
    url_mock = MagicMock()
    url_mock.name = "tier1_url_lexical"
    url_mock.analyze.return_value = AnalysisResult(
        analyzer_name="tier1_url_lexical",
        verdict=Verdict.UNCERTAIN,
        nsfw_probability=0.15,
        latency_ms=0.5,
    )

    html_mock = MagicMock()
    html_mock.name = "tier2_html_metadata"
    html_mock.analyze.return_value = AnalysisResult(
        analyzer_name="tier2_html_metadata",
        verdict=Verdict.SAFE,
        nsfw_probability=0.08,
        latency_ms=15.0,
        metadata={"reason": "clean_metadata_and_text", "candidate_images": []},
    )

    vision_mock = MagicMock()

    orchestrator = CascadeOrchestrator(
        url_analyzer=url_mock,
        html_analyzer=html_mock,
        vision_analyzer=vision_mock,
    )

    report = orchestrator.classify_website("https://example.com")
    assert report.final_verdict == Verdict.SAFE
    assert report.triggered_by == "tier2_html_metadata"
    assert len(report.steps) == 2
    vision_mock.analyze.assert_not_called()
