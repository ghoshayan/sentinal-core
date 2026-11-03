import pytest
from pydantic import ValidationError
from app.schemas import Verdict, AnalysisResult, PipelineReport


def test_verdict_enum_values():
    assert Verdict.SAFE == "SAFE"
    assert Verdict.NSFW == "NSFW"
    assert Verdict.UNCERTAIN == "UNCERTAIN"


def test_analysis_result_validation_success():
    result = AnalysisResult(
        analyzer_name="test_analyzer",
        verdict=Verdict.SAFE,
        nsfw_probability=0.05,
        latency_ms=12.5,
        metadata={"normal": 0.95, "nsfw": 0.05, "complex_obj": ["a", "b"]},
    )
    assert result.analyzer_name == "test_analyzer"
    assert result.verdict == Verdict.SAFE
    assert result.nsfw_probability == 0.05
    assert result.latency_ms == 12.5
    assert result.metadata["normal"] == 0.95


def test_analysis_result_probability_bounds():
    with pytest.raises(ValidationError):
        AnalysisResult(
            analyzer_name="test_analyzer",
            verdict=Verdict.NSFW,
            nsfw_probability=1.2,
            latency_ms=10.0,
        )
    with pytest.raises(ValidationError):
        AnalysisResult(
            analyzer_name="test_analyzer",
            verdict=Verdict.SAFE,
            nsfw_probability=-0.1,
            latency_ms=10.0,
        )


def test_pipeline_report_validation():
    step = AnalysisResult(
        analyzer_name="tier1_url_lexical",
        verdict=Verdict.SAFE,
        nsfw_probability=0.01,
        latency_ms=0.5,
    )
    report = PipelineReport(
        target="https://example.com",
        final_verdict=Verdict.SAFE,
        confidence=0.01,
        triggered_by="tier1_url_lexical",
        total_latency_ms=0.5,
        steps=[step],
    )
    assert report.target == "https://example.com"
    assert report.final_verdict == Verdict.SAFE
    assert len(report.steps) == 1
