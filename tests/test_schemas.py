import pytest
from pydantic import ValidationError
from app.schemas import Verdict, AnalysisResult


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
        metadata={"normal": 0.95, "nsfw": 0.05},
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
