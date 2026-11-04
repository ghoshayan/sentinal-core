from app.core.contracts import AnalysisContext, AnalysisResult, BaseAnalyzer, Verdict
from app.core.registry import AnalyzerRegistry


class MockTier1Analyzer(BaseAnalyzer):
    tier = 10

    @property
    def name(self) -> str:
        return "mock_tier1"

    def analyze_context(self, ctx: AnalysisContext) -> AnalysisResult:
        return AnalysisResult(
            analyzer_name=self.name,
            tier=self.tier,
            verdict=Verdict.SAFE,
            nsfw_probability=0.01,
            latency_ms=1.0,
            short_circuit=False,
        )


class MockTier2Analyzer(BaseAnalyzer):
    tier = 20

    @property
    def name(self) -> str:
        return "mock_tier2"

    def analyze_context(self, ctx: AnalysisContext) -> AnalysisResult:
        return AnalysisResult(
            analyzer_name=self.name,
            tier=self.tier,
            verdict=Verdict.SAFE,
            nsfw_probability=0.05,
            latency_ms=5.0,
            short_circuit=False,
        )


def test_registry_ordering():
    # Manually register mock analyzers into registry dict
    AnalyzerRegistry._plugins["mock_tier2"] = MockTier2Analyzer()
    AnalyzerRegistry._plugins["mock_tier1"] = MockTier1Analyzer()

    ordered = AnalyzerRegistry.get_ordered_pipeline()
    tier_order = [a.tier for a in ordered if a.name.startswith("mock_")]
    assert tier_order == [10, 20]
