import time
from app.core.contracts import AnalysisContext, PipelineReport, Verdict
from app.core.registry import AnalyzerRegistry

class CascadeOrchestrator:
    def __init__(self):
        # Auto-discover all plugins in app/analyzers/ on startup
        AnalyzerRegistry.autodiscover("app.analyzers")

    def run_pipeline(self, ctx: AnalysisContext, target_label: str) -> PipelineReport:
        start_time = time.perf_counter()

        for analyzer in AnalyzerRegistry.get_ordered_pipeline():
            if not analyzer.should_run(ctx):
                continue

            result = analyzer.analyze_context(ctx)
            ctx.history.append(result)

            # If the analyzer explicitly requests a short-circuit (e.g., Tier 1 Blocklist hit)
            if result.short_circuit and result.verdict in (Verdict.NSFW, Verdict.SAFE):
                break

        total_ms = round((time.perf_counter() - start_time) * 1000, 2)
        deciding = ctx.history[-1]

        return PipelineReport(
            target=target_label,
            final_verdict=deciding.verdict,
            confidence=deciding.nsfw_probability,
            triggered_by=deciding.analyzer_name,
            total_latency_ms=total_ms,
            steps=ctx.history
        )