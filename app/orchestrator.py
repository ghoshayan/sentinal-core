from pathlib import Path
from typing import Any, List, Optional, Union
from PIL import Image
from pydantic import BaseModel, Field

from app.analyzers.base import BaseAnalyzer
from app.analyzers.vision_onnx import OfflineVisionAnalyzer
from app.schemas import AnalysisResult, Verdict


class PipelineVerdict(BaseModel):
    """Aggregated moderation result across all configured analyzers."""
    overall_verdict: Verdict
    max_nsfw_probability: float = Field(..., ge=0.0, le=1.0)
    total_latency_ms: float
    results: List[AnalysisResult] = Field(default_factory=list)


class ModerationOrchestrator:
    """Coordinates execution across registered analyzers and aggregates safety verdicts."""

    def __init__(self, analyzers: Optional[List[BaseAnalyzer]] = None):
        self.analyzers: List[BaseAnalyzer] = (
            analyzers if analyzers is not None else [OfflineVisionAnalyzer()]
        )

    def register_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """Register an additional analyzer engine dynamically."""
        self.analyzers.append(analyzer)

    def scan_image(self, payload: Union[str, Path, Image.Image]) -> PipelineVerdict:
        """Execute all analyzers against the image payload and synthesize the verdict."""
        results: List[AnalysisResult] = []
        total_latency = 0.0
        max_prob = 0.0

        for analyzer in self.analyzers:
            res = analyzer.analyze(payload)
            results.append(res)
            total_latency += res.latency_ms
            if res.nsfw_probability > max_prob:
                max_prob = res.nsfw_probability

        if any(r.verdict == Verdict.NSFW for r in results):
            overall = Verdict.NSFW
        elif any(r.verdict == Verdict.UNCERTAIN for r in results):
            overall = Verdict.UNCERTAIN
        else:
            overall = Verdict.SAFE

        return PipelineVerdict(
            overall_verdict=overall,
            max_nsfw_probability=round(max_prob, 4),
            total_latency_ms=round(total_latency, 2),
            results=results,
        )
