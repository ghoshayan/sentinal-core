from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from pydantic import BaseModel, Field

class Verdict(str, Enum):
    SAFE = "SAFE"
    NSFW = "NSFW"
    UNCERTAIN = "UNCERTAIN"

class AnalysisResult(BaseModel):
    analyzer_name: str
    tier: int
    verdict: Verdict
    nsfw_probability: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float
    short_circuit: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AnalysisContext(BaseModel):
    """
    The Blackboard: Passed through the pipeline so analyzers can share state
    (like extracted HTML text or candidate image URLs) without importing each other.
    """
    model_config = {"arbitrary_types_allowed": True}

    url: Optional[str] = None
    image_path: Optional[Path] = None
    image_obj: Optional[Image.Image] = None

    # Shared evidence populated along the pipeline for downstream tiers/agents
    page_title: str = ""
    extracted_text: str = ""
    candidate_image_urls: List[str] = Field(default_factory=list)
    needs_dynamic_render: bool = False
    history: List[AnalysisResult] = Field(default_factory=list)

class PipelineReport(BaseModel):
    target: str
    final_verdict: Verdict
    confidence: float
    triggered_by: str
    total_latency_ms: float
    steps: List[AnalysisResult]

class BaseAnalyzer(ABC):
    """Strict plugin contract. Every analyzer takes AnalysisContext and returns AnalysisResult."""
    tier: int = 1

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def should_run(self, ctx: AnalysisContext) -> bool:
        """Override to run conditionally (e.g., Playwright only runs if ctx.needs_dynamic_render is True)."""
        return True

    @abstractmethod
    def analyze_context(self, ctx: AnalysisContext) -> AnalysisResult:
        pass