from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class Verdict(str, Enum):
    SAFE = "SAFE"
    NSFW = "NSFW"
    UNCERTAIN = "UNCERTAIN"

class AnalysisResult(BaseModel):
    analyzer_name: str
    verdict: Verdict
    nsfw_probability: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = None

class PipelineReport(BaseModel):
    target: str
    final_verdict: Verdict
    confidence: float
    triggered_by: str
    total_latency_ms: float
    steps: List[AnalysisResult]