from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class Verdict(str, Enum):
    SAFE = "SAFE"
    NSFW = "NSFW"
    UNCERTAIN = "UNCERTAIN"

class NsfwSubcategory(str, Enum):
    NONE = "none"
    BDSM = "bdsm"
    BONDAGE = "bondage"
    FETISH_KINK = "fetish_kink"
    HARDCORE_EXPLICIT = "hardcore_explicit"
    NUDITY_EROTICA = "nudity_erotica"
    HENTAI_ANIME = "hentai_anime"
    LIVE_CAM = "live_cam"

class AnalysisResult(BaseModel):
    analyzer_name: str
    verdict: Verdict
    nsfw_probability: float = Field(..., ge=0.0, le=1.0)
    primary_category: NsfwSubcategory = NsfwSubcategory.NONE
    subcategories: List[NsfwSubcategory] = Field(default_factory=list)
    category_scores: Dict[str, float] = Field(default_factory=dict)
    latency_ms: float
    metadata: Optional[Dict[str, Any]] = None

class PipelineReport(BaseModel):
    target: str
    final_verdict: Verdict
    confidence: float
    primary_category: NsfwSubcategory = NsfwSubcategory.NONE
    subcategories: List[NsfwSubcategory] = Field(default_factory=list)
    category_scores: Dict[str, float] = Field(default_factory=dict)
    triggered_by: str
    total_latency_ms: float
    steps: List[AnalysisResult]