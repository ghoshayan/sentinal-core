from PIL import Image
from app.orchestrator import ModerationOrchestrator
from app.schemas import Verdict


def test_orchestrator_execution():
    orchestrator = ModerationOrchestrator()
    img = Image.new("RGB", (224, 224), color="white")
    pipeline_verdict = orchestrator.scan_image(img)

    assert pipeline_verdict.overall_verdict in [Verdict.SAFE, Verdict.NSFW, Verdict.UNCERTAIN]
    assert 0.0 <= pipeline_verdict.max_nsfw_probability <= 1.0
    assert pipeline_verdict.total_latency_ms > 0
    assert len(pipeline_verdict.results) == 1
