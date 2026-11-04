from PIL import Image
from app.analyzers.vision_subcategory import HierarchicalVisionAnalyzer
from app.schemas import NsfwSubcategory, Verdict


def test_hierarchical_vision_analyzer_init():
    analyzer = HierarchicalVisionAnalyzer()
    assert analyzer.name == "hierarchical_vision_onnx"
    assert analyzer.nsfw_threshold == 0.70


def test_hierarchical_vision_analyze_safe():
    analyzer = HierarchicalVisionAnalyzer()
    img = Image.new("RGB", (224, 224), color="blue")
    result = analyzer.analyze(img)
    assert result.verdict == Verdict.SAFE
    assert result.primary_category == NsfwSubcategory.NONE
    assert result.subcategories == []
    assert 0.0 <= result.nsfw_probability <= 1.0
