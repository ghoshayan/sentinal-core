from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from app.analyzers.vision_onnx import OfflineVisionAnalyzer
from app.schemas import Verdict


@pytest.fixture(scope="module")
def analyzer():
    return OfflineVisionAnalyzer()


def test_analyzer_initialization(analyzer):
    assert analyzer.name == "offline_vision_rekib23r_int8"
    assert analyzer.threshold == 0.80
    assert analyzer.input_name is not None


def test_preprocess_tensor_shape(analyzer):
    img = Image.new("RGB", (300, 400), color="red")
    tensor = analyzer._preprocess(img)
    # Expected NCHW shape: (1, 3, 224, 224)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    # Verify normalization range is within [-1, 1]
    assert np.all(tensor >= -1.0) and np.all(tensor <= 1.0)


def test_analyze_synthetic_safe_image(analyzer):
    # Pure solid color image should reliably yield non-NSFW / safe verdict
    img = Image.new("RGB", (224, 224), color="blue")
    result = analyzer.analyze(img)
    assert result.analyzer_name == analyzer.name
    assert result.verdict == Verdict.SAFE
    assert 0.0 <= result.nsfw_probability <= 1.0
    assert result.latency_ms > 0
    assert "normal" in result.metadata
    assert "nsfw" in result.metadata

def test_analyze_offline_file_path(analyzer, tmp_path: Path):
    # Verify offline disk file loading via pathlib.Path
    img_path: Path = tmp_path / "offline_test.jpg"
    Image.new("RGB", (224, 224), color="green").save(img_path)

    result = analyzer.analyze(img_path)
    assert img_path.exists()
    assert result.verdict == Verdict.SAFE
    assert 0.0 <= result.nsfw_probability <= 1.0

