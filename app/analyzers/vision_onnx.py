import time
from pathlib import Path
from typing import Union
import numpy as np
import onnxruntime as ort
from PIL import Image

from app.analyzers.base import BaseAnalyzer
from app.schemas import AnalysisResult, Verdict

class OfflineVisionAnalyzer(BaseAnalyzer):
    def __init__(
        self,
        model_path: str = "models/nsfw_vit_onnx/onnx/model_int8.onnx",
        threshold: float = 0.80
    ):
        self._name = "offline_vision_rekib23r_int8"
        self.threshold = threshold

        onnx_file = Path(model_path)
        if not onnx_file.exists():
            raise FileNotFoundError(f"ONNX file not found at {onnx_file}")

        # Initialize pure ONNX Runtime session (no PyTorch or Optimum required)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_file),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.labels = {0: "normal", 1: "nsfw"}  # From Rekib23r/nsfw_image_detection-ONNX config.json

    @property
    def name(self) -> str:
        return self._name

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        # Matches Rekib23r preprocessor_config.json (224x224, rescale 1/255, mean/std 0.5)
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        img_array = np.asarray(image, dtype=np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5
        # Transpose HWC -> CHW and add batch dimension (1, 3, 224, 224)
        return np.expand_dims( np.transpose(img_array, (2, 0, 1)), axis=0 )

    def analyze(self, payload: Union[str, Path, Image.Image]) -> AnalysisResult:
        start_time = time.perf_counter()

        image = Image.open(payload) if isinstance(payload, (str, Path)) else payload
        input_tensor = self._preprocess(image)

        # Run raw ONNX inference
        logits = self.session.run(None, {self.input_name: input_tensor})[0][0]
        
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        scores = {self.labels[i]: float(probs[i]) for i in range(len(probs))}
        nsfw_prob = scores.get("nsfw", 0.0)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if nsfw_prob >= self.threshold:
            verdict = Verdict.NSFW
        elif nsfw_prob <= 0.30:
            verdict = Verdict.SAFE
        else:
            verdict = Verdict.UNCERTAIN

        return AnalysisResult(
            analyzer_name=self.name,
            verdict=verdict,
            nsfw_probability=round(nsfw_prob, 4),
            latency_ms=latency_ms,
            metadata=scores
        )