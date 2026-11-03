import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from app.analyzers.vision_onnx import OfflineVisionAnalyzer
from app.schemas import AnalysisResult

app = FastAPI(title="Sentinel Local Content Classifier", version="1.0.0")
vision_engine = OfflineVisionAnalyzer()

@app.post("/v1/classify/image", response_model=AnalysisResult)
async def classify_image(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    image = Image.open(io.BytesIO(raw_bytes))
    return vision_engine.analyze(image)