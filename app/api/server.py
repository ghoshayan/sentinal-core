import io
from pathlib import Path
from typing import List
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.analyzers.url_lexical import UrlLexicalAnalyzer
from app.analyzers.html_metadata import HtmlMetadataAnalyzer
from app.analyzers.playwright_dynamic import PlaywrightDynamicAnalyzer
from app.analyzers.vision_subcategory import HierarchicalVisionAnalyzer  # Replaced OfflineVisionAnalyzer
from app.orchestrator import CascadeOrchestrator
from app.schemas import AnalysisResult, PipelineReport

app = FastAPI(title="Sentinel Local Content Classifier", version="2.2.0")

# Initialize modular engines once at startup
url_engine = UrlLexicalAnalyzer()
html_engine = HtmlMetadataAnalyzer()
playwright_engine = PlaywrightDynamicAnalyzer()
vision_engine = HierarchicalVisionAnalyzer()  # Uses two-stage nudity + subcategory classifier

orchestrator = CascadeOrchestrator(
    url_analyzer=url_engine,
    html_analyzer=html_engine,
    vision_analyzer=vision_engine,
    playwright_analyzer=playwright_engine
)

class UrlRequest(BaseModel):
    url: str

class LocalPathRequest(BaseModel):
    path: str

# 1. OFFLINE IMAGE UPLOAD (Drag & drop an image in /docs)
@app.post("/v1/classify/image", response_model=AnalysisResult)
async def classify_image(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    image = Image.open(io.BytesIO(raw_bytes))
    return vision_engine.analyze(image)

# 2. OFFLINE LOCAL FILE OR FOLDER SCANNER (Pass a local disk path like "C:/Pictures")
@app.post("/v1/classify/local-path", response_model=List[AnalysisResult])
def classify_local_path(
    path: str = Query(..., description="Paste any local file or folder path (e.g. C:\\Pictures)")
):
    # Strip accidental wrapping quotes from Windows 'Copy as path'
    clean_path = path.strip().strip('"').strip("'")
    target = Path(clean_path)

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found on disk: {target}")

    supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [target] if target.is_file() else [
        p for p in target.rglob("*") if p.suffix.lower() in supported_exts
    ]

    results = []
    for file_path in files:
        res = vision_engine.analyze(file_path)
        if res.metadata is not None:
            res.metadata["file_name"] = file_path.name
            res.metadata["file_path"] = str(file_path)
        results.append(res)
    return results

# 3. LIVE WEBSITE CLASSIFIER (Tier 1 URL -> Tier 2A httpx -> Tier 2B Playwright -> Tier 3 Vision)
@app.post("/v1/classify/url", response_model=PipelineReport)
def classify_url(payload: UrlRequest):
    return orchestrator.classify_website(payload.url)