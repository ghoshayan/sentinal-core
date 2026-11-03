# Sentinel Core

> High-throughput, low-latency, fully offline content moderation engine powered by INT8-quantized Vision Transformers (ViT) on ONNX Runtime.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU%20Optimized-orange.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey.svg)](LICENSE)

---

## 1. Overview

**Sentinel Core** is an enterprise-grade, privacy-first content moderation microservice designed to run entirely **on-premise / edge / offline** with zero external network calls.

### Key Architectural Tenets
* **Deterministic Privacy & Air-Gap Ready:** Zero telemetry, no third-party cloud API dependencies, and fully operational in isolated environments.
* **Lightweight CPU Inference:** Strips away heavyweight dependencies like PyTorch, TorchVision, or Hugging Face Transformers. Utilizes optimized `onnxruntime` with graph-level optimizations (`ORT_ENABLE_ALL`) and pure NumPy preprocessing.
* **Quantized ViT Backbone:** Runs an INT8-quantized Vision Transformer (`model_int8.onnx`, ~83MB), achieving sub-100ms classification on standard consumer CPU cores while preserving classification fidelity.
* **Dual Interface:** Serves both real-time HTTP requests via FastAPI and batch/bulk recursive sweeps via an administrative CLI scanner.

---

## 2. Architecture & Design

```
                     ┌──────────────────────────────────────────────┐
                     │            Client Request / CLI              │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │        FastAPI Server / CLI Scanner          │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │         Moderation Orchestrator              │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             OfflineVisionAnalyzer                                │
│                                                                                  │
│   ┌───────────────────┐    ┌────────────────────┐    ┌───────────────────────┐   │
│   │ Image Resampling  │───▶│ Tensor Normalization│───▶│ ONNX Runtime Session  │   │
│   │ 224x224 (Bilinear)│    │ [-1.0, 1.0] CHW    │    │ (model_int8.onnx)     │   │
│   └───────────────────┘    └────────────────────┘    └───────────┬───────────┘   │
│                                                                  │               │
│                                                                  ▼               │
│                                                      ┌───────────────────────┐   │
│                                                      │ Logits Softmax & Gating│  │
│                                                      └───────────┬───────────┘   │
└──────────────────────────────────────────────────────────────────┼───────────────┘
                                                                   │
                                                                   ▼
                                                       ┌───────────────────────┐
                                                       │ AnalysisResult Schema │
                                                       │ (SAFE/NSFW/UNCERTAIN) │
                                                       └───────────────────────┘
```

### Classification Gating Policy
The visual analyzer calculates normalized class probabilities via a numerically stable Softmax over ViT logits:
* **`NSFW`**: $\text{nsfw\_prob} \ge 0.80$
* **`SAFE`**: $\text{nsfw\_prob} \le 0.30$
* **`UNCERTAIN`**: $0.30 < \text{nsfw\_prob} < 0.80$ (triggers human review or secondary inspection)

---

## 3. Directory Layout

```
sentinel-core/
├── app/
│   ├── __init__.py
│   ├── schemas.py             # Pydantic v2 domain schemas (Verdict, AnalysisResult)
│   ├── orchestrator.py        # Pipeline coordinator for multi-analyzer workflows
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseAnalyzer abstract interface contract
│   │   └── vision_onnx.py     # Quantized ViT ONNX CPU inference implementation
│   └── api/
│       ├── __init__.py
│       └── server.py          # FastAPI HTTP application (/v1/classify/image)
├── cli_scanner.py             # Recursive folder and single-file CLI scanner
├── models/
│   └── nsfw_vit_onnx/         # ViT configurations and INT8 quantized weights
│       ├── config.json
│       ├── preprocessor_config.json
│       └── onnx/
│           ├── model_int8.onnx (~83MB, primary production runtime)
│           └── model.onnx      (~343MB, unquantized FP32 base artifact)
├── tests/                     # Pytest automated test suite
│   ├── __init__.py
│   ├── test_schemas.py
│   ├── test_vision_analyzer.py
│   ├── test_orchestrator.py
│   └── test_api.py
├── .gitattributes             # Git LFS tracking specifications
├── .gitignore                 # Exclusion rules for environments and caches
├── requirements.txt           # Production and testing runtime dependencies
└── README.md
```

---

## 4. Getting Started

### Prerequisites
* Python 3.10+
* Git with [Git LFS](https://git-lfs.com/) installed

### Installation

1. Clone repository with LFS hooks:
```bash
git clone https://github.com/ghoshayan/sentinal-core.git
cd sentinal-core
git lfs pull
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate
```

3. Install runtime dependencies:
```bash
pip install -r requirements.txt
```

---

## 5. Usage

### CLI Batch Scanner
Scan an individual image or recursively audit an entire asset directory:

```bash
# Scan a single asset
python cli_scanner.py path/to/sample.jpg

# Scan a directory recursively (.jpg, .png, .webp)
python cli_scanner.py ./assets/media/
```

**Example Output:**
```
[SAFE] hero_banner.png | NSFW Prob: 0.12% | 42.15ms
[NSFW] flagged_upload.jpg | NSFW Prob: 98.40% | 51.30ms
[UNCERTAIN] artistic_render.webp | NSFW Prob: 52.80% | 46.10ms
```

### Microservice HTTP API
Launch the FastAPI development server:

```bash
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --workers 1
```

Submit an image for analysis:
```bash
curl -X POST "http://localhost:8000/v1/classify/image" \
  -H "accept: application/json" \
  -F "file=@sample.jpg;type=image/jpeg"
```

**Response Payload (`200 OK`):**
```json
{
  "analyzer_name": "offline_vision_rekib23r_int8",
  "verdict": "SAFE",
  "nsfw_probability": 0.002,
  "latency_ms": 48.72,
  "metadata": {
    "normal": 0.998,
    "nsfw": 0.002
  }
}
```

---

## 6. Running Tests

Execute the automated test suite using `pytest`:

```bash
pytest -v
```

---

## 7. Model Provenance

The visual classification architecture is derived from the Vision Transformer (ViT) model published by `Rekib23r/nsfw_image_detection-ONNX` on Hugging Face. The model is quantized to INT8 static precision, yielding a compact ~83MB binary optimized for x86_64 AVX2/AVX-512 and ARM NEON vector instructions without requiring specialized accelerator hardware.
