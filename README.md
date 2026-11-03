# Sentinel Core

> High-throughput, multi-tier content moderation and safety engine powered by a cascading URL lexical, HTML metadata, dynamic browser, and INT8-quantized Vision Transformer (ViT) pipeline on ONNX Runtime.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v2.1.0-green.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU%20Optimized-orange.svg)](https://onnxruntime.ai/)
[![Playwright](https://img.shields.io/badge/Playwright-Dynamic%20DOM-purple.svg)](https://playwright.dev/)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey.svg)](LICENSE)

---

## 1. Overview

**Sentinel Core** is an enterprise-grade content safety engine and moderation microservice designed to inspect images, local filesystem storage, and live web domains through an intelligent, latency-tiered cascade.

### Cascading Architectural Philosophy
Inspection follows a cost-aware funnel: cheap, fast heuristics run first; expensive deep learning or dynamic headless rendering runs only when earlier tiers yield ambiguity:

1. **Tier 1: Lexical & Domain Verification (~1ms)**
   * Instant offline domain blocklist matching (`data/nsfw_domains.txt`).
   * Explicit Adult Top-Level Domain (TLD) filtering (`.xxx`, `.porn`, `.adult`, `.cam`, etc.).
   * Tokenized hostname and path lexical pattern matching.
   * *Short-circuits immediately on high-confidence blocklist/TLD matches.*

2. **Tier 2A: Fast Static HTML & Metadata Inspection (~15-50ms)**
   * HTTP `Rating` response header inspection (RTA-5042 standard).
   * HTML `<meta name="rating">` and Dublin Core mature tags.
   * High-density explicit keyword scanning across `<title>`, `<meta description>`, and heading elements (`<h1>`-`<h2>`).
   * Extracts candidate image URLs (OpenGraph `og:image`, top-level content `<img>`).

3. **Tier 2B: Dynamic Headless Browser Fallback (~800ms-1.5s)**
   * Invoked conditionally when static HTTP encounters blank JavaScript Single Page Applications (React/Next/Vue) or anti-scraping challenges.
   * Headless Chromium via Playwright aborts video, audio, font, and CSS downloads to minimize overhead while rendering client-side DOM.

4. **Tier 3: Quantized ViT Visual Inspection (~50-80ms)**
   * INT8-quantized Vision Transformer (`model_int8.onnx`, ~83MB) on ONNX Runtime.
   * Inspects extracted candidate DOM thumbnails and standalone images with zero cloud API dependencies.

---

## 2. Multi-Tier Cascade Pipeline

```
                              ┌─────────────────────────┐
                              │    Incoming Target      │
                              │ (URL, Image, or Folder) │
                              └────────────┬────────────┘
                                           │
                                           ▼
                 ┌──────────────────────────────────────────────────┐
                 │          Tier 1: URL & Domain Lexical            │
                 │   (Domain Blocklist / Adult TLDs / Tokens)       │
                 └─────────┬──────────────────────────────┬─────────┘
                    NSFW   │                              │ UNCERTAIN
                     ┌─────┘                              ▼
                     │            ┌─────────────────────────────────┐
                     │            │    Tier 2A: Static HTML Meta    │
                     │            │    (HTTP Headers / RTA / Meta)  │
                     │            └──────┬────────────────────┬─────┘
                     │       SAFE / NSFW │                    │ UNCERTAIN (JS SPA / Anti-Bot)
                     │     ┌─────────────┘                    ▼
                     │     │              ┌─────────────────────────────────┐
                     │     │              │    Tier 2B: Dynamic Playwright  │
                     │     │              │    (Headless Chromium Render)   │
                     │     │              └──────┬────────────────────┬─────┘
                     │     │         SAFE / NSFW │                    │ UNCERTAIN (Images Found)
                     │     │     ┌───────────────┘                    ▼
                     │     │     │        ┌─────────────────────────────────┐
                     │     │     │        │    Tier 3: Offline ViT ONNX     │
                     │     │     │        │   (model_int8.onnx CPU Infer)   │
                     │     │     │        └───────────────────┬─────────────┘
                     ▼     ▼     ▼                            │
                 ┌────────────────────────────────────────────┴─────┐
                 │                 PipelineReport                   │
                 │ (final_verdict, confidence, audit steps, latency)│
                 └──────────────────────────────────────────────────┘
```

---

## 3. Directory Layout

```
sentinel-core/
├── app/
│   ├── __init__.py
│   ├── schemas.py             # Pydantic domain models (Verdict, AnalysisResult, PipelineReport)
│   ├── orchestrator.py        # CascadeOrchestrator multi-tier execution coordinator
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseAnalyzer abstract interface contract
│   │   ├── url_lexical.py     # Tier 1: Lexical URL & Domain blocklist engine
│   │   ├── html_metadata.py   # Tier 2A: Fast HTTP metadata & RTA scraper
│   │   ├── playwright_dynamic.py # Tier 2B: Headless Chromium SPA analyzer
│   │   └── vision_onnx.py     # Tier 3: Quantized ViT ONNX vision analyzer
│   └── api/
│       ├── __init__.py
│       └── server.py          # FastAPI service (Image upload, Local path, URL pipeline)
├── cli_scanner.py             # Recursive folder and single-file CLI scanner
├── models/
│   └── nsfw_vit_onnx/         # ViT configurations and INT8 quantized weights
│       ├── config.json
│       ├── preprocessor_config.json
│       └── onnx/
│           └── model_int8.onnx (~83MB INT8 runtime artifact)
├── tests/                     # Pytest automated test suite (16 test cases)
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_url_lexical.py
│   ├── test_vision_analyzer.py
│   ├── test_orchestrator.py
│   └── test_api.py
├── .gitattributes             # Line endings and LF normalization
├── .gitignore                 # Exclusion rules for environments, caches, and FP32 base models
├── requirements.txt           # Production dependencies
└── README.md
```

---

## 4. Getting Started

### Installation

```bash
git clone https://github.com/ghoshayan/sentinal-core.git
cd sentinal-core

python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
playwright install chromium   # Optional: only needed for dynamic SPA tier
```

---

## 5. API Reference & Endpoints

Start the FastAPI application:
```bash
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### 1. Standalone Image Classification
```http
POST /v1/classify/image
Content-Type: multipart/form-data
```
Returns an `AnalysisResult` evaluating an uploaded image binary directly on the local ViT ONNX engine.

### 2. Local Filesystem Scanner
```http
POST /v1/classify/local-path?path=C:\Users\Username\Pictures
```
Recursively analyzes image assets (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`) on disk without network payload serialization.

### 3. Live Website Cascade
```http
POST /v1/classify/url
Content-Type: application/json

{
  "url": "https://example.com"
}
```
Executes the tiered cascade (Tier 1 $\rightarrow$ Tier 2A $\rightarrow$ Tier 2B $\rightarrow$ Tier 3) and outputs a detailed `PipelineReport` with timing and per-tier trace:
```json
{
  "target": "https://example.com",
  "final_verdict": "SAFE",
  "confidence": 0.08,
  "triggered_by": "tier2_html_metadata",
  "total_latency_ms": 32.14,
  "steps": [
    {
      "analyzer_name": "tier1_url_lexical",
      "verdict": "UNCERTAIN",
      "nsfw_probability": 0.15,
      "latency_ms": 0.35,
      "metadata": { "domain": "example.com" }
    },
    {
      "analyzer_name": "tier2_html_metadata",
      "verdict": "SAFE",
      "nsfw_probability": 0.08,
      "latency_ms": 31.79,
      "metadata": { "reason": "clean_metadata_and_text" }
    }
  ]
}
```

---

## 6. Running Tests

```bash
pytest -v
```
All 16 test cases run offline in isolated environments.
