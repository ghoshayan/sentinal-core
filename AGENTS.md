# Sentinel-Core Coding Agent Instructions

## Context Boundary Rules (CRITICAL: READ BEFORE SCANNING REPO)
This repository uses a strict **Blackboard + Self-Registering Plugin Architecture**.
1. **DO NOT scan the whole repository** when asked to create, debug, or modify an analyzer.
2. **To add or edit a classification module:**
   - Read ONLY `app/core/contracts.py` (for `AnalysisContext`, `AnalysisResult`, and `BaseAnalyzer`).
   - Create or edit ONLY the single target file inside `app/analyzers/`.
   - Decorate the class with `@AnalyzerRegistry.register(tier=N)` from `app.core.registry`.
   - **NEVER** edit `app/core/orchestrator.py` or `app/api/server.py` when adding a new analyzer; plugins are auto-discovered at runtime.
3. **Zero Cross-Analyzer Imports:**
   - Files in `app/analyzers/` must NEVER import other files in `app/analyzers/`.
   - Shared stateless helpers belong in `app/utils/`.
   - Runtime state shared between tiers must be stored on `AnalysisContext` (`app/core/contracts.py`).

## Tier Allocation Map
- `tier=10`: URL & Lexical Blocklists (`app/analyzers/url_lexical.py`)
- `tier=20`: Static HTML & Metadata (`app/analyzers/html_metadata.py`)
- `tier=25`: Dynamic Headless Browser Fallback (`app/analyzers/playwright_dynamic.py`)
- `tier=30`: Offline ONNX Vision Classifier (`app/analyzers/vision_onnx.py`)
- `tier=40`: Future LLM/VLM Contextual Judge Agents (`app/analyzers/agent_*.py`)