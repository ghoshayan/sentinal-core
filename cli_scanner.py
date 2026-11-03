import sys
from pathlib import Path
from app.analyzers.vision_onnx import OfflineVisionAnalyzer

def scan_target(target_path: str):
    analyzer = OfflineVisionAnalyzer()
    path = Path(target_path)
    files = [path] if path.is_file() else list(path.rglob("*.jpg")) + list(path.rglob("*.png")) + list(path.rglob("*.webp"))

    for file_path in files:
        result = analyzer.analyze(file_path)
        print(f"[{result.verdict.value}] {file_path.name} | NSFW Prob: {result.nsfw_probability:.2%} | {result.latency_ms}ms")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli_scanner.py <path_to_image_or_folder>")
    else:
        scan_target(sys.argv[1])