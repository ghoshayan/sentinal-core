from app.api.server import app, vision_engine, url_engine, html_engine, orchestrator
from app.schemas import AnalysisResult, PipelineReport, Verdict


def test_api_routes_registered():
    routes = {route.path: route for route in app.routes}
    assert "/v1/classify/image" in routes
    assert "/v1/classify/local-path" in routes
    assert "/v1/classify/url" in routes


def test_api_openapi_schema():
    schema = app.openapi()
    assert schema["info"]["title"] == "Sentinel Local Content Classifier"
    assert schema["info"]["version"] == "2.2.0"
    assert "/v1/classify/image" in schema["paths"]
    assert "/v1/classify/local-path" in schema["paths"]
    assert "/v1/classify/url" in schema["paths"]


def test_api_engines_initialized():
    assert vision_engine is not None
    assert url_engine is not None
    assert html_engine is not None
    assert orchestrator is not None
