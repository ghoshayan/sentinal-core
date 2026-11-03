from app.api.server import app, vision_engine
from app.schemas import AnalysisResult, Verdict


def test_api_route_registered():
    routes = {route.path: route for route in app.routes}
    assert "/v1/classify/image" in routes
    route = routes["/v1/classify/image"]
    assert "POST" in route.methods
    assert route.response_model == AnalysisResult


def test_api_openapi_schema():
    schema = app.openapi()
    assert schema["info"]["title"] == "Sentinel Local Content Classifier"
    assert schema["info"]["version"] == "1.0.0"
    assert "/v1/classify/image" in schema["paths"]
    post_op = schema["paths"]["/v1/classify/image"]["post"]
    assert post_op["operationId"] == "classify_image_v1_classify_image_post"


def test_api_vision_engine_attached():
    assert vision_engine is not None
    assert vision_engine.name == "offline_vision_rekib23r_int8"
