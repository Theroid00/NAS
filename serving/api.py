"""Local FastAPI prediction service and experiment viewer."""
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import time
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

from serving.artifact import Predictor

LOGGER = logging.getLogger("nas.service")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: list[Annotated[list[FiniteFloat], Field(min_length=1, max_length=54)]] = Field(min_length=1, max_length=1024)

    @field_validator("features", mode="before")
    @classmethod
    def numeric_rows(cls, rows):
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise ValueError("features must be an array of rows")
        if any(type(value) not in (int, float) for row in rows for value in row):
            raise ValueError("Feature values must be numbers, not strings or booleans")
        return rows


def create_app(artifact_dir=None, report_path=None, device="cpu"):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = Predictor(artifact_dir, device=device) if artifact_dir else None
        if report_path:
            info = json.loads(Path(report_path).read_text(encoding="utf-8"))
            if info.get("report_version") != 1:
                raise ValueError("Unsupported comparison report")
            if app.state.predictor and info["dataset"] != app.state.predictor.manifest["dataset_name"]:
                raise ValueError("Report and served model belong to different datasets")
        yield
        app.state.predictor = None

    app = FastAPI(title="Tabular Model Search", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Do not echo NaN/Infinity into a JSON error response (which would itself fail).
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]} for item in error.errors()]})

    @app.middleware("http")
    async def timing(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.3f}"
        LOGGER.info("method=%s path=%s status=%s elapsed_ms=%.3f", request.method,
                    request.url.path, response.status_code, elapsed)
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.predictor is not None,
                "device": str(next(app.state.predictor.model.parameters()).device) if app.state.predictor else None}

    @app.get("/model")
    def model_info():
        if app.state.predictor is None:
            raise HTTPException(503, "No model artifact configured")
        return app.state.predictor.manifest

    @app.post("/predict")
    def predict(payload: PredictRequest):
        if app.state.predictor is None:
            raise HTTPException(503, "No model artifact configured")
        try:
            return app.state.predictor.predict(payload.features)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/example")
    def example():
        if artifact_dir is None or not app.state.predictor.manifest.get("example_available"):
            raise HTTPException(404, "No example row bundled")
        return json.loads((Path(artifact_dir) / "example.json").read_text(encoding="utf-8"))

    @app.get("/dashboard.js")
    def script():
        return FileResponse(Path(__file__).with_name("dashboard.js"), media_type="text/javascript")

    @app.get("/report")
    def report():
        if report_path is None:
            raise HTTPException(404, "No comparison report configured")
        return json.loads(Path(report_path).read_text(encoding="utf-8"))

    @app.get("/")
    def dashboard():
        return FileResponse(Path(__file__).with_name("dashboard.html"))

    return app
