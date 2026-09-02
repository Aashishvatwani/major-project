"""
Aerospace Mission Control Telemetry Dashboard Backend
FastAPI server streaming real-time HITL pipeline telemetry over WebSockets,
providing REST endpoints for dynamic fault injection, human operator authorization, and research benchmarks.
"""

import asyncio
import json
import os
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline.hitl_pipeline import SatelliteHITLPipeline
from src.ingestion.data_loader import TelemetryDataLoader

app = FastAPI(title="Satellite Telemetry HITL Mission Control", version="3.0.0")

# Setup CORS and static assets
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Global Pipeline and Connection Manager
pipeline_instance: Optional[SatelliteHITLPipeline] = None
active_websockets: List[WebSocket] = []
is_streaming = False
stream_delay = 0.4  # seconds
autopilot_mode = True


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for dc in dead_connections:
            self.disconnect(dc)


manager = ConnectionManager()


class FaultInjectionRequest(BaseModel):
    fault_type: str  # "thermal_runaway", "internal_short", "undervoltage", "high_impedance", "sensor_drift"
    duration_sec: float = 25.0


class MitigationAuthRequest(BaseModel):
    authorized: bool
    autopilot: Optional[bool] = None


def init_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        if not os.path.exists("saved_models/ensemble_model.joblib"):
            from src.models.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            trainer.train_full_pipeline(duration_minutes=360.0)

        pipeline_instance = SatelliteHITLPipeline(
            models_dir="saved_models",
            serial_port="AUTO",
            enable_hardware=True
        )

        loader = TelemetryDataLoader("data/raw/synthetic_telemetry.csv")
        df = loader.load_or_generate_dataset(duration_minutes=360.0)
        pipeline_instance.stream.set_dataset(df)


@app.on_event("startup")
async def startup_event():
    init_pipeline()
    asyncio.create_task(telemetry_broadcaster_task())


async def telemetry_broadcaster_task():
    global is_streaming, pipeline_instance, stream_delay, autopilot_mode
    is_streaming = True
    while True:
        if is_streaming and pipeline_instance:
            sample = pipeline_instance.stream.get_next_sample()
            if sample is not None:
                record = pipeline_instance.process_single_sample(sample)
                record["autopilot_mode"] = autopilot_mode
                await manager.broadcast(record)
        await asyncio.sleep(stream_delay)


@app.get("/favicon.ico")
async def get_favicon():
    svg_icon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🛰️</text></svg>'
    return Response(content=svg_icon, media_type="image/svg+xml")


@app.get("/")
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/inject_fault")
async def inject_fault_endpoint(req: FaultInjectionRequest):
    global pipeline_instance
    if pipeline_instance and pipeline_instance.stream:
        pipeline_instance.stream.inject_fault(req.fault_type, duration_sec=req.duration_sec)
        return {"status": "success", "injected": req.fault_type, "duration": req.duration_sec}
    return {"status": "error", "message": "Pipeline not initialized"}


@app.post("/api/clear_fault")
async def clear_fault_endpoint():
    global pipeline_instance
    if pipeline_instance and pipeline_instance.stream:
        pipeline_instance.stream.clear_fault()
        return {"status": "success", "message": "Fault cleared"}
    return {"status": "error"}


@app.post("/api/authorize_mitigation")
async def authorize_mitigation_endpoint(req: MitigationAuthRequest):
    global pipeline_instance, autopilot_mode
    if req.autopilot is not None:
        autopilot_mode = req.autopilot
    if pipeline_instance:
        pipeline_instance.human_approval_override = req.authorized
    return {"status": "success", "authorized": req.authorized, "autopilot": autopilot_mode}


@app.get("/api/ablation_results")
async def get_ablation_results_endpoint():
    json_path = "docs/ablation_results.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "not_found", "message": "Run `python scripts/run_ablation_study.py` first."}


@app.get("/api/hardware_status")
async def get_hardware_status_endpoint():
    global pipeline_instance
    if pipeline_instance and pipeline_instance.serial_bridge:
        return pipeline_instance.serial_bridge.get_status()
    return {"status": "hardware bridge inactive"}


@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
