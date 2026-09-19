"""
Aerospace Mission Control Telemetry Dashboard Backend
FastAPI server streaming real-time HITL pipeline telemetry over WebSockets,
providing REST endpoints for dynamic fault injection, human operator authorization, and research benchmarks.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline.hitl_pipeline import SatelliteHITLPipeline
from src.ingestion.data_loader import TelemetryDataLoader
from src.utils.audit_logger import audit_logger
from src.utils.pdf_generator import generate_mission_audit_pdf

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


class OperatorActionRequest(BaseModel):
    action: str
    details: Optional[str] = ""
    operator: Optional[str] = "MissionCommander"
    metadata: Optional[Dict[str, Any]] = None


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

        from data.synthetic_generator import SatelliteTelemetryGenerator
        generator = SatelliteTelemetryGenerator()
        # Generate pure nominal stream baseline so faults only happen when injected
        df = generator.generate_telemetry_dataset(duration_minutes=360.0, inject_anomalies=False)
        pipeline_instance.stream.set_dataset(df)

        audit_logger.log_event(
            event_type="SYSTEM",
            level="INFO",
            subsystem="HITL_PIPELINE",
            message="AERO-GUARD Master Pipeline initialized. Synthetic telemetry baseline loaded."
        )


@app.on_event("startup")
async def startup_event():
    init_pipeline()
    asyncio.create_task(telemetry_broadcaster_task())


# State tracker for audit transitions
pipeline_state = {
    "was_in_anomaly": False,
    "last_fault": "NOMINAL",
    "last_rl_action": "NOMINAL_MONITOR",
    "early_warning_fired": False,
    "frame_counter": 0
}


async def telemetry_broadcaster_task():
    global is_streaming, pipeline_instance, stream_delay, autopilot_mode, pipeline_state
    is_streaming = True
    while True:
        if is_streaming and pipeline_instance:
            sample = pipeline_instance.stream.get_next_sample()
            if sample is not None:
                record = pipeline_instance.process_single_sample(sample)
                record["autopilot_mode"] = autopilot_mode
                await manager.broadcast(record)

                # =============================================================
                # Structured Mission Audit Logging (Max 10,000 FIFO Buffer)
                # =============================================================
                pipeline_state["frame_counter"] += 1
                fc = pipeline_state["frame_counter"]
                
                p_ens = float(record.get("p_ensemble", 0.0))
                pri_fault = record.get("primary_fault", "NOMINAL")
                severity = record.get("final_severity", "NOMINAL")
                rl_action = record.get("rl_action_name", "NOMINAL_MONITOR")
                dyn_thresh = float(record.get("dynamic_threshold", 0.70))
                override_act = bool(record.get("safety_override_active", False))

                is_anomaly = (p_ens >= 0.30) or (pri_fault != "NOMINAL") or (severity != "NOMINAL") or override_act

                # 1. Early Precursor Warning Detection (before hard threshold trip)
                if not is_anomaly and (p_ens >= 0.15 or abs(record.get("thermal_gradient", 0.0)) > 0.45):
                    if not pipeline_state["early_warning_fired"]:
                        pipeline_state["early_warning_fired"] = True
                        precursor_name = "Thermal Gradient Rising" if abs(record.get("thermal_gradient", 0.0)) > 0.45 else "Ensemble Precursor Drift"
                        audit_logger.log_early_prediction(
                            precursor_feature=precursor_name,
                            probability=p_ens,
                            threshold=dyn_thresh,
                            recommended_action=rl_action,
                            telemetry={"voltage": record["voltage"], "current": record["current"], "temperature": record["temperature"], "soc": record["soc"]},
                            models={"p_rf": record["p_rf"], "p_xgboost": record["p_xgboost"], "p_extra_trees": record["p_extra_trees"], "p_ensemble": p_ens}
                        )
                elif not is_anomaly:
                    pipeline_state["early_warning_fired"] = False

                # 2. Anomaly State Ingress & Progression
                if is_anomaly:
                    if not pipeline_state["was_in_anomaly"] or (pri_fault != pipeline_state["last_fault"]):
                        pipeline_state["was_in_anomaly"] = True
                        pipeline_state["last_fault"] = pri_fault
                        audit_logger.log_fault_event(
                            fault_type=pri_fault,
                            severity=severity,
                            details=f"P(Anomaly)={p_ens:.3f} >= Tau={dyn_thresh:.2f}. Status: {record.get('final_status', 'CRITICAL')}",
                            telemetry={"voltage": record["voltage"], "current": record["current"], "temperature": record["temperature"], "soc": record["soc"], "power_watts": record["power_watts"]},
                            models={"p_rf": record["p_rf"], "p_xgboost": record["p_xgboost"], "p_extra_trees": record["p_extra_trees"], "p_ensemble": p_ens}
                        )

                    # 3. RL Mitigation Trigger & Counterfactual Tracking
                    if rl_action != "NOMINAL_MONITOR" and rl_action != pipeline_state["last_rl_action"]:
                        pipeline_state["last_rl_action"] = rl_action
                        cfs = record.get("counterfactuals", [])
                        best_cf = next((c for c in cfs if c.get("is_recommended")), cfs[0] if cfs else {})
                        cf_temp = best_cf.get("projected_temp_60s", record["temperature"])
                        cf_safe = best_cf.get("safety_score", 1.0)
                        audit_logger.log_mitigation_event(
                            action_name=rl_action,
                            target_fault=pri_fault,
                            counterfactual_temp=cf_temp,
                            safety_score=cf_safe
                        )

                    # 4. Deterministic Safety Override
                    if override_act:
                        audit_logger.log_event(
                            event_type="SAFETY_OVERRIDE",
                            level="SAFETY",
                            subsystem="SAFETY_GATEWAY",
                            message=f"Deterministic Safety Guardrail Active! Violations: {', '.join(record.get('safety_violations', []))}",
                            telemetry={"voltage": record["voltage"], "current": record["current"], "temperature": record["temperature"]}
                        )
                else:
                    # 5. Recovery Detection
                    if pipeline_state["was_in_anomaly"]:
                        prev_f = pipeline_state["last_fault"]
                        prev_act = pipeline_state["last_rl_action"]
                        pipeline_state["was_in_anomaly"] = False
                        pipeline_state["last_fault"] = "NOMINAL"
                        pipeline_state["last_rl_action"] = "NOMINAL_MONITOR"
                        audit_logger.log_recovery_event(
                            mitigating_action=prev_act,
                            resolved_fault=prev_f,
                            details=f"Subsystems normalized to nominal operating limits (V={record['voltage']:.2f}V, T={record['temperature']:.1f}°C)."
                        )

                # 6. Periodic Baseline Telemetry Frame Logging (every 4 nominal frames)
                if fc % 4 == 0:
                    audit_logger.log_event(
                        event_type="TELEMETRY_FRAME",
                        level="INFO",
                        subsystem="EPS_CORE",
                        message=f"Telemetry Cruise: V={record['voltage']:.3f}V | I={record['current']:.3f}A | T={record['temperature']:.1f}°C | SOC={record['soc']*100:.1f}% | P_ens={p_ens:.3f}",
                        telemetry={"voltage": record["voltage"], "current": record["current"], "temperature": record["temperature"], "soc": record["soc"], "power_watts": record["power_watts"], "impedance_proxy": record["impedance_proxy"]},
                        models={"p_rf": record["p_rf"], "p_xgboost": record["p_xgboost"], "p_extra_trees": record["p_extra_trees"], "p_ensemble": p_ens, "dynamic_threshold": dyn_thresh}
                    )

        await asyncio.sleep(stream_delay)


@app.get("/favicon.ico")
async def get_favicon():
    svg_icon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🛰️</text></svg>'
    return Response(content=svg_icon, media_type="image/svg+xml")


@app.get("/")
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/research")
@app.get("/rd")
@app.get("/whitepaper")
async def get_research_paper(request: Request):
    return templates.TemplateResponse("research.html", {"request": request})


@app.post("/api/inject_fault")
async def inject_fault_endpoint(req: FaultInjectionRequest):
    global pipeline_instance
    if pipeline_instance and pipeline_instance.stream:
        pipeline_instance.stream.inject_fault(req.fault_type, duration_sec=req.duration_sec)
        audit_logger.log_operator_action(
            action_name=f"INJECT_FAULT_{req.fault_type.upper()}",
            details=f"Duration: {req.duration_sec} seconds."
        )
        return {"status": "success", "injected": req.fault_type, "duration": req.duration_sec}
    return {"status": "error", "message": "Pipeline not initialized"}


@app.post("/api/clear_fault")
async def clear_fault_endpoint():
    global pipeline_instance
    if pipeline_instance and pipeline_instance.stream:
        pipeline_instance.stream.clear_fault()
        if hasattr(pipeline_instance, "feature_extractor"):
            pipeline_instance.feature_extractor.reset_stream_buffer()
        if hasattr(pipeline_instance, "recent_alerts"):
            pipeline_instance.recent_alerts.clear()
        audit_logger.log_operator_action(
            action_name="CLEAR_ALL_FAULTS",
            details="All fault bus lines cleared, EPS telemetry reset to nominal."
        )
        return {"status": "success", "message": "Fault cleared and telemetry returned to nominal"}
    return {"status": "error"}


@app.post("/api/authorize_mitigation")
async def authorize_mitigation_endpoint(req: MitigationAuthRequest):
    global pipeline_instance, autopilot_mode
    if req.autopilot is not None:
        autopilot_mode = req.autopilot
    if pipeline_instance:
        pipeline_instance.human_approval_override = req.authorized
    audit_logger.log_operator_action(
        action_name="AUTHORIZE_MITIGATION" if req.authorized else "MODE_CHANGE",
        details=f"Authorized={req.authorized}, AutopilotMode={autopilot_mode}"
    )
    return {"status": "success", "authorized": req.authorized, "autopilot": autopilot_mode}


@app.get("/api/logs/all")
async def get_logs_endpoint(
    limit: int = Query(default=200, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
    category: Optional[str] = Query(default=None),
    level: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None)
):
    """Fetches structured logs from the 10,000-line buffer with stats"""
    logs = audit_logger.get_logs(
        limit=limit,
        offset=offset,
        category=category,
        level=level,
        search=search,
        reverse=True
    )
    stats = audit_logger.get_stats()
    return {"logs": logs, "stats": stats}


@app.post("/api/logs/action")
async def post_operator_action_endpoint(req: OperatorActionRequest):
    """Records frontend operator interactions (button clicks, camera shifts, view changes)"""
    record = audit_logger.log_operator_action(
        action_name=req.action,
        details=req.details or "",
        operator=req.operator or "MissionCommander",
        metadata=req.metadata
    )
    return {"status": "success", "logged_record": record}


@app.get("/api/logs/export_pdf")
async def export_logs_pdf_endpoint():
    """Generates and serves a publication-grade aerospace PDF report of the audit logs"""
    logs = audit_logger.get_logs(limit=10000, reverse=True)
    stats = audit_logger.get_stats()
    output_pdf = "docs/AERO_GUARD_MISSION_AUDIT_REPORT.pdf"
    
    generate_mission_audit_pdf(
        logs=logs,
        stats=stats,
        output_path=output_pdf,
        title="AERO-GUARD MISSION CONTROL AUDIT LEDGER"
    )
    
    filename = f"AERO_GUARD_MISSION_AUDIT_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"
    return FileResponse(
        path=output_pdf,
        filename=filename,
        media_type="application/pdf"
    )


@app.post("/api/logs/clear")
async def clear_logs_endpoint():
    """Resets the 10,000-line buffer"""
    audit_logger.clear_buffer()
    return {"status": "success", "message": "10,000-line buffer cleared successfully."}


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
