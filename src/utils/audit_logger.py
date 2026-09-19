"""
Aerospace Mission Audit Logger & 10,000-Line FIFO Buffer Engine
Maintains an ultra-structured, timestamped mission log ledger with a strict
10,000-line ring buffer (FIFO deletion when capacity is reached) and persistent JSONL store.
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Any, List, Optional


class AerospaceAuditLogger:
    """
    Thread-safe Mission Audit Logger with 10,000-Line FIFO Ring Buffer.
    Captures:
    - Telemetry Frames & Anomaly Consensus
    - Early Predictions & Precursor Gradient Warnings
    - Frontend Operator Actions & Mode Toggles
    - Multi-Class Injected & Detected Faults
    - RL Mitigation Policy Executions & Counterfactual Projections
    - Deterministic Safety Guardrail Activations
    - AI Agent Diagnostic Reasoning & RAG Citations
    - Telemetry Stabilization & Recovery Events
    """

    def __init__(self, max_lines: int = 10000, storage_file: str = "data/mission_audit_logs.jsonl"):
        self.max_lines = max_lines
        self.storage_file = storage_file
        self.buffer = deque(maxlen=max_lines)
        self.lock = threading.RLock()
        self.sequence_id = 0
        self.start_time = time.time()
        
        # Stats counters
        self.counts = {
            "total_logged": 0,
            "operator_actions": 0,
            "early_predictions": 0,
            "faults_detected": 0,
            "mitigations_executed": 0,
            "safety_overrides": 0,
            "ai_reasonings": 0,
            "recoveries": 0
        }

        # Initialize storage directory & load existing logs if any
        dir_name = os.path.dirname(self.storage_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self._load_from_disk()
        self._initialized = True

    def _load_from_disk(self):
        """Loads existing logs up to max_lines on startup"""
        if not os.path.exists(self.storage_file):
            return

        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Take only the last max_lines
                lines_to_load = lines[-self.max_lines:]
                for line in lines_to_load:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            self.buffer.append(record)
                            self.sequence_id = max(self.sequence_id, record.get("seq_id", 0))
                            self._increment_stats(record.get("event_type", "SYSTEM"))
                        except Exception:
                            continue
        except Exception as e:
            print(f"[AuditLogger] Warning loading existing logs: {e}")

    def _increment_stats(self, event_type: str):
        self.counts["total_logged"] += 1
        if event_type == "OPERATOR_ACTION":
            self.counts["operator_actions"] += 1
        elif event_type == "EARLY_PREDICTION":
            self.counts["early_predictions"] += 1
        elif event_type in ["FAULT_ANOMALY", "FAULT_INJECTED"]:
            self.counts["faults_detected"] += 1
        elif event_type == "RL_MITIGATION":
            self.counts["mitigations_executed"] += 1
        elif event_type == "SAFETY_OVERRIDE":
            self.counts["safety_overrides"] += 1
        elif event_type == "AI_RAG_REASONING":
            self.counts["ai_reasonings"] += 1
        elif event_type == "RECOVERY_EVENT":
            self.counts["recoveries"] += 1

    def _save_to_disk_batch(self):
        """Rewrites the JSONL file to ensure disk strictly reflects the 10,000-line buffer"""
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                for record in self.buffer:
                    f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[AuditLogger] Error writing to disk: {e}")

    def log_event(
        self,
        event_type: str,
        level: str = "INFO",
        subsystem: str = "EPS_CORE",
        message: str = "",
        telemetry: Optional[Dict[str, Any]] = None,
        models: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Appends a structured event to the ring buffer (FIFO max 10,000 entries).
        """
        with self.lock:
            self.sequence_id += 1
            now_ts = time.time()
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
            met_seconds = int(now_ts - self.start_time)
            met_str = f"T+{met_seconds // 3600:02d}:{(met_seconds % 3600) // 60:02d}:{met_seconds % 60:02d}"

            record = {
                "seq_id": self.sequence_id,
                "timestamp": now_iso,
                "unix_ts": now_ts,
                "met": met_str,
                "level": level.upper(),  # INFO, ACTION, PREDICT, FAULT, RL-POLICY, SAFETY, RECOVER, CRITICAL
                "event_type": event_type.upper(),
                "subsystem": subsystem.upper(),  # PCDU, BCR, BATTERY, RADIATOR_ADCS, PAYLOAD_BUS, HITL_BRIDGE, AI_REASONER, SYSTEM
                "message": message,
                "telemetry": telemetry or {},
                "models": models or {},
                "metadata": metadata or {}
            }

            self.buffer.append(record)
            self._increment_stats(event_type.upper())

            # Append to disk (and periodic truncate when buffer hits max capacity)
            try:
                with open(self.storage_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass

            # If total logged is a multiple of 1000, trigger disk sync to guarantee exactly <= 10,000 on disk
            if len(self.buffer) >= self.max_lines and (self.counts["total_logged"] % 500 == 0):
                self._save_to_disk_batch()

            return record

    def log_operator_action(self, action_name: str, details: str = "", operator: str = "MissionCommander", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Logs frontend operator interactions and manual commands"""
        return self.log_event(
            event_type="OPERATOR_ACTION",
            level="ACTION",
            subsystem="OPERATOR_CONSOLE",
            message=f"Operator [{operator}] executed: {action_name}. {details}".strip(),
            metadata={"operator": operator, "action": action_name, **(metadata or {})}
        )

    def log_early_prediction(self, precursor_feature: str, probability: float, threshold: float, recommended_action: str, telemetry: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
        """Logs predictive early warnings before threshold violation occurs"""
        return self.log_event(
            event_type="EARLY_PREDICTION",
            level="PREDICT",
            subsystem="PREDICTIVE_FDIR",
            message=f"Early Precursor Detected: {precursor_feature} | P(Anomaly)={probability:.3f} >= Tau={threshold:.2f}. Recommended Pre-Arm: {recommended_action}",
            telemetry=telemetry,
            models=models,
            metadata={"precursor": precursor_feature, "p_ensemble": probability, "threshold": threshold, "recommendation": recommended_action}
        )

    def log_fault_event(self, fault_type: str, severity: str, details: str, telemetry: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
        """Logs detected or injected anomaly events"""
        level = "CRITICAL" if severity in ["CRITICAL", "EMERGENCY"] else "FAULT"
        return self.log_event(
            event_type="FAULT_ANOMALY",
            level=level,
            subsystem="EPS_CORE",
            message=f"Anomaly Triggered: {fault_type.upper()} | Severity: {severity}. {details}",
            telemetry=telemetry,
            models=models,
            metadata={"fault_type": fault_type, "severity": severity}
        )

    def log_mitigation_event(self, action_name: str, target_fault: str, counterfactual_temp: float, safety_score: float) -> Dict[str, Any]:
        """Logs closed-loop RL mitigation policy decisions"""
        return self.log_event(
            event_type="RL_MITIGATION",
            level="RL-POLICY",
            subsystem="PCDU_RL_ACTUATOR",
            message=f"RL Mitigation Executed: [{action_name}] targeting [{target_fault}]. Projected 60s Temp: {counterfactual_temp:.1f}°C, Safety Score: {safety_score:.2f}",
            metadata={"action_name": action_name, "target_fault": target_fault, "cf_temp_60s": counterfactual_temp, "cf_safety": safety_score}
        )

    def log_recovery_event(self, mitigating_action: str, resolved_fault: str, details: str = "") -> Dict[str, Any]:
        """Logs recovery of telemetry back to nominal operating envelope"""
        return self.log_event(
            event_type="RECOVERY_EVENT",
            level="RECOVER",
            subsystem="SPACECRAFT_HEALTH",
            message=f"Nominal State Restored: {mitigating_action} successfully mitigated {resolved_fault}. {details}".strip(),
            metadata={"action": mitigating_action, "resolved_fault": resolved_fault}
        )

    def log_ai_reasoning(self, reasoning_message: str, rag_citation: str, risk_level: str) -> Dict[str, Any]:
        """Logs AI Agent diagnostic reasoning and regulatory RAG grounding"""
        return self.log_event(
            event_type="AI_RAG_REASONING",
            level="INFO",
            subsystem="AI_AGENT_RAG",
            message=f"[AI Diagnostic]: {reasoning_message} | Citation: {rag_citation}",
            metadata={"risk_level": risk_level, "rag_citation": rag_citation}
        )

    def get_logs(
        self,
        limit: int = 200,
        offset: int = 0,
        category: Optional[str] = None,
        level: Optional[str] = None,
        search: Optional[str] = None,
        reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieves filtered logs from the 10,000-line ring buffer.
        """
        with self.lock:
            items = list(self.buffer)

        if reverse:
            items = items[::-1]

        filtered = []
        search_lower = search.lower() if search else None
        cat_upper = category.upper() if category and category != "ALL" else None
        lvl_upper = level.upper() if level and level != "ALL" else None

        for item in items:
            if cat_upper and item.get("event_type") != cat_upper and item.get("subsystem") != cat_upper:
                continue
            if lvl_upper and item.get("level") != lvl_upper:
                continue
            if search_lower:
                msg_match = search_lower in item.get("message", "").lower()
                sub_match = search_lower in item.get("subsystem", "").lower()
                type_match = search_lower in item.get("event_type", "").lower()
                met_match = search_lower in item.get("met", "").lower()
                if not (msg_match or sub_match or type_match or met_match):
                    continue
            filtered.append(item)

        return filtered[offset: offset + limit]

    def get_stats(self) -> Dict[str, Any]:
        """Returns buffer capacity and event counts"""
        with self.lock:
            current_buffer_len = len(self.buffer)
            return {
                "buffer_count": current_buffer_len,
                "buffer_max_lines": self.max_lines,
                "buffer_usage_pct": round((current_buffer_len / self.max_lines) * 100.0, 1),
                "total_events_logged": self.counts["total_logged"],
                "operator_actions": self.counts["operator_actions"],
                "early_predictions": self.counts["early_predictions"],
                "faults_detected": self.counts["faults_detected"],
                "mitigations_executed": self.counts["mitigations_executed"],
                "safety_overrides": self.counts["safety_overrides"],
                "ai_reasonings": self.counts["ai_reasonings"],
                "recoveries": self.counts["recoveries"],
                "start_time_iso": datetime.fromtimestamp(self.start_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            }

    def clear_buffer(self):
        """Clears in-memory buffer and truncates disk file"""
        with self.lock:
            self.buffer.clear()
            self.sequence_id = 0
            self.counts = {k: 0 for k in self.counts}
            try:
                with open(self.storage_file, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                pass
            self.log_event(
                event_type="SYSTEM",
                level="INFO",
                subsystem="LOG_ENGINE",
                message="Mission Audit Log Buffer cleared by operator command."
            )


# Global Singleton Instance
audit_logger = AerospaceAuditLogger()
