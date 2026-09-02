import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.pipeline.hitl_pipeline import SatelliteHITLPipeline
from src.ingestion.data_loader import TelemetryDataLoader
import time

pipeline = SatelliteHITLPipeline(models_dir='saved_models', serial_port='AUTO', enable_hardware=False)
loader = TelemetryDataLoader('data/raw/synthetic_telemetry.csv')
df = loader.load_or_generate_dataset(duration_minutes=360.0)
pipeline.stream.set_dataset(df)

for fault in ["thermal_runaway", "internal_short", "undervoltage", "high_impedance", "sensor_fault"]:
    print(f"\n==================== FAULT: {fault.upper()} ====================")
    pipeline.stream.inject_fault(fault, duration_sec=30.0)
    for step in range(15):
        sample = pipeline.stream.get_next_sample()
        res = pipeline.process_single_sample(sample)
        if step % 3 == 0 or step == 14:
            print(f"Step {step+1:02d}: V={res['voltage']:.2f}V, I={res['current']:.2f}A, T={res['temperature']:.1f}°C | P_ens={res['p_ensemble']:.3f} | Fault={res['primary_fault']} (Conf: {res['diagnosis_confidence']:.2f}) | Sev={res['final_severity']}")
        time.sleep(0.05)
    pipeline.stream.clear_fault()
