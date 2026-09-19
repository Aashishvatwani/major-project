"""
Verification script for FastAPI audit endpoints
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.dashboard.app import app

client = TestClient(app)

def test_api():
    print("Testing GET /api/logs/all...")
    res = client.get("/api/logs/all?limit=50")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert "logs" in data
    assert "stats" in data
    print(f"[PASS] /api/logs/all returned {len(data['logs'])} logs, stats: {data['stats']}")

    print("\nTesting POST /api/logs/action...")
    res = client.post("/api/logs/action", json={
        "action": "TEST_CAMERA_VIEW_IR",
        "details": "Operator switched view to IR thermal",
        "operator": "FlightDirector",
        "metadata": {"mode": "thermal"}
    })
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    print("[PASS] /api/logs/action logged operator action successfully!")

    print("\nTesting GET /api/logs/export_pdf...")
    res = client.get("/api/logs/export_pdf")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert len(res.content) > 1000
    print(f"[PASS] /api/logs/export_pdf returned valid PDF ({len(res.content):,} bytes)!")

    print("\n==========================================")
    print(" ALL FASTAPI AUDIT ENDPOINTS VERIFIED 100%!")
    print("==========================================")

if __name__ == "__main__":
    test_api()
