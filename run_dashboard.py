"""
Launch Script for Satellite Mission Control Web Dashboard
Runs FastAPI + Uvicorn server hosting real-time WebSocket telemetry stream.
"""

import uvicorn
import webbrowser
import threading
import time
import argparse


def open_browser(url: str, delay: float = 1.5):
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Satellite HITL Mission Control Dashboard Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 65)
    print(" [*] AERO-GUARD SATELLITE TELEMETRY HITL MISSION CONTROL DASHBOARD ")
    print("=" * 65)
    print(f" Dashboard URL: {url}")
    print(" Real-time WebSocket: ws://127.0.0.1:8000/ws/telemetry")
    print("=" * 65)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    uvicorn.run("src.dashboard.app:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()
