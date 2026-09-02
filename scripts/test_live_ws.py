import asyncio
import websockets
import json

async def test_ws():
    async with websockets.connect('ws://127.0.0.1:8000/ws/telemetry') as ws:
        for i in range(5):
            msg = await ws.recv()
            data = json.loads(msg)
            print("Frame", i+1, "V:", round(data["voltage"], 2), "I:", round(data["current"], 2), "T:", round(data["temperature"], 1), "P_ens:", round(data["p_ensemble"], 3), "Fault:", data["primary_fault"], "Sev:", data["final_severity"])

asyncio.run(test_ws())
