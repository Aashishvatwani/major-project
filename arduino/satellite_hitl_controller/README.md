# Arduino UNO Hardware-in-the-Loop (HITL) Controller

This directory contains the production C++ firmware for the **Arduino Uno / Nano / Mega** used in the Satellite Telemetry Anomaly Detection Hardware-in-the-Loop testbed.

---

## Pinout Configuration

| Arduino Pin | Connection / Component | Function | Behavior |
|---|---|---|---|
| **Digital Pin 13** | Built-in LED / Optocoupler | Primary Anomaly Actuator | **OFF**: Nominal<br>**SOLID ON**: Anomaly Alert<br>**FAST STROBE (50ms)**: Critical Safety Override<br>**PULSE (250ms)**: Load Shedding Warning |
| **Digital Pin 12** | Relay Module / Buzzer (Optional) | Safe Mode Bus Isolation | **LOW**: Bus Active<br>**HIGH**: Bus Isolated / Alarm Sound |
| **USB Type-B / COM** | Python Host (PySerial) | Full Duplex Serial Telemetry | 115200 baud, 8-N-1 |

---

## Flashing Instructions

1. Connect your **Arduino Uno** via USB.
2. Open the **Arduino IDE** (or use `arduino-cli`).
3. Open `satellite_hitl_controller.ino`.
4. Select **Board**: *Arduino Uno* and **Port**: *(e.g. COM3 or /dev/ttyACM0)*.
5. Click **Upload** ($\rightarrow$).

---

## Serial Protocol Specification

- **Baud Rate**: `115200`
- **Format**: ASCII bracketed frame `<ALERT_STATE,SEVERITY_CODE,RL_ACTION>\n`
  - `ALERT_STATE`: `0` (Normal) or `1` (Anomaly)
  - `SEVERITY_CODE`: `0` (Nominal), `1` (Elevated), `2` (Warning), `3` (Critical), `4` (Critical Safety Override)
  - `RL_ACTION`: `0` (Nominal), `1` (High Sensitivity Pre-arm), `2` (Load Shedding), `3` (Safe Mode)
- **Arduino Response**: `<ACK,PIN13_STATE,SEVERITY,RL_ACTION,UPTIME_MS>\n`
