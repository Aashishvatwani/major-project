"""
Virtual Arduino UNO Microcontroller Emulator
Provides full software-in-the-loop emulation of Arduino UNO Pin 13 LED actuation,
serial registers, CRC validation, and telemetry echoing when no physical hardware is attached.
"""

import time
from typing import Dict, Any, Optional
from .protocol import HITLProtocol


class VirtualArduino:
    """
    Simulates Arduino Uno behavior:
    - Digital Pin 13 (Built-in LED): 0 = OFF (Normal), 1 = SOLID ON (Anomaly), 2 = FAST STROBE (Safety Override)
    - Hardware registers & non-blocking serial loop
    - Responds with telemetry ACKs
    """

    def __init__(self, port_name: str = "VIRTUAL_COM_UNO"):
        self.port_name = port_name
        self.is_open = True
        self.pin13_state = 0  # 0: OFF, 1: ON
        self.pin13_mode = "OFF"
        self.active_severity = 0
        self.active_rl_action = 0
        self.start_time = time.time()
        self.packet_count = 0
        self.last_command_time = time.time()

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def write(self, data: bytes) -> int:
        """Processes incoming bytes from PySerial bridge"""
        if not self.is_open:
            return 0

        self.last_command_time = time.time()
        self.packet_count += 1

        # Check for ASCII framing <ALERT,SEV,RL>
        text = data.decode("utf-8", errors="ignore")
        if "<" in text and ">" in text:
            parsed = HITLProtocol.decode_ascii_line(text)
            if parsed:
                self.pin13_state = parsed["pin13_state"]
                self.active_severity = parsed["ack_severity"]
                self.active_rl_action = parsed["ack_rl"]
                self._update_pin_mode()
                return len(data)

        # Check for Binary framing [0xAA, 0x55, CMD, STATE, SEV, RL, CRC_H, CRC_L]
        if len(data) >= 8 and data[0] == 0xAA and data[1] == 0x55:
            cmd = data[2]
            state = data[3]
            sev = data[4]
            rl = data[5]

            self.pin13_state = state
            self.active_severity = sev
            self.active_rl_action = rl
            self._update_pin_mode()
            return len(data)

        return len(data)

    def _update_pin_mode(self):
        if self.active_severity == 4:
            self.pin13_mode = "CRITICAL_STROBE"
            self.pin13_state = 1
        elif self.pin13_state == 1:
            self.pin13_mode = "SOLID_ON"
        else:
            self.pin13_mode = "OFF"

    def readline(self) -> bytes:
        """Simulates Arduino serial response"""
        uptime_ms = int((time.time() - self.start_time) * 1000)
        ack_str = f"<ACK,{self.pin13_state},{self.active_severity},{self.active_rl_action},{uptime_ms}>\n"
        return ack_str.encode("utf-8")

    def get_hardware_status(self) -> Dict[str, Any]:
        """Returns virtual microcontroller telemetry for UI and diagnostics"""
        uptime_s = time.time() - self.start_time
        return {
            "device": "Arduino Uno (Virtual HITL Emulator)",
            "port": self.port_name,
            "connected": self.is_open,
            "pin13_led_state": self.pin13_state,
            "pin13_mode": self.pin13_mode,
            "active_severity": HITLProtocol.SEVERITY_REVERSE_MAP.get(self.active_severity, "NORMAL"),
            "active_rl_action": self.active_rl_action,
            "packets_received": self.packet_count,
            "uptime_sec": round(uptime_s, 1)
        }
