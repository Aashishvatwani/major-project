"""
PySerial Hardware-in-the-Loop Bridge
Connects Python runtime to physical Arduino UNO over USB Serial.
Automatically discovers ports, handles timeouts, and falls back to Virtual Arduino Emulator.
"""

import time
import os
from typing import Dict, Any, Optional, List
from .protocol import HITLProtocol
from .virtual_arduino import VirtualArduino

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False


class HITLSerialBridge:
    """
    Manages physical / virtual serial connection to Arduino UNO.
    Transmits alert states to drive Digital Pin 13 (Built-in LED) and receives status telemetry.
    """

    def __init__(
        self,
        port: str = "AUTO",
        baud_rate: int = 115200,
        timeout: float = 0.5,
        auto_fallback_virtual: bool = True
    ):
        self.port_setting = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.auto_fallback_virtual = auto_fallback_virtual

        self.serial_conn = None
        self.virtual_device: Optional[VirtualArduino] = None
        self.is_virtual = False
        self.active_port_name = "NONE"
        self.last_sent_alert = 0
        self.last_sent_severity = "NORMAL"

        self.connect()

    def discover_ports(self) -> List[str]:
        """Scans system for available serial ports with Arduino/USB descriptions"""
        if not HAS_PYSERIAL:
            return []
        try:
            matched_ports = []
            for p in serial.tools.list_ports.comports():
                desc = (p.description or "").lower()
                mfg = (p.manufacturer or "").lower()
                # Prioritize devices matching microcontroller/USB signatures
                if any(kw in desc or kw in mfg for kw in ["arduino", "ch340", "cp210", "ftdi", "usb serial", "ch341"]):
                    matched_ports.insert(0, p.device)
                else:
                    matched_ports.append(p.device)
            return matched_ports
        except Exception:
            return []

    def connect(self) -> bool:
        """Attempts connection to physical serial port, or initializes virtual emulator"""
        if self.port_setting.upper() != "VIRTUAL" and HAS_PYSERIAL:
            ports = self.discover_ports() if self.port_setting.upper() == "AUTO" else [self.port_setting]
            for p in ports:
                try:
                    conn = serial.Serial(port=p, baudrate=self.baud_rate, timeout=self.timeout, write_timeout=self.timeout)
                    time.sleep(0.5)  # Allow Arduino reset on DTR toggle
                    self.serial_conn = conn
                    self.is_virtual = False
                    self.active_port_name = p
                    self.virtual_device = None
                    print(f"[*] HITL Hardware Bridge connected to physical Arduino on {p} @ {self.baud_rate} baud")
                    return True
                except Exception:
                    continue

        if self.auto_fallback_virtual or not HAS_PYSERIAL:
            self.virtual_device = VirtualArduino(port_name="VIRTUAL_COM_UNO")
            self.is_virtual = True
            self.active_port_name = "VIRTUAL_COM_UNO"
            self.serial_conn = None
            return True

        return False

    def send_anomaly_alert(
        self,
        alert_state: int,
        severity: str = "NORMAL",
        rl_action: int = 0
    ) -> Dict[str, Any]:
        """
        Sends binary and ASCII command to Arduino to drive Pin 13 LED.
        alert_state: 0 (Normal / LED OFF), 1 (Anomaly Alert / LED ON)
        severity: "NORMAL", "ELEVATED", "WARNING", "CRITICAL", "OVERRIDE"
        """
        self.last_sent_alert = int(alert_state)
        self.last_sent_severity = severity

        # Encode formatted ASCII packet: <ALERT,SEV,RL>
        ascii_pkt = HITLProtocol.encode_ascii_line(alert_state, severity, rl_action)
        pkt_bytes = ascii_pkt.encode("utf-8")

        response_info = {}

        if self.is_virtual and self.virtual_device:
            self.virtual_device.write(pkt_bytes)
            ack = self.virtual_device.readline()
            response_info = self.virtual_device.get_hardware_status()
        elif self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(pkt_bytes)
                self.serial_conn.flush()
                raw_ack = self.serial_conn.readline().decode("utf-8", errors="ignore")
                parsed = HITLProtocol.decode_ascii_line(raw_ack)
                response_info = {
                    "device": "Arduino Uno (Physical Hardware)",
                    "port": self.active_port_name,
                    "connected": True,
                    "pin13_led_state": self.last_sent_alert,
                    "active_severity": severity,
                    "active_rl_action": rl_action,
                    "raw_ack": raw_ack.strip(),
                    "parsed_ack": parsed
                }
            except Exception as e:
                print(f"[!] Serial write error: {e}. Switching to virtual emulator.")
                self.is_virtual = True
                self.virtual_device = VirtualArduino(port_name="VIRTUAL_COM_UNO")
                if self.serial_conn:
                    try:
                        self.serial_conn.close()
                    except Exception:
                        pass
                self.serial_conn = None

        return response_info

    def get_status(self) -> Dict[str, Any]:
        """Returns current bridge status"""
        if self.is_virtual and self.virtual_device:
            return self.virtual_device.get_hardware_status()
        return {
            "device": "Arduino Uno (Physical Hardware)",
            "port": self.active_port_name,
            "connected": self.serial_conn is not None and self.serial_conn.is_open,
            "pin13_led_state": self.last_sent_alert,
            "active_severity": self.last_sent_severity,
            "is_virtual": self.is_virtual
        }

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        if self.virtual_device:
            self.virtual_device.close()
