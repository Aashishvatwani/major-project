"""
Hardware-in-the-Loop Serial Protocol Specification
Implements robust binary and ASCII packet framing with CRC-16 checksum verification
for bidirectional Python-to-Arduino communication.
"""

import struct
from typing import Dict, Any, Tuple, Optional


class HITLProtocol:
    """
    Protocol Encoder / Decoder for Satellite HITL Actuation.
    Packet Structure (Binary):
      [0] SYNC_1     = 0xAA
      [1] SYNC_2     = 0x55
      [2] CMD_ID     = 0x01 (SET_ALERT), 0x02 (HEARTBEAT), 0x03 (QUERY_STATUS)
      [3] ALERT_BIT  = 0 (OFF), 1 (ON)
      [4] SEVERITY   = 0 (NOMINAL), 1 (ELEVATED), 2 (WARNING), 3 (CRITICAL), 4 (OVERRIDE)
      [5] RL_ACTION  = 0..3
      [6:8] CRC16    = uint16 (CCITT)
    """

    SYNC_1 = 0xAA
    SYNC_2 = 0x55

    CMD_SET_ALERT = 0x01
    CMD_HEARTBEAT = 0x02
    CMD_QUERY = 0x03

    SEVERITY_MAP = {
        "NORMAL": 0,
        "NOMINAL": 0,
        "ELEVATED": 1,
        "WARNING": 2,
        "CRITICAL": 3,
        "CRITICAL_SAFETY_OVERRIDE": 4,
        "OVERRIDE": 4
    }

    SEVERITY_REVERSE_MAP = {v: k for k, v in SEVERITY_MAP.items()}

    @staticmethod
    def calculate_crc16(data: bytes) -> int:
        """Standard CRC-16-CCITT calculation"""
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    @classmethod
    def encode_command_packet(
        cls,
        alert_state: int,
        severity_str: str = "NORMAL",
        rl_action: int = 0,
        cmd_id: int = CMD_SET_ALERT
    ) -> bytes:
        """Encodes binary packet with CRC16"""
        sev_code = cls.SEVERITY_MAP.get(severity_str.upper(), 0)
        payload = bytes([cmd_id, int(alert_state) & 0x01, sev_code, int(rl_action) & 0x07])
        crc = cls.calculate_crc16(payload)
        crc_bytes = struct.pack(">H", crc)
        return bytes([cls.SYNC_1, cls.SYNC_2]) + payload + crc_bytes

    @classmethod
    def encode_ascii_line(
        cls,
        alert_state: int,
        severity_str: str = "NORMAL",
        rl_action: int = 0
    ) -> str:
        """
        Lightweight ASCII format: <ALERT:0|1,SEV:0..4,RL:0..3>\n
        """
        sev_code = cls.SEVERITY_MAP.get(severity_str.upper(), 0)
        return f"<{alert_state},{sev_code},{rl_action}>\n"

    @classmethod
    def decode_ascii_line(cls, line: str) -> Optional[Dict[str, Any]]:
        """Parses ASCII response from Arduino"""
        clean = line.strip()
        if clean.startswith("<") and clean.endswith(">"):
            parts = clean[1:-1].split(",")
            if len(parts) >= 3:
                try:
                    return {
                        "ack_alert": int(parts[0]),
                        "pin13_state": int(parts[0]),
                        "ack_severity": int(parts[1]),
                        "ack_rl": int(parts[2]),
                        "uptime_ms": int(parts[3]) if len(parts) > 3 else 0
                    }
                except ValueError:
                    return None
        return None
