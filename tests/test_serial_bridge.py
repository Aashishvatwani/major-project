"""
Unit Tests for HITL Serial Protocol, Virtual Arduino, and PySerial Bridge
"""

import pytest
from src.hitl.protocol import HITLProtocol
from src.hitl.virtual_arduino import VirtualArduino
from src.hitl.serial_bridge import HITLSerialBridge


def test_protocol_crc_and_encoding():
    """Verify CRC16 CCITT and packet encoding/decoding"""
    pkt = HITLProtocol.encode_command_packet(alert_state=1, severity_str="CRITICAL", rl_action=2)
    assert len(pkt) == 8
    assert pkt[0] == 0xAA and pkt[1] == 0x55
    assert pkt[2] == HITLProtocol.CMD_SET_ALERT
    assert pkt[3] == 1  # Alert ON
    assert pkt[4] == 3  # Critical Severity
    assert pkt[5] == 2  # RL action

    # ASCII framing
    ascii_line = HITLProtocol.encode_ascii_line(alert_state=1, severity_str="OVERRIDE", rl_action=3)
    assert ascii_line == "<1,4,3>\n"

    parsed = HITLProtocol.decode_ascii_line("<1,4,3,1500>")
    assert parsed is not None
    assert parsed["pin13_state"] == 1
    assert parsed["ack_severity"] == 4
    assert parsed["ack_rl"] == 3


def test_virtual_arduino_actuation():
    """Verify Virtual Arduino receives packets, switches Pin 13, and sends ACKs"""
    uno = VirtualArduino()
    assert uno.pin13_state == 0

    # Send Alert ON
    uno.write(b"<1,2,0>\n")
    assert uno.pin13_state == 1
    assert uno.pin13_mode == "SOLID_ON"

    # Send Critical Safety Override
    uno.write(b"<1,4,3>\n")
    assert uno.pin13_state == 1
    assert uno.pin13_mode == "CRITICAL_STROBE"

    ack_bytes = uno.readline()
    assert b"<ACK," in ack_bytes

    status = uno.get_hardware_status()
    assert status["connected"] is True
    assert status["pin13_led_state"] == 1


def test_serial_bridge_virtual_fallback():
    """Verify Serial Bridge operates seamlessly with virtual fallback"""
    bridge = HITLSerialBridge(port="VIRTUAL", auto_fallback_virtual=True)
    assert bridge.is_virtual is True

    hw_info = bridge.send_anomaly_alert(alert_state=1, severity="CRITICAL", rl_action=3)
    assert hw_info["pin13_led_state"] == 1

    status = bridge.get_status()
    assert status["pin13_led_state"] == 1
    bridge.close()
