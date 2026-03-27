"""Unit tests for examples/common/sts3215_bus.py (no hardware required)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make common importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from common.sts3215_bus import SO101LeaderDriver, _decode_sign_magnitude, _normalize_degrees, _normalize_range_0_100


# ---------------------------------------------------------------------------
# Pure-math helpers
# ---------------------------------------------------------------------------

class TestDecodeSignMagnitude:
    def test_positive_value(self):
        # bit 15 = 0 → positive
        assert _decode_sign_magnitude(100) == 100

    def test_negative_value(self):
        # bit 15 = 1, magnitude = 100 → -100
        assert _decode_sign_magnitude((1 << 15) | 100) == -100

    def test_zero(self):
        assert _decode_sign_magnitude(0) == 0

    def test_max_positive(self):
        # 0x7FFF = 32767
        assert _decode_sign_magnitude(0x7FFF) == 32767

    def test_max_negative(self):
        # 0xFFFF → sign=1, magnitude=32767 → -32767
        assert _decode_sign_magnitude(0xFFFF) == -32767


class TestNormalizeDegrees:
    def test_midpoint_is_zero(self):
        # mid = (851 + 3408) / 2 = 2129.5; val = mid → 0.0
        result = _normalize_degrees(val=2129, range_min=851, range_max=3408)
        assert abs(result) < 0.1  # close to 0

    def test_positive_direction(self):
        # val above mid → positive degrees
        result = _normalize_degrees(val=3000, range_min=851, range_max=3408)
        assert result > 0

    def test_negative_direction(self):
        result = _normalize_degrees(val=1000, range_min=851, range_max=3408)
        assert result < 0

    def test_known_value(self):
        # mid = (0 + 4095) / 2 = 2047.5; val = 4095; expected = (4095 - 2047.5) * 360 / 4095
        result = _normalize_degrees(val=4095, range_min=0, range_max=4095)
        assert abs(result - (4095 - 2047.5) * 360 / 4095) < 0.01


class TestNormalizeRange0100:
    def test_min_is_zero(self):
        assert _normalize_range_0_100(val=1968, range_min=1968, range_max=3526) == pytest.approx(0.0)

    def test_max_is_hundred(self):
        assert _normalize_range_0_100(val=3526, range_min=1968, range_max=3526) == pytest.approx(100.0)

    def test_midpoint(self):
        mid = (1968 + 3526) // 2
        result = _normalize_range_0_100(val=mid, range_min=1968, range_max=3526)
        assert 49 < result < 51


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------

SAMPLE_CALIBRATION = {
    "shoulder_pan":  {"id": 1, "drive_mode": 0, "homing_offset": 1526,  "range_min": 851,  "range_max": 3408},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1377, "range_min": 646,  "range_max": 2982},
    "elbow_flex":    {"id": 3, "drive_mode": 0, "homing_offset": 1156,  "range_min": 824,  "range_max": 3030},
    "wrist_flex":    {"id": 4, "drive_mode": 0, "homing_offset": 801,   "range_min": 84,   "range_max": 2408},
    "wrist_roll":    {"id": 5, "drive_mode": 0, "homing_offset": -597,  "range_min": 0,    "range_max": 4095},
    "gripper":       {"id": 6, "drive_mode": 0, "homing_offset": 1605,  "range_min": 1968, "range_max": 3526},
}


class TestCalibrationLoading:
    def test_loads_from_json(self, tmp_path):
        cal_file = tmp_path / "test_arm.json"
        cal_file.write_text(json.dumps(SAMPLE_CALIBRATION))
        driver = SO101LeaderDriver.__new__(SO101LeaderDriver)
        cal = driver._load_calibration(cal_file)
        assert cal["shoulder_pan"]["range_min"] == 851
        assert cal["gripper"]["range_max"] == 3526

    def test_raises_if_missing(self, tmp_path):
        driver = SO101LeaderDriver.__new__(SO101LeaderDriver)
        with pytest.raises(FileNotFoundError):
            driver._load_calibration(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# SO101LeaderDriver with mocked scservo_sdk
# ---------------------------------------------------------------------------

def _make_mock_scs(raw_values: dict[int, int]):
    """Return a mock scservo_sdk module where sync_read returns raw_values keyed by motor id."""
    scs = MagicMock()
    scs.COMM_SUCCESS = 0

    port_handler = MagicMock()
    port_handler.is_open = False

    def open_port():
        port_handler.is_open = True
        return True

    port_handler.openPort.side_effect = open_port
    port_handler.closePort.side_effect = lambda: setattr(port_handler, "is_open", False)
    scs.PortHandler.return_value = port_handler

    packet_handler = MagicMock()
    packet_handler.write1ByteTxRx.return_value = (0, 0)
    scs.PacketHandler.return_value = packet_handler

    sync_reader = MagicMock()
    sync_reader.txRxPacket.return_value = scs.COMM_SUCCESS
    sync_reader.getData.side_effect = lambda id_, addr, length: raw_values[id_]
    scs.GroupSyncRead.return_value = sync_reader

    return scs, port_handler, packet_handler, sync_reader


class TestSO101LeaderDriver:
    def _make_driver(self, tmp_path, raw_values):
        """Create a driver with a temp calibration file and mocked scservo_sdk."""
        cal_file = tmp_path / "arm.json"
        cal_file.write_text(json.dumps(SAMPLE_CALIBRATION))

        scs_mock, ph, pah, sr = _make_mock_scs(raw_values)
        with patch.dict("sys.modules", {"scservo_sdk": scs_mock}):
            driver = SO101LeaderDriver(
                port="/dev/ttyACM0",
                calibration_path=cal_file,
            )
        driver._scs = scs_mock
        driver._port_handler = ph
        driver._packet_handler = pah
        driver._sync_reader = sr
        return driver

    def test_is_connected_false_before_connect(self, tmp_path):
        driver = self._make_driver(tmp_path, {})
        assert driver.is_connected is False

    def test_connect_opens_port(self, tmp_path):
        driver = self._make_driver(tmp_path, {})
        driver.connect()
        driver._port_handler.openPort.assert_called_once()
        assert driver.is_connected is True

    def test_disconnect_closes_port(self, tmp_path):
        driver = self._make_driver(tmp_path, {})
        driver.connect()
        driver.disconnect()
        driver._port_handler.closePort.assert_called_once()

    def test_get_action_returns_six_keys(self, tmp_path):
        raw = {
            1: 2129,  # shoulder_pan mid
            2: 1814,  # shoulder_lift mid
            3: 1927,  # elbow_flex mid
            4: 1246,  # wrist_flex mid
            5: 2047,  # wrist_roll mid
            6: 2747,  # gripper mid
        }
        driver = self._make_driver(tmp_path, raw)
        driver.connect()
        action = driver.get_action()
        assert set(action.keys()) == {
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        }

    def test_get_action_body_joint_near_zero_at_mid(self, tmp_path):
        mid_shoulder_pan = int((851 + 3408) / 2)  # 2129
        raw = {1: mid_shoulder_pan, 2: 1814, 3: 1927, 4: 1246, 5: 2047, 6: 2747}
        driver = self._make_driver(tmp_path, raw)
        driver.connect()
        action = driver.get_action()
        assert abs(action["shoulder_pan.pos"]) < 1.0

    def test_get_action_gripper_near_fifty_at_mid(self, tmp_path):
        mid_gripper = (1968 + 3526) // 2  # 2747
        raw = {1: 2129, 2: 1814, 3: 1927, 4: 1246, 5: 2047, 6: mid_gripper}
        driver = self._make_driver(tmp_path, raw)
        driver.connect()
        action = driver.get_action()
        assert 49 < action["gripper.pos"] < 51
