"""Standalone STS3215 servo bus driver for the SO101 leader and follower arms.

Replaces the lerobot dependency with a direct scservo_sdk implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import scservo_sdk as scs

# ---------------------------------------------------------------------------
# Register addresses (Protocol 0, STS/SMS series)
# ---------------------------------------------------------------------------
ADDR_RETURN_DELAY_TIME = 7
ADDR_MAX_TORQUE_LIMIT = 16
ADDR_P_COEFFICIENT = 21
ADDR_D_COEFFICIENT = 22
ADDR_I_COEFFICIENT = 23
ADDR_PROTECTION_CURRENT = 28
ADDR_OVERLOAD_TORQUE = 36
ADDR_OPERATING_MODE = 33
ADDR_TORQUE_ENABLE = 40
ADDR_ACCELERATION = 41
ADDR_GOAL_POSITION = 42
ADDR_LOCK = 55
ADDR_PRESENT_POSITION = 56
ADDR_MAXIMUM_ACCELERATION = 85

BAUD_RATE = 1_000_000

# Motor name -> id mapping (fixed hardware)
MOTOR_NAMES = [
    "shoulder_pan",  # id=1
    "shoulder_lift",  # id=2
    "elbow_flex",  # id=3
    "wrist_flex",  # id=4
    "wrist_roll",  # id=5
    "gripper",  # id=6
]

GRIPPER_MOTOR = "gripper"
GRIPPER_ID = 6


# ---------------------------------------------------------------------------
# Pure-math helpers
# ---------------------------------------------------------------------------


def _decode_sign_magnitude(val: int) -> int:
    """Decode a sign-magnitude encoded 16-bit integer.

    Bit 15 is the sign bit; bits 0-14 are the magnitude.
    """
    sign = (val >> 15) & 1
    magnitude = val & 0x7FFF
    return -magnitude if sign else magnitude


def _normalize_degrees(val: int, range_min: int, range_max: int) -> float:
    """Normalise a raw position to degrees, centred at the midpoint of the range."""
    mid = (range_min + range_max) / 2.0
    return (val - mid) * 360.0 / 4095.0


def _normalize_range_0_100(val: int, range_min: int, range_max: int) -> float:
    """Normalise a raw position to the 0-100 range (used for the gripper)."""
    return (val - range_min) / (range_max - range_min) * 100.0


def _encode_sign_magnitude(val: int) -> int:
    """Encode a signed integer using sign-magnitude (bit 15 is sign)."""
    return (1 << 15 | abs(val)) if val < 0 else val


def _unnormalize_degrees(val: float, range_min: int, range_max: int) -> int:
    """Convert degrees back to raw position (inverse of _normalize_degrees)."""
    mid = (range_min + range_max) / 2.0
    return int(round(val * 4095.0 / 360.0 + mid))


def _unnormalize_range_0_100(val: float, range_min: int, range_max: int) -> int:
    """Convert 0-100 gripper value back to raw position."""
    return int(round(val / 100.0 * (range_max - range_min) + range_min))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class SO101LeaderDriver:
    """Driver for the SO101 leader arm using the STS3215 servo bus."""

    def __init__(
        self,
        port: str,
        calibration_path: Optional[Path] = None,
        calibration_id: Optional[str] = None,
    ) -> None:
        """Initialize the leader driver with either a path or calibration id."""
        self._port = port

        # Resolve calibration
        if calibration_path is not None:
            cal_path = Path(calibration_path)
        elif calibration_id is not None:
            base = (
                Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
                / "lerobot"
                / "calibration"
                / "teleoperators"
                / "so_leader"
            )
            cal_path = base / f"{calibration_id}.json"
        else:
            raise ValueError(
                "Either calibration_path or calibration_id must be provided."
            )

        self._calibration = self._load_calibration(cal_path)

        # Build id -> name mapping from calibration
        self._id_to_name: dict[int, str] = {
            info["id"]: name for name, info in self._calibration.items()
        }

        # scservo_sdk objects
        self._scs = scs
        self._port_handler = scs.PortHandler(port)
        self._packet_handler = scs.PacketHandler(0)
        self._sync_reader = scs.GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            ADDR_PRESENT_POSITION,
            2,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _load_calibration(self, path: Path) -> dict[str, dict]:
        """Load calibration from a JSON file.

        Raises FileNotFoundError if the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with path.open() as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return whether the serial port is currently open."""
        return self._port_handler.is_open

    def connect(self, calibrate: bool = False) -> None:
        """Open the serial port and configure all motors."""
        if not self._port_handler.openPort():
            raise ConnectionError(f"Failed to open port '{self._port}'.")
        self._port_handler.setBaudRate(BAUD_RATE)
        self._configure_motors()

    def disconnect(self) -> None:
        """Close the serial port."""
        self._port_handler.closePort()

    # ------------------------------------------------------------------
    # Motor configuration
    # ------------------------------------------------------------------

    def _configure_motors(self) -> None:
        """Configure all motors in position-control mode.

        Disables lock and torque, sets timing/acceleration registers, and
        configures operating mode before normal command traffic begins.
        """
        for name, info in self._calibration.items():
            motor_id = info["id"]
            # Disable lock (EEPROM write protection)
            self._write_1byte(motor_id, ADDR_LOCK, 0)
            # Disable torque
            self._write_1byte(motor_id, ADDR_TORQUE_ENABLE, 0)
            # Set return delay time to 0
            self._write_1byte(motor_id, ADDR_RETURN_DELAY_TIME, 0)
            # Set maximum acceleration
            self._write_1byte(motor_id, ADDR_MAXIMUM_ACCELERATION, 254)
            # Set acceleration
            self._write_1byte(motor_id, ADDR_ACCELERATION, 254)
            # Set operating mode (0 = position control)
            self._write_1byte(motor_id, ADDR_OPERATING_MODE, 0)

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    def _write_1byte(self, motor_id: int, addr: int, value: int) -> None:
        """Write a single byte to a motor register."""
        self._packet_handler.write1ByteTxRx(self._port_handler, motor_id, addr, value)

    def _sync_read_raw(self) -> dict[int, int]:
        """Sync-read Present_Position from all motors.

        Returns a dict mapping motor id -> raw position value.
        """
        motor_ids = [info["id"] for info in self._calibration.values()]

        self._sync_reader.clearParam()
        self._sync_reader.start_address = ADDR_PRESENT_POSITION
        self._sync_reader.data_length = 2
        for id_ in motor_ids:
            self._sync_reader.addParam(id_)

        comm = self._sync_reader.txRxPacket()
        if comm != self._scs.COMM_SUCCESS:
            raise IOError(f"Sync read failed with error code {comm}")

        result: dict[int, int] = {}
        for id_ in motor_ids:
            result[id_] = self._sync_reader.getData(id_, ADDR_PRESENT_POSITION, 2)
        return result

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def get_action(self) -> dict[str, float]:
        """Read all motor positions and return normalised values.

        Returns a dict with keys like "shoulder_pan.pos", "gripper.pos", etc.
        """
        raw_positions = self._sync_read_raw()
        action: dict[str, float] = {}

        for name, info in self._calibration.items():
            motor_id = info["id"]
            raw = raw_positions[motor_id]
            decoded = _decode_sign_magnitude(raw)

            if name == GRIPPER_MOTOR:
                value = _normalize_range_0_100(
                    decoded, info["range_min"], info["range_max"]
                )
            else:
                value = _normalize_degrees(
                    decoded, info["range_min"], info["range_max"]
                )

            action[f"{name}.pos"] = value

        return action


# ---------------------------------------------------------------------------
# Follower driver
# ---------------------------------------------------------------------------


class SO101FollowerDriver:
    """Driver for the SO101 follower arm.

    Replaces lerobot.robots.so_follower.SO101Follower. Exposes:
    - connect() / disconnect()
    - is_connected
    - get_observation() -> dict[str, float]  (same format as SO101LeaderDriver.get_action)
    - send_action(action: dict[str, float])

    Calibration JSON path (default):
      ~/.cache/huggingface/lerobot/calibration/robots/so_follower/{follower_id}.json
    """

    def __init__(
        self,
        port: str,
        calibration_path: Optional[Path] = None,
        follower_id: Optional[str] = None,
    ) -> None:
        """Initialize the follower driver with either a path or follower id."""
        self._port = port

        if calibration_path is not None:
            cal_path = Path(calibration_path)
        elif follower_id is not None:
            base = (
                Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
                / "lerobot"
                / "calibration"
                / "robots"
                / "so_follower"
            )
            cal_path = base / f"{follower_id}.json"
        else:
            raise ValueError("Either calibration_path or follower_id must be provided.")

        self._calibration = self._load_calibration(cal_path)

        self._scs = scs
        self._port_handler = scs.PortHandler(port)
        self._packet_handler = scs.PacketHandler(0)
        self._sync_reader = scs.GroupSyncRead(
            self._port_handler, self._packet_handler, ADDR_PRESENT_POSITION, 2
        )
        self._sync_writer = scs.GroupSyncWrite(
            self._port_handler, self._packet_handler, ADDR_GOAL_POSITION, 2
        )

    def _load_calibration(self, path: Path) -> dict[str, dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {path}\n"
                "Run: lerobot-calibrate --robot.type=so101_follower "
                "--robot.port=<port> --robot.id=<id>"
            )
        with path.open() as f:
            return json.load(f)

    @property
    def is_connected(self) -> bool:
        """Return whether the serial port is currently open."""
        return self._port_handler.is_open

    def connect(self, calibrate: bool = False) -> None:
        """Open serial port, configure motors, and enable torque."""
        if not self._port_handler.openPort():
            raise ConnectionError(f"Failed to open port '{self._port}'.")
        self._port_handler.setBaudRate(BAUD_RATE)
        self._configure_motors()

    def disconnect(self, disable_torque: bool = True) -> None:
        """Close the serial port, optionally disabling torque first."""
        if disable_torque:
            for info in self._calibration.values():
                self._write_1byte(info["id"], ADDR_TORQUE_ENABLE, 0)
                self._write_1byte(info["id"], ADDR_LOCK, 0)
        self._port_handler.closePort()

    def _write_1byte(self, motor_id: int, addr: int, value: int) -> None:
        self._packet_handler.write1ByteTxRx(self._port_handler, motor_id, addr, value)

    def _write_2byte(self, motor_id: int, addr: int, value: int) -> None:
        self._packet_handler.write2ByteTxRx(self._port_handler, motor_id, addr, value)

    def _configure_motors(self) -> None:
        """Disable torque, set PID + operating mode, re-enable torque on all motors."""
        for name, info in self._calibration.items():
            mid = info["id"]
            self._write_1byte(mid, ADDR_LOCK, 0)
            self._write_1byte(mid, ADDR_TORQUE_ENABLE, 0)
            self._write_1byte(mid, ADDR_RETURN_DELAY_TIME, 0)
            self._write_1byte(mid, ADDR_MAXIMUM_ACCELERATION, 254)
            self._write_1byte(mid, ADDR_ACCELERATION, 254)
            self._write_1byte(mid, ADDR_OPERATING_MODE, 0)
            self._write_1byte(mid, ADDR_P_COEFFICIENT, 16)
            self._write_1byte(mid, ADDR_I_COEFFICIENT, 0)
            self._write_1byte(mid, ADDR_D_COEFFICIENT, 32)
            if name == GRIPPER_MOTOR:
                self._write_2byte(mid, ADDR_MAX_TORQUE_LIMIT, 500)
                self._write_2byte(mid, ADDR_PROTECTION_CURRENT, 250)
                self._write_1byte(mid, ADDR_OVERLOAD_TORQUE, 25)
            self._write_1byte(mid, ADDR_TORQUE_ENABLE, 1)
            self._write_1byte(mid, ADDR_LOCK, 1)

    def _sync_read_raw(self) -> dict[int, int]:
        motor_ids = [info["id"] for info in self._calibration.values()]
        self._sync_reader.clearParam()
        self._sync_reader.start_address = ADDR_PRESENT_POSITION
        self._sync_reader.data_length = 2
        for id_ in motor_ids:
            self._sync_reader.addParam(id_)
        comm = self._sync_reader.txRxPacket()
        if comm != self._scs.COMM_SUCCESS:
            raise IOError(f"Sync read failed with error code {comm}")
        return {
            id_: self._sync_reader.getData(id_, ADDR_PRESENT_POSITION, 2)
            for id_ in motor_ids
        }

    def get_observation(self) -> dict[str, float]:
        """Read all motor positions. Returns dict with keys like 'shoulder_pan.pos'."""
        raw_positions = self._sync_read_raw()
        obs: dict[str, float] = {}
        for name, info in self._calibration.items():
            decoded = _decode_sign_magnitude(raw_positions[info["id"]])
            if name == GRIPPER_MOTOR:
                value = _normalize_range_0_100(
                    decoded, info["range_min"], info["range_max"]
                )
            else:
                value = _normalize_degrees(
                    decoded, info["range_min"], info["range_max"]
                )
            obs[f"{name}.pos"] = value
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Send goal positions to all motors.

        action: dict with keys like 'shoulder_pan.pos' (degrees) and 'gripper.pos' (0-100).
        Returns the action dict unchanged.
        """
        self._sync_writer.clearParam()
        self._sync_writer.start_address = ADDR_GOAL_POSITION
        self._sync_writer.data_length = 2
        for name, info in self._calibration.items():
            val = action.get(f"{name}.pos")
            if val is None:
                continue
            raw = (
                _unnormalize_range_0_100(val, info["range_min"], info["range_max"])
                if name == GRIPPER_MOTOR
                else _unnormalize_degrees(val, info["range_min"], info["range_max"])
            )
            encoded = _encode_sign_magnitude(raw)
            self._sync_writer.addParam(
                info["id"], [encoded & 0xFF, (encoded >> 8) & 0xFF]
            )
        self._sync_writer.txPacket()
        return action
