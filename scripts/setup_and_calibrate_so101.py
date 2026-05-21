#!/usr/bin/env python3
"""
Assign motor IDs then auto-calibrate an SO-101 follower arm.
No lerobot library required — only feetech-servo-sdk.

Phase 1 — Motor ID assignment:
    Connect one servo at a time when prompted.
    Each servo is found at whatever ID it currently has and
    reassigned to its correct slot:
        gripper=6, wrist_roll=5, wrist_flex=4,
        elbow_flex=3, shoulder_lift=2, shoulder_pan=1

Phase 2 — Sweep auto-calibration:
    All 6 servos connected. The arm moves on its own to find
    mechanical limits. Stand clear during this phase.

Calibration is saved to:
    ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json

Install dependency (once, inside your conda env):
    pip install feetech-servo-sdk

Usage:
    python scripts/setup_and_calibrate_so101.py \\
        --port /dev/ttyACM0 --id my_follower

    # Skip ID assignment if motors are already set up:
    python scripts/setup_and_calibrate_so101.py \\
        --port /dev/ttyACM0 --id my_follower --skip-setup

    # Dry run (print calibration JSON, do not save):
    python scripts/setup_and_calibrate_so101.py \\
        --port /dev/ttyACM0 --id test --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import scservo_sdk as scs
except ImportError:
    print(
        "ERROR: scservo_sdk not found.\n"
        "Install it:  pip install feetech-servo-sdk\n"
        "Or activate the conda env:  conda activate lerobot"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Register map — STS3215, Protocol 0
# ---------------------------------------------------------------------------

BAUD_RATE = 1_000_000

ADDR_ID                = 5
ADDR_RETURN_DELAY_TIME = 7
ADDR_MIN_POSITION_LIMIT = 9
ADDR_MAX_POSITION_LIMIT = 11
ADDR_MAX_TORQUE_LIMIT  = 16
ADDR_P_COEFFICIENT     = 21
ADDR_D_COEFFICIENT     = 22
ADDR_I_COEFFICIENT     = 23
ADDR_TORQUE_LIMIT      = 28
ADDR_OPERATING_MODE    = 33
ADDR_TORQUE_ENABLE     = 40
ADDR_ACCELERATION      = 41
ADDR_GOAL_POSITION     = 42
ADDR_GOAL_VELOCITY     = 46
ADDR_LOCK              = 55
ADDR_PRESENT_POSITION  = 56
ADDR_PRESENT_VELOCITY  = 58
ADDR_STATUS            = 65

ENCODER_MAX = 4095
ENCODER_MID = 2048

MOTOR_IDS: dict[str, int] = {
    "shoulder_pan":  1,
    "shoulder_lift": 2,
    "elbow_flex":    3,
    "wrist_flex":    4,
    "wrist_roll":    5,
    "gripper":       6,
}

# Distal-to-proximal — same order as lerobot_setup_motors
SETUP_ORDER = [
    ("gripper",       6),
    ("wrist_roll",    5),
    ("wrist_flex",    4),
    ("elbow_flex",    3),
    ("shoulder_lift", 2),
    ("shoulder_pan",  1),
]

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

SWEEP_SPEED         = 150
SWEEP_ACCEL         = 10
STALL_TIMEOUT       = 8.0
SWEEP_SETTLE        = 0.3
BACK_OFF_COUNTS     = 80
STALL_VEL_THRESHOLD = 10
STALL_POS_THRESHOLD = 3
STALL_CONFIRM       = 4
INITIAL_MOVE_DELAY  = 0.5
SAMPLE_INTERVAL     = 0.05

FOLLOWER_UNFOLD_POS: dict[str, int] = {
    "shoulder_lift": 1500,
    "elbow_flex":    2300,
}
FOLLOWER_REST_POS: dict[str, int] = {
    "shoulder_lift": 2004,
    "elbow_flex":    2028,
}
SWEEP_ORDER_OUTER = ["gripper", "wrist_roll", "wrist_flex"]
SWEEP_ORDER_INNER = ["elbow_flex", "shoulder_lift"]


# ---------------------------------------------------------------------------
# Bus wrapper
# ---------------------------------------------------------------------------

class ArmBus:
    def __init__(self, port: str) -> None:
        self.ph   = scs.PortHandler(port)
        self.pkth = scs.PacketHandler(0)
        if not self.ph.openPort():
            raise ConnectionError(
                f"Cannot open port '{port}'.\n"
                f"Check the USB cable and run:  sudo chmod 666 {port}"
            )
        self.ph.setBaudRate(BAUD_RATE)

    def close(self) -> None:
        self.ph.closePort()

    def write1(self, mid: int, addr: int, val: int) -> None:
        self.pkth.write1ByteTxRx(self.ph, mid, addr, val)

    def write2(self, mid: int, addr: int, val: int) -> None:
        self.pkth.write2ByteTxRx(self.ph, mid, addr, val)

    def read2(self, mid: int, addr: int) -> int | None:
        try:
            val, comm, _ = self.pkth.read2ByteTxRx(self.ph, mid, addr)
            return val if comm == scs.COMM_SUCCESS else None
        except Exception:
            return None

    def probe_ids(self) -> list[int]:
        """Return IDs (1-6) that respond on the bus."""
        return [i for i in range(1, 7) if self.read2(i, ADDR_PRESENT_POSITION) is not None]

    def enable_torque(self, mid: int, on: bool) -> None:
        self.write1(mid, ADDR_TORQUE_ENABLE, 1 if on else 0)

    def set_velocity(self, mid: int, velocity: int) -> None:
        encoded = (1 << 15 | abs(velocity)) if velocity < 0 else velocity
        self.write2(mid, ADDR_GOAL_VELOCITY, encoded & 0xFFFF)

    def read_position(self, mid: int) -> int | None:
        return self.read2(mid, ADDR_PRESENT_POSITION)

    def init_motor(self, mid: int) -> None:
        """Write safe EEPROM defaults and re-lock for normal operation."""
        self.write1(mid, ADDR_LOCK, 0)
        self.write1(mid, ADDR_TORQUE_ENABLE, 0)
        self.write1(mid, ADDR_RETURN_DELAY_TIME, 0)
        # Clear any previously written position limits so the full range is sweepable
        self.write2(mid, ADDR_MIN_POSITION_LIMIT, 0)
        self.write2(mid, ADDR_MAX_POSITION_LIMIT, ENCODER_MAX)
        self.write2(mid, ADDR_MAX_TORQUE_LIMIT, 1000)
        self.write2(mid, ADDR_TORQUE_LIMIT, 380)
        self.write1(mid, ADDR_P_COEFFICIENT, 16)
        self.write1(mid, ADDR_D_COEFFICIENT, 32)
        self.write1(mid, ADDR_I_COEFFICIENT, 0)
        self.write1(mid, ADDR_ACCELERATION, SWEEP_ACCEL)
        self.write1(mid, ADDR_LOCK, 1)
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Phase 1: Motor ID assignment
# ---------------------------------------------------------------------------

def _set_motor_id(bus: ArmBus, current_id: int, target_id: int) -> bool:
    bus.write1(current_id, ADDR_LOCK, 0)
    time.sleep(0.05)
    bus.write1(current_id, ADDR_ID, target_id)
    time.sleep(0.1)
    bus.write1(target_id, ADDR_LOCK, 1)
    time.sleep(0.15)
    return bus.read2(target_id, ADDR_PRESENT_POSITION) is not None


def setup_motor_ids(bus: ArmBus) -> bool:
    print("\n" + "=" * 60)
    print("  PHASE 1: Motor ID Assignment")
    print("  You will connect ONE motor at a time.")
    print("  ALL other motors must be DISCONNECTED each time.")
    print("=" * 60)

    for name, target_id in SETUP_ORDER:
        print(f"\n  Next motor : {name}  (will be assigned ID {target_id})")
        input(f"  Connect ONLY [{name}] to the controller board, then press Enter... ")
        time.sleep(0.4)

        found = bus.probe_ids()

        if not found:
            print(f"  ERROR: No motor found. Check that [{name}] is connected and power is on.")
            return False

        if len(found) > 1:
            print(f"  WARNING: {len(found)} motors detected: {found}")
            print(f"           Disconnect all except [{name}] and try again.")
            input("  Press Enter when only one motor is connected... ")
            time.sleep(0.4)
            found = bus.probe_ids()
            if len(found) != 1:
                print(f"  ERROR: Still seeing {len(found)} motors. Aborting.")
                return False

        current_id = found[0]

        if current_id == target_id:
            print(f"  [{name}] already has correct ID {target_id}. OK.")
            continue

        if _set_motor_id(bus, current_id, target_id):
            print(f"  [{name}] ID {current_id} -> {target_id}  OK")
        else:
            print(f"  ERROR: Could not verify [{name}] at ID {target_id} after assignment.")
            return False

    print("\n  All motor IDs assigned.")
    print("  Connect ALL 6 motors to the controller board now.")
    input("  Press Enter when all motors are connected... ")
    time.sleep(0.5)

    present = bus.probe_ids()
    missing = sorted(set(range(1, 7)) - set(present))
    if missing:
        print(f"  WARNING: Motors at IDs {missing} not responding. Check cables.")
    else:
        print(f"  All 6 motors responding. Ready to calibrate.")
    return True


# ---------------------------------------------------------------------------
# Phase 2: Sweep auto-calibration
# ---------------------------------------------------------------------------

def _decode_sm(val: int) -> int:
    return -(val & 0x7FFF) if (val >> 15) & 1 else val


def _wait_for_stall(bus: ArmBus, mid: int, timeout: float) -> int:
    stable = overload = 0
    prev_pos: int | None = None
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        time.sleep(SAMPLE_INTERVAL)
        vel_raw = bus.read2(mid, ADDR_PRESENT_VELOCITY)
        pos_raw = bus.read_position(mid)
        status  = bus.read2(mid, ADDR_STATUS)

        if vel_raw is None or pos_raw is None:
            continue

        vel     = abs(_decode_sm(vel_raw))
        pos     = _decode_sm(pos_raw)
        pos_ok  = prev_pos is None or abs(pos - prev_pos) < STALL_POS_THRESHOLD
        vel_ok  = vel < STALL_VEL_THRESHOLD

        if vel_ok and pos_ok:
            stable += 1
            if stable >= STALL_CONFIRM:
                time.sleep(SWEEP_SETTLE)
                final = bus.read_position(mid)
                return _decode_sm(final) if final is not None else pos
        else:
            stable = 0

        if status is not None and (status & 0x20):
            overload += 1
            if overload >= STALL_CONFIRM:
                time.sleep(SWEEP_SETTLE)
                final = bus.read_position(mid)
                return _decode_sm(final) if final is not None else pos
        else:
            overload = 0

        prev_pos = pos

    final = bus.read_position(mid)
    return _decode_sm(final) if final is not None else (prev_pos or ENCODER_MID)


def _pos_move(bus: ArmBus, mid: int, target: int, speed: int = 300) -> None:
    bus.write1(mid, ADDR_LOCK, 0)
    bus.write1(mid, ADDR_TORQUE_ENABLE, 0)
    bus.write1(mid, ADDR_OPERATING_MODE, 0)
    bus.write1(mid, ADDR_LOCK, 1)
    bus.enable_torque(mid, True)
    bus.write2(mid, ADDR_GOAL_VELOCITY, speed)
    bus.write2(mid, ADDR_GOAL_POSITION, target & 0xFFFF)


def _move_joints(bus: ArmBus, positions: dict[str, int], speed: int = 250) -> None:
    for name in positions:
        mid = MOTOR_IDS[name]
        bus.init_motor(mid)
        bus.write1(mid, ADDR_LOCK, 0)
        bus.write1(mid, ADDR_TORQUE_ENABLE, 0)
        bus.write1(mid, ADDR_OPERATING_MODE, 0)
        bus.write1(mid, ADDR_LOCK, 1)
        bus.enable_torque(mid, True)
    time.sleep(0.1)
    for name, pos in positions.items():
        mid = MOTOR_IDS[name]
        bus.write2(mid, ADDR_GOAL_VELOCITY, speed)
        bus.write2(mid, ADDR_GOAL_POSITION, pos & 0xFFFF)
    time.sleep(2.0)


def sweep_one_joint(bus: ArmBus, name: str, mid: int) -> tuple[int, int]:
    print(f"  Sweeping {name} (id={mid})... ", end="", flush=True)

    bus.init_motor(mid)
    bus.write1(mid, ADDR_LOCK, 0)
    bus.write1(mid, ADDR_TORQUE_ENABLE, 0)
    bus.write1(mid, ADDR_OPERATING_MODE, 1)
    bus.write1(mid, ADDR_LOCK, 1)
    bus.enable_torque(mid, True)
    time.sleep(0.1)

    bus.set_velocity(mid, SWEEP_SPEED)
    time.sleep(INITIAL_MOVE_DELAY)
    raw_max = _wait_for_stall(bus, mid, STALL_TIMEOUT)
    bus.set_velocity(mid, 0)
    raw_max = max(0, min(ENCODER_MAX, raw_max))
    time.sleep(0.2)
    _pos_move(bus, mid, max(0, raw_max - BACK_OFF_COUNTS))
    time.sleep(0.8)

    bus.set_velocity(mid, -SWEEP_SPEED)
    time.sleep(INITIAL_MOVE_DELAY)
    raw_min = _wait_for_stall(bus, mid, STALL_TIMEOUT)
    bus.set_velocity(mid, 0)
    raw_min = max(0, min(ENCODER_MAX, raw_min))
    time.sleep(0.2)
    _pos_move(bus, mid, min(ENCODER_MAX, raw_min + BACK_OFF_COUNTS))
    time.sleep(0.8)

    _pos_move(bus, mid, (raw_min + raw_max) // 2, speed=300)
    time.sleep(1.2)
    bus.enable_torque(mid, False)

    span_deg = (raw_max - raw_min) * 360.0 / 4095.0
    print(f"range=[{raw_min}, {raw_max}]  span={span_deg:.1f}")
    return raw_min, raw_max


def sweep_calibration(bus: ArmBus) -> dict[str, dict]:
    print("\n" + "=" * 60)
    print("  PHASE 2: Auto-Calibration")
    print("  The arm will move on its own. STAND CLEAR.")
    print("=" * 60)
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1.0)

    measured: dict[str, tuple[int, int]] = {}

    print("\n[1/5] Unfolding arm to clearance pose...")
    _move_joints(bus, FOLLOWER_UNFOLD_POS)

    print("\n[2/5] Sweeping outer joints (wrist / gripper)...")
    for name in SWEEP_ORDER_OUTER:
        measured[name] = sweep_one_joint(bus, name, MOTOR_IDS[name])

    print("\n[3/5] Sweeping inner joints (elbow / shoulder_lift)...")
    for name in SWEEP_ORDER_INNER:
        measured[name] = sweep_one_joint(bus, name, MOTOR_IDS[name])

    print("\n[4/5] Returning arm to rest pose...")
    _move_joints(bus, FOLLOWER_REST_POS)

    print("\n[5/5] Sweeping shoulder_pan...")
    measured["shoulder_pan"] = sweep_one_joint(bus, "shoulder_pan", MOTOR_IDS["shoulder_pan"])

    print("\nEnabling torque — arm is ready for commands.")
    for mid in MOTOR_IDS.values():
        bus.write1(mid, ADDR_LOCK, 0)
        bus.write1(mid, ADDR_OPERATING_MODE, 0)
        bus.write1(mid, ADDR_LOCK, 1)
        bus.enable_torque(mid, True)

    calibration: dict[str, dict] = {}
    for name, mid in MOTOR_IDS.items():
        r_min, r_max = measured[name]
        calibration[name] = {
            "id":            mid,
            "drive_mode":    0,
            "homing_offset": int((r_min + r_max) / 2.0 - ENCODER_MID),
            "range_min":     int(r_min),
            "range_max":     int(r_max),
        }
    return calibration


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

def calibration_path(calibration_id: str) -> Path:
    hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return (
        Path(hf_home) / "lerobot" / "calibration"
        / "robots" / "so_follower" / f"{calibration_id}.json"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port",       required=True, help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--id",         required=True, metavar="CALIBRATION_ID",
                        help="Name for the saved calibration file, e.g. my_follower")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip motor ID assignment (use if IDs are already set)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print calibration JSON without saving to disk")
    args = parser.parse_args()

    out_path = calibration_path(args.id)

    print(f"\nSO-101 follower — setup & auto-calibration")
    print(f"  Port           : {args.port}")
    print(f"  Calibration ID : {args.id}")
    print(f"  Output path    : {out_path}")

    bus = ArmBus(args.port)
    try:
        if not args.skip_setup:
            if not setup_motor_ids(bus):
                print("\nMotor ID setup failed. Aborting.")
                return

        calibration = sweep_calibration(bus)
    finally:
        bus.close()

    print("\nCalibration results:")
    for name, info in calibration.items():
        span_deg = (info["range_max"] - info["range_min"]) * 360.0 / 4095.0
        print(
            f"  {name:<15} range=[{info['range_min']:4d}, {info['range_max']:4d}]"
            f"  span={span_deg:.1f}  homing_offset={info['homing_offset']:+d}"
        )

    json_str = json.dumps(calibration, indent=4)

    if args.dry_run:
        print("\n[DRY RUN] Would write to:", out_path)
        print(json_str)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json_str)
    print(f"\nCalibration saved to: {out_path}")


if __name__ == "__main__":
    main()
