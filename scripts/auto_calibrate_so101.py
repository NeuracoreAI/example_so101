#!/usr/bin/env python3
"""
Calibrate an SO-101 arm without dismounting any cables.

Three modes:

  --mode interactive  (RECOMMENDED)
    Torque is disabled so you can move each joint freely by hand.
    The script walks you joint-by-joint from the base outward, asking you to:
      1. Hold the joint at its CENTER / zero position  → press Enter
      2. Move the joint to its MAX CLOCKWISE limit     → press Enter
      3. Move the joint to its MAX ANTI-CLOCKWISE limit → press Enter
    No motor commands are sent — the arm never moves on its own.

  --mode quick
    Read current encoder positions and use them as the zero reference with
    typical range half-spans. No movement at all. Use when the arm is already
    in a reasonable neutral pose.

  --mode sweep
    Motor-driven: each joint is spun slowly to its mechanical limits via
    velocity mode and stall detection. Arm moves on its own.

Activate the conda environment first:
    conda activate so101-teleop

Usage:
    # Interactive (recommended — you move the arm by hand):
    python scripts/auto_calibrate_so101.py \\
        --port /dev/ttyACM0 --arm follower --id my_awesome_follower_arm --mode interactive

    # Quick (no movement, current pose = zero):
    python scripts/auto_calibrate_so101.py \\
        --port /dev/ttyACM0 --arm follower --id my_awesome_follower_arm

    # Dry run (print JSON, do not save):
    python scripts/auto_calibrate_so101.py \\
        --port /dev/ttyACM0 --arm follower --id test --mode interactive --dry-run

Calibration is saved to:
    Leader:   ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<id>.json
    Follower: ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json
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
        "Activate the conda environment:  conda activate so101-teleop\n"
        "Or install:  pip install feetech-servo-sdk"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Hardware constants  (STS3215 / SMS3215 register map, Protocol 0)
# ---------------------------------------------------------------------------

BAUD_RATE = 1_000_000

ADDR_RETURN_DELAY_TIME   = 7
ADDR_MAX_TORQUE_LIMIT    = 16   # 2 bytes, EEPROM
ADDR_P_COEFFICIENT       = 21   # EEPROM
ADDR_D_COEFFICIENT       = 22   # EEPROM
ADDR_I_COEFFICIENT       = 23   # EEPROM
ADDR_TORQUE_LIMIT        = 28   # 2 bytes, EEPROM
ADDR_OPERATING_MODE      = 33   # EEPROM: 0=position, 1=velocity
ADDR_TORQUE_ENABLE       = 40   # RAM
ADDR_ACCELERATION        = 41   # EEPROM
ADDR_GOAL_POSITION       = 42   # RAM, 2 bytes
ADDR_GOAL_VELOCITY       = 46   # RAM, 2 bytes, sign-magnitude (velocity mode)
ADDR_LOCK                = 55   # RAM: 0=EEPROM unlocked, 1=EEPROM locked (normal op)
ADDR_PRESENT_POSITION    = 56   # RAM, 2 bytes, read-only
ADDR_PRESENT_VELOCITY    = 58   # RAM, 2 bytes, sign-magnitude, read-only

ENCODER_MAX  = 4095
ENCODER_MID  = 2048

# Motor name -> bus id
MOTOR_IDS: dict[str, int] = {
    "shoulder_pan":  1,
    "shoulder_lift": 2,
    "elbow_flex":    3,
    "wrist_flex":    4,
    "wrist_roll":    5,
    "gripper":       6,
}

# ── Differences between leader and follower ────────────────────────────────
#
# | Aspect           | Leader                     | Follower                  |
# |------------------|----------------------------|---------------------------|
# | Gear ratio       | Mixed (1/191, 1/345, 1/147)| All 1/345                 |
# | Post-cal torque  | DISABLED (moved by hand)   | ENABLED (drives commands) |
# | Calibration path | teleoperators/so_leader/   | robots/so_follower/       |
# | Sweep speed      | Slightly slower (1/191 on  | Standard speed            |
# |                  | pan & elbow → faster joint)|                           |
#
# The encoder range (0-4095) measures the OUTPUT shaft. Because the follower
# uses identical mechanical design to the leader (same physical joint limits),
# the range_min/range_max half-spans are effectively the same. The gear ratio
# only affects torque and speed, not the encoder's position window.
# ──────────────────────────────────────────────────────────────────────────

# Typical half-spans (raw encoder units) derived from real SO-101 calibration data.
# Used in --mode quick to set range_min/range_max around the current raw position.
# Same defaults for both arm types — physical limits are set by mechanical design.
HALF_SPANS: dict[str, int] = {
    "shoulder_pan":  1213,   # ~106° each side
    "shoulder_lift": 1268,   # ~111° each side
    "elbow_flex":    1104,   # ~97°  each side
    "wrist_flex":    1171,   # ~103° each side
    "wrist_roll":    1915,   # ~169° each side (nearly full encoder range)
    "gripper":        806,   # open/close range
}

# Sweep speed per joint for leader vs follower.
# Leader joints 1 and 3 (pan, elbow) use 1/191 gearing — same encoder counts but
# the joint rotates ~1.8× faster per count, so use a lower speed to stay gentle.
SWEEP_SPEEDS_LEADER: dict[str, int] = {
    "shoulder_pan":   80,   # 1/191 gearing → faster joint per encoder count
    "shoulder_lift": 150,
    "elbow_flex":     80,   # 1/191 gearing
    "wrist_flex":    100,
    "wrist_roll":    100,
    "gripper":       100,
}
SWEEP_SPEEDS_FOLLOWER: dict[str, int] = {name: 150 for name in MOTOR_IDS}

# Sweep calibration parameters
SWEEP_SPEED      = 150    # raw speed units (slow, safe). Range: 0-32767
SWEEP_ACCEL      = 10     # low acceleration for gentle ramp-up
STALL_WINDOW     = 0.15   # seconds of position not changing → stall detected
STALL_TOLERANCE  = 3      # encoder counts; less than this = "not moving"
STALL_TIMEOUT    = 8.0    # max seconds per direction per joint
SWEEP_SETTLE     = 0.3    # seconds to wait before reading final stalled position
BACK_OFF_COUNTS  = 80     # how far to back off from each limit before returning

# ── Sweep orders ─────────────────────────────────────────────────────────────
#
# FOLLOWER sweep order matters for mechanical safety:
#   The follower starts folded (resting pose). Sweeping shoulder_lift or
#   elbow_flex first while the arm is down can cause the wrist/gripper to
#   collide with the table or arm body. Safe sequence:
#     1. Unfold: raise shoulder_lift + elbow_flex to a clearance pose.
#     2. Sweep end-effector joints (gripper, wrist_roll, wrist_flex) while elevated.
#     3. Sweep elbow_flex and shoulder_lift (inner joints).
#     4. Fold back to rest pose.
#     5. Sweep shoulder_pan last (widest base rotation, safest to do solo).
#
# LEADER sweep order is less critical — lighter gearing, designed to be
# moved freely — so a simple distal-to-proximal order is fine.
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_ORDER_LEADER = [
    "gripper",
    "wrist_roll",
    "wrist_flex",
    "elbow_flex",
    "shoulder_lift",
    "shoulder_pan",
]

# Interactive mode: base → end-effector so stable joints are already zeroed
# before the user handles more distal ones.
INTERACTIVE_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Follower: outer joints swept after unfold, inner joints after.
SWEEP_ORDER_FOLLOWER_OUTER = ["gripper", "wrist_roll", "wrist_flex"]
SWEEP_ORDER_FOLLOWER_INNER = ["elbow_flex", "shoulder_lift"]

# Typical midpoints from real SO-101 calibration, used for unfold/rest steps.
FOLLOWER_UNFOLD_POS: dict[str, int] = {
    "shoulder_lift": 1500,   # ~44° above midpoint — arm raised for clearance
    "elbow_flex":    2300,   # slightly extended to give wrist room
}
FOLLOWER_REST_POS: dict[str, int] = {
    "shoulder_lift": 2004,
    "elbow_flex":    2028,
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _decode_sign_magnitude(val: int) -> int:
    sign = (val >> 15) & 1
    return -(val & 0x7FFF) if sign else val


class ArmBus:
    """Minimal STS3215 bus wrapper used by the calibration routines."""

    def __init__(self, port: str) -> None:
        self.ph   = scs.PortHandler(port)
        self.pkth = scs.PacketHandler(0)
        if not self.ph.openPort():
            raise ConnectionError(
                f"Cannot open port '{port}'.\n"
                f"  Check USB cable and run:  sudo chmod 666 {port}"
            )
        self.ph.setBaudRate(BAUD_RATE)

    def close(self) -> None:
        self.ph.closePort()

    # ---- single-byte helpers ------------------------------------------------

    def write1(self, motor_id: int, addr: int, val: int) -> None:
        self.pkth.write1ByteTxRx(self.ph, motor_id, addr, val)

    def read2(self, motor_id: int, addr: int) -> int | None:
        try:
            val, comm, _ = self.pkth.read2ByteTxRx(self.ph, motor_id, addr)
            return val if comm == scs.COMM_SUCCESS else None
        except (IndexError, Exception):
            return None

    def write2(self, motor_id: int, addr: int, val: int) -> None:
        self.pkth.write2ByteTxRx(self.ph, motor_id, addr, val)

    # ---- batch read ---------------------------------------------------------

    def read_all_positions(self) -> dict[str, int]:
        """Sync-read Present_Position for all 6 motors. Raises IOError on failure."""
        sr = scs.GroupSyncRead(self.ph, self.pkth, ADDR_PRESENT_POSITION, 2)
        for mid in MOTOR_IDS.values():
            sr.addParam(mid)
        comm = sr.txRxPacket()
        if comm != scs.COMM_SUCCESS:
            raise IOError(f"Sync read failed (code {comm}). Check power and cables.")
        missing = []
        positions: dict[str, int] = {}
        for name, mid in MOTOR_IDS.items():
            if sr.isAvailable(mid, ADDR_PRESENT_POSITION, 2):
                positions[name] = sr.getData(mid, ADDR_PRESENT_POSITION, 2)
            else:
                missing.append(f"{name}(id={mid})")
        if missing:
            raise IOError(f"No response from: {', '.join(missing)}. Check cables/power.")
        return positions

    # ---- motor control ------------------------------------------------------

    def enable_torque(self, motor_id: int, enable: bool) -> None:
        self.write1(motor_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)

    def set_position(self, motor_id: int, pos: int, speed: int = 300) -> None:
        """Write goal position + speed using WritePosEx (7-byte packet)."""
        # WritePosEx writes: pos(2) + time(2) + speed(2) + accel(1) starting at ADDR_ACCELERATION
        # We use the simpler write2ByteTxRx pair instead.
        self.write2(motor_id, ADDR_GOAL_VELOCITY, speed)
        self.write2(motor_id, ADDR_GOAL_POSITION, pos & 0xFFFF)

    def read_position(self, motor_id: int) -> int | None:
        raw = self.read2(motor_id, ADDR_PRESENT_POSITION)
        return raw

    def set_velocity(self, motor_id: int, velocity: int) -> None:
        """Write Goal_Velocity for constant-speed (velocity) mode.
        Positive = CW, negative = CCW (sign-magnitude encoding)."""
        encoded = (1 << 15 | abs(velocity)) if velocity < 0 else velocity
        self.write2(motor_id, ADDR_GOAL_VELOCITY, encoded & 0xFFFF)

    def read_velocity(self, motor_id: int) -> int | None:
        raw = self.read2(motor_id, ADDR_PRESENT_VELOCITY)
        if raw is None:
            return None
        return _decode_sign_magnitude(raw)

    def init_motor(self, motor_id: int) -> None:
        """Configure motor EEPROM registers for calibration sweep.

        Sequence: unlock EEPROM (Lock=0) → write config → re-lock (Lock=1).
        Lock=1 is REQUIRED for normal servo operation — with Lock=0 the servo
        ignores Goal_Position and Goal_Velocity commands (EEPROM programming mode).
        """
        self.write1(motor_id, ADDR_LOCK, 0)            # unlock EEPROM
        self.write1(motor_id, ADDR_TORQUE_ENABLE, 0)   # disable torque before config
        self.write1(motor_id, ADDR_RETURN_DELAY_TIME, 0)
        self.write2(motor_id, ADDR_MAX_TORQUE_LIMIT, 1000)
        self.write2(motor_id, ADDR_TORQUE_LIMIT,      380)
        self.write1(motor_id, ADDR_P_COEFFICIENT,      16)
        self.write1(motor_id, ADDR_D_COEFFICIENT,      32)
        self.write1(motor_id, ADDR_I_COEFFICIENT,       0)
        self.write1(motor_id, ADDR_ACCELERATION,  SWEEP_ACCEL)
        self.write1(motor_id, ADDR_LOCK, 1)            # re-lock → normal servo operation
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Quick calibration (no movement)
# ---------------------------------------------------------------------------

def quick_calibration(bus: ArmBus) -> tuple[dict[str, int], dict[str, dict]]:
    """Read current positions and generate calibration centred on them."""
    print("Reading raw encoder positions from all motors…")
    raw_positions = bus.read_all_positions()

    calibration: dict[str, dict] = {}
    for name, mid in MOTOR_IDS.items():
        raw  = raw_positions[name]
        half = HALF_SPANS[name]
        r_min = max(0,           raw - half)
        r_max = min(ENCODER_MAX, raw + half)

        actual_half = min(raw - r_min, r_max - raw)
        if actual_half < half * 0.75:
            print(
                f"  WARNING: {name} raw={raw} is near encoder edge — "
                f"range clamped to [{r_min}, {r_max}]"
            )

        calibration[name] = {
            "id":           mid,
            "drive_mode":   0,
            "homing_offset": int(raw - ENCODER_MID),  # for lerobot std compatibility
            "range_min":    int(r_min),
            "range_max":    int(r_max),
        }
    return raw_positions, calibration


# ---------------------------------------------------------------------------
# Interactive calibration — user moves each joint by hand
# ---------------------------------------------------------------------------

def _read_positions_sync(bus: ArmBus) -> dict[str, int]:
    """Read all motor positions via GroupSyncRead with up to 3 retries.

    GroupSyncRead is far more reliable than individual read2ByteTxRx on the
    Waveshare USB adapter (individual reads silently fail with COMM error).
    Returns raw 0-4095 encoder values (no sign-magnitude decoding needed).
    """
    for attempt in range(3):
        try:
            return bus.read_all_positions()
        except IOError:
            if attempt == 2:
                raise
            time.sleep(0.1)
    return {}


def interactive_calibration(bus: ArmBus, arm: str) -> dict[str, dict]:
    """Manual calibration: torque off, user moves each joint to CENTER then CW/CCW limits.

    No motor commands are ever sent — the arm never moves on its own.
    Walk order: shoulder_pan → shoulder_lift → elbow_flex →
                wrist_flex → wrist_roll → gripper
    """
    print("\nDisabling torque on all motors so you can move every joint freely…")
    for mid in MOTOR_IDS.values():
        bus.write1(mid, ADDR_LOCK, 0)
        bus.write1(mid, ADDR_TORQUE_ENABLE, 0)
        bus.write1(mid, ADDR_LOCK, 1)
    time.sleep(0.1)  # let bus settle after bulk writes
    print("Done. All joints are now free-wheeling.\n")
    print("─" * 60)
    print("  For each joint you will be asked to:")
    print("    1. Hold it at CENTER / zero  → press Enter")
    print("    2. Move it to MAX CLOCKWISE  → press Enter")
    print("    3. Move it to MAX ANTI-CW    → press Enter")
    print("─" * 60)

    calibration: dict[str, dict] = {}

    for name in INTERACTIVE_ORDER:
        mid = MOTOR_IDS[name]
        print(f"\n{'═' * 60}")
        print(f"  Joint: {name}  (motor id={mid})")
        print(f"{'═' * 60}")

        input(f"  1. Hold  [{name}]  at CENTER / zero position, then press Enter … ")
        time.sleep(0.05)
        raw_centre = _read_positions_sync(bus).get(name, ENCODER_MID)
        print(f"     → CENTER recorded: raw={raw_centre}")

        input(f"  2. Move  [{name}]  to MAX CLOCKWISE limit, then press Enter … ")
        time.sleep(0.05)
        raw_cw = _read_positions_sync(bus).get(name, raw_centre)
        print(f"     → CW MAX recorded:  raw={raw_cw}")

        input(f"  3. Move  [{name}]  to MAX ANTI-CLOCKWISE limit, then press Enter … ")
        time.sleep(0.05)
        raw_ccw = _read_positions_sync(bus).get(name, raw_centre)
        print(f"     → CCW MAX recorded: raw={raw_ccw}")

        raw_min = min(raw_cw, raw_ccw)
        raw_max = max(raw_cw, raw_ccw)

        span_deg = (raw_max - raw_min) * 360.0 / 4095.0
        homing_offset = int(raw_centre - ENCODER_MID)
        print(f"     → span={span_deg:.1f}°  centre(raw)={raw_centre}  homing_offset={homing_offset:+d}")

        calibration[name] = {
            "id":            mid,
            "drive_mode":    0,
            "homing_offset": homing_offset,
            "range_min":     int(raw_min),
            "range_max":     int(raw_max),
        }

    # Follower: re-enable torque in position mode so it's ready for commands
    if arm == "follower":
        print("\nRe-enabling torque on follower arm (position mode)…")
        for name, mid in MOTOR_IDS.items():
            bus.write1(mid, ADDR_LOCK, 0)
            bus.write1(mid, ADDR_OPERATING_MODE, 0)
            bus.write1(mid, ADDR_P_COEFFICIENT, 16)
            bus.write1(mid, ADDR_D_COEFFICIENT, 32)
            bus.write1(mid, ADDR_I_COEFFICIENT,  0)
            bus.write1(mid, ADDR_LOCK, 1)
            bus.enable_torque(mid, True)
        print("Done. Follower arm is ready for tele-operation commands.")
    else:
        print("\nTorque remains disabled on leader arm (passive operation).")

    return calibration


# ---------------------------------------------------------------------------
# Sweep calibration — velocity mode, Maker-Mods approach
#
# Mirrors lerobot-MakerMods/src/lerobot/motors/feetech/auto_calibration.py:
#   1. Set Operating_Mode=1 (velocity/constant-speed mode)
#   2. Write Goal_Velocity (positive = CW, negative = CCW)
#   3. Wait 0.5 s initial delay (motor needs time to actually start moving)
#   4. Poll Present_Velocity + Present_Position delta until both near zero
#   5. Stop, back off, sweep other direction
#   6. Return to centre; restore Operating_Mode=0 (position mode)
#
# Critical fix from Maker-Mods code review:
#   Lock=0 disables normal servo operation (EEPROM programming mode).
#   Lock=1 is required BEFORE issuing any motion command.
#   init_motor() now restores Lock=1 after EEPROM writes.
# ---------------------------------------------------------------------------

STALL_VEL_THRESHOLD  = 10    # |Present_Velocity| below this = "not moving"
STALL_POS_THRESHOLD  = 3     # position delta below this = "not moving"
STALL_CONFIRM        = 4     # consecutive samples meeting both conditions
INITIAL_MOVE_DELAY   = 0.5   # seconds to wait after Goal_Velocity before polling
SAMPLE_INTERVAL      = 0.05  # seconds between stall-detection polls


def _wait_for_stall(bus: ArmBus, motor_id: int, timeout: float) -> int:
    """Poll velocity + position until the motor has stopped (stall / end-stop reached).

    Returns the final raw position. Mirrors Maker-Mods _wait_for_stall logic:
      - velocity near zero AND position delta near zero → stall confirmed
      - Status register BIT5 (0x20) overload → stall confirmed
    """
    stable: int        = 0
    overload: int      = 0
    prev_pos: int | None = None
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        time.sleep(SAMPLE_INTERVAL)

        vel_raw = bus.read2(motor_id, ADDR_PRESENT_VELOCITY)
        pos_raw = bus.read_position(motor_id)
        # Read status byte (address 65 in most STS3215 firmware)
        status_raw = bus.read2(motor_id, 65)

        if vel_raw is None or pos_raw is None:
            continue

        vel     = abs(_decode_sign_magnitude(vel_raw))
        pos_dec = _decode_sign_magnitude(pos_raw)
        pos_ok  = (prev_pos is None) or (abs(pos_dec - prev_pos) < STALL_POS_THRESHOLD)
        vel_ok  = vel < STALL_VEL_THRESHOLD

        if vel_ok and pos_ok:
            stable += 1
            if stable >= STALL_CONFIRM:
                time.sleep(SWEEP_SETTLE)
                final = bus.read_position(motor_id)
                return _decode_sign_magnitude(final) if final is not None else pos_dec
        else:
            stable = 0

        # Overload flag (Status BIT5)
        if status_raw is not None and (status_raw & 0x20):
            overload += 1
            if overload >= STALL_CONFIRM:
                time.sleep(SWEEP_SETTLE)
                final = bus.read_position(motor_id)
                return _decode_sign_magnitude(final) if final is not None else pos_dec
        else:
            overload = 0

        prev_pos = pos_dec

    # Timeout — return current position
    final = bus.read_position(motor_id)
    return _decode_sign_magnitude(final) if final is not None else (prev_pos or ENCODER_MID)


def sweep_one_joint(bus: ArmBus, name: str, motor_id: int, speed: int = SWEEP_SPEED) -> tuple[int, int]:
    """Sweep one joint to both mechanical limits using velocity mode.

    Returns (range_min, range_max) in raw encoder units.
    """
    print(f"  Sweeping {name} (id={motor_id})…", end="", flush=True)

    # ── Configure motor: EEPROM settings + Lock=1 (critical!) ──────────────
    bus.init_motor(motor_id)   # ends with Lock=1 → normal servo operation

    # Switch to velocity mode (Operating_Mode=1, requires Lock=0 briefly)
    bus.write1(motor_id, ADDR_LOCK, 0)
    bus.write1(motor_id, ADDR_TORQUE_ENABLE, 0)
    bus.write1(motor_id, ADDR_OPERATING_MODE, 1)   # velocity/constant-speed mode
    bus.write1(motor_id, ADDR_LOCK, 1)             # re-lock before enabling torque
    bus.enable_torque(motor_id, True)
    time.sleep(0.1)

    # ── Sweep CW (positive velocity → encoder increases) ───────────────────
    bus.set_velocity(motor_id, speed)
    time.sleep(INITIAL_MOVE_DELAY)                 # let motor actually start moving
    raw_max = _wait_for_stall(bus, motor_id, STALL_TIMEOUT)
    bus.set_velocity(motor_id, 0)
    raw_max = max(0, min(ENCODER_MAX, raw_max))
    time.sleep(0.2)

    # Back off from the CW end-stop
    _pos_move(bus, motor_id, max(0, raw_max - BACK_OFF_COUNTS), speed=200)
    time.sleep(0.8)

    # ── Sweep CCW (negative velocity → encoder decreases) ──────────────────
    bus.set_velocity(motor_id, -speed)
    time.sleep(INITIAL_MOVE_DELAY)
    raw_min = _wait_for_stall(bus, motor_id, STALL_TIMEOUT)
    bus.set_velocity(motor_id, 0)
    raw_min = max(0, min(ENCODER_MAX, raw_min))
    time.sleep(0.2)

    # Back off from the CCW end-stop
    _pos_move(bus, motor_id, min(ENCODER_MAX, raw_min + BACK_OFF_COUNTS), speed=200)
    time.sleep(0.8)

    # ── Return to centre; restore position mode ─────────────────────────────
    centre = (raw_min + raw_max) // 2
    _pos_move(bus, motor_id, centre, speed=300)
    time.sleep(1.2)

    bus.enable_torque(motor_id, False)

    span_deg = (raw_max - raw_min) * 360.0 / 4095.0
    print(f"  range=[{raw_min}, {raw_max}]  span={span_deg:.1f}°")
    return raw_min, raw_max


def _pos_move(bus: ArmBus, motor_id: int, target: int, speed: int = 300) -> None:
    """Switch to position mode, write goal position, then restore velocity mode."""
    bus.write1(motor_id, ADDR_LOCK, 0)
    bus.write1(motor_id, ADDR_TORQUE_ENABLE, 0)
    bus.write1(motor_id, ADDR_OPERATING_MODE, 0)   # position mode
    bus.write1(motor_id, ADDR_LOCK, 1)
    bus.enable_torque(motor_id, True)
    bus.write2(motor_id, ADDR_GOAL_VELOCITY, speed)
    bus.write2(motor_id, ADDR_GOAL_POSITION, target & 0xFFFF)


def _move_to_position_and_wait(bus: ArmBus, positions: dict[str, int], speed: int = 250) -> None:
    """Configure motors in position mode (Lock=1) and drive to target positions."""
    for name in positions:
        mid = MOTOR_IDS[name]
        bus.init_motor(mid)                    # EEPROM config, ends Lock=1
        bus.write1(mid, ADDR_LOCK, 0)
        bus.write1(mid, ADDR_TORQUE_ENABLE, 0)
        bus.write1(mid, ADDR_OPERATING_MODE, 0)  # position mode
        bus.write1(mid, ADDR_LOCK, 1)
        bus.enable_torque(mid, True)
    time.sleep(0.1)
    for name, pos in positions.items():
        mid = MOTOR_IDS[name]
        bus.write2(mid, ADDR_GOAL_VELOCITY, speed)
        bus.write2(mid, ADDR_GOAL_POSITION, pos & 0xFFFF)
    time.sleep(2.0)


def sweep_calibration(bus: ArmBus, arm: str) -> dict[str, dict]:
    """Sweep all joints to find actual mechanical limits.

    Follower sequence (collision-safe):
      1. Unfold — raise shoulder_lift + elbow_flex to give the wrist joints room.
      2. Sweep outer joints (gripper, wrist_roll, wrist_flex) while arm is elevated.
      3. Sweep inner joints (elbow_flex, shoulder_lift).
      4. Return to rest pose.
      5. Sweep shoulder_pan last (widest base footprint).
      Torque enabled after completion (arm ready for commands).

    Leader sequence (simple distal-to-proximal, lighter gearing):
      gripper → wrist_roll → wrist_flex → elbow_flex → shoulder_lift → shoulder_pan
      Torque disabled after completion (passive leader, moved by hand).
    """
    print("\nSweep calibration — arm will move. Stand clear.\n")
    for i in range(3, 0, -1):
        print(f"  Starting in {i}…")
        time.sleep(1.0)

    speeds   = SWEEP_SPEEDS_LEADER if arm == "leader" else SWEEP_SPEEDS_FOLLOWER
    measured: dict[str, tuple[int, int]] = {}

    if arm == "follower":
        # ── Step 1: Unfold ──────────────────────────────────────────────────
        print("\n[1/5] Unfolding arm to clearance pose…")
        _move_to_position_and_wait(bus, FOLLOWER_UNFOLD_POS)

        # ── Step 2: Sweep outer joints (wrist / gripper) ────────────────────
        print("\n[2/5] Sweeping outer joints (arm elevated)…")
        for name in SWEEP_ORDER_FOLLOWER_OUTER:
            mid = MOTOR_IDS[name]
            r_min, r_max = sweep_one_joint(bus, name, mid, speed=speeds[name])
            measured[name] = (r_min, r_max)

        # ── Step 3: Sweep inner joints ──────────────────────────────────────
        print("\n[3/5] Sweeping inner joints…")
        for name in SWEEP_ORDER_FOLLOWER_INNER:
            mid = MOTOR_IDS[name]
            r_min, r_max = sweep_one_joint(bus, name, mid, speed=speeds[name])
            measured[name] = (r_min, r_max)

        # ── Step 4: Return to rest ──────────────────────────────────────────
        print("\n[4/5] Returning arm to rest pose…")
        _move_to_position_and_wait(bus, FOLLOWER_REST_POS)

        # ── Step 5: Sweep shoulder_pan ──────────────────────────────────────
        print("\n[5/5] Sweeping shoulder_pan…")
        name = "shoulder_pan"
        r_min, r_max = sweep_one_joint(bus, name, MOTOR_IDS[name], speed=speeds[name])
        measured[name] = (r_min, r_max)

        # Re-enable torque in position mode — follower is active
        print("\nEnabling torque on all joints (follower ready for commands)…")
        for mid in MOTOR_IDS.values():
            bus.write1(mid, ADDR_LOCK, 0)
            bus.write1(mid, ADDR_OPERATING_MODE, 0)
            bus.write1(mid, ADDR_LOCK, 1)
            bus.enable_torque(mid, True)

    else:
        # ── Leader: simple distal-to-proximal sweep ─────────────────────────
        for name in SWEEP_ORDER_LEADER:
            mid = MOTOR_IDS[name]
            r_min, r_max = sweep_one_joint(bus, name, mid, speed=speeds[name])
            measured[name] = (r_min, r_max)
        print("\nTorque remains disabled on leader arm (passive operation).")

    print()
    calibration: dict[str, dict] = {}
    for name, mid in MOTOR_IDS.items():
        r_min, r_max = measured[name]
        centre = (r_min + r_max) / 2.0
        calibration[name] = {
            "id":            mid,
            "drive_mode":    0,
            "homing_offset": int(centre - ENCODER_MID),
            "range_min":     int(r_min),
            "range_max":     int(r_max),
        }
    return calibration


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def default_calibration_path(arm: str, calibration_id: str) -> Path:
    hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    base    = Path(hf_home) / "lerobot" / "calibration"
    if arm == "leader":
        return base / "teleoperators" / "so_leader" / f"{calibration_id}.json"
    return base / "robots" / "so_follower" / f"{calibration_id}.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port",  required=True, help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--arm",   required=True, choices=["leader", "follower"])
    parser.add_argument("--id",    required=True, metavar="CALIBRATION_ID",
                        help="Calibration ID (e.g. my_awesome_leader_arm)")
    parser.add_argument("--mode",  choices=["interactive", "quick", "sweep"], default="interactive",
                        help="interactive: you move each joint by hand to MIN/MAX (recommended, no arm movement); "
                             "quick: use current pose as zero (no movement); "
                             "sweep: drive joints to limits automatically (arm moves!)")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="Override output JSON path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print calibration JSON without writing to disk")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else default_calibration_path(args.arm, args.id)

    print(f"\nSO-101 auto-calibration")
    print(f"  Arm            : {args.arm}")
    print(f"  Port           : {args.port}")
    print(f"  Calibration ID : {args.id}")
    print(f"  Mode           : {args.mode}")
    print(f"  Output path    : {out_path}")
    print()

    bus = ArmBus(args.port)
    try:
        if args.mode == "interactive":
            calibration = interactive_calibration(bus, arm=args.arm)
        elif args.mode == "quick":
            raw_positions, calibration = quick_calibration(bus)
            print("\nRaw positions (current pose = zero for every joint):")
            for name, raw in raw_positions.items():
                deg = (raw - ENCODER_MID) * 360.0 / 4095.0
                print(f"  {name:<15} id={MOTOR_IDS[name]}  raw={raw:4d}  ({deg:+.1f}° from encoder centre)")
        else:
            calibration = sweep_calibration(bus, arm=args.arm)
    finally:
        bus.close()

    print("\nGenerated calibration:")
    for name, info in calibration.items():
        span_deg = (info["range_max"] - info["range_min"]) * 360.0 / 4095.0
        print(
            f"  {name:<15} range=[{info['range_min']:4d}, {info['range_max']:4d}]"
            f"  span={span_deg:.1f}°  homing_offset={info['homing_offset']:+d}"
        )

    json_str = json.dumps(calibration, indent=4)

    if args.dry_run:
        print("\n[DRY RUN] Would write to:", out_path)
        print(json_str)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json_str)
    print(f"\nCalibration saved to: {out_path}")

    if args.mode == "interactive":
        print("\nInteractive calibration complete. Each joint is zeroed at the centre of the range you swept.")
    elif args.mode == "quick":
        print(
            "\nNote: the current arm pose is now the 'zero' for every joint.\n"
            "If you want a different zero, move the arm to that pose and re-run."
        )
    else:
        print("\nSweep complete. The arm has been returned to the centre of each joint's range.")


if __name__ == "__main__":
    main()
