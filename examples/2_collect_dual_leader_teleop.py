#!/usr/bin/env python3
"""Dual SO101 leader arms → dual SO101 follower arms teleop with Neuracore data collection.

Left leader drives the left follower; right leader drives the right follower.
Each pair uses an independent offset configuration: SO101_OFFSETS_DEG for the
left pair and SO101_OFFSETS_DEG_2 for the right pair.

Controls:
  e     - enable / disable both follower arms
  r     - start / stop Neuracore recording
  Ctrl+C - exit
"""

import argparse
import multiprocessing
import select
import sys
import termios
import threading
import time
import traceback
import tty
from pathlib import Path
from typing import Any

import neuracore as nc
import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

from common.configs import (  # type: ignore  # noqa: E402
    CAMERA_2_DEVICE_INDEX,
    CAMERA_2_HEIGHT,
    CAMERA_2_WIDTH,
    CAMERA_DEVICE_INDEX,
    CAMERA_FRAME_STREAMING_RATE,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONTROLLER_DATA_RATE,
    DUAL_URDF_PATH,
    LEADER_TO_SO101_JOINT,
    NEUTRAL_JOINT_ANGLES,
    ROBOT_RATE,
    SO101_DIRECTIONS,
    SO101_FIXED_JOINTS,
    SO101_JOINT_LIMITS_DEG,
    SO101_OFFSETS_DEG,
    SO101_OFFSETS_DEG_2,
)
from common.data_manager_dual import DualDataManager, RobotActivityState  # type: ignore  # noqa: E402
from common.leader_arm import LerobotSO101LeaderArm  # type: ignore  # noqa: E402
from common.threads.dual_joint_state import dual_joint_state_thread  # type: ignore  # noqa: E402
from common.threads.dual_leader_reader import dual_leader_reader_thread  # type: ignore  # noqa: E402
from so101_dual_controller import SO101DualController  # type: ignore  # noqa: E402

_NC_ROBOT_NAME = "LeRobot SO101 Dual"
_NC_LOGGING_RATE_HZ = 30.0
_CAMERA_NAMES = ["rgb", "rgb_2"]
_BODY_DOF = 5

_BODY_JOINT_SUFFIXES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def _neuracore_logging_thread(data_manager: DualDataManager, rate_hz: float) -> None:
    """Poll DualDataManager and log joint, gripper, and camera state to Neuracore."""
    dt = 1.0 / rate_hz
    print("📡 Neuracore logging thread started")
    try:
        while not data_manager.is_shutdown_requested():
            t0 = time.time()
            ts = t0

            joint_angles = data_manager.get_current_joint_angles()
            if joint_angles is not None and len(joint_angles) >= _BODY_DOF * 2:
                left_rad = np.radians(joint_angles[:_BODY_DOF])
                right_rad = np.radians(joint_angles[_BODY_DOF : _BODY_DOF * 2])
                positions: dict[str, float] = {}
                for name, val in zip(_BODY_JOINT_SUFFIXES, left_rad):
                    positions[f"left_{name}"] = float(val)
                for name, val in zip(_BODY_JOINT_SUFFIXES, right_rad):
                    positions[f"right_{name}"] = float(val)
                nc.log_joint_positions(positions, timestamp=ts)

            target_angles = data_manager.get_target_joint_angles()
            if target_angles is not None and len(target_angles) >= _BODY_DOF * 2:
                left_rad = np.radians(target_angles[:_BODY_DOF])
                right_rad = np.radians(target_angles[_BODY_DOF : _BODY_DOF * 2])
                targets: dict[str, float] = {}
                for name, val in zip(_BODY_JOINT_SUFFIXES, left_rad):
                    targets[f"left_{name}"] = float(val)
                for name, val in zip(_BODY_JOINT_SUFFIXES, right_rad):
                    targets[f"right_{name}"] = float(val)
                nc.log_joint_target_positions(targets, timestamp=ts)

            for side in ("left", "right"):
                gripper_val = data_manager.get_current_gripper_open_value(side)
                if gripper_val is not None:
                    nc.log_parallel_gripper_open_amounts(
                        {f"{side}_gripper": float(gripper_val)}, timestamp=ts
                    )
                target_gripper = data_manager.get_target_gripper_open_value(side)
                if target_gripper is not None:
                    nc.log_parallel_gripper_target_open_amounts(
                        {f"{side}_gripper": float(target_gripper)}, timestamp=ts
                    )

            for cam_name in _CAMERA_NAMES:
                img = data_manager.get_rgb_image(cam_name)
                if img is not None:
                    nc.log_rgb(cam_name, img, timestamp=ts)

            elapsed = time.time() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as e:
        print(f"❌ Neuracore logging thread error: {e}")
        traceback.print_exc()
    finally:
        print("📡 Neuracore logging thread stopped")


def _run_camera(
    dm: DualDataManager,
    camera_name: str,
    device_index: int,
    width: int,
    height: int,
) -> None:
    """Capture frames from one USB camera and write to DualDataManager."""
    import cv2

    print(f"📷 Camera thread started (device {device_index}, name='{camera_name}')")
    dt: float = 1.0 / CAMERA_FRAME_STREAMING_RATE
    cap: cv2.VideoCapture | None = None

    try:
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            print(
                f"❌ Could not open camera device {device_index} ('{camera_name}'). "
                "Check connection and device index."
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FRAME_STREAMING_RATE)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  '{camera_name}' opened: {actual_w}x{actual_h} @ ~{CAMERA_FRAME_STREAMING_RATE} Hz")

        while not dm.is_shutdown_requested():
            iteration_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"⚠️  Camera '{camera_name}' read failed, skipping frame")
                time.sleep(dt)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dm.set_rgb_image(rgb, camera_name)

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Camera thread error ('{camera_name}'): {e}")
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
            print(f"  ✓ Camera '{camera_name}' released")
        print(f"📷 Camera thread stopped ('{camera_name}')")


def _dual_teleop_loop(
    data_manager: DualDataManager,
    loop_rate_hz: float,
) -> None:
    """Map both leader-mapped states into follower targets.

    Mirrors the single-arm _teleop_loop from 2_collect_teleop_data_with_neuracore,
    extended for dual arm. Hardware targets are 10-DOF body-only; gripper targets
    are set via per-arm controller trigger state for dual_joint_state_thread.
    """
    dt = 1.0 / loop_rate_hz
    print("🌀 Dual teleop loop started")
    try:
        while not data_manager.is_shutdown_requested():
            t0 = time.time()

            left_angles, left_gripper = data_manager.get_leader_mapped_state("left")
            right_angles, right_gripper = data_manager.get_leader_mapped_state("right")

            if (
                left_angles is not None and left_gripper is not None
                and right_angles is not None and right_gripper is not None
            ):
                combined = np.concatenate([
                    np.asarray(left_angles, dtype=np.float64).flatten()[:5],
                    np.asarray(right_angles, dtype=np.float64).flatten()[:5],
                ])
                data_manager.set_target_joint_angles(combined)

                # dual_joint_state_thread reads trigger_value from controller state
                # and computes gripper_target = 1.0 - trigger_value.
                data_manager.set_controller_state("left", None, 1.0, 1.0 - float(left_gripper))
                data_manager.set_controller_state("right", None, 1.0, 1.0 - float(right_gripper))
                data_manager.set_teleop_state(True)

            elapsed = time.time() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as e:
        print(f"❌ Dual teleop loop error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🌀 Dual teleop loop stopped")


def main() -> None:
    """Run dual SO101 leader → dual SO101 follower teleop with Neuracore logging."""
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Dual SO101 leader → dual SO101 follower teleop with Neuracore data collection.",
    )
    parser.add_argument("--left-leader-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--left-leader-id", type=str, default="my_awesome_left_leader")
    parser.add_argument("--right-leader-port", type=str, default="/dev/ttyACM2")
    parser.add_argument("--right-leader-id", type=str, default="my_awesome_right_leader")
    parser.add_argument("--left-follower-port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--left-follower-id", type=str, default="my_awesome_left_follower")
    parser.add_argument("--right-follower-port", type=str, default="/dev/ttyACM3")
    parser.add_argument("--right-follower-id", type=str, default="my_awesome_right_follower")
    parser.add_argument("--leader-rate", type=float, default=50.0)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name in Neuracore (default: timestamp-based name).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DUAL SO101 LEADER → DUAL SO101 FOLLOWER TELEOP WITH NEURACORE")
    print("=" * 60)
    print("Thread frequencies:")
    print(f"  🦾 Leader Reader (×2): {args.leader_rate:.1f} Hz")
    print(f"  🔁 Teleop Loop:        {CONTROLLER_DATA_RATE:.1f} Hz")
    print(f"  🤖 Robot Controller:   {ROBOT_RATE:.1f} Hz")
    print(f"  📡 NC Logging:         {_NC_LOGGING_RATE_HZ:.1f} Hz")
    print(f"  📸 Camera Frame (×2):  {CAMERA_FRAME_STREAMING_RATE:.1f} Hz")

    # ── Neuracore ─────────────────────────────────────────────────────────────

    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name=_NC_ROBOT_NAME,
        urdf_path=str(DUAL_URDF_PATH),
        overwrite=True,
    )

    dataset_name = (
        args.dataset_name
        or f"so101-dual-leader-teleop-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    print(f"\n🔧 Creating dataset '{dataset_name}'...")
    nc.create_dataset(
        name=dataset_name,
        description="Dual-arm SO101 teleop data collection using dual SO101 leader arms.",
    )

    # ── Leader arms ───────────────────────────────────────────────────────────

    print("\n🦾 Initializing SO101 leader arms...")
    left_leader = LerobotSO101LeaderArm(
        port=args.left_leader_port,
        calibration_id=args.left_leader_id,
    )
    left_leader.configure_follower(
        follower_limits_deg=SO101_JOINT_LIMITS_DEG,
        follower_offsets_deg=SO101_OFFSETS_DEG,
        follower_directions=SO101_DIRECTIONS,
        leader_to_follower_joint=LEADER_TO_SO101_JOINT,
        fixed_joints=SO101_FIXED_JOINTS,
    )

    right_leader = LerobotSO101LeaderArm(
        port=args.right_leader_port,
        calibration_id=args.right_leader_id,
    )
    right_leader.configure_follower(
        follower_limits_deg=SO101_JOINT_LIMITS_DEG,
        follower_offsets_deg=SO101_OFFSETS_DEG_2,
        follower_directions=SO101_DIRECTIONS,
        leader_to_follower_joint=LEADER_TO_SO101_JOINT,
        fixed_joints=SO101_FIXED_JOINTS,
    )

    for side, leader, port, lid in (
        ("left", left_leader, args.left_leader_port, args.left_leader_id),
        ("right", right_leader, args.right_leader_port, args.right_leader_id),
    ):
        try:
            leader.connect(calibrate=False)
        except Exception as e:
            print(f"✗ Failed to connect to {side} leader: {e}")
            if "no calibration registered" in str(e).lower():
                print(
                    f"Run: lerobot-calibrate --teleop.type=so101_leader "
                    f"--teleop.port={port} --teleop.id={lid}"
                )
            raise SystemExit(1) from e
    print("✓ Both leader arms connected")

    # ── Shared state ──────────────────────────────────────────────────────────

    data_manager = DualDataManager()

    # ── Follower controller ───────────────────────────────────────────────────

    print("\n🤖 Initializing dual SO101 follower controller...")
    dual_ctrl = SO101DualController(
        left_port=args.left_follower_port,
        left_follower_id=args.left_follower_id,
        right_port=args.right_follower_port,
        right_follower_id=args.right_follower_id,
        robot_rate=ROBOT_RATE,
        neutral_joint_angles=np.asarray(NEUTRAL_JOINT_ANGLES, dtype=np.float64),
    )
    dual_ctrl.start_control_loop()

    # ── Joint state threads ───────────────────────────────────────────────────

    print("📊 Starting joint state threads (left + right)...")
    left_joint_thread = threading.Thread(
        target=dual_joint_state_thread,
        args=(data_manager, dual_ctrl.left, "left"),
        daemon=True,
    )
    right_joint_thread = threading.Thread(
        target=dual_joint_state_thread,
        args=(data_manager, dual_ctrl.right, "right"),
        daemon=True,
    )
    left_joint_thread.start()
    right_joint_thread.start()

    # ── Leader reader threads ─────────────────────────────────────────────────

    print("\n🎮 Starting leader reader threads (left + right)...")
    left_leader_thread = threading.Thread(
        target=dual_leader_reader_thread,
        args=(data_manager, left_leader, "left", args.leader_rate),
        daemon=True,
    )
    right_leader_thread = threading.Thread(
        target=dual_leader_reader_thread,
        args=(data_manager, right_leader, "right", args.leader_rate),
        daemon=True,
    )
    left_leader_thread.start()
    right_leader_thread.start()

    # ── Teleop loop thread ────────────────────────────────────────────────────

    print("\n🔁 Starting dual teleop loop thread...")
    teleop_thread = threading.Thread(
        target=_dual_teleop_loop,
        args=(data_manager, CONTROLLER_DATA_RATE),
        daemon=True,
    )
    teleop_thread.start()

    # ── Neuracore logging thread ───────────────────────────────────────────────

    print("\n📡 Starting Neuracore logging thread...")
    nc_logging_thread = threading.Thread(
        target=_neuracore_logging_thread,
        args=(data_manager, _NC_LOGGING_RATE_HZ),
        daemon=True,
    )
    nc_logging_thread.start()

    # ── Camera threads ────────────────────────────────────────────────────────

    print("\n📷 Starting camera threads (×2)...")
    cam1_thread = threading.Thread(
        target=_run_camera,
        args=(data_manager, "rgb", CAMERA_DEVICE_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT),
        daemon=True,
    )
    cam2_thread = threading.Thread(
        target=_run_camera,
        args=(data_manager, "rgb_2", CAMERA_2_DEVICE_INDEX, CAMERA_2_WIDTH, CAMERA_2_HEIGHT),
        daemon=True,
    )
    cam1_thread.start()
    cam2_thread.start()

    print()
    print("🚀 Starting dual-arm teleoperation with Neuracore data collection...")
    print("   - Move the SO101 leader arms to drive the followers.")
    print("   - The followers start disabled; press 'e' to enable them.")
    if sys.stdin.isatty():
        print("   - Press 'e' (no Enter) to enable / disable both followers.")
        print("   - Press 'r' (no Enter) to start / stop Neuracore recording.")
    else:
        print("   - Stdin is not a TTY; use the Neuracore UI for recording.")
    print("⚠️  Press Ctrl+C to exit")
    print()

    # ── Keyboard input ────────────────────────────────────────────────────────

    stdin_fd = sys.stdin.fileno()
    old_termios: Any = None
    if sys.stdin.isatty():
        old_termios = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)

    def toggle_robot_enabled() -> None:
        state = data_manager.get_robot_activity_state()
        if state == RobotActivityState.ENABLED:
            data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
            dual_ctrl.left.graceful_stop()
            dual_ctrl.right.graceful_stop()
            data_manager.set_teleop_state(False)
            print("\n✓ Both arms disabled (press 'e' to enable)\n", flush=True)
        elif state == RobotActivityState.DISABLED:
            left_ok = dual_ctrl.left.resume_robot()
            right_ok = dual_ctrl.right.resume_robot()
            if left_ok and right_ok:
                data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
                print("\n✓ Both arms enabled (press 'e' to disable)\n", flush=True)
            else:
                print("\n✗ Failed to enable one or both arms\n", flush=True)
        else:
            print("\n⚠️  Robot is homing; wait before toggling enable\n", flush=True)

    _toggle_lock = threading.Lock()

    def _do_toggle_recording() -> None:
        if not _toggle_lock.acquire(blocking=False):
            return
        try:
            if not nc.is_recording():
                try:
                    nc.start_recording()
                    print("\n✓ 🔴 Neuracore recording started (press 'r' to stop)\n", flush=True)
                except Exception as e:
                    print(f"\n✗ Failed to start recording: {e}\n", flush=True)
                    traceback.print_exc()
            else:
                try:
                    nc.stop_recording()
                    print("\n✓ ⏹️  Neuracore recording stopped (press 'r' to start)\n", flush=True)
                except Exception as e:
                    print(f"\n✗ Failed to stop recording: {e}\n", flush=True)
                    traceback.print_exc()
        finally:
            _toggle_lock.release()

    def toggle_neuracore_recording() -> None:
        threading.Thread(target=_do_toggle_recording, daemon=True).start()

    # ── Main loop ─────────────────────────────────────────────────────────────

    try:
        try:
            while not data_manager.is_shutdown_requested():
                if old_termios is not None:
                    readable, _, _ = select.select([sys.stdin], [], [], 1.0)
                    if readable:
                        ch = sys.stdin.read(1)
                        if ch:
                            key = ch.lower()
                            if key == "e":
                                toggle_robot_enabled()
                            elif key == "r":
                                toggle_neuracore_recording()
                else:
                    time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n👋 Interrupt received – shutting down gracefully...")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            traceback.print_exc()
    finally:
        if old_termios is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_termios)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    print("\n🧹 Cleaning up...")

    if nc.is_recording():
        try:
            print("⏹️  Stopping active recording...")
            nc.stop_recording()
            print("✓ Recording stopped")
        except Exception as e:
            print(f"⚠️  Error stopping recording: {e}")
            traceback.print_exc()
            try:
                print("⚠️  Cancelling recording as fallback...")
                nc.cancel_recording()
                print("✓ Recording cancelled")
            except Exception as inner_e:
                print(f"⚠️  Error cancelling recording: {inner_e}")

    nc.logout()
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)

    left_leader_thread.join(timeout=2.0)
    right_leader_thread.join(timeout=2.0)
    teleop_thread.join(timeout=2.0)
    nc_logging_thread.join(timeout=2.0)
    cam1_thread.join(timeout=2.0)
    cam2_thread.join(timeout=2.0)
    left_joint_thread.join(timeout=3.0)
    right_joint_thread.join(timeout=3.0)

    dual_ctrl.cleanup()

    for leader in (left_leader, right_leader):
        try:
            leader.disconnect()
        except Exception:
            pass

    print("\n👋 Done.")


if __name__ == "__main__":
    main()
