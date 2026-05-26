#!/usr/bin/env python3
"""Dual-arm SO101 teleoperation with Meta Quest and Neuracore data collection.

Left Meta Quest hand → left SO101 arm.
Right Meta Quest hand → right SO101 arm.
Single 10-DOF IK solver on the dual-arm URDF.

Controls:
  Hold LEFT + RIGHT grip  - activate dual-arm teleoperation
  Button A                - enable / disable both arms
  Button B                - move both arms to home
  Button X                - start / stop Neuracore recording
  Ctrl+C                  - exit
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import neuracore as nc
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.configs import (  # type: ignore  # noqa: E402
    CAMERA_2_DEVICE_INDEX,
    CAMERA_2_HEIGHT,
    CAMERA_2_WIDTH,
    CAMERA_DEVICE_INDEX,
    CAMERA_FRAME_STREAMING_RATE,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_MIN_CUTOFF,
    DAMPING_COST,
    DUAL_URDF_PATH,
    END_EFFECTOR_FRAME_NAMES,
    FRAME_TASK_GAIN,
    IK_SOLVER_RATE,
    JOINT_STATE_STREAMING_RATE,
    LM_DAMPING,
    NEUTRAL_JOINT_ANGLES,
    NEUTRAL_JOINT_ANGLES_DUAL,
    ORIENTATION_COST,
    POSITION_COST,
    POSTURE_COST_VECTOR_DUAL,
    ROBOT_RATE,
    ROTATION_SCALE,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    TRANSLATION_SCALE,
)
from common.data_manager_dual import DualDataManager, RobotActivityState  # type: ignore  # noqa: E402
from common.threads.dual_ik_solver import dual_ik_solver_thread  # type: ignore  # noqa: E402
from common.threads.dual_joint_state import dual_joint_state_thread  # type: ignore  # noqa: E402
from meta_quest_teleop.reader import MetaQuestReader
from pink_ik_solver import PinkIKSolver
from so101_dual_controller import SO101DualController

_BODY_DOF = 5
_NC_ROBOT_NAME = "LeRobot SO101 Dual"
_NC_LOGGING_RATE_HZ = 30.0  # Lower than control rate to avoid GIL contention
_CAMERA_NAMES = ["rgb", "rgb_2"]

# Body joint suffixes per arm — prefixed with "left_" / "right_" for Neuracore.
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

            if nc.is_recording():
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
    """Camera capture loop — mirrors common/threads/camera.py for the dual-arm setup.

    Writes frames to DualDataManager only. NC logging is handled by the shared
    Neuracore logging thread so this loop is never blocked by frame encoding.
    """
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
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
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


def main() -> None:
    """Run dual-arm SO101 Meta Quest teleop with Neuracore data collection."""
    parser = argparse.ArgumentParser(
        description="Dual-arm SO101 Meta Quest teleop with Neuracore data collection.",
    )
    parser.add_argument("--left-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--left-id", type=str, default="L1")
    parser.add_argument("--right-port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--right-id", type=str, default="L1")
    parser.add_argument("--ip-address", type=str, default=None)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name in Neuracore (default: timestamp-based name).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DUAL-ARM SO101 TELEOP WITH NEURACORE DATA COLLECTION")
    print("=" * 60)
    print(f"  Left arm:  port={args.left_port}  id={args.left_id}")
    print(f"  Right arm: port={args.right_port}  id={args.right_id}")
    print(f"  🧮 IK Solver:      {IK_SOLVER_RATE} Hz  (10-DOF, dual-arm URDF)")
    print(f"  🤖 Joint State:    {JOINT_STATE_STREAMING_RATE} Hz  (per arm)")
    print(f"  📡 NC Logging:     {_NC_LOGGING_RATE_HZ} Hz")
    print(f"  📷 Camera:         {CAMERA_FRAME_STREAMING_RATE} Hz  (×2)")

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
        or f"so101-dual-teleop-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    print(f"\n🔧 Creating dataset '{dataset_name}'...")
    nc.create_dataset(
        name=dataset_name,
        description="Dual-arm SO101 teleop data collection via Meta Quest.",
    )

    # ── Shared state ──────────────────────────────────────────────────────────

    data_manager = DualDataManager()
    data_manager.set_controller_filter_params(
        CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF
    )
    data_manager.set_teleop_scaling(TRANSLATION_SCALE, ROTATION_SCALE)

    # ── Robot hardware ────────────────────────────────────────────────────────

    print("\n🤖 Initializing dual SO101 controller...")
    dual_ctrl = SO101DualController(
        left_port=args.left_port,
        left_follower_id=args.left_id,
        right_port=args.right_port,
        right_follower_id=args.right_id,
        robot_rate=ROBOT_RATE,
        neutral_joint_angles=np.array(NEUTRAL_JOINT_ANGLES),
    )
    dual_ctrl.start_control_loop()

    # ── Joint state threads ───────────────────────────────────────────────────

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

    # ── IK solver ─────────────────────────────────────────────────────────────

    initial_joint_angles = np.radians(NEUTRAL_JOINT_ANGLES_DUAL)
    posture_cost_vec = np.array(POSTURE_COST_VECTOR_DUAL, dtype=float)

    print("\n🔧 Creating dual-arm Pink IK solver...")
    ik_solver = PinkIKSolver(
        urdf_path=DUAL_URDF_PATH,
        end_effector_frames=END_EFFECTOR_FRAME_NAMES,
        solver_name=SOLVER_NAME,
        position_cost=POSITION_COST,
        orientation_cost=ORIENTATION_COST,
        frame_task_gain=FRAME_TASK_GAIN,
        lm_damping=LM_DAMPING,
        damping_cost=DAMPING_COST,
        solver_damping_value=SOLVER_DAMPING_VALUE,
        integration_time_step=1.0 / IK_SOLVER_RATE,
        initial_configuration=initial_joint_angles,
        posture_cost_vector=posture_cost_vec,
    )

    # ── Quest reader ──────────────────────────────────────────────────────────

    print("\n🎮 Initializing Meta Quest reader...")
    quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

    print("\n🧮 Starting dual IK solver thread...")
    ik_thread = threading.Thread(
        target=dual_ik_solver_thread,
        args=(data_manager, ik_solver, quest_reader),
        daemon=True,
    )
    ik_thread.start()

    # ── Neuracore logging thread ───────────────────────────────────────────────

    print("\n📡 Starting Neuracore logging thread...")
    nc_logging_thread = threading.Thread(
        target=_neuracore_logging_thread,
        args=(data_manager, _NC_LOGGING_RATE_HZ),
        daemon=True,
    )
    nc_logging_thread.start()

    # ── Camera threads ────────────────────────────────────────────────────────

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

    # ── Button callbacks ──────────────────────────────────────────────────────

    def toggle_robot_enabled_status() -> None:
        state = data_manager.get_robot_activity_state()
        if state == RobotActivityState.ENABLED:
            data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
            data_manager.set_teleop_state(False)
            dual_ctrl.left.graceful_stop()
            dual_ctrl.right.graceful_stop()
            print("✓ 🔴 Both arms disabled")
        elif state in (RobotActivityState.DISABLED, RobotActivityState.HOMING):
            left_ok = dual_ctrl.left.resume_robot()
            right_ok = dual_ctrl.right.resume_robot()
            if left_ok and right_ok:
                data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
                print("✓ 🟢 Both arms enabled")
            else:
                print("✗ Failed to enable one or both arms")

    def on_go_home() -> None:
        state = data_manager.get_robot_activity_state()
        if state in (RobotActivityState.ENABLED, RobotActivityState.HOMING):
            print("🏠 Moving both arms to home...")
            data_manager.set_robot_activity_state(RobotActivityState.HOMING)
            data_manager.set_teleop_state(False)
            dual_ctrl.left.move_to_home()
            dual_ctrl.right.move_to_home()
        else:
            print("⚠️  Cannot home: arms not enabled")

    _toggle_lock = threading.Lock()

    def _do_toggle_recording() -> None:
        if not _toggle_lock.acquire(blocking=False):
            return  # HTTP request already in-flight — drop extra presses
        try:
            if not nc.is_recording():
                print("\n⏳ Starting recording...\n", flush=True)
                try:
                    nc.start_recording()
                    print("\n✓ 🔴 Recording started (press X to stop)\n", flush=True)
                except Exception as e:
                    print(f"\n✗ Failed to start recording: {e}\n", flush=True)
                    traceback.print_exc()
            else:
                print("\n⏳ Stopping recording...\n", flush=True)
                try:
                    nc.stop_recording()
                    print("\n✓ ⏹️  Recording stopped (press X to start)\n", flush=True)
                except Exception as e:
                    print(f"\n✗ Failed to stop recording: {e}\n", flush=True)
                    traceback.print_exc()
        finally:
            _toggle_lock.release()

    def toggle_neuracore_recording() -> None:
        threading.Thread(target=_do_toggle_recording, daemon=True).start()

    quest_reader.on("button_a_pressed", toggle_robot_enabled_status)
    quest_reader.on("button_b_pressed", on_go_home)
    quest_reader.on("button_x_pressed", toggle_neuracore_recording)

    print()
    print("🚀 Dual-arm teleoperation with Neuracore data collection ready.")
    print("   1. Press BUTTON A to enable/disable both arms")
    print("   2. Hold LEFT + RIGHT GRIP to activate teleoperation")
    print("   3. Move controllers — arms follow!")
    print("   4. Hold triggers to close grippers")
    print("   5. Press BUTTON B to home both arms")
    print("   6. Press BUTTON X to start / stop recording")
    print("⚠️  Press Ctrl+C to exit")
    print()

    try:
        while not data_manager.is_shutdown_requested():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupt received — shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        traceback.print_exc()

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
    quest_reader.stop()

    ik_thread.join(timeout=3.0)
    left_joint_thread.join(timeout=3.0)
    right_joint_thread.join(timeout=3.0)
    nc_logging_thread.join(timeout=2.0)
    cam1_thread.join(timeout=2.0)
    cam2_thread.join(timeout=2.0)

    dual_ctrl.cleanup()
    print("\n👋 Done.")


if __name__ == "__main__":
    main()
