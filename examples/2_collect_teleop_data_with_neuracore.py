#!/usr/bin/env python3
"""SO100 leader arm → SO100 follower teleop with Neuracore data collection.

This demo:
- Uses a LeRobot SO100 leader arm as the teleoperation device
- Optionally drives a real SO100 follower arm (via `SO100Controller`)
- Streams RGB frames from a simple USB webcam (OpenCV-based `camera_thread`)
- Logs joint states, joint targets, gripper states, and RGB images to Neuracore
"""

import argparse
import multiprocessing
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import neuracore as nc
import numpy as np

# Repo root for so100_controller; examples for common.*
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

from common.configs import (  # type: ignore  # noqa: E402
    CAMERA_FRAME_STREAMING_RATE,
    CAMERA_LOGGING_NAME,
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_DATA_RATE,
    CONTROLLER_MIN_CUTOFF,
    GRIPPER_LOGGING_NAME,
    JOINT_NAMES,
    LEADER_TO_SO100_JOINT,
    NEUTRAL_JOINT_ANGLES,
    ROBOT_RATE,
    SO100_DIRECTIONS,
    SO100_FIXED_JOINTS,
    SO100_JOINT_LIMITS_DEG,
    SO100_OFFSETS_DEG,
    URDF_PATH,
)
from common.data_manager import DataManager, RobotActivityState  # type: ignore  # noqa: E402
from common.leader_arm import LerobotSO100LeaderArm  # type: ignore  # noqa: E402
from common.threads.camera import camera_thread  # type: ignore  # noqa: E402
from common.threads.joint_state import joint_state_thread  # type: ignore  # noqa: E402
from common.threads.leader_arm_controller import leader_arm_controller_thread  # type: ignore  # noqa: E402
from so100_controller import SO100Controller  # type: ignore  # noqa: E402


def log_to_neuracore_on_change_callback(
    name: str, value: Any, timestamp: float
) -> None:
    """Log data to Neuracore when DataManager state changes."""
    try:
        if name == "log_joint_positions":
            # DataManager stores degrees; Neuracore expects radians.
            data_value = np.radians(value)
            data_dict = {
                joint_name: angle
                for joint_name, angle in zip(JOINT_NAMES, data_value)
            }
            nc.log_joint_positions(data_dict, timestamp=timestamp)
        elif name == "log_joint_target_positions":
            data_value = np.radians(value)
            data_dict = {
                joint_name: angle
                for joint_name, angle in zip(JOINT_NAMES, data_value)
            }
            nc.log_joint_target_positions(data_dict, timestamp=timestamp)
        elif name == "log_parallel_gripper_open_amounts":
            data_dict = {GRIPPER_LOGGING_NAME: float(value)}
            nc.log_parallel_gripper_open_amounts(data_dict, timestamp=timestamp)
        elif name == "log_parallel_gripper_target_open_amounts":
            data_dict = {GRIPPER_LOGGING_NAME: float(value)}
            nc.log_parallel_gripper_target_open_amounts(
                data_dict, timestamp=timestamp
            )
        elif name == "log_rgb":
            camera_name = CAMERA_LOGGING_NAME
            image_array = value
            nc.log_rgb(camera_name, image_array, timestamp=timestamp)
        else:
            print(f"\n⚠️  Unknown logging stream name for Neuracore: {name}")
    except Exception as e:  # pragma: no cover - logging should never crash demo
        print(f"\n⚠️  Failed to log {name} to Neuracore. Exception: {e}")
        print("Traceback:")
        traceback.print_exc()


def _teleop_loop(
    data_manager: DataManager,
    use_real_robot: bool,
    loop_rate_hz: float,
) -> None:
    """Map leader-mapped state into follower targets and controller fields.

    This mirrors the leader → follower mapping behavior in X_leader_arm_teleop_so100,
    but without visualization or IK. Joint_state_thread handles sending commands
    to the real robot when enabled.
    """
    dt = 1.0 / loop_rate_hz
    print("🌀 Teleop loop started")
    try:
        while not data_manager.is_shutdown_requested():
            t0 = time.time()

            mapped_angles, mapped_gripper = data_manager.get_leader_mapped_state()
            if mapped_angles is not None and mapped_gripper is not None:
                # Target joints in degrees (SO100 controller convention)
                # For Neuracore visualization, we also append a pseudo "gripper joint"
                # to the target vector so arm + gripper can be shown together.
                pseudo_gripper_deg = float(np.clip(mapped_gripper, 0.0, 1.0) * 100.0)
                target_with_gripper = np.concatenate(
                    [np.asarray(mapped_angles, dtype=np.float64).flatten(), [pseudo_gripper_deg]]
                )
                data_manager.set_target_joint_angles(target_with_gripper)

                # Reuse controller grip/trigger channels: grip=1.0, trigger = 1 - gripper_open.
                # Joint_state_thread interprets trigger_value as "closedness"
                # and inverts it back to an open amount for the gripper target.
                data_manager.set_controller_data(
                    transform=None,
                    grip=1.0,
                    trigger=1.0 - float(mapped_gripper),
                )
                data_manager.set_teleop_state(True, None, None)

                # In URDF-only mode, reflect targets as current state for logging.
                if not use_real_robot:
                    current_with_gripper = target_with_gripper
                    data_manager.set_current_joint_angles(current_with_gripper)
                    data_manager.set_current_gripper_open_value(float(mapped_gripper))

            elapsed = time.time() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as e:
        print(f"❌ Teleop loop error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🌀 Teleop loop stopped")


def main() -> None:
    """Run SO100 leader → SO100 follower teleop with Neuracore logging."""
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="SO100 leader → SO100 follower teleop with Neuracore data collection.",
    )
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader-id", type=str, default="my_awesome_leader_arm")
    parser.add_argument("--leader-rate", type=float, default=50.0)
    parser.add_argument(
        "--real-robot",
        action="store_true",
        help="Drive the real SO100 follower arm (default: URDF-only logging).",
    )
    parser.add_argument("--follower-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--follower-id", type=str, default="my_awesome_follower_arm")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name in Neuracore (default: timestamp-based name).",
    )
    args = parser.parse_args()

    use_real_robot = args.real_robot

    print("=" * 60)
    print(
        "SO100 LEADER → SO100 FOLLOWER TELEOP WITH NEURACORE"
        + (" – REAL ROBOT" if use_real_robot else " – URDF only")
    )
    print("=" * 60)
    print("Thread frequencies:")
    print(f"  🦾 Leader Reader:    {args.leader_rate:.1f} Hz")
    print(f"  🔁 Teleop Loop:      {CONTROLLER_DATA_RATE:.1f} Hz")
    if use_real_robot:
        print(f"  🤖 Robot Controller: {ROBOT_RATE:.1f} Hz")
        print(f"  📊 Joint State:      {CAMERA_FRAME_STREAMING_RATE:.1f} Hz")
    print(f"  📸 Camera Frame:     {CAMERA_FRAME_STREAMING_RATE:.1f} Hz")

    # Connect to Neuracore
    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name="LeRobot SO100",
        urdf_path=str(URDF_PATH),
        overwrite=True,
    )

    # Create dataset
    dataset_name = (
        args.dataset_name or f"so100-teleop-data-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    print(f"\n🔧 Creating dataset {dataset_name}...")
    nc.create_dataset(
        name=dataset_name,
        description="Teleop data collection for SO100 follower using LeRobot SO100 leader arm.",
    )

    # Initialize shared state
    data_manager = DataManager()
    data_manager.set_on_change_callback(log_to_neuracore_on_change_callback)
    data_manager.set_controller_filter_params(
        CONTROLLER_MIN_CUTOFF,
        CONTROLLER_BETA,
        CONTROLLER_D_CUTOFF,
    )

    # Initialize leader arm and follower mapping
    print("\n🦾 Initializing SO100 leader arm...")
    leader = LerobotSO100LeaderArm(
        port=args.leader_port,
        calibration_id=args.leader_id,
    )
    leader.configure_follower(
        follower_limits_deg=SO100_JOINT_LIMITS_DEG,
        follower_offsets_deg=SO100_OFFSETS_DEG,
        follower_directions=SO100_DIRECTIONS,
        leader_to_follower_joint=LEADER_TO_SO100_JOINT,
        fixed_joints=SO100_FIXED_JOINTS,
    )
    try:
        leader.connect(calibrate=False)
    except Exception as e:
        print(f"✗ Failed to connect to leader arm: {e}")
        if "no calibration registered" in str(e).lower():
            print(
                "Run: lerobot-calibrate --teleop.type=so100_leader "
                "--teleop.port=... --teleop.id=..."
            )
        raise SystemExit(1) from e
    print("✓ Leader arm connected")

    robot_controller: SO100Controller | None = None
    joint_state_thread_obj: threading.Thread | None = None

    # Initialize follower controller (optional)
    if use_real_robot:
        print("\n🤖 Initializing SO100 follower controller...")
        robot_controller = SO100Controller(
            port=args.follower_port,
            follower_id=args.follower_id,
            robot_rate=ROBOT_RATE,
            neutral_joint_angles=np.asarray(NEUTRAL_JOINT_ANGLES, dtype=np.float64),
            debug_mode=False,
        )
        robot_controller.start_control_loop()
        print("📊 Starting joint state thread...")
        joint_state_thread_obj = threading.Thread(
            target=joint_state_thread,
            args=(data_manager, robot_controller),
            daemon=True,
        )
        joint_state_thread_obj.start()
        # Enable robot activity state and resume controller
        data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
        if not robot_controller.resume_robot():
            print("⚠️  Failed to resume SO100 robot; commands will not be sent.")

    # Start leader arm controller thread (same pattern as Meta Quest controller thread)
    print("\n🎮 Starting leader arm controller thread...")
    leader_thread = threading.Thread(
        target=leader_arm_controller_thread,
        args=(data_manager, leader, args.leader_rate),
        daemon=True,
    )
    leader_thread.start()

    # Start teleop loop thread
    print("\n🔁 Starting teleop loop thread...")
    teleop_thread = threading.Thread(
        target=_teleop_loop,
        args=(data_manager, use_real_robot, CONTROLLER_DATA_RATE),
        daemon=True,
    )
    teleop_thread.start()

    # Start camera thread (USB webcam)
    print("\n📷 Starting camera thread (USB webcam)...")
    camera_thread_obj = threading.Thread(
        target=camera_thread,
        args=(data_manager,),
        daemon=True,
    )
    camera_thread_obj.start()

    print()
    print("🚀 Starting teleoperation with Neuracore data collection...")
    print("   - Move the SO100 leader arm to drive the follower.")
    if use_real_robot:
        print("   - The real SO100 follower is being commanded.")
    else:
        print("   - No real robot: only leader, camera, and Neuracore logging are active.")
    print("   - Recording is controlled via Neuracore; this script attempts to auto-start.")
    print("⚠️  Press Ctrl+C to exit")
    print()

    try:
        while not data_manager.is_shutdown_requested():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n👋 Interrupt received – shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Demo error. Exception: {e}")
        print("Traceback:")
        traceback.print_exc()

    # Cleanup
    print("\n🧹 Cleaning up...")

    # Stop or cancel recording if active
    if nc.is_recording():
        try:
            print("⏹️  Stopping active recording...")
            nc.stop_recording()
            print("✓ Recording stopped")
        except Exception as e:
            print(f"⚠️  Error stopping recording. Exception: {e}")
            print("Traceback:")
            traceback.print_exc()
            try:
                print("⚠️  Cancelling recording as fallback...")
                nc.cancel_recording()
                print("✓ Recording cancelled")
            except Exception as inner_e:
                print(
                    f"⚠️  Error cancelling recording. Exception: {inner_e}",
                )

    # Request shutdown for all threads
    nc.logout()
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)

    # Join threads
    leader_thread.join(timeout=2.0)
    teleop_thread.join(timeout=2.0)
    camera_thread_obj.join(timeout=2.0)
    if joint_state_thread_obj is not None:
        joint_state_thread_obj.join(timeout=2.0)
    if robot_controller is not None:
        robot_controller.cleanup()

    # Disconnect leader
    try:
        leader.disconnect()
    except Exception:
        pass

    print("\n👋 Demo stopped.")


if __name__ == "__main__":
    main()

