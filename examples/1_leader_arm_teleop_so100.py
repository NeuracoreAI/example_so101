#!/usr/bin/env python3
"""SO100 leader arm → SO100 follower arm teleop (direct joint mapping).

"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

# Repo root for so100_controller; examples for common.*
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

from common.configs import (
    LEADER_TO_SO100_JOINT,
    NEUTRAL_JOINT_ANGLES,
    ROBOT_RATE,
    SO100_DIRECTIONS,
    SO100_FIXED_JOINTS,
    SO100_JOINT_LIMITS_DEG,
    SO100_OFFSETS_DEG,
    URDF_JOINT_ORDER_FROM_OURS,
    VISUALIZATION_RATE,
)
from common.data_manager import DataManager, RobotActivityState
from common.leader_arm import LerobotSO100LeaderArm
from common.robot_visualizer import RobotVisualizer
from common.threads.leader_reader import leader_reader_thread


def _joint_cfg_6_from_5_and_gripper(
    joint_angles_deg: np.ndarray, gripper_open: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build 6-DOF config (rad): (ours order for display, URDF order for update_robot_pose)."""
    body_rad = np.radians(np.asarray(joint_angles_deg, dtype=np.float64).flatten()[:5])
    g = float(np.clip(gripper_open, 0.0, 1.0))
    GRIPPER_RAD_CLOSED = -0.174533
    GRIPPER_RAD_OPEN = 1.74533
    gripper_rad = GRIPPER_RAD_CLOSED + g * (GRIPPER_RAD_OPEN - GRIPPER_RAD_CLOSED)
    ours = np.append(body_rad, gripper_rad)
    for_urdf = ours[URDF_JOINT_ORDER_FROM_OURS]
    return ours, for_urdf


def main() -> None:
    """Run SO100 leader → SO100 follower teleop (URDF and optionally real robot)."""
    from common.configs import URDF_PATH

    parser = argparse.ArgumentParser(
        description="SO100 URDF / real-robot teleop with LeRobot SO100 leader arm."
    )
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader-id", type=str, default="my_awesome_leader_arm")
    parser.add_argument("--leader-rate", type=float, default=50.0)
    parser.add_argument(
        "--real-robot",
        action="store_true",
        help="Drive the real SO100 follower arm (default: URDF only)",
    )
    parser.add_argument("--follower-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--follower-id", type=str, default="my_awesome_follower_arm")
    args = parser.parse_args()

    use_real_robot = args.real_robot
    print("=" * 60)
    print(
        "SO100 LEADER → SO100 FOLLOWER TELEOP"
        + (" – REAL ROBOT" if use_real_robot else " – URDF only")
    )
    print("=" * 60)

    leader = LerobotSO100LeaderArm(port=args.leader_port, calibration_id=args.leader_id)
    leader.configure_follower(
        follower_limits_deg=SO100_JOINT_LIMITS_DEG,
        follower_offsets_deg=SO100_OFFSETS_DEG,
        follower_directions=SO100_DIRECTIONS,
        leader_to_follower_joint=LEADER_TO_SO100_JOINT,
        fixed_joints=SO100_FIXED_JOINTS,
    )
    print("\n🦾 Connecting to SO100 leader arm...")
    try:
        leader.connect(calibrate=False)
    except Exception as e:
        print(f"Failed to connect to leader: {e}")
        if "no calibration registered" in str(e).lower():
            print(
                "Run: lerobot-calibrate --teleop.type=so100_leader --teleop.port=... --teleop.id=..."
            )
        sys.exit(1)
    print("✓ Leader arm connected")

    data_manager = DataManager()

    robot_controller = None
    joint_state_thread_obj = None
    if use_real_robot:
        from common.threads.joint_state import joint_state_thread

        from so100_controller import SO100Controller

        print("\n🤖 Initializing SO100 follower controller...")
        robot_controller = SO100Controller(
            port=args.follower_port,
            follower_id=args.follower_id,
            robot_rate=ROBOT_RATE,
            neutral_joint_angles=NEUTRAL_JOINT_ANGLES,
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

    leader_thread = threading.Thread(
        target=leader_reader_thread,
        args=(data_manager, leader, args.leader_rate),
        daemon=True,
    )
    leader_thread.start()

    visualizer = RobotVisualizer(urdf_path=URDF_PATH)
    visualizer.add_basic_controls()
    visualizer.add_teleop_controls()
    visualizer.add_gripper_status_controls()
    if use_real_robot:
        visualizer.add_homing_controls()
        visualizer.add_toggle_robot_enabled_status_button()

        def toggle_robot_enabled() -> None:
            assert robot_controller is not None
            state = data_manager.get_robot_activity_state()
            if state == RobotActivityState.ENABLED:
                data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
                robot_controller.graceful_stop()
                data_manager.set_teleop_state(False, None, None)
                visualizer.update_toggle_robot_enabled_status(False)
                print("✓ 🔴 Robot disabled")
            elif state == RobotActivityState.DISABLED:
                if robot_controller.resume_robot():
                    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
                    visualizer.update_toggle_robot_enabled_status(True)
                    print("✓ 🟢 Robot enabled")
                else:
                    print("✗ Failed to enable robot")

        def on_go_home() -> None:
            assert robot_controller is not None
            state = data_manager.get_robot_activity_state()
            if state == RobotActivityState.ENABLED:
                data_manager.set_robot_activity_state(RobotActivityState.HOMING)
                data_manager.set_teleop_state(False, None, None)
                ok = robot_controller.move_to_home()
                if not ok:
                    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            else:
                print("⚠️ Cannot home: robot not enabled")

        visualizer.set_toggle_robot_enabled_status_callback(toggle_robot_enabled)
        visualizer.set_go_home_callback(on_go_home)

    print()
    if use_real_robot:
        print(
            "🚀 Leader arm driving REAL SO100 follower. Enable robot in GUI, then move leader. Ctrl+C to exit."
        )
    else:
        print("🚀 Leader arm driving SO100 URDF. Move the leader arm. Ctrl+C to exit.")
    print()

    dt_viz = 1.0 / VISUALIZATION_RATE
    try:
        while True:
            t0 = time.time()
            mapped_angles, mapped_gripper = data_manager.get_leader_mapped_state()
            if mapped_angles is not None and mapped_gripper is not None:
                data_manager.set_target_joint_angles(mapped_angles)
                data_manager.set_controller_data(None, 1.0, 1.0 - mapped_gripper)
                data_manager.set_teleop_state(True, None, None)
                if not use_real_robot:
                    data_manager.set_current_joint_angles(mapped_angles)

            current_joint_angles = data_manager.get_current_joint_angles()
            _, grip_value, trigger_value = data_manager.get_controller_data()
            target_joint_angles = data_manager.get_target_joint_angles()

            visualizer.set_grip_value(grip_value)
            visualizer.set_trigger_value(trigger_value)
            visualizer.update_teleop_status(data_manager.get_teleop_active())

            if use_real_robot:
                robot_activity_state = data_manager.get_robot_activity_state()
                if robot_activity_state == RobotActivityState.ENABLED:
                    visualizer.update_robot_status("Robot Status: Enabled")
                elif robot_activity_state == RobotActivityState.HOMING:
                    visualizer.update_robot_status("Robot Status: Homing")
                else:
                    visualizer.update_robot_status("Robot Status: Disabled")
                if (
                    target_joint_angles is not None
                    and robot_activity_state == RobotActivityState.ENABLED
                ):
                    visualizer.update_ghost_robot_visibility(True)
                    target_gripper = 1.0 - trigger_value if trigger_value is not None else 0.5
                    _, ghost_cfg_urdf = _joint_cfg_6_from_5_and_gripper(
                        target_joint_angles, target_gripper
                    )
                    visualizer.update_ghost_robot_pose(ghost_cfg_urdf)
                else:
                    visualizer.update_ghost_robot_visibility(False)
                visualizer.update_gripper_status(
                    trigger_value,
                    robot_enabled=(robot_activity_state == RobotActivityState.ENABLED),
                )
            else:
                visualizer.update_robot_status("URDF only – SO100 leader driving")
                visualizer.update_ghost_robot_visibility(False)
                visualizer.update_gripper_status(trigger_value, robot_enabled=True)

            if current_joint_angles is not None:
                current_gripper = data_manager.get_current_gripper_open_value()
                gripper_open = current_gripper if current_gripper is not None else 0.5
                current_cfg_ours, current_cfg_urdf = _joint_cfg_6_from_5_and_gripper(
                    current_joint_angles, gripper_open
                )
                visualizer.update_robot_pose(current_cfg_urdf)
                visualizer.update_joint_angles_display(current_cfg_ours)

            elapsed = time.time() - t0
            if dt_viz - elapsed > 0:
                time.sleep(dt_viz - elapsed)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupt – shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

    print("\n🧹 Cleaning up...")
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
    leader_thread.join(timeout=2.0)
    if joint_state_thread_obj is not None:
        joint_state_thread_obj.join(timeout=2.0)
    if robot_controller is not None:
        robot_controller.cleanup()
    leader.disconnect()
    visualizer.stop()
    print("👋 Done.")


if __name__ == "__main__":
    main()
