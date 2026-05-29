#!/usr/bin/env python3
"""Dual SO101 leader arms → dual SO101 follower arms teleop (direct joint mapping).

Left leader drives the left follower; right leader drives the right follower.
Each pair uses an independent offset configuration: SO101_OFFSETS_DEG for the
left pair and SO101_OFFSETS_DEG_2 for the right pair.
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

from common.configs import (
    DUAL_URDF_PATH,
    DUAL_URDF_JOINT_ORDER_FROM_OURS,
    LEADER_TO_SO101_JOINT,
    NEUTRAL_JOINT_ANGLES,
    ROBOT_RATE,
    SO101_DIRECTIONS,
    SO101_FIXED_JOINTS,
    SO101_JOINT_LIMITS_DEG,
    SO101_OFFSETS_DEG,
    SO101_OFFSETS_DEG_2,
    VISUALIZATION_RATE,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.leader_arm import LerobotSO101LeaderArm
from common.robot_visualizer import RobotVisualizer
from common.threads.dual_leader_reader import dual_leader_reader_thread

_GRIPPER_RAD_CLOSED = -0.174533
_GRIPPER_RAD_OPEN = 1.74533


def _joint_cfg_12_from_10_and_grippers(
    joint_angles_deg: np.ndarray,
    left_gripper_open: float,
    right_gripper_open: float,
) -> np.ndarray:
    """Build 12-DOF URDF config (rad) from 10-DOF body angles and two gripper values.

    Our order: [left×5, left_gripper, right×5, right_gripper]
    DUAL_URDF_JOINT_ORDER_FROM_OURS is identity, so URDF order matches our order.
    """
    body = np.asarray(joint_angles_deg, dtype=np.float64).flatten()
    left_body = np.radians(body[:5])
    right_body = np.radians(body[5:10])

    def _gripper_rad(open_val: float) -> float:
        g = float(np.clip(open_val, 0.0, 1.0))
        return _GRIPPER_RAD_CLOSED + g * (_GRIPPER_RAD_OPEN - _GRIPPER_RAD_CLOSED)

    ours = np.array(
        [*left_body, _gripper_rad(left_gripper_open), *right_body, _gripper_rad(right_gripper_open)],
        dtype=np.float64,
    )
    return ours[DUAL_URDF_JOINT_ORDER_FROM_OURS]


def main() -> None:
    """Run dual SO101 leader → dual SO101 follower teleop."""
    parser = argparse.ArgumentParser(
        description="Dual SO101 leader → dual SO101 follower teleop (direct joint mapping)."
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
        "--real-robot",
        action="store_true",
        help="Drive the real dual SO101 follower arms (default: URDF only)",
    )
    args = parser.parse_args()

    use_real_robot = args.real_robot
    print("=" * 60)
    print(
        "DUAL SO101 LEADER → DUAL SO101 FOLLOWER TELEOP"
        + (" – REAL ROBOT" if use_real_robot else " – URDF only")
    )
    print("=" * 60)

    # ── Leader arms ───────────────────────────────────────────────────────────

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

    print("\n🦾 Connecting to SO101 leader arms...")
    for side, leader in (("left", left_leader), ("right", right_leader)):
        try:
            leader.connect(calibrate=False)
        except Exception as e:
            print(f"✗ Failed to connect to {side} leader: {e}")
            if "no calibration registered" in str(e).lower():
                print(
                    f"Run: lerobot-calibrate --teleop.type=so101_leader "
                    f"--teleop.port={args.left_leader_port if side == 'left' else args.right_leader_port} "
                    f"--teleop.id={args.left_leader_id if side == 'left' else args.right_leader_id}"
                )
            sys.exit(1)
    print("✓ Both leader arms connected")

    # ── Shared state ──────────────────────────────────────────────────────────

    data_manager = DualDataManager()

    # ── Follower controller (real robot only) ─────────────────────────────────

    dual_ctrl = None
    left_joint_thread_obj = None
    right_joint_thread_obj = None

    if use_real_robot:
        from common.threads.dual_joint_state import dual_joint_state_thread
        from so101_dual_controller import SO101DualController

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

        print("📊 Starting joint state threads (left + right)...")
        left_joint_thread_obj = threading.Thread(
            target=dual_joint_state_thread,
            args=(data_manager, dual_ctrl.left, "left"),
            daemon=True,
        )
        right_joint_thread_obj = threading.Thread(
            target=dual_joint_state_thread,
            args=(data_manager, dual_ctrl.right, "right"),
            daemon=True,
        )
        left_joint_thread_obj.start()
        right_joint_thread_obj.start()

    # ── Leader reader threads ─────────────────────────────────────────────────

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

    # ── Visualizer ────────────────────────────────────────────────────────────

    visualizer = RobotVisualizer(urdf_path=DUAL_URDF_PATH)
    visualizer.add_basic_controls()
    visualizer.add_teleop_controls()
    visualizer.add_gripper_status_controls()

    if use_real_robot:
        visualizer.add_robot_status_controls()
        visualizer.add_homing_controls()
        visualizer.add_toggle_robot_enabled_status_button()

        def toggle_robot_enabled() -> None:
            assert dual_ctrl is not None
            state = data_manager.get_robot_activity_state()
            if state == RobotActivityState.ENABLED:
                data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
                dual_ctrl.left.graceful_stop()
                dual_ctrl.right.graceful_stop()
                data_manager.set_teleop_state(False)
                visualizer.update_toggle_robot_enabled_status(False)
                print("✓ 🔴 Both arms disabled")
            elif state == RobotActivityState.DISABLED:
                left_ok = dual_ctrl.left.resume_robot()
                right_ok = dual_ctrl.right.resume_robot()
                if left_ok and right_ok:
                    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
                    visualizer.update_toggle_robot_enabled_status(True)
                    print("✓ 🟢 Both arms enabled")
                else:
                    print("✗ Failed to enable one or both arms")
            else:
                print("⚠️  Robot is homing; wait before toggling enable")

        def on_go_home() -> None:
            assert dual_ctrl is not None
            state = data_manager.get_robot_activity_state()
            if state == RobotActivityState.ENABLED:
                data_manager.set_robot_activity_state(RobotActivityState.HOMING)
                data_manager.set_teleop_state(False)
                dual_ctrl.left.move_to_home()
                dual_ctrl.right.move_to_home()
            else:
                print("⚠️  Cannot home: robot not enabled")

        visualizer.set_toggle_robot_enabled_status_callback(toggle_robot_enabled)
        visualizer.set_go_home_callback(on_go_home)

    print()
    if use_real_robot:
        print("🚀 Dual leaders driving REAL dual SO101 followers. Enable robot in GUI, then move leaders. Ctrl+C to exit.")
    else:
        print("🚀 Dual leaders driving dual SO101 URDF. Move the leader arms. Ctrl+C to exit.")
    print()

    # ── Main visualization loop ───────────────────────────────────────────────

    dt_viz = 1.0 / VISUALIZATION_RATE
    try:
        while True:
            t0 = time.time()

            left_angles, left_gripper = data_manager.get_leader_mapped_state("left")
            right_angles, right_gripper = data_manager.get_leader_mapped_state("right")

            both_available = (
                left_angles is not None and left_gripper is not None
                and right_angles is not None and right_gripper is not None
            )

            if both_available:
                combined = np.concatenate([
                    np.asarray(left_angles, dtype=np.float64).flatten()[:5],
                    np.asarray(right_angles, dtype=np.float64).flatten()[:5],
                ])
                data_manager.set_target_joint_angles(combined)
                data_manager.set_controller_state("left", None, 1.0, 1.0 - float(left_gripper))
                data_manager.set_controller_state("right", None, 1.0, 1.0 - float(right_gripper))
                data_manager.set_teleop_state(True)

                if not use_real_robot:
                    data_manager.set_current_joint_angles(combined)
                    data_manager.set_current_gripper_open_value("left", float(left_gripper))
                    data_manager.set_current_gripper_open_value("right", float(right_gripper))

            current_joint_angles = data_manager.get_current_joint_angles()
            left_grip_current = data_manager.get_current_gripper_open_value("left") or 0.5
            right_grip_current = data_manager.get_current_gripper_open_value("right") or 0.5

            target_joint_angles = data_manager.get_target_joint_angles()
            _, _, left_trigger = data_manager.get_controller_state("left")

            # Use left arm values for the single-slot GUI displays
            visualizer.set_grip_value(1.0 if both_available else 0.0)
            visualizer.set_trigger_value(left_trigger)
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
                    _, _, left_trig = data_manager.get_controller_state("left")
                    _, _, right_trig = data_manager.get_controller_state("right")
                    target_left_gripper = 1.0 - left_trig
                    target_right_gripper = 1.0 - right_trig
                    ghost_cfg = _joint_cfg_12_from_10_and_grippers(
                        target_joint_angles, target_left_gripper, target_right_gripper
                    )
                    visualizer.update_ghost_robot_pose(ghost_cfg)
                else:
                    visualizer.update_ghost_robot_visibility(False)

                visualizer.update_gripper_status(
                    left_trigger,
                    robot_enabled=(robot_activity_state == RobotActivityState.ENABLED),
                )
            else:
                visualizer.update_ghost_robot_visibility(False)
                visualizer.update_gripper_status(left_trigger, robot_enabled=True)

            if current_joint_angles is not None and len(current_joint_angles) >= 10:
                current_cfg = _joint_cfg_12_from_10_and_grippers(
                    current_joint_angles, left_grip_current, right_grip_current
                )
                visualizer.update_robot_pose(current_cfg)
                visualizer.update_joint_angles_display(current_cfg)

            elapsed = time.time() - t0
            if dt_viz - elapsed > 0:
                time.sleep(dt_viz - elapsed)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupt – shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    print("\n🧹 Cleaning up...")
    data_manager.request_shutdown()
    data_manager.set_robot_activity_state(RobotActivityState.DISABLED)

    left_leader_thread.join(timeout=2.0)
    right_leader_thread.join(timeout=2.0)
    if left_joint_thread_obj is not None:
        left_joint_thread_obj.join(timeout=2.0)
    if right_joint_thread_obj is not None:
        right_joint_thread_obj.join(timeout=2.0)
    if dual_ctrl is not None:
        dual_ctrl.cleanup()

    for leader in (left_leader, right_leader):
        try:
            leader.disconnect()
        except Exception:
            pass

    visualizer.stop()
    print("👋 Done.")


if __name__ == "__main__":
    main()
