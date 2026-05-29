"""Dual-arm IK solver thread for SO101 Quest teleoperation.

Ported from example_openarm's ik_solver.py. Reads both hand transforms
directly from the MetaQuestReader and drives a single 10-DOF PinkIKSolver
(dual-arm URDF, grippers locked). Each arm has its own calibration transform
computed on teleop activation.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

import numpy as np

from common.configs import (
    IK_SOLVER_RATE,
    LEFT_END_EFFECTOR_FRAME_NAME,
    RIGHT_END_EFFECTOR_FRAME_NAME,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.utils import (
    compute_hand_to_robot_calibration,
    map_head_frame_hand_to_robot_target,
    map_quest_hands_to_robot_arms,
)

from common.pink_ik_solver import PinkIKSolver

_DIVERGENCE_TOLERANCE_DEG = 0.1


def _sync_targets_from_ik(
    data_manager: DualDataManager,
    ik_solver: PinkIKSolver,
    joint_angles_deg: np.ndarray,
) -> None:
    """Push joint and EEF targets from the current IK config into DataManager."""
    data_manager.set_target_joint_angles(joint_angles_deg)
    ee_poses = ik_solver.get_current_end_effector_poses()
    data_manager.set_target_pose("left", ee_poses.get(LEFT_END_EFFECTOR_FRAME_NAME))
    data_manager.set_target_pose("right", ee_poses.get(RIGHT_END_EFFECTOR_FRAME_NAME))
    data_manager.set_ik_success(True)
    data_manager.set_ik_solve_time_ms(0.0)


def dual_ik_solver_thread(
    data_manager: DualDataManager,
    ik_solver: PinkIKSolver,
    quest_reader: Any | None = None,
) -> None:
    """Dual-arm IK solver thread.

    With quest_reader: absolute pointer poses in the head frame map to robot
    TCP targets each frame. A one-time calibration at grip-press aligns head
    frame to the robot. Left hand → left arm, right hand → right arm.
    Without quest_reader: Viser gizmos drive absolute TCP targets via DataManager.
    """
    input_label = "Quest" if quest_reader is not None else "Viser gizmo"
    print(f"🧮 Dual IK solver thread started ({input_label})")

    dt: float = 1.0 / IK_SOLVER_RATE
    left_hand_to_robot: np.ndarray | None = None
    right_hand_to_robot: np.ndarray | None = None
    left_hand_reference: np.ndarray | None = None
    right_hand_reference: np.ndarray | None = None
    teleop_active_prev = False
    mirror_control_prev = False

    def _reset_teleop_calibration() -> None:
        nonlocal left_hand_to_robot, right_hand_to_robot
        nonlocal left_hand_reference, right_hand_reference
        left_hand_to_robot = None
        right_hand_to_robot = None
        left_hand_reference = None
        right_hand_reference = None

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()
            mirror_control = data_manager.get_mirror_control_enabled()
            if mirror_control != mirror_control_prev:
                if left_hand_to_robot is not None or right_hand_to_robot is not None:
                    _reset_teleop_calibration()
                    print(
                        "↔️  Mirror control "
                        + ("enabled" if mirror_control else "disabled")
                        + " — release grips and re-press to recalibrate teleop"
                    )
                mirror_control_prev = mirror_control

            if quest_reader is not None:
                quest_left_tf = quest_reader.get_hand_controller_transform_ros(hand="left")
                quest_right_tf = quest_reader.get_hand_controller_transform_ros(hand="right")
                right_grip = quest_reader.get_grip_value("right")
                right_trigger = quest_reader.get_trigger_value("right")
                left_grip_for_dm = quest_reader.get_grip_value("left")
                left_trigger_for_dm = quest_reader.get_trigger_value("left")

                if quest_left_tf is not None and quest_right_tf is not None:
                    left_tf_for_dm, right_tf_for_dm = map_quest_hands_to_robot_arms(
                        quest_left_tf, quest_right_tf, mirror_control=mirror_control
                    )
                else:
                    left_tf_for_dm = quest_left_tf
                    right_tf_for_dm = quest_right_tf

                data_manager.set_controller_state(
                    "right", right_tf_for_dm, right_grip, right_trigger
                )
                data_manager.set_controller_state(
                    "left", left_tf_for_dm, left_grip_for_dm, left_trigger_for_dm
                )

            current_joint_angles = data_manager.get_current_joint_angles()
            robot_activity_state = data_manager.get_robot_activity_state()

            if quest_reader is not None:
                left_grip = quest_reader.get_grip_value("left")
                right_grip = quest_reader.get_grip_value("right")
                quest_left_tf = quest_reader.get_hand_controller_transform_ros(hand="left")
                quest_right_tf = quest_reader.get_hand_controller_transform_ros(hand="right")
                teleop_active = (
                    quest_left_tf is not None
                    and quest_right_tf is not None
                    and left_grip >= 0.9
                    and right_grip >= 0.9
                )
                data_manager.set_teleop_state(teleop_active)
                if quest_left_tf is not None and quest_right_tf is not None:
                    left_tf, right_tf = map_quest_hands_to_robot_arms(
                        quest_left_tf, quest_right_tf, mirror_control=mirror_control
                    )
                else:
                    left_tf = None
                    right_tf = None
            else:
                teleop_active = data_manager.get_teleop_active()
                left_tf = None
                right_tf = None

            # Anchor IK to measured joints whenever teleop is not active.
            if current_joint_angles is not None:
                if not teleop_active:
                    ik_solver.set_configuration_no_task_update(
                        np.radians(current_joint_angles)
                    )
                else:
                    current_ik_joint_angles = np.degrees(ik_solver.get_current_configuration())
                    if current_ik_joint_angles is not None and np.all(
                        np.abs(current_joint_angles - current_ik_joint_angles)
                        <= _DIVERGENCE_TOLERANCE_DEG
                    ):
                        ik_solver.set_configuration_no_task_update(
                            np.radians(current_joint_angles)
                        )

            current_poses = ik_solver.get_current_end_effector_poses()
            left_pose = current_poses.get(LEFT_END_EFFECTOR_FRAME_NAME)
            right_pose = current_poses.get(RIGHT_END_EFFECTOR_FRAME_NAME)
            if right_pose is not None:
                data_manager.set_current_end_effector_pose("right", right_pose)
            if left_pose is not None:
                data_manager.set_current_end_effector_pose("left", left_pose)

            # On rising edge of teleop, compute per-arm calibration transforms.
            if (
                quest_reader is not None
                and teleop_active
                and not teleop_active_prev
                and robot_activity_state == RobotActivityState.ENABLED
                and left_tf is not None
                and right_tf is not None
                and left_pose is not None
                and right_pose is not None
            ):
                translation_scale, rotation_scale = data_manager.get_teleop_scaling()
                left_hand_reference = left_tf.copy()
                right_hand_reference = right_tf.copy()
                left_hand_to_robot = compute_hand_to_robot_calibration(
                    left_pose, left_tf, left_hand_reference, translation_scale, rotation_scale
                )
                right_hand_to_robot = compute_hand_to_robot_calibration(
                    right_pose, right_tf, right_hand_reference, translation_scale, rotation_scale
                )
                mode = "mirror" if mirror_control else "direct"
                print(f"✓ Dual-arm teleop activated (absolute head-frame mapping, {mode})")

            if quest_reader is not None and not teleop_active and teleop_active_prev:
                _reset_teleop_calibration()
                print("✗ Dual-arm teleop deactivated")

            teleop_active_prev = teleop_active

            if robot_activity_state == RobotActivityState.POLICY_CONTROLLED:
                if current_joint_angles is not None:
                    ik_solver.set_configuration(np.radians(current_joint_angles))
                    _sync_targets_from_ik(data_manager, ik_solver, current_joint_angles)

            elif (
                quest_reader is not None
                and teleop_active
                and robot_activity_state == RobotActivityState.ENABLED
                and left_tf is not None
                and right_tf is not None
                and left_hand_to_robot is not None
                and right_hand_to_robot is not None
                and left_hand_reference is not None
                and right_hand_reference is not None
            ):
                translation_scale, rotation_scale = data_manager.get_teleop_scaling()

                left_target = map_head_frame_hand_to_robot_target(
                    left_tf, left_hand_to_robot, left_hand_reference,
                    translation_scale, rotation_scale,
                )
                right_target = map_head_frame_hand_to_robot_target(
                    right_tf, right_hand_to_robot, right_hand_reference,
                    translation_scale, rotation_scale,
                )

                ik_solver.set_target_poses(
                    {
                        LEFT_END_EFFECTOR_FRAME_NAME: (left_target[:3, 3], left_target[:3, :3]),
                        RIGHT_END_EFFECTOR_FRAME_NAME: (right_target[:3, 3], right_target[:3, :3]),
                    }
                )

                success = ik_solver.solve_ik()
                if success:
                    joint_config = np.degrees(ik_solver.get_current_configuration())
                    data_manager.set_target_joint_angles(joint_config)
                    data_manager.set_target_pose("left", left_target)
                    data_manager.set_target_pose("right", right_target)
                    data_manager.set_ik_success(True)
                    data_manager.set_ik_solve_time_ms(
                        float(ik_solver.get_statistics()["last_solve_time_ms"])
                    )
                else:
                    data_manager.set_ik_success(False)
                    data_manager.set_ik_solve_time_ms(0.0)

            elif (
                quest_reader is None
                and teleop_active
                and robot_activity_state == RobotActivityState.ENABLED
            ):
                left_target = data_manager.get_gizmo_target_pose("left")
                right_target = data_manager.get_gizmo_target_pose("right")
                if left_target is not None and right_target is not None:
                    ik_solver.set_target_poses(
                        {
                            LEFT_END_EFFECTOR_FRAME_NAME: (left_target[:3, 3], left_target[:3, :3]),
                            RIGHT_END_EFFECTOR_FRAME_NAME: (right_target[:3, 3], right_target[:3, :3]),
                        }
                    )
                    success = ik_solver.solve_ik()
                    if success:
                        joint_config = np.degrees(ik_solver.get_current_configuration())
                        data_manager.set_target_joint_angles(joint_config)
                        data_manager.set_target_pose("left", left_target)
                        data_manager.set_target_pose("right", right_target)
                        data_manager.set_ik_success(True)
                        data_manager.set_ik_solve_time_ms(
                            float(ik_solver.get_statistics()["last_solve_time_ms"])
                        )
                    else:
                        data_manager.set_ik_success(False)
                        data_manager.set_ik_solve_time_ms(0.0)

            elif robot_activity_state in (RobotActivityState.HOMING, RobotActivityState.DISABLED):
                if current_joint_angles is not None:
                    ik_solver.set_configuration(np.radians(current_joint_angles))
                    _sync_targets_from_ik(data_manager, ik_solver, current_joint_angles)

            elif not teleop_active:
                joint_config = np.degrees(ik_solver.get_current_configuration())
                _sync_targets_from_ik(data_manager, ik_solver, joint_config)

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as exc:
        print(f"❌ Dual IK solver thread error: {exc}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🧮 Dual IK solver thread stopped")
