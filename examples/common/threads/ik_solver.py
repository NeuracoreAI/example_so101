"""IK solver thread - solves IK and updates state."""

import time
import traceback

import numpy as np

from common.configs import IK_SOLVER_RATE
from common.data_manager import DataManager, RobotActivityState
from common.pink_ik_solver import PinkIKSolver
from common.utils import scale_and_add_delta_transform

# Pinocchio (reduced 5-DOF) order matches DataManager "our" order exactly:
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
# No reordering is needed between Pinocchio and DataManager.


def ik_solver_thread(data_manager: DataManager, ik_solver: PinkIKSolver) -> None:
    """IK solver thread - solves IK and updates state."""
    print("🧮 IK solver thread started")

    dt: float = 1.0 / IK_SOLVER_RATE
    DIVERGENCE_TOLERANCE = 0.1

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start: float = time.time()

            # Get current robot joint angles from state (our order, 6 values including pseudo-gripper)
            current_joint_angles = data_manager.get_current_joint_angles()
            # Skip this iteration if we haven't received at least 5 body joint readings yet
            if current_joint_angles is not None and len(current_joint_angles) < 5:
                current_joint_angles = None

            current_ik_joint_angles = ik_solver.get_current_configuration()

            if current_ik_joint_angles is not None:
                # Pinocchio 5-DOF order == "our" order; convert to degrees for comparison
                current_ik_joint_angles_deg = np.degrees(current_ik_joint_angles)

                # Sync IK solver with actual joint angles if close enough
                if current_joint_angles is not None and np.all(
                    np.abs(current_joint_angles[:5] - current_ik_joint_angles_deg)
                    <= DIVERGENCE_TOLERANCE
                ):
                    angles_pinocchio = np.radians(current_joint_angles[:5])
                    ik_solver.set_configuration_no_task_update(angles_pinocchio)

            # Get current end effector pose from IK solver and set in state
            ik_ee_pose = ik_solver.get_current_end_effector_pose()
            data_manager.set_current_end_effector_pose(ik_ee_pose)

            robot_activity_state = data_manager.get_robot_activity_state()
            controller_transform, _, _ = data_manager.get_controller_data()
            teleop_active = data_manager.get_teleop_active()

            if robot_activity_state == RobotActivityState.POLICY_CONTROLLED:
                if current_joint_angles is not None:
                    angles_pinocchio = np.radians(current_joint_angles[:5])
                    ik_solver.set_configuration(angles_pinocchio)
                    current_end_effector_pose = data_manager.get_current_end_effector_pose()
                    data_manager.set_target_pose(current_end_effector_pose)
                    data_manager.set_ik_solve_time_ms(0.0)
                    data_manager.set_ik_success(True)

            elif teleop_active and controller_transform is not None:
                controller_initial, robot_initial = (
                    data_manager.get_initial_robot_controller_transforms()
                )
                if controller_initial is None or robot_initial is None:
                    elapsed = time.time() - iteration_start
                    sleep_time = dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                delta_position = controller_transform[:3, 3] - controller_initial[:3, 3]
                delta_orientation = (
                    controller_transform[:3, :3] @ controller_initial[:3, :3].T
                )

                translation_scale, rotation_scale = data_manager.get_scaling_params()
                T_robot_target = scale_and_add_delta_transform(
                    delta_position,
                    delta_orientation,
                    translation_scale,
                    rotation_scale,
                    robot_initial,
                )

                ik_solver.set_target_pose(T_robot_target[:3, 3], T_robot_target[:3, :3])
                data_manager.set_target_pose(T_robot_target)

                success = ik_solver.solve_ik()

                if success:
                    # Pinocchio 5-DOF output is already in "our" order — no reordering needed
                    joint_config_our = np.degrees(ik_solver.get_current_configuration())
                    stats = ik_solver.get_statistics()
                    solve_time_ms = stats["last_solve_time_ms"]

                    data_manager.set_target_joint_angles(joint_config_our)
                    data_manager.set_ik_solve_time_ms(solve_time_ms)
                    data_manager.set_ik_success(success)
                else:
                    data_manager.set_ik_solve_time_ms(0.0)
                    data_manager.set_ik_success(False)

            else:  # HOMING or DISABLED
                if current_joint_angles is not None:
                    angles_pinocchio = np.radians(current_joint_angles[:5])
                    ik_solver.set_configuration(angles_pinocchio)
                    current_end_effector_pose = data_manager.get_current_end_effector_pose()
                    data_manager.set_target_pose(current_end_effector_pose)
                    data_manager.set_target_joint_angles(current_joint_angles[:5])
                    data_manager.set_ik_solve_time_ms(0.0)
                    data_manager.set_ik_success(True)

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ IK solver thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🧮 IK solver thread stopped")
