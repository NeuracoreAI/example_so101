"""Joint state thread for one arm in a dual-arm SO101 setup.

Reads joint angles from one SO101Controller, writes them into the shared
10-DOF DualDataManager array (left: [0:5], right: [5:10]), and forwards
IK target angles to the physical arm when teleop is active.
"""

import time
import traceback

import numpy as np

from common.configs import JOINT_STATE_STREAMING_RATE
from common.data_manager_dual import DualDataManager, RobotActivityState
from common.so101_controller import SO101Controller

_BODY_DOF = 5  # SO101 has 5 actuated body joints per arm


def dual_joint_state_thread(
    data_manager: DualDataManager,
    robot_controller: SO101Controller,
    arm_side: str,
) -> None:
    """Read/write joint state for one arm in the dual-arm setup.

    Args:
        data_manager: Shared dual-arm DataManager.
        robot_controller: SO101Controller for this arm.
        arm_side: "left" or "right" — determines which slice of the
                  10-DOF array this thread owns.
    """
    if arm_side not in ("left", "right"):
        raise ValueError("arm_side must be 'left' or 'right'")

    print(f"🔧 Dual joint state thread started ({arm_side})")
    dt: float = 1.0 / JOINT_STATE_STREAMING_RATE

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            # ── Read current state from hardware ─────────────────────────────
            current_joint_angles = robot_controller.get_current_joint_angles()
            if current_joint_angles is not None:
                combined = data_manager.get_current_joint_angles()
                if combined is None or len(combined) != _BODY_DOF * 2:
                    combined = np.zeros(_BODY_DOF * 2, dtype=np.float64)
                else:
                    combined = combined.copy()
                if arm_side == "left":
                    combined[:_BODY_DOF] = current_joint_angles[:_BODY_DOF]
                else:
                    combined[_BODY_DOF:] = current_joint_angles[:_BODY_DOF]
                data_manager.set_current_joint_angles(combined)

            gripper_open_value = robot_controller.get_current_gripper_open_value()
            if gripper_open_value is not None:
                data_manager.set_current_gripper_open_value(arm_side, gripper_open_value)

            # ── Commands ─────────────────────────────────────────────────────
            target_joint_angles = data_manager.get_target_joint_angles()
            _, _, trigger_value = data_manager.get_controller_state(arm_side)

            robot_activity_state = data_manager.get_robot_activity_state()

            if robot_activity_state == RobotActivityState.HOMING:
                if robot_controller.is_robot_homed():
                    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
                    print(f"✓ {arm_side} arm reached home and re-enabled")

            elif robot_activity_state == RobotActivityState.ENABLED:
                if (
                    target_joint_angles is not None
                    and data_manager.get_teleop_active()
                    and len(target_joint_angles) >= _BODY_DOF * 2
                ):
                    if arm_side == "left":
                        arm_targets = target_joint_angles[:_BODY_DOF]
                    else:
                        arm_targets = target_joint_angles[_BODY_DOF:]
                    robot_controller.set_target_joint_angles(arm_targets)

                if data_manager.get_teleop_active():
                    # Trigger controls gripper: fully pressed = fully closed.
                    gripper_target = 1.0 - trigger_value
                    robot_controller.set_gripper_open_value(gripper_target)
                    data_manager.set_target_gripper_open_value(arm_side, gripper_target)

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Dual joint state thread error ({arm_side}): {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print(f"🔧 Dual joint state thread stopped ({arm_side})")
