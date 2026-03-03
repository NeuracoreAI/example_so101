"""Joint state thread - reads joint state and sends commands to SO101 follower."""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
# Repo root (example_lerobot_so101) so so101_controller is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from common.configs import JOINT_STATE_STREAMING_RATE
from common.data_manager import DataManager, RobotActivityState

from so101_controller import SO101Controller


def joint_state_thread(
    data_manager: DataManager, robot_controller: SO101Controller
) -> None:
    """Joint state thread - reads joint state and sends commands."""
    print("🔧 Joint state thread started")

    dt: float = 1.0 / JOINT_STATE_STREAMING_RATE

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start: float = time.time()

            # Get current joint angles and gripper value
            current_joint_angles = robot_controller.get_current_joint_angles()
            gripper_open_value = robot_controller.get_current_gripper_open_value()

            if current_joint_angles is not None:
                if gripper_open_value is not None:
                    # NOTE: For Neuracore dataset visualization we also append a sixth
                    # "gripper" entry to the joint vector as a pseudo-angle so the
                    # visualizer can show the gripper alongside the arm. This couples
                    # joints and gripper and is not generally recommended, but works
                    # for now.
                    pseudo_gripper_deg = float(np.clip(gripper_open_value, 0.0, 1.0) * 100.0)
                    joint_with_gripper = np.concatenate(
                        [np.asarray(current_joint_angles, dtype=np.float64).flatten(), [pseudo_gripper_deg]]
                    )
                    data_manager.set_current_joint_angles(joint_with_gripper)
                else:
                    data_manager.set_current_joint_angles(current_joint_angles)

            # Get current gripper open value and set in state (also logged separately)
            if gripper_open_value is not None:
                data_manager.set_current_gripper_open_value(gripper_open_value)

            target_joint_angles = data_manager.get_target_joint_angles()
            _, _, trigger_value = data_manager.get_controller_data()

            # Check if robot is homing
            robot_activity_state = data_manager.get_robot_activity_state()
            if robot_activity_state == RobotActivityState.HOMING:
                if robot_controller.is_robot_homed():
                    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
                    print("✓ Robot reached home position and is re-enabled")

            elif robot_activity_state == RobotActivityState.ENABLED:
                if (
                    target_joint_angles is not None
                    and trigger_value is not None
                    and data_manager.get_teleop_active()
                ):
                    robot_controller.set_target_joint_angles(target_joint_angles)
                    target_gripper_open_value = 1.0 - trigger_value
                    data_manager.set_target_gripper_open_value(
                        target_gripper_open_value
                    )
                    robot_controller.set_gripper_open_value(target_gripper_open_value)

            # Sleep to maintain streaming rate
            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Joint state thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🔧 Joint state thread stopped")
