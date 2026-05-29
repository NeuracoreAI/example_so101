#!/usr/bin/env python3
"""SO101 Robot Teleoperation with Meta Quest Controller - REAL ROBOT CONTROL.

Uses Pink IK control with Meta Quest controller input to control the REAL robot.
- REAL ROBOT CONTROL - sends commands to physical robot!
- Uses right hand controller grip as dead man's button
- Applies relative transformations to convert controller motion to robot motion
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

from common.configs import (
    CAMERA_FRAME_STREAMING_RATE,
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_DATA_RATE,
    CONTROLLER_MIN_CUTOFF,
    DAMPING_COST,
    FRAME_TASK_GAIN,
    GRIPPER_FRAME_NAME,
    IK_SOLVER_RATE,
    JOINT_STATE_STREAMING_RATE,
    LM_DAMPING,
    NEUTRAL_JOINT_ANGLES,
    ORIENTATION_COST,
    POSITION_COST,
    POSTURE_COST_VECTOR,
    ROBOT_RATE,
    ROTATION_SCALE,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    TRANSLATION_SCALE,
    URDF_JOINT_ORDER_FROM_OURS,
    URDF_PATH,
    VISUALIZATION_RATE,
)
from common.data_manager import DataManager, RobotActivityState
from common.robot_visualizer import RobotVisualizer
from common.threads.camera import camera_thread
from common.threads.ik_solver import ik_solver_thread
from common.threads.joint_state import joint_state_thread
from common.threads.quest_reader import quest_reader_thread
from meta_quest_teleop.reader import MetaQuestReader

from examples.common.pink_ik_solver import PinkIKSolver
from examples.common.so101_controller import SO101Controller


parser = argparse.ArgumentParser(
    description="SO101 Robot Teleoperation with Meta Quest - REAL ROBOT CONTROL"
)
parser.add_argument(
    "--port",
    type=str,
    default="/dev/ttyACM0",
    help="Serial port for the SO101 follower arm (e.g. /dev/ttyACM0)",
)
parser.add_argument(
    "--follower-id",
    type=str,
    default="my_awesome_follower_arm",
    help="Calibration ID for the SO101 follower arm",
)
parser.add_argument(
    "--ip-address",
    type=str,
    default=None,
    help="IP address of Meta Quest device (optional, defaults to None for auto-discovery)",
)
args = parser.parse_args()

print("=" * 60)
print("SO101 ROBOT TELEOPERATION - REAL ROBOT CONTROL")
print("=" * 60)
print("Thread frequencies:")
print(f"  🎮 Quest Reader:     {CONTROLLER_DATA_RATE} Hz")
print(f"  🧮 IK Solver:        {IK_SOLVER_RATE} Hz")
print(f"  🖥️ Visualization:    {VISUALIZATION_RATE} Hz (running in the main thread)")
print(f"  🤖 Robot Controller: {ROBOT_RATE} Hz")
print(f"  📊 Joint State:      {JOINT_STATE_STREAMING_RATE} Hz")
print(f"  📷 Camera:           {CAMERA_FRAME_STREAMING_RATE} Hz")


# Initialize shared state
data_manager = DataManager()
data_manager.set_controller_filter_params(
    CONTROLLER_MIN_CUTOFF,
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
)
data_manager.set_scaling_params(TRANSLATION_SCALE, ROTATION_SCALE)

# Initialize robot controller
print("\n🤖 Initializing SO101 robot controller...")
robot_controller = SO101Controller(
    port=args.port,
    follower_id=args.follower_id,
    robot_rate=ROBOT_RATE,
    neutral_joint_angles=NEUTRAL_JOINT_ANGLES,
    debug_mode=False,
)

# Start robot control loop
print("\n🚀 Starting robot control loop...")
robot_controller.start_control_loop()

# Start joint state thread
print("\n📊 Starting joint state thread...")
joint_state_thread_obj = threading.Thread(
    target=joint_state_thread, args=(data_manager, robot_controller), daemon=True
)
joint_state_thread_obj.start()

# Initialize Meta Quest reader
print("\n🎮 Initializing Meta Quest reader...")
quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

# Start quest reader thread
print("\n🎮 Starting quest reader thread...")
quest_thread = threading.Thread(
    target=quest_reader_thread, args=(data_manager, quest_reader), daemon=True
)
quest_thread.start()

# Build 5-DOF initial configuration for the reduced Pinocchio model.
# Pinocchio (reduced, gripper locked) order matches "our" DataManager order:
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] — no reordering needed.
current_joint_angles = data_manager.get_current_joint_angles()
if current_joint_angles is not None:
    body_joints = np.asarray(current_joint_angles[:5], dtype=float)
else:
    body_joints = np.asarray(NEUTRAL_JOINT_ANGLES, dtype=float)
initial_joint_angles = np.radians(body_joints)  # 5 values, direct

posture_cost_vec = np.array(POSTURE_COST_VECTOR, dtype=float)  # 5 values, direct

# Create Pink IK solver
print("\n🔧 Creating Pink IK solver...")
ik_solver = PinkIKSolver(
    urdf_path=URDF_PATH,
    end_effector_frame=GRIPPER_FRAME_NAME,
    solver_name=SOLVER_NAME,
    position_cost=POSITION_COST,
    orientation_cost=ORIENTATION_COST,
    frame_task_gain=FRAME_TASK_GAIN,
    lm_damping=LM_DAMPING,
    damping_cost=DAMPING_COST,
    solver_damping_value=SOLVER_DAMPING_VALUE,
    integration_time_step=1 / IK_SOLVER_RATE,
    initial_configuration=initial_joint_angles,
    posture_cost_vector=posture_cost_vec,
)

# Start IK solver thread
print("\n🧮 Starting IK solver thread...")
ik_thread = threading.Thread(
    target=ik_solver_thread, args=(data_manager, ik_solver), daemon=True
)
ik_thread.start()

# Start camera thread
print("\n📷 Starting camera thread...")
camera_thread_obj = threading.Thread(
    target=camera_thread, args=(data_manager,), daemon=True
)
camera_thread_obj.start()

# Set up visualizer
print("\n🖥️  Starting visualization...")
visualizer = RobotVisualizer(urdf_path=URDF_PATH)
visualizer.add_basic_controls()
visualizer.add_teleop_controls()
visualizer.add_gripper_status_controls()
visualizer.add_homing_controls()
visualizer.add_toggle_robot_enabled_status_button()
visualizer.add_controller_filter_controls(
    initial_min_cutoff=CONTROLLER_MIN_CUTOFF,
    initial_beta=CONTROLLER_BETA,
    initial_d_cutoff=CONTROLLER_D_CUTOFF,
)
visualizer.add_scaling_controls(
    initial_translation_scale=TRANSLATION_SCALE,
    initial_rotation_scale=ROTATION_SCALE,
)
visualizer.add_pink_parameter_controls(
    position_cost=POSITION_COST,
    orientation_cost=ORIENTATION_COST,
    frame_task_gain=FRAME_TASK_GAIN,
    lm_damping=LM_DAMPING,
    damping_cost=DAMPING_COST,
    solver_damping_value=SOLVER_DAMPING_VALUE,
    posture_cost_vector=POSTURE_COST_VECTOR,
)
visualizer.add_controller_visualization()
visualizer.add_target_frame_visualization()


def toggle_robot_enabled_status() -> None:
    """Toggle robot enabled/disabled state."""
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state == RobotActivityState.ENABLED:
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        robot_controller.graceful_stop()
        data_manager.set_teleop_state(False, None, None)
        data_manager.set_leader_teleop_engaged(False)
        visualizer.update_toggle_robot_enabled_status(False)
        print("✓ 🔴 Robot disabled")
    elif robot_activity_state in (RobotActivityState.DISABLED, RobotActivityState.HOMING):
        if robot_controller.resume_robot():
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            visualizer.update_toggle_robot_enabled_status(True)
            print("✓ 🟢 Robot enabled")
        else:
            print("✗ Failed to enable robot")


def on_go_home() -> None:
    """Move robot to home position."""
    robot_activity_state = data_manager.get_robot_activity_state()
    if robot_activity_state in (RobotActivityState.ENABLED, RobotActivityState.HOMING):
        print("🏠 Moving to home position...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_teleop_state(False, None, None)
        data_manager.set_leader_teleop_engaged(False)
        if not robot_controller.move_to_home():
            print("✗ Failed to initiate home move")
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    else:
        print("⚠️  Cannot home: robot not enabled")


# Register Quest controller button callbacks
quest_reader.on("button_a_pressed", toggle_robot_enabled_status)
quest_reader.on("button_b_pressed", on_go_home)

visualizer.set_toggle_robot_enabled_status_callback(toggle_robot_enabled_status)
visualizer.set_go_home_callback(on_go_home)

print()
print("🚀 Starting teleoperation with REAL ROBOT CONTROL...")
print("🎮 CONTROLS:")
print("   1. Press BUTTON A to enable/disable robot (or use GUI)")
print("   2. Hold RIGHT GRIP to activate teleoperation")
print("   3. Move controller - robot follows!")
print("   4. Hold RIGHT TRIGGER to close gripper")
print("   5. Press BUTTON B to send robot home (or use GUI)")
print("   6. Release grip to stop")
print("   7. Use 'Emergency Stop' in GUI if needed")
print("⚠️  Press Ctrl+C to exit")
print()

dt: float = 1.0 / VISUALIZATION_RATE

try:
    while True:
        iteration_start: float = time.time()

        # Push GUI filter params to shared state (read by quest_reader_thread)
        min_cutoff, beta, d_cutoff = visualizer.get_controller_filter_params()
        data_manager.set_controller_filter_params(min_cutoff, beta, d_cutoff)

        # Push GUI scaling params to shared state (read by ik_solver_thread)
        data_manager.set_scaling_params(
            visualizer.get_translation_scale(),
            visualizer.get_rotation_scale(),
        )

        # Push GUI Pink IK params directly to the solver.
        # The GUI returns posture_cost_vector as 5 values in "our" order, which matches
        # the reduced Pinocchio model's order — pass through directly.
        pink_params = visualizer.get_pink_parameters()
        pcv = pink_params.get("posture_cost_vector")
        if pcv is not None:
            pink_params["posture_cost_vector"] = np.asarray(pcv, dtype=float)
        ik_solver.update_task_parameters(**pink_params)

        # Read shared state for this frame
        controller_transform, grip_value, trigger_value = data_manager.get_controller_data()
        teleop_active = data_manager.get_teleop_active()
        robot_activity_state = data_manager.get_robot_activity_state()
        current_joint_angles = data_manager.get_current_joint_angles()
        target_joint_angles = data_manager.get_target_joint_angles()
        solve_time_ms = data_manager.get_ik_solve_time_ms()
        target_pose = data_manager.get_target_pose()

        # Gripper control: trigger acts independently of arm teleop (no grip required).
        # joint_state_thread only drives the gripper inside the full teleop block; this
        # covers trigger presses when the user is not holding grip.
        if robot_activity_state == RobotActivityState.ENABLED:
            gripper_target = 1.0 - trigger_value
            data_manager.set_target_gripper_open_value(gripper_target)
            robot_controller.set_gripper_open_value(gripper_target)

        # Update GUI displays
        visualizer.set_grip_value(grip_value)
        visualizer.set_trigger_value(trigger_value)
        visualizer.update_timing(solve_time_ms)

        # Update controller visualization
        visualizer.update_controller_visualization(controller_transform)
        if controller_transform is not None:
            visualizer.update_controller_status_display(
                controller_transform[:3, 3], connected=True
            )
        else:
            visualizer.update_controller_status_display(None, connected=False)

        visualizer.update_teleop_status(teleop_active)
        visualizer.update_target_visualization(target_pose)

        # Update main robot visualization.
        # DataManager stores degrees in "our" order; yourdfpy expects radians in URDF order.
        if current_joint_angles is not None:
            current_joint_rad = np.radians(current_joint_angles[URDF_JOINT_ORDER_FROM_OURS])
            visualizer.update_robot_pose(current_joint_rad)
            visualizer.update_joint_angles_display(current_joint_rad)

        # Ghost robot shows IK target when teleop is active.
        # IK returns 5 body joints; pad with pseudo-gripper to get 6 values for yourdfpy.
        if target_joint_angles is not None and robot_activity_state == RobotActivityState.ENABLED:
            pseudo_gripper = current_joint_angles[5] if current_joint_angles is not None else 0.0
            target_6dof = np.concatenate([np.asarray(target_joint_angles), [pseudo_gripper]])
            target_joint_rad = np.radians(target_6dof[URDF_JOINT_ORDER_FROM_OURS])
            visualizer.update_ghost_robot_visibility(True)
            visualizer.update_ghost_robot_pose(target_joint_rad)
        else:
            visualizer.update_ghost_robot_visibility(False)

        # Sync robot status text and enable/disable toggle button.
        # Also catches auto-transitions (e.g. HOMING → ENABLED set by joint_state_thread).
        if robot_activity_state == RobotActivityState.ENABLED:
            visualizer.update_robot_status("Robot Status: Enabled")
            visualizer.update_toggle_robot_enabled_status(True)
        elif robot_activity_state == RobotActivityState.HOMING:
            visualizer.update_robot_status("Robot Status: Homing")
            visualizer.update_toggle_robot_enabled_status(False)
        else:
            visualizer.update_robot_status("Robot Status: Disabled")
            visualizer.update_toggle_robot_enabled_status(False)

        # Update gripper status
        visualizer.update_gripper_status(
            trigger_value,
            robot_enabled=(robot_activity_state == RobotActivityState.ENABLED),
        )

        elapsed = time.time() - iteration_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n\n👋 Interrupt received - shutting down gracefully...")
except Exception as e:
    print(f"\n❌ Demo error: {e}")
    traceback.print_exc()

# Cleanup
print("\n🧹 Cleaning up...")

data_manager.request_shutdown()
data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
quest_thread.join()
quest_reader.stop()
ik_thread.join()
joint_state_thread_obj.join()
robot_controller.cleanup()
visualizer.stop()

print("\n👋 Demo stopped.")
