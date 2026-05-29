#!/usr/bin/env python3
"""Dual-arm SO101 teleoperation with Meta Quest.

Left Meta Quest hand → left SO101 arm.
Right Meta Quest hand → right SO101 arm.
Single 10-DOF IK solver on the dual-arm URDF.

Controls:
  Hold LEFT + RIGHT grip  - activate dual-arm teleoperation
  Button A                - enable / disable both arms
  Button B                - move both arms to home
  Ctrl+C                  - exit
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
    DUAL_URDF_JOINT_ORDER_FROM_OURS,
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
    # VISUALIZATION_RATE,
)
from common.data_manager_dual import DualDataManager, RobotActivityState
# from common.robot_visualizer import RobotVisualizer
from common.threads.dual_camera import dual_camera_thread
from common.threads.dual_ik_solver import dual_ik_solver_thread
from common.threads.dual_joint_state import dual_joint_state_thread
from meta_quest_teleop.reader import MetaQuestReader

from common.pink_ik_solver import PinkIKSolver
from common.so101_dual_controller import SO101DualController

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Dual-arm SO101 teleoperation with Meta Quest"
)
parser.add_argument("--left-port", type=str, default="/dev/ttyACM0")
parser.add_argument("--left-id", type=str, default="L1")
parser.add_argument("--right-port", type=str, default="/dev/ttyACM1")
parser.add_argument("--right-id", type=str, default="L1")
parser.add_argument("--ip-address", type=str, default=None)
args = parser.parse_args()

print("=" * 60)
print("DUAL-ARM SO101 TELEOPERATION")
print("=" * 60)
print(f"  Left arm:  port={args.left_port}  id={args.left_id}")
print(f"  Right arm: port={args.right_port}  id={args.right_id}")
print(f"  🧮 IK Solver:   {IK_SOLVER_RATE} Hz  (10-DOF, dual-arm URDF)")
print(f"  🤖 Joint State: {JOINT_STATE_STREAMING_RATE} Hz  (per arm)")
print(f"  📷 Camera:      {CAMERA_FRAME_STREAMING_RATE} Hz  (×2)")

# ── Shared state ──────────────────────────────────────────────────────────────

data_manager = DualDataManager()
data_manager.set_controller_filter_params(
    CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF
)
data_manager.set_teleop_scaling(TRANSLATION_SCALE, ROTATION_SCALE)

# ── Robot hardware ────────────────────────────────────────────────────────────

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

# ── Joint state threads ───────────────────────────────────────────────────────

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

# ── IK solver ─────────────────────────────────────────────────────────────────

initial_joint_angles = np.radians(NEUTRAL_JOINT_ANGLES_DUAL)  # 10 body joints
posture_cost_vec = np.array(POSTURE_COST_VECTOR_DUAL, dtype=float)  # 10 values

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

# ── Quest reader ──────────────────────────────────────────────────────────────

print("\n🎮 Initializing Meta Quest reader...")
quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)

# IK thread reads Quest directly (openarm pattern — no separate quest thread).
print("\n🧮 Starting dual IK solver thread...")
ik_thread = threading.Thread(
    target=dual_ik_solver_thread,
    args=(data_manager, ik_solver, quest_reader),
    daemon=True,
)
ik_thread.start()

# ── Camera threads ────────────────────────────────────────────────────────────

cam1_thread = threading.Thread(
    target=dual_camera_thread,
    args=(data_manager, "rgb", CAMERA_DEVICE_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT),
    daemon=True,
)
cam2_thread = threading.Thread(
    target=dual_camera_thread,
    args=(data_manager, "rgb_2", CAMERA_2_DEVICE_INDEX, CAMERA_2_WIDTH, CAMERA_2_HEIGHT),
    daemon=True,
)
cam1_thread.start()
cam2_thread.start()

# ── Visualizer ────────────────────────────────────────────────────────────────
# DISABLED: Optional visualiser

# print("\n🖥️  Starting visualization...")
# visualizer = RobotVisualizer(urdf_path=DUAL_URDF_PATH)
# visualizer.add_basic_controls()
# visualizer.add_teleop_controls()
# visualizer.add_homing_controls()
# visualizer.add_toggle_robot_enabled_status_button()
# visualizer.add_controller_filter_controls(
#     initial_min_cutoff=CONTROLLER_MIN_CUTOFF,
#     initial_beta=CONTROLLER_BETA,
#     initial_d_cutoff=CONTROLLER_D_CUTOFF,
# )
# visualizer.add_scaling_controls(
#     initial_translation_scale=TRANSLATION_SCALE,
#     initial_rotation_scale=ROTATION_SCALE,
# )

# ── Button callbacks ──────────────────────────────────────────────────────────

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


quest_reader.on("button_a_pressed", toggle_robot_enabled_status)
quest_reader.on("button_b_pressed", on_go_home)
# visualizer.set_toggle_robot_enabled_status_callback(toggle_robot_enabled_status)
# visualizer.set_go_home_callback(on_go_home)

print()
print("🚀 Dual-arm teleoperation ready. (visualizer disabled)")
print("   1. Press BUTTON A to enable/disable both arms")
print("   2. Hold LEFT + RIGHT GRIP to activate teleoperation")
print("   3. Move controllers — arms follow!")
print("   4. Hold triggers to close grippers")
print("   5. Press BUTTON B to home both arms")
print("⚠️  Press Ctrl+C to exit")
print()

try:
    while True:
        time.sleep(1.0)

except KeyboardInterrupt:
    print("\n\n👋 Interrupt received — shutting down...")
except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()

# ── Cleanup ───────────────────────────────────────────────────────────────────

print("\n🧹 Cleaning up...")
data_manager.request_shutdown()
data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
quest_reader.stop()
ik_thread.join(timeout=3.0)
left_joint_thread.join(timeout=3.0)
right_joint_thread.join(timeout=3.0)
dual_ctrl.cleanup()
print("\n👋 Done.")
