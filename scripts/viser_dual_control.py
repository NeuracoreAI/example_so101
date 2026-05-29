#!/usr/bin/env python3
"""Dual-arm SO101 simulation with Viser — joint sliders + real-time EEF gizmo IK.

Sim-only: no real robot required.

Usage:
    python scripts/viser_dual_control.py [--ip-address <Quest IP>]
Then open http://localhost:8080 in a browser.

Controls:
  - Drag EEF gizmos            → arms track in real time (IK at 250 Hz)
  - Drag gripper sliders        → open/close grippers in visualization
  Meta Quest (USB auto-discovered or via --ip-address):
  - Button A                    → enable / disable Quest teleoperation
  - Button B                    → move both arms to home position
  - Hold LEFT + RIGHT grip      → activate hand tracking (arms follow controllers)
  - Left/right trigger          → open/close left/right gripper
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from common.configs import (
    CONTROLLER_DATA_RATE,
    DAMPING_COST,
    DUAL_URDF_PATH,
    END_EFFECTOR_FRAME_NAMES,
    FRAME_TASK_GAIN,
    GRIP_THRESHOLD,
    LM_DAMPING,
    NEUTRAL_JOINT_ANGLES_DUAL,
    ORIENTATION_COST,
    POSITION_COST,
    POSTURE_COST_VECTOR_DUAL,
    ROTATION_SCALE,
    SOLVER_DAMPING_VALUE,
    SOLVER_NAME,
    TRANSLATION_SCALE,
)
from common.utils import (
    compute_hand_to_robot_calibration,
    map_head_frame_hand_to_robot_target,
    map_quest_hands_to_robot_arms,
)
from pink_ik_solver import PinkIKSolver

parser = argparse.ArgumentParser(description="Dual-arm SO101 Viser simulation")
parser.add_argument(
    "--ip-address", type=str, default=None,
    help="Meta Quest IP address. Omit to auto-discover via USB.",
)
args = parser.parse_args()

_IK_RATE = 250.0  # Hz

LEFT_EEF  = END_EFFECTOR_FRAME_NAMES[0]  # "left_eef_link"
RIGHT_EEF = END_EFFECTOR_FRAME_NAMES[1]  # "right_eef_link"

_BODY_JOINTS = [
    ("Shoulder Pan",  -150.0, 150.0),
    ("Shoulder Lift", -180.0, 180.0),
    ("Elbow Flex",    -150.0, 150.0),
    ("Wrist Flex",    -150.0, 150.0),
    ("Wrist Roll",    -180.0, 180.0),
]
_LEFT_NEUTRAL  = NEUTRAL_JOINT_ANGLES_DUAL[:5]
_RIGHT_NEUTRAL = NEUTRAL_JOINT_ANGLES_DUAL[5:10]

# ── Meta Quest discovery ──────────────────────────────────────────────────────

from meta_quest_teleop.reader import MetaQuestReader

quest_reader: MetaQuestReader | None = None
_quest_connected = False

try:
    connect_msg = (
        f"🎮 Connecting to Meta Quest at {args.ip_address}:5555 ..."
        if args.ip_address
        else "🎮 Auto-discovering Meta Quest over USB ..."
    )
    print(connect_msg)
    quest_reader = MetaQuestReader(ip_address=args.ip_address, port=5555, run=True)
    _quest_connected = True
    print("✓ Meta Quest connected")
except SystemExit:
    print("⚠️  Meta Quest not found — falling back to gizmo-only mode")
    quest_reader = None
except Exception as e:
    print(f"⚠️  Meta Quest connection failed ({e}) — falling back to gizmo-only mode")
    quest_reader = None


# ── IK solver ─────────────────────────────────────────────────────────────────
print("📁 Loading dual-arm Pink IK solver...")
ik = PinkIKSolver(
    urdf_path=DUAL_URDF_PATH,
    end_effector_frames=END_EFFECTOR_FRAME_NAMES,
    solver_name=SOLVER_NAME,
    position_cost=POSITION_COST,
    orientation_cost=ORIENTATION_COST,
    frame_task_gain=FRAME_TASK_GAIN,
    lm_damping=LM_DAMPING,
    damping_cost=DAMPING_COST,
    solver_damping_value=SOLVER_DAMPING_VALUE,
    integration_time_step=1.0 / _IK_RATE,
    initial_configuration=np.radians(NEUTRAL_JOINT_ANGLES_DUAL),
    posture_cost_vector=np.array(POSTURE_COST_VECTOR_DUAL),
)

# ── Shared state (IK thread ↔ Quest thread → main thread) ────────────────────
_state_lock  = threading.Lock()
_body_rad    = np.radians(NEUTRAL_JOINT_ANGLES_DUAL).copy()  # 10-DOF
_ik_solve_ms = 0.0
_shutdown    = threading.Event()

# Quest controller state written by _quest_thread, read by _ik_thread.
_quest_lock           = threading.Lock()
_quest_enabled        = False  # toggled by button A / GUI button
_quest_teleop_active  = False  # both grips held while enabled
_quest_left_gizmo_tf: np.ndarray | None  = None
_quest_right_gizmo_tf: np.ndarray | None = None
_quest_left_trigger   = 0.0
_quest_right_trigger  = 0.0

# Homing flag: set by button B / GUI button, cleared once IK resets.
_homing = threading.Event()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _pose_to_wxyz_pos(tf4: np.ndarray) -> tuple:
    pos  = tuple(float(v) for v in tf4[:3, 3])
    q    = Rotation.from_matrix(tf4[:3, :3]).as_quat()
    wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    return wxyz, pos


def _wxyz_pos_to_tf4(wxyz, pos) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, 3] = pos
    tf[:3, :3] = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
    return tf

# ── Viser server ──────────────────────────────────────────────────────────────
server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

urdf_model = yourdfpy.URDF.load(DUAL_URDF_PATH)
urdf_vis   = ViserUrdf(server, urdf_model, root_node_name="/robot")

# ── GUI ───────────────────────────────────────────────────────────────────────

left_sliders:  list[viser.GuiSliderHandle] = []
right_sliders: list[viser.GuiSliderHandle] = []

with server.gui.add_folder("Left Arm"):
    for i, (name, lo, hi) in enumerate(_BODY_JOINTS):
        s = server.gui.add_slider(
            f"L {name}", min=lo, max=hi, step=0.5,
            initial_value=float(_LEFT_NEUTRAL[i]), disabled=True,
        )
        left_sliders.append(s)
    left_gripper_slider = server.gui.add_slider(
        "L Gripper", min=0.0, max=100.0, step=0.5, initial_value=0.0
    )

with server.gui.add_folder("Right Arm"):
    for i, (name, lo, hi) in enumerate(_BODY_JOINTS):
        s = server.gui.add_slider(
            f"R {name}", min=lo, max=hi, step=0.5,
            initial_value=float(_RIGHT_NEUTRAL[i]), disabled=True,
        )
        right_sliders.append(s)
    right_gripper_slider = server.gui.add_slider(
        "R Gripper", min=0.0, max=100.0, step=0.5, initial_value=0.0
    )

server.gui.add_markdown("---")
ik_status_text  = server.gui.add_text("IK Status",     "Starting...")
eef_text        = server.gui.add_text("EEF Positions", "Loading...")
quest_status_text = server.gui.add_text(
    "Quest Status",
    "Quest: connected — press Enable or Button A" if _quest_connected else "Quest: not connected (gizmo mode)",
)

with server.gui.add_folder("Controls"):
    enable_button = server.gui.add_button(
        "Enable Quest Teleop" if _quest_connected else "Enable (no Quest)",
        disabled=not _quest_connected,
    )
    home_button = server.gui.add_button("Go Home 🏠")

with server.gui.add_folder("IK Parameters", expand_by_default=False):
    pos_cost_slider = server.gui.add_slider(
        "Position Cost", min=0.0, max=5.0, step=0.05, initial_value=POSITION_COST
    )
    ori_cost_slider = server.gui.add_slider(
        "Orientation Cost", min=0.0, max=2.0, step=0.05, initial_value=ORIENTATION_COST
    )
    gain_slider = server.gui.add_slider(
        "Frame Task Gain", min=0.0, max=2.0, step=0.05, initial_value=FRAME_TASK_GAIN
    )

# ── EEF gizmos ────────────────────────────────────────────────────────────────
init_poses = ik.get_current_end_effector_poses()
left_wxyz,  left_pos  = _pose_to_wxyz_pos(init_poses[LEFT_EEF])
right_wxyz, right_pos = _pose_to_wxyz_pos(init_poses[RIGHT_EEF])

left_gizmo  = server.scene.add_transform_controls("/eef_left",  scale=0.12, wxyz=left_wxyz,  position=left_pos)
right_gizmo = server.scene.add_transform_controls("/eef_right", scale=0.12, wxyz=right_wxyz, position=right_pos)

# Axis frames showing actual FK EEF position on the robot mesh
left_frame  = server.scene.add_frame("/eef_left_fk",  axes_length=0.08, axes_radius=0.003)
right_frame = server.scene.add_frame("/eef_right_fk", axes_length=0.08, axes_radius=0.003)

# ── Control callbacks (shared by GUI buttons and Quest buttons) ───────────────

def toggle_quest_enabled() -> None:
    global _quest_enabled
    with _quest_lock:
        _quest_enabled = not _quest_enabled
        enabled = _quest_enabled
    if enabled:
        print("✓ 🟢 Quest teleop enabled — hold both grips to track hands")
    else:
        print("✓ 🔴 Quest teleop disabled")


def go_home() -> None:
    print("🏠 Homing both arms...")
    _homing.set()


if quest_reader is not None:
    quest_reader.on("button_a_pressed", toggle_quest_enabled)
    quest_reader.on("button_b_pressed", go_home)

@enable_button.on_click
def _on_enable_click(_) -> None:
    toggle_quest_enabled()

@home_button.on_click
def _on_home_click(_) -> None:
    go_home()

# ── Meta Quest controller thread ──────────────────────────────────────────────

def _quest_thread(reader: MetaQuestReader) -> None:
    """Read Meta Quest hand data and update shared gizmo targets.

    Uses the calibration-based approach from the real teleop, but overrides
    the reference hand rotation with the current EEF rotation at grip-press.
    This makes R_ee @ R_ref^T = I so that translation deltas map 1:1 between
    the ROS hand frame and the URDF gizmo frame, regardless of how the
    controller is held at calibration time.
    """
    global _quest_teleop_active, _quest_left_gizmo_tf, _quest_right_gizmo_tf
    global _quest_left_trigger, _quest_right_trigger

    dt = 1.0 / CONTROLLER_DATA_RATE
    prev_both_grips = False

    left_hand_to_robot:  np.ndarray | None = None
    right_hand_to_robot: np.ndarray | None = None
    left_hand_reference: np.ndarray | None = None
    right_hand_reference: np.ndarray | None = None

    print(f"🎮 Quest controller thread started ({CONTROLLER_DATA_RATE} Hz)")

    while not _shutdown.is_set():
        t0 = time.time()

        left_raw   = reader.get_hand_controller_transform_ros(hand="left")
        right_raw  = reader.get_hand_controller_transform_ros(hand="right")
        left_grip  = reader.get_grip_value("left")
        right_grip = reader.get_grip_value("right")
        left_trig  = reader.get_trigger_value("left")
        right_trig = reader.get_trigger_value("right")

        with _quest_lock:
            enabled = _quest_enabled

        if left_raw is not None and right_raw is not None:
            left_tf, right_tf = map_quest_hands_to_robot_arms(
                left_raw, right_raw, mirror_control=False
            )
        else:
            left_tf = right_tf = None

        both_grips = (
            enabled
            and left_tf is not None
            and right_tf is not None
            and left_grip  >= GRIP_THRESHOLD
            and right_grip >= GRIP_THRESHOLD
        )

        # Rising edge: build calibration with R_ref forced to R_ee so that
        # the translation mapping is R_ee @ R_ee^T = I (axis-aligned).
        if both_grips and not prev_both_grips and left_tf is not None and right_tf is not None:
            current_poses = ik.get_current_end_effector_poses()
            left_ee  = current_poses.get(LEFT_EEF)
            right_ee = current_poses.get(RIGHT_EEF)
            if left_ee is not None and right_ee is not None:
                # Build hand references with the EEF rotation substituted in so
                # that R_ee @ R_ref^T = I and translation is 1:1.
                left_ref  = left_tf.copy();  left_ref[:3, :3]  = left_ee[:3, :3]
                right_ref = right_tf.copy(); right_ref[:3, :3] = right_ee[:3, :3]
                left_hand_reference  = left_ref
                right_hand_reference = right_ref
                left_hand_to_robot  = compute_hand_to_robot_calibration(
                    left_ee,  left_ref,  left_ref,  TRANSLATION_SCALE, ROTATION_SCALE
                )
                right_hand_to_robot = compute_hand_to_robot_calibration(
                    right_ee, right_ref, right_ref, TRANSLATION_SCALE, ROTATION_SCALE
                )
                print("✓ Quest hand tracking activated — gizmos now follow controllers")

        # Falling edge — clear calibration.
        if not both_grips and prev_both_grips:
            left_hand_to_robot  = None
            right_hand_to_robot = None
            left_hand_reference  = None
            right_hand_reference = None
            print("✗ Quest hand tracking deactivated")

        prev_both_grips = both_grips

        if (
            both_grips
            and left_tf is not None and right_tf is not None
            and left_hand_to_robot is not None and right_hand_to_robot is not None
            and left_hand_reference is not None and right_hand_reference is not None
        ):
            new_left  = map_head_frame_hand_to_robot_target(
                left_tf,  left_hand_to_robot,  left_hand_reference,  TRANSLATION_SCALE, ROTATION_SCALE
            )
            new_right = map_head_frame_hand_to_robot_target(
                right_tf, right_hand_to_robot, right_hand_reference, TRANSLATION_SCALE, ROTATION_SCALE
            )
            with _quest_lock:
                _quest_teleop_active  = True
                _quest_left_gizmo_tf  = new_left
                _quest_right_gizmo_tf = new_right
                _quest_left_trigger   = left_trig
                _quest_right_trigger  = right_trig
        else:
            with _quest_lock:
                _quest_teleop_active = False
                _quest_left_trigger  = left_trig
                _quest_right_trigger = right_trig

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    print("🎮 Quest controller thread stopped")


# ── IK solver thread ──────────────────────────────────────────────────────────

def _ik_thread() -> None:
    global _body_rad, _ik_solve_ms, _quest_teleop_active
    dt = 1.0 / _IK_RATE
    print(f"🧮 IK thread started ({_IK_RATE} Hz)")
    while not _shutdown.is_set():
        t0 = time.time()

        # Handle homing: reset IK config and snap gizmos to neutral EEF poses.
        if _homing.is_set():
            _homing.clear()
            ik.set_configuration_no_task_update(np.radians(NEUTRAL_JOINT_ANGLES_DUAL))
            neutral_poses = ik.get_current_end_effector_poses()
            if LEFT_EEF in neutral_poses and RIGHT_EEF in neutral_poses:
                lwxyz, lpos = _pose_to_wxyz_pos(neutral_poses[LEFT_EEF])
                rwxyz, rpos = _pose_to_wxyz_pos(neutral_poses[RIGHT_EEF])
                left_gizmo.wxyz      = lwxyz
                left_gizmo.position  = lpos
                right_gizmo.wxyz     = rwxyz
                right_gizmo.position = rpos
            with _quest_lock:
                _quest_teleop_active  = False   # type: ignore[assignment]  # cleared on home
            print("✓ Arms homed")

        # Update IK task weights from GUI sliders.
        ik.update_task_parameters(
            position_cost=pos_cost_slider.value,
            orientation_cost=ori_cost_slider.value,
            frame_task_gain=gain_slider.value,
        )

        # Determine targets: Quest overrides gizmos when hand tracking is active.
        with _quest_lock:
            quest_active   = _quest_teleop_active
            quest_left_tf  = _quest_left_gizmo_tf
            quest_right_tf = _quest_right_gizmo_tf

        if quest_active and quest_left_tf is not None and quest_right_tf is not None:
            # Push Quest targets back onto gizmos so they move visually in the browser.
            lwxyz, lpos = _pose_to_wxyz_pos(quest_left_tf)
            rwxyz, rpos = _pose_to_wxyz_pos(quest_right_tf)
            left_gizmo.wxyz      = lwxyz
            left_gizmo.position  = lpos
            right_gizmo.wxyz     = rwxyz
            right_gizmo.position = rpos
            left_tf  = quest_left_tf
            right_tf = quest_right_tf
        else:
            left_tf  = _wxyz_pos_to_tf4(left_gizmo.wxyz,  left_gizmo.position)
            right_tf = _wxyz_pos_to_tf4(right_gizmo.wxyz, right_gizmo.position)

        ik.set_target_poses({
            LEFT_EEF:  (left_tf[:3, 3],  left_tf[:3, :3]),
            RIGHT_EEF: (right_tf[:3, 3], right_tf[:3, :3]),
        })

        t_solve = time.time()
        success = ik.solve_ik()
        solve_ms = (time.time() - t_solve) * 1000.0

        if success:
            result = ik.get_current_configuration().copy()
            with _state_lock:
                _body_rad    = result
                _ik_solve_ms = solve_ms

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    print("🧮 IK thread stopped")


ik_thread = threading.Thread(target=_ik_thread, daemon=True)
ik_thread.start()

quest_thread = None
if quest_reader is not None:
    quest_thread = threading.Thread(target=_quest_thread, args=(quest_reader,), daemon=True)
    quest_thread.start()

# ── Main loop (visualization @ 60 Hz) ────────────────────────────────────────
print("🖥️  Viser running — open http://localhost:8080")
if _quest_connected:
    print("   Button A (or GUI) → enable/disable Quest teleop.")
    print("   Button B (or GUI) → home both arms.")
    print("   Hold LEFT + RIGHT grip → activate hand tracking.")
    print("   Squeeze triggers → open/close grippers.")
else:
    print("   Drag EEF gizmos — arms track in real time.")
    print("   'Go Home' button → reset to neutral.")
print("   Ctrl+C to exit.")

dt_vis = 1.0 / 60.0

try:
    while True:
        t0 = time.time()

        with _state_lock:
            body_rad = _body_rad.copy()
            solve_ms = _ik_solve_ms

        with _quest_lock:
            quest_active   = _quest_teleop_active
            enabled        = _quest_enabled
            left_trig_val  = _quest_left_trigger
            right_trig_val = _quest_right_trigger

        body_deg = np.degrees(body_rad)

        # Update read-only body sliders.
        for i in range(5):
            left_sliders[i].value  = float(body_deg[i])
            right_sliders[i].value = float(body_deg[i + 5])

        # Triggers drive gripper sliders when Quest is connected.
        if quest_reader is not None:
            left_gripper_slider.value  = float((1.0 - left_trig_val)  * 100.0)
            right_gripper_slider.value = float((1.0 - right_trig_val) * 100.0)

        # Compose 12-DOF for yourdfpy.
        cfg_12_deg = np.array([
            *body_deg[:5],
            left_gripper_slider.value,
            *body_deg[5:],
            right_gripper_slider.value,
        ])
        urdf_vis.update_cfg(np.radians(cfg_12_deg))

        # FK frames.
        poses = ik.get_current_end_effector_poses()

        lwxyz, lpos = _pose_to_wxyz_pos(poses[LEFT_EEF])
        left_frame.wxyz     = lwxyz
        left_frame.position = lpos

        rwxyz, rpos = _pose_to_wxyz_pos(poses[RIGHT_EEF])
        right_frame.wxyz     = rwxyz
        right_frame.position = rpos

        # Status text.
        ik_status_text.value = f"IK solve: {solve_ms:.2f} ms"
        eef_text.value = (
            f"Left EEF:  [{lpos[0]:.3f}, {lpos[1]:.3f}, {lpos[2]:.3f}]\n"
            f"Right EEF: [{rpos[0]:.3f}, {rpos[1]:.3f}, {rpos[2]:.3f}]"
        )

        if _quest_connected:
            if quest_active:
                quest_status_text.value = "Quest: HAND TRACKING ACTIVE 🟢"
            elif enabled:
                quest_status_text.value = "Quest: enabled — hold both grips to track"
            else:
                quest_status_text.value = "Quest: disabled (press A or Enable)"

        elapsed = time.time() - t0
        if dt_vis - elapsed > 0:
            time.sleep(dt_vis - elapsed)

except KeyboardInterrupt:
    print("\n👋 Shutting down...")
finally:
    _shutdown.set()
    ik_thread.join(timeout=1.0)
    if quest_thread is not None:
        quest_thread.join(timeout=1.0)
    if quest_reader is not None:
        quest_reader.stop()
    print("Done.")
