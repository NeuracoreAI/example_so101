#!/usr/bin/env python3
"""SO101 URDF in Viser with sliders; control the real robot from slider values.

Loads the SO101 URDF in Viser (like datasets_previewer/urdf_visualiser.py) and
drives the real SO101 follower arm from the same joint angles. Use the sliders
to move both the 3D preview and the physical robot (when --real-robot and
robot is enabled).

Requirements:
- pip/conda: viser, yourdfpy, scipy; lerobot with [feetech]
- Real robot: SO101 follower on USB (e.g. /dev/ttyUSB0), motors set up.

Usage:
  cd example_lerobot_so101/examples
  python viser_so101_control.py
  python viser_so101_control.py --real-robot --follower-port /dev/ttyUSB0 --follower-id my_awesome_follower_arm

  Open http://localhost:8080, move sliders. Enable robot and use Home/Sync as needed. Ctrl+C to exit.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import viser
import yourdfpy
from common.configs import URDF_PATH
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

# Paths: repo root = parent of examples/
_root = Path(__file__).resolve().parent.parent
_examples = _root / "examples"
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_examples))

# SO101 controller order (degrees): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
CONTROLLER_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
# URDF gripper joint range (robot.urdf)
GRIPPER_RAD_CLOSED = -0.174533
GRIPPER_RAD_OPEN = 1.74533
SHOW_FRAME_PATH = "/robot/show_frame"


def _rad_to_gripper_01(rad: float) -> float:
    """Map gripper angle in rad to open amount in [0, 1]."""
    return float(
        np.clip(
            (rad - GRIPPER_RAD_CLOSED) / (GRIPPER_RAD_OPEN - GRIPPER_RAD_CLOSED),
            0.0,
            1.0,
        )
    )


def _gripper_01_to_rad(g01: float) -> float:
    """Map open amount [0, 1] to gripper angle in rad."""
    return GRIPPER_RAD_CLOSED + float(np.clip(g01, 0.0, 1.0)) * (
        GRIPPER_RAD_OPEN - GRIPPER_RAD_CLOSED
    )


def _transform_to_position_wxyz(
    T: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert 4x4 homogeneous matrix to position and quaternion wxyz for Viser."""
    position = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    R = T[:3, :3]
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    wxyz = (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )
    return position, wxyz


def main() -> None:
    """Run the Viser UI and optionally mirror slider commands to the real robot."""
    parser = argparse.ArgumentParser(
        description="SO101 URDF in Viser; control real robot from sliders."
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")

    parser.add_argument(
        "--frame",
        type=str,
        default=None,
        metavar="LINK_OR_JOINT",
        help="Link or joint to show as coordinate frame.",
    )
    parser.add_argument(
        "--real-robot",
        action="store_true",
        help="Connect to real SO101 follower and send slider values to it.",
    )
    parser.add_argument("--follower-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--follower-id", type=str, default="my_awesome_follower_arm")
    args = parser.parse_args()

    print("=" * 60)
    print("SO101 VISER CONTROL")
    print("=" * 60)
    print(f"URDF: {URDF_PATH}")
    print(f"Real robot: {args.real_robot}")
    if args.real_robot:
        print(f"  Port: {args.follower_port}, ID: {args.follower_id}")
    print()

    # Load URDF
    print("📦 Loading URDF...")
    urdf_path = Path(URDF_PATH)
    urdf = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(urdf_path.parent))
    joint_names = list(urdf.actuated_joint_names)
    print(f"✓ Actuated joints: {joint_names}")

    # Build mapping: URDF index -> controller body index (0..4) or None for gripper
    # Controller order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
    urdf_to_body_index: list[int | None] = []
    gripper_urdf_index: int | None = None
    for i, name in enumerate(joint_names):
        if name == "gripper":
            gripper_urdf_index = i
            urdf_to_body_index.append(None)
        elif name in CONTROLLER_JOINT_NAMES:
            urdf_to_body_index.append(CONTROLLER_JOINT_NAMES.index(name))
        else:
            urdf_to_body_index.append(None)
    if gripper_urdf_index is None:
        print("Error: no 'gripper' in actuated joints")
        sys.exit(1)

    # Joint limits from URDF (radians)
    joint_limits_rad: list[tuple[float, float]] = []
    for name in joint_names:
        joint = urdf.joint_map[name]
        if (
            joint.limit is not None
            and joint.limit.lower is not None
            and joint.limit.upper is not None
        ):
            low, high = joint.limit.lower, joint.limit.upper
        else:
            low, high = -np.pi, np.pi
        joint_limits_rad.append((low, high))

    initial_config = np.array(
        [(lo + hi) / 2 for lo, hi in joint_limits_rad],
        dtype=np.float64,
    )
    joint_config = initial_config.copy()

    # Robot controller (optional)
    robot_controller = None
    if args.real_robot:
        from common.configs import NEUTRAL_JOINT_ANGLES

        from so101_controller import SO101Controller

        print("🤖 Connecting to SO101 follower...")
        robot_controller = SO101Controller(
            port=args.follower_port,
            follower_id=args.follower_id,
            robot_rate=100.0,
            neutral_joint_angles=NEUTRAL_JOINT_ANGLES,
            debug_mode=False,
        )
        robot_controller.start_control_loop()
        print("✓ Follower controller ready (robot disabled until you click Enable)")

    # Viser server
    print("\n🌐 Starting Viser server...")
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=1, height=1, cell_size=0.05)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Joint sliders (degrees for body, 0–100% for gripper in UI)
    joint_handles: list = []
    with server.gui.add_folder("Joint Angles"):
        for i, jname in enumerate(joint_names):
            lo_rad, hi_rad = joint_limits_rad[i]
            if jname == "gripper":
                # Slider 0–100 for open %
                g01 = _rad_to_gripper_01(joint_config[i])
                handle = server.gui.add_slider(
                    jname,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    initial_value=g01 * 100.0,
                )
            else:
                lo_deg = float(np.degrees(lo_rad))
                hi_deg = float(np.degrees(hi_rad))
                angle_deg = np.degrees(joint_config[i])
                handle = server.gui.add_slider(
                    jname,
                    min=lo_deg,
                    max=hi_deg,
                    step=1.0,
                    initial_value=angle_deg,
                )
            joint_handles.append(handle)

    # Frame dropdown (optional)
    frame_options = sorted(set(urdf.link_map.keys()) | set(urdf.joint_map.keys()))
    initial_frame = args.frame if args.frame in frame_options else "(none)"
    selected_frame_name: list[str | None] = [
        None if initial_frame == "(none)" else initial_frame
    ]
    show_frame_handle: list = [None]

    with server.gui.add_folder("Frame"):
        frame_dropdown = server.gui.add_dropdown(
            "Show frame",
            options=["(none)"] + frame_options,
            initial_value=initial_frame,
        )

    @frame_dropdown.on_update
    def _on_frame_choice(_: Any) -> None:
        val = frame_dropdown.value
        selected_frame_name[0] = None if val == "(none)" else val
        _refresh_vis()

    # Real-robot controls
    robot_enabled = [False]

    def _refresh_vis() -> None:
        """Update joint_config from sliders, then URDF and optional frame."""
        for i, handle in enumerate(joint_handles):
            if joint_names[i] == "gripper":
                joint_config[i] = _gripper_01_to_rad(handle.value / 100.0)
            else:
                joint_config[i] = np.radians(handle.value)
        urdf_vis.update_cfg(joint_config)
        name = selected_frame_name[0]
        if name is None:
            if show_frame_handle[0] is not None:
                show_frame_handle[0].remove()
                show_frame_handle[0] = None
        else:
            urdf.update_cfg(joint_config)
            T = urdf.get_transform(name)
            position, wxyz = _transform_to_position_wxyz(T)
            show_frame_handle[0] = server.scene.add_frame(
                SHOW_FRAME_PATH,
                position=position,
                wxyz=wxyz,
                axes_length=0.05,
                axes_radius=0.002,
            )

    def _slider_to_controller() -> None:
        """Send current slider values to the real robot (when enabled)."""
        if robot_controller is None or not robot_enabled[0]:
            return
        body_deg = np.zeros(5, dtype=np.float64)
        for i, ctrl_idx in enumerate(urdf_to_body_index):
            if ctrl_idx is not None:
                body_deg[ctrl_idx] = np.degrees(joint_config[i])
        gripper_01 = _rad_to_gripper_01(joint_config[gripper_urdf_index])
        robot_controller.set_target_joint_angles(body_deg)
        robot_controller.set_gripper_open_value(gripper_01)

    def on_slider_update(_: Any) -> None:
        _refresh_vis()
        _slider_to_controller()

    for handle in joint_handles:
        handle.on_update(on_slider_update)

    if args.real_robot and robot_controller is not None:
        with server.gui.add_folder("Robot"):
            enable_btn = server.gui.add_button(
                "Enable robot" if not robot_enabled[0] else "Disable robot"
            )

            @enable_btn.on_click
            def _toggle_enable(_: Any) -> None:
                if robot_enabled[0]:
                    robot_controller.graceful_stop()
                    robot_enabled[0] = False
                    enable_btn.name = "Enable robot"
                    print("🔴 Robot disabled")
                else:
                    if robot_controller.resume_robot():
                        robot_enabled[0] = True
                        enable_btn.name = "Disable robot"
                        print("🟢 Robot enabled")
                    else:
                        print("✗ Failed to enable robot")

            home_btn = server.gui.add_button("Home")

            @home_btn.on_click
            def _go_home(_: Any) -> None:
                if not robot_enabled[0]:
                    print("⚠️ Enable robot first")
                    return
                robot_controller.move_to_home()
                # Update sliders to home (body only; gripper unchanged)
                cur = robot_controller.get_target_joint_angles()
                if cur is not None:
                    for i, ctrl_idx in enumerate(urdf_to_body_index):
                        if ctrl_idx is not None:
                            joint_handles[i].value = float(cur[ctrl_idx])
                    _refresh_vis()

            sync_btn = server.gui.add_button("Sync from robot")

            @sync_btn.on_click
            def _sync_from_robot(_: Any) -> None:
                cur = robot_controller.get_current_joint_angles()
                g = robot_controller.get_current_gripper_open_value()
                if cur is not None and len(cur) >= 5:
                    for i, ctrl_idx in enumerate(urdf_to_body_index):
                        if ctrl_idx is not None:
                            joint_handles[i].value = float(cur[ctrl_idx])
                if g is not None and gripper_urdf_index is not None:
                    joint_handles[gripper_urdf_index].value = g * 100.0
                _refresh_vis()
                _slider_to_controller()
                print("✓ Synced sliders from robot")

    # Reset to neutral
    with server.gui.add_folder("Controls"):
        reset_btn = server.gui.add_button("Reset to Neutral")

    @reset_btn.on_click
    def _reset(_: Any) -> None:
        for i, handle in enumerate(joint_handles):
            if joint_names[i] == "gripper":
                handle.value = _rad_to_gripper_01(initial_config[i]) * 100.0
            else:
                handle.value = np.degrees(initial_config[i])
        _refresh_vis()
        _slider_to_controller()

    _refresh_vis()

    print("✓ Viser server ready")
    print(f"\n🌐 Open http://localhost:{args.port}")
    print(
        "   Move sliders to drive the URDF"
        + (" and the real robot (when enabled)." if args.real_robot else ".")
    )
    print("   Ctrl+C to exit\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
    finally:
        print("🧹 Cleaning up...")
        if robot_controller is not None:
            robot_controller.cleanup()
        server.stop()
        print("✓ Done")


if __name__ == "__main__":
    main()
