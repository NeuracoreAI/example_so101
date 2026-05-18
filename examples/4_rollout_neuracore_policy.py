#!/usr/bin/env python3
"""SO101 follower policy rollout with leader-arm teleop and Viser preview.

Loads a Neuracore policy, streams follower state (joints, gripper, RGB camera),
optionally teleoperates via the SO101 leader arm, previews the prediction horizon
on a ghost URDF, and can execute the horizon on the real follower.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import neuracore as nc
import numpy as np
from neuracore_types import DataType, EmbodimentDescription

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

from common.configs import (  # noqa: E402
    CONTROLLER_BETA,
    CONTROLLER_D_CUTOFF,
    CONTROLLER_DATA_RATE,
    CONTROLLER_MIN_CUTOFF,
    JOINT_STATE_STREAMING_RATE,
    LEADER_TO_SO101_JOINT,
    MAX_ACTION_ERROR_THRESHOLD,
    MAX_SAFETY_THRESHOLD,
    NEUTRAL_JOINT_ANGLES,
    POLICY_EXECUTION_RATE,
    PREDICTION_HORIZON_EXECUTION_RATIO,
    ROBOT_RATE,
    SO101_DIRECTIONS,
    SO101_FIXED_JOINTS,
    SO101_JOINT_LIMITS_DEG,
    SO101_OFFSETS_DEG,
    TARGETING_POSE_TIME_THRESHOLD,
    URDF_PATH,
    VISUALIZATION_RATE,
)
from common.data_manager import DataManager, RobotActivityState  # noqa: E402
from common.leader_arm import LerobotSO101LeaderArm  # noqa: E402
from common.policy_helpers import (  # noqa: E402
    DEFAULT_ROBOT_NAME,
    arm_joint_angles_deg,
    arm_joint_names_in_horizon,
    arm_joint_targets_deg_at_index,
    convert_predictions_to_horizon,
    get_policy_embodiments,
    gripper_open_at_index,
    joint_rad_ours_from_horizon,
    log_robot_state_for_policy,
    ours_joint_rad_to_urdf_cfg,
    print_policy_embodiments,
    target_with_pseudo_gripper_deg,
)
from common.policy_state import PolicyState  # noqa: E402
from common.robot_visualizer import RobotVisualizer  # noqa: E402
from common.threads.camera import camera_thread  # noqa: E402
from common.threads.joint_state import joint_state_thread  # noqa: E402
from common.threads.leader_arm_controller import leader_arm_controller_thread  # noqa: E402
from so101_controller import SO101Controller  # noqa: E402


def _teleop_loop(
    data_manager: DataManager,
    loop_rate_hz: float,
) -> None:
    """Map leader state to follower targets (same pattern as example 2)."""
    dt = 1.0 / loop_rate_hz
    print("🌀 Teleop loop started")
    try:
        while not data_manager.is_shutdown_requested():
            t0 = time.time()
            if data_manager.get_robot_activity_state() == RobotActivityState.POLICY_CONTROLLED:
                time.sleep(dt)
                continue

            if not data_manager.get_leader_teleop_engaged():
                data_manager.set_teleop_state(False, None, None)
                time.sleep(dt)
                continue

            mapped_angles, mapped_gripper = data_manager.get_leader_mapped_state()
            if mapped_angles is not None and mapped_gripper is not None:
                target_with_gripper = target_with_pseudo_gripper_deg(
                    mapped_angles, mapped_gripper
                )
                data_manager.set_target_joint_angles(target_with_gripper)
                data_manager.set_controller_data(
                    transform=None,
                    grip=1.0,
                    trigger=1.0 - float(mapped_gripper),
                )
                data_manager.set_teleop_state(True, None, None)

            elapsed = time.time() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as e:
        print(f"❌ Teleop loop error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🌀 Teleop loop stopped")


def toggle_robot_enabled_status(
    data_manager: DataManager,
    robot_controller: SO101Controller,
    visualizer: RobotVisualizer,
) -> None:
    """Toggle follower enable/disable."""
    state = data_manager.get_robot_activity_state()
    if state == RobotActivityState.ENABLED:
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        robot_controller.graceful_stop()
        data_manager.set_leader_teleop_engaged(False)
        visualizer.update_leader_teleop_button_status(False)
        visualizer.update_toggle_robot_enabled_status(False)
        print("✓ 🔴 Follower disabled")
    elif state == RobotActivityState.DISABLED:
        if robot_controller.resume_robot():
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
            visualizer.update_toggle_robot_enabled_status(True)
            print("✓ 🟢 Follower enabled")
        else:
            print("✗ Failed to enable follower")


def toggle_leader_teleop(
    data_manager: DataManager,
    visualizer: RobotVisualizer,
) -> None:
    """Engage/disengage leader-arm teleop (replaces Quest grip-to-teleop)."""
    if data_manager.get_leader_teleop_engaged():
        data_manager.set_leader_teleop_engaged(False)
        visualizer.update_leader_teleop_button_status(False)
        print("✓ Leader teleop disengaged")
        return

    state = data_manager.get_robot_activity_state()
    if state != RobotActivityState.ENABLED:
        print("⚠️  Enable the follower before engaging leader teleop")
        return

    data_manager.set_leader_teleop_engaged(True)
    visualizer.update_leader_teleop_button_status(True)
    print("✓ 🎮 Leader teleop engaged – move the leader arm")


def home_robot(
    data_manager: DataManager,
    robot_controller: SO101Controller,
    visualizer: RobotVisualizer,
) -> None:
    """Move follower to neutral pose."""
    if data_manager.get_robot_activity_state() == RobotActivityState.ENABLED:
        print("🏠 Moving to home...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        data_manager.set_leader_teleop_engaged(False)
        visualizer.update_leader_teleop_button_status(False)
        if not robot_controller.move_to_home():
            print("✗ Failed to initiate home move")
            data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    else:
        print("⚠️  Enable the follower before homing")


def run_policy(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    input_embodiment_description: EmbodimentDescription,
) -> bool:
    """Capture follower state, run policy, store horizon for preview/execution."""
    print("Running policy...")

    if not log_robot_state_for_policy(data_manager, input_embodiment_description):
        print("✗ No data available to run policy (check policy input channels)")
        return False
    print("  ✓ Logged policy inputs")

    try:
        start_time = time.time()
        predictions = policy.predict(timeout=60)
        prediction_horizon = convert_predictions_to_horizon(predictions)
        elapsed = time.time() - start_time
        horizon_length = policy_state.get_prediction_horizon_length()
        print(f"  ✓ Got {horizon_length} actions in {elapsed:.3f} s")

        policy_state.set_execution_ratio(visualizer.get_prediction_ratio())
        rgb_image = data_manager.get_rgb_image()
        if rgb_image is not None:
            policy_state.set_policy_rgb_image_input(rgb_image)
        current_joint_angles = data_manager.get_current_joint_angles()
        if current_joint_angles is not None:
            policy_state.set_policy_state_input(current_joint_angles)

        policy_state.set_prediction_horizon(prediction_horizon)
        visualizer.update_ghost_robot_visibility(True)
        policy_state.set_ghost_robot_playing(True)
        policy_state.reset_ghost_action_index()
        return True

    except Exception as e:
        print(f"✗ Policy prediction failed: {e}")
        traceback.print_exc()
        return False


def start_policy_execution(
    data_manager: DataManager, policy_state: PolicyState
) -> bool:
    """Validate safety and begin horizon execution on the follower."""
    if (
        data_manager.get_robot_activity_state() == RobotActivityState.POLICY_CONTROLLED
        and not policy_state.get_continuous_play_active()
    ):
        print("⚠️  Policy execution already in progress")
        return False
    if data_manager.get_robot_activity_state() == RobotActivityState.DISABLED:
        print("⚠️  Enable the follower before executing policy")
        return False

    prediction_horizon = policy_state.get_prediction_horizon()
    if policy_state.get_prediction_horizon_length() == 0:
        print("⚠️  Run policy first to obtain a horizon")
        return False

    if not arm_joint_names_in_horizon(prediction_horizon):
        print("⚠️  Horizon missing body joint targets")
        return False

    current_deg = arm_joint_angles_deg(data_manager.get_current_joint_angles())
    if current_deg is None:
        print("⚠️  No current joint angles")
        return False

    first_targets_deg = arm_joint_targets_deg_at_index(prediction_horizon, 0)
    if first_targets_deg is None:
        print("⚠️  Invalid first action in horizon")
        return False

    joint_differences = np.abs(current_deg - first_targets_deg)
    if np.any(joint_differences > MAX_SAFETY_THRESHOLD):
        print("⚠️  Follower too far from first policy action")
        print(f"   Differences (deg): {[f'{d:.2f}' for d in joint_differences]}")
        print(f"   Threshold: {MAX_SAFETY_THRESHOLD}°")
        return False

    policy_state.set_ghost_robot_playing(False)
    data_manager.set_leader_teleop_engaged(False)
    policy_state.start_policy_execution()

    if policy_state.get_locked_prediction_horizon_length() == 0:
        print("⚠️  Failed to lock prediction horizon")
        policy_state.end_policy_execution()
        return False

    print(
        f"✓ Executing {policy_state.get_locked_prediction_horizon_length()} actions"
    )
    data_manager.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED)
    return True


def run_and_start_policy_execution(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    input_embodiment_description: EmbodimentDescription,
) -> None:
    run_policy(
        data_manager, policy, policy_state, visualizer, input_embodiment_description
    )
    start_policy_execution(data_manager, policy_state)


def end_policy_play(
    data_manager: DataManager,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    policy_status_message: str,
) -> None:
    if policy_state.get_continuous_play_active():
        policy_state.set_continuous_play_active(False)
    visualizer.update_play_policy_button_status(False)
    policy_state.end_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
    data_manager.set_leader_teleop_engaged(False)
    visualizer.update_leader_teleop_button_status(False)
    visualizer.update_policy_status(policy_status_message)


def play_policy(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    input_embodiment_description: EmbodimentDescription,
) -> None:
    if not policy_state.get_continuous_play_active():
        print("▶️  Starting continuous policy play...")
        if not run_policy(
            data_manager, policy, policy_state, visualizer, input_embodiment_description
        ):
            end_policy_play(
                data_manager,
                policy_state,
                visualizer,
                "Continuous play stopped – prediction failed",
            )
            return
        if not start_policy_execution(data_manager, policy_state):
            end_policy_play(
                data_manager,
                policy_state,
                visualizer,
                "Continuous play stopped – execution failed",
            )
            return
        policy_state.set_continuous_play_active(True)
        visualizer.update_play_policy_button_status(True)
    else:
        print("⏹️  Stopping continuous policy play...")
        end_policy_play(
            data_manager, policy_state, visualizer, "Policy execution stopped"
        )


def policy_execution_thread(
    policy: nc.policy,
    data_manager: DataManager,
    policy_state: PolicyState,
    robot_controller: SO101Controller,
    visualizer: RobotVisualizer,
    input_embodiment_description: EmbodimentDescription,
) -> None:
    """Send locked horizon actions to the follower at the configured rate."""
    last_visualization_update = 0.0
    visualization_update_interval = 1.0 / 30.0

    while True:
        start_time = time.time()

        if (
            data_manager.get_robot_activity_state()
            == RobotActivityState.POLICY_CONTROLLED
        ):
            locked_horizon = policy_state.get_locked_prediction_horizon()
            execution_index = policy_state.get_execution_action_index()
            locked_horizon_length = policy_state.get_locked_prediction_horizon_length()

            if execution_index == 0 and locked_horizon_length > 0:
                print(
                    f"🔄 Policy execution: {locked_horizon_length} steps, "
                    f"enabled={robot_controller.is_robot_enabled()}"
                )

            if execution_index < locked_horizon_length:
                current_deg = arm_joint_angles_deg(
                    data_manager.get_current_joint_angles()
                )
                if (
                    execution_index > 0
                    and current_deg is not None
                    and policy_state.get_execution_mode()
                    == PolicyState.ExecutionMode.TARGETING_POSE
                ):
                    targeting_start = time.time()
                    while time.time() - targeting_start < TARGETING_POSE_TIME_THRESHOLD:
                        prev_deg = arm_joint_targets_deg_at_index(
                            locked_horizon, execution_index - 1
                        )
                        if prev_deg is None:
                            break
                        if np.any(
                            np.abs(current_deg - prev_deg) <= MAX_ACTION_ERROR_THRESHOLD
                        ):
                            break
                        time.sleep(0.001)
                        current_deg = arm_joint_angles_deg(
                            data_manager.get_current_joint_angles()
                        )
                        if current_deg is None:
                            break

                body_deg = arm_joint_targets_deg_at_index(locked_horizon, execution_index)
                gripper = gripper_open_at_index(locked_horizon, execution_index)

                if body_deg is not None:
                    if gripper is not None:
                        target_vec = target_with_pseudo_gripper_deg(body_deg, gripper)
                    else:
                        target_vec = target_with_pseudo_gripper_deg(body_deg, 0.5)
                    data_manager.set_target_joint_angles(target_vec)

                    if robot_controller.is_robot_enabled():
                        robot_controller.set_target_joint_angles(body_deg)
                    else:
                        print(
                            f"⚠️  Follower disabled at step {execution_index}, skipping"
                        )

                if gripper is not None and robot_controller.is_robot_enabled():
                    robot_controller.set_gripper_open_value(gripper)

                policy_state.increment_execution_action_index()
                visualizer.update_policy_status(
                    f"Executing policy: {execution_index + 1}/{locked_horizon_length}"
                )

            elif policy_state.get_continuous_play_active():
                try:
                    policy_state.end_policy_execution()
                    if not run_policy(
                        data_manager,
                        policy,
                        policy_state,
                        visualizer,
                        input_embodiment_description,
                    ):
                        end_policy_play(
                            data_manager,
                            policy_state,
                            visualizer,
                            "Continuous play stopped – prediction failed",
                        )
                        continue
                    if not start_policy_execution(data_manager, policy_state):
                        end_policy_play(
                            data_manager,
                            policy_state,
                            visualizer,
                            "Continuous play stopped – execution failed",
                        )
                except Exception as e:
                    print(f"✗ Continuous play error: {e}")
                    traceback.print_exc()
                    end_policy_play(
                        data_manager,
                        policy_state,
                        visualizer,
                        "Continuous play stopped – error",
                    )
            else:
                print("✓ Policy execution completed")
                end_policy_play(
                    data_manager,
                    policy_state,
                    visualizer,
                    "Policy execution completed",
                )

        current_time = time.time()
        if current_time - last_visualization_update >= visualization_update_interval:
            update_visualization(data_manager, policy_state, visualizer)
            last_visualization_update = current_time

        dt_execution = 1.0 / visualizer.get_policy_execution_rate()
        elapsed = time.time() - start_time
        if elapsed < dt_execution:
            time.sleep(dt_execution - elapsed)


def _update_policy_control_buttons(
    data_manager: DataManager,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
    robot_activity_state: RobotActivityState,
    prediction_horizon_length: int,
    ghost_robot_playing: bool,
) -> None:
    """Update policy GUI buttons (always — not gated on teleop state)."""
    robot_enabled = robot_activity_state == RobotActivityState.ENABLED
    has_horizon = prediction_horizon_length > 0
    policy_controlled = robot_activity_state == RobotActivityState.POLICY_CONTROLLED

    if policy_controlled:
        visualizer.set_start_policy_execution_button_disabled(True)
        visualizer.set_run_policy_button_disabled(True)
        visualizer.set_run_and_start_policy_execution_button_disabled(True)
        visualizer.set_play_policy_button_disabled(False)
    elif ghost_robot_playing and has_horizon:
        visualizer.set_start_policy_execution_button_disabled(False)
        visualizer.set_run_policy_button_disabled(not robot_enabled)
        visualizer.set_run_and_start_policy_execution_button_disabled(not robot_enabled)
        visualizer.set_play_policy_button_disabled(not robot_enabled)
    else:
        visualizer.set_start_policy_execution_button_disabled(
            not (robot_enabled and has_horizon)
        )
        visualizer.set_run_policy_button_disabled(not robot_enabled)
        visualizer.set_run_and_start_policy_execution_button_disabled(not robot_enabled)
        visualizer.set_play_policy_button_disabled(not robot_enabled)

    if policy_controlled:
        return
    if not has_horizon:
        visualizer.update_policy_status(
            "Ready – enable follower, then 'Run Policy'"
        )
    elif not robot_enabled:
        visualizer.update_policy_status("Follower not enabled")
    else:
        visualizer.update_policy_status(
            f"Ready – {prediction_horizon_length} actions in horizon"
        )


def update_visualization(
    data_manager: DataManager,
    policy_state: PolicyState,
    visualizer: RobotVisualizer,
) -> None:
    """Update Viser robot, ghost, camera, and policy button states."""
    visualizer.update_leader_teleop_button_status(
        data_manager.get_leader_teleop_engaged()
    )

    current_joint_angles = data_manager.get_current_joint_angles()
    if current_joint_angles is not None:
        joint_rad = np.radians(current_joint_angles)
        visualizer.update_robot_pose(ours_joint_rad_to_urdf_cfg(joint_rad))

    rgb_image = data_manager.get_rgb_image()
    if rgb_image is not None:
        visualizer.update_rgb_image(rgb_image)

    prediction_horizon = policy_state.get_prediction_horizon()
    prediction_horizon_length = policy_state.get_prediction_horizon_length()
    ghost_robot_playing = policy_state.get_ghost_robot_playing()
    ghost_action_index = policy_state.get_ghost_action_index()
    robot_activity_state = data_manager.get_robot_activity_state()

    if robot_activity_state == RobotActivityState.POLICY_CONTROLLED:
        visualizer.update_ghost_robot_visibility(True)
        target_joint_angles = data_manager.get_target_joint_angles()
        if target_joint_angles is not None:
            visualizer.update_ghost_robot_pose(
                ours_joint_rad_to_urdf_cfg(np.radians(target_joint_angles))
            )

    elif (
        robot_activity_state == RobotActivityState.ENABLED
        and data_manager.get_leader_teleop_engaged()
        and data_manager.get_teleop_active()
    ):
        visualizer.update_ghost_robot_visibility(True)
        target_joint_angles = data_manager.get_target_joint_angles()
        if target_joint_angles is not None:
            visualizer.update_ghost_robot_pose(
                ours_joint_rad_to_urdf_cfg(np.radians(target_joint_angles))
            )

    elif ghost_robot_playing and prediction_horizon_length > 0:
        visualizer.update_ghost_robot_visibility(True)
        if ghost_action_index < prediction_horizon_length:
            ghost_cfg = joint_rad_ours_from_horizon(
                prediction_horizon, ghost_action_index
            )
            if ghost_cfg is not None:
                visualizer.update_ghost_robot_pose(
                    ours_joint_rad_to_urdf_cfg(ghost_cfg)
                )
            policy_state.set_ghost_action_index(
                (ghost_action_index + 1) % prediction_horizon_length
            )
        else:
            policy_state.reset_ghost_action_index()

    else:
        visualizer.update_ghost_robot_visibility(False)

    _update_policy_control_buttons(
        data_manager,
        policy_state,
        visualizer,
        robot_activity_state,
        prediction_horizon_length,
        ghost_robot_playing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SO101 policy rollout with leader teleop and Viser"
    )
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--train-run-name", type=str, default=None)
    policy_group.add_argument("--model-path", type=str, default=None)
    policy_group.add_argument("--remote-endpoint-name", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default=DEFAULT_ROBOT_NAME)
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader-id", type=str, default="my_awesome_leader_arm")
    parser.add_argument("--leader-rate", type=float, default=50.0)
    parser.add_argument("--follower-port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--follower-id", type=str, default="my_awesome_follower_arm")
    args = parser.parse_args()

    print("=" * 60)
    print("SO101 POLICY ROLLOUT (LEADER TELEOP + VISER)")
    print("=" * 60)
    print("Thread frequencies:")
    print(f"  🦾 Leader reader:    {args.leader_rate:.1f} Hz")
    print(f"  🔁 Teleop loop:      {CONTROLLER_DATA_RATE:.1f} Hz")
    print(f"  🤖 Follower control: {ROBOT_RATE:.1f} Hz")
    print(f"  📊 Joint state:      {JOINT_STATE_STREAMING_RATE:.1f} Hz")
    print(f"  🖥️  Visualization:   {VISUALIZATION_RATE:.1f} Hz")

    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name=args.robot_name,
        urdf_path=str(URDF_PATH),
        overwrite=False,
    )

    if args.remote_endpoint_name is not None:
        print(f"\n🤖 Remote endpoint: {args.remote_endpoint_name}...")
        try:
            policy = nc.policy_remote_server(args.remote_endpoint_name)
        except nc.EndpointError:
            print(
                f"❌ Endpoint '{args.remote_endpoint_name}' not available. "
                "Start it from the Neuracore dashboard."
            )
            sys.exit(1)
    elif args.train_run_name is not None:
        print(f"\n🤖 Training run: {args.train_run_name}...")
        policy = nc.policy(
            train_run_name=args.train_run_name,
            device="cuda",
            robot_name=args.robot_name,
        )
    else:
        print(f"\n🤖 Model: {args.model_path}...")
        policy = nc.policy(
            model_file=args.model_path,
            device="cuda",
            robot_name=args.robot_name,
        )
    print("  ✓ Policy loaded")
    input_embodiment_description, output_embodiment_description = get_policy_embodiments(
        policy
    )
    print_policy_embodiments(
        input_embodiment_description, output_embodiment_description
    )

    policy_state = PolicyState()
    policy_state.set_execution_mode(PolicyState.ExecutionMode.TARGETING_TIME)

    data_manager = DataManager()
    data_manager.set_controller_filter_params(
        CONTROLLER_MIN_CUTOFF, CONTROLLER_BETA, CONTROLLER_D_CUTOFF
    )

    print("\n🦾 Connecting leader arm...")
    leader = LerobotSO101LeaderArm(
        port=args.leader_port, calibration_id=args.leader_id
    )
    leader.configure_follower(
        follower_limits_deg=SO101_JOINT_LIMITS_DEG,
        follower_offsets_deg=SO101_OFFSETS_DEG,
        follower_directions=SO101_DIRECTIONS,
        leader_to_follower_joint=LEADER_TO_SO101_JOINT,
        fixed_joints=SO101_FIXED_JOINTS,
    )
    try:
        leader.connect(calibrate=False)
    except Exception as e:
        print(f"✗ Leader connect failed: {e}")
        sys.exit(1)
    print("✓ Leader connected")

    print("\n🤖 Initializing follower...")
    robot_controller = SO101Controller(
        port=args.follower_port,
        follower_id=args.follower_id,
        robot_rate=ROBOT_RATE,
        neutral_joint_angles=np.asarray(NEUTRAL_JOINT_ANGLES, dtype=np.float64),
        debug_mode=False,
    )
    robot_controller.start_control_loop()

    joint_state_thread_obj = threading.Thread(
        target=joint_state_thread,
        args=(data_manager, robot_controller),
        daemon=True,
    )
    joint_state_thread_obj.start()

    leader_thread = threading.Thread(
        target=leader_arm_controller_thread,
        args=(data_manager, leader, args.leader_rate),
        daemon=True,
    )
    leader_thread.start()

    teleop_thread = threading.Thread(
        target=_teleop_loop,
        args=(data_manager, CONTROLLER_DATA_RATE),
        daemon=True,
    )
    teleop_thread.start()

    camera_thread_obj = threading.Thread(
        target=camera_thread,
        args=(data_manager,),
        daemon=True,
    )
    camera_thread_obj.start()

    print("\n🖥️  Starting Viser...")
    visualizer = RobotVisualizer(str(URDF_PATH))
    visualizer.add_rgb_image_placeholder()
    visualizer.add_policy_controls(
        initial_prediction_ratio=PREDICTION_HORIZON_EXECUTION_RATIO,
        initial_policy_rate=POLICY_EXECUTION_RATE,
        initial_robot_rate=ROBOT_RATE,
        initial_execution_mode=PolicyState.ExecutionMode.TARGETING_TIME.value,
    )
    visualizer.add_toggle_robot_enabled_status_button()
    visualizer.add_leader_teleop_button()
    visualizer.add_homing_controls()
    visualizer.add_policy_buttons()

    visualizer.set_toggle_robot_enabled_status_callback(
        lambda: toggle_robot_enabled_status(data_manager, robot_controller, visualizer)
    )
    visualizer.set_leader_teleop_callback(
        lambda: toggle_leader_teleop(data_manager, visualizer)
    )
    visualizer.set_go_home_callback(
        lambda: home_robot(data_manager, robot_controller, visualizer)
    )
    visualizer.set_run_policy_callback(
        lambda: run_policy(
            data_manager,
            policy,
            policy_state,
            visualizer,
            input_embodiment_description,
        )
    )
    visualizer.set_start_policy_execution_callback(
        lambda: start_policy_execution(data_manager, policy_state)
    )
    visualizer.set_run_and_start_policy_execution_callback(
        lambda: run_and_start_policy_execution(
            data_manager,
            policy,
            policy_state,
            visualizer,
            input_embodiment_description,
        )
    )
    visualizer.set_play_policy_callback(
        lambda: play_policy(
            data_manager,
            policy,
            policy_state,
            visualizer,
            input_embodiment_description,
        )
    )
    visualizer.set_execution_mode_callback(
        lambda: policy_state.set_execution_mode(
            PolicyState.ExecutionMode(visualizer.get_execution_mode())
        )
    )

    policy_execution_thread_obj = threading.Thread(
        target=policy_execution_thread,
        args=(
            policy,
            data_manager,
            policy_state,
            robot_controller,
            visualizer,
            input_embodiment_description,
        ),
        daemon=True,
    )
    policy_execution_thread_obj.start()

    print()
    print("🚀 Policy rollout with leader-arm teleop")
    print("CONTROLS (Viser GUI at http://localhost:8080):")
    print("  1. Enable Robot")
    print("  2. Engage Leader Teleop — then move the leader arm (like Quest grip)")
    print("  3. Run / Execute / Play policy when ready")
    print("  • Home disengages teleop")
    print("⚠️  Ctrl+C in this terminal to exit")
    print()

    try:
        while not data_manager.is_shutdown_requested():
            update_visualization(data_manager, policy_state, visualizer)
            time.sleep(1.0 / VISUALIZATION_RATE)

    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up...")
        policy.disconnect()
        data_manager.request_shutdown()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        leader_thread.join(timeout=2.0)
        teleop_thread.join(timeout=2.0)
        camera_thread_obj.join(timeout=2.0)
        joint_state_thread_obj.join(timeout=2.0)
        try:
            leader.disconnect()
        except Exception:
            pass
        robot_controller.cleanup()
        nc.logout()
        print("\n👋 Done.")


if __name__ == "__main__":
    main()
