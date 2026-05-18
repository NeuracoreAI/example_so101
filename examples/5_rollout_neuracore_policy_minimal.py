#!/usr/bin/env python3
"""Minimal SO101 policy rollout – terminal only, no GUI.

1. Enables the follower arm
2. Sends the arm home
3. Runs policy in a continuous loop (log state, predict, execute horizon)
4. On Ctrl+C: homes the arm and exits
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

import neuracore as nc
import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

from common.configs import (  # noqa: E402
    NEUTRAL_JOINT_ANGLES,
    POLICY_EXECUTION_RATE,
    PREDICTION_HORIZON_EXECUTION_RATIO,
    ROBOT_RATE,
    URDF_PATH,
)
from common.data_manager import DataManager, RobotActivityState  # noqa: E402
from common.policy_helpers import (  # noqa: E402
    DEFAULT_ROBOT_NAME,
    arm_joint_targets_deg_at_index,
    convert_predictions_to_horizon,
    get_policy_embodiments,
    gripper_open_at_index,
    log_robot_state_for_policy,
    print_policy_embodiments,
)
from common.policy_state import PolicyState  # noqa: E402
from common.threads.camera import camera_thread  # noqa: E402
from common.threads.joint_state import joint_state_thread  # noqa: E402
from so101_controller import SO101Controller  # noqa: E402


def run_policy(
    data_manager: DataManager,
    policy: nc.policy,
    policy_state: PolicyState,
    input_embodiment_description,
) -> bool:
    """Run policy and store the prediction horizon."""
    if not log_robot_state_for_policy(data_manager, input_embodiment_description):
        print("⚠️  No policy inputs logged")
        return False

    try:
        start_time = time.time()
        predictions = policy.predict(timeout=60)
        prediction_horizon = convert_predictions_to_horizon(predictions)
        elapsed = time.time() - start_time

        policy_state.set_prediction_horizon(prediction_horizon)
        horizon_length = policy_state.get_prediction_horizon_length()
        print(f"✓ Got {horizon_length} actions in {elapsed:.3f}s")
        return True

    except Exception as e:
        print(f"✗ Policy prediction failed: {e}")
        traceback.print_exc()
        return False


def execute_horizon(
    data_manager: DataManager,
    policy_state: PolicyState,
    robot_controller: SO101Controller,
    frequency: float,
) -> None:
    """Execute the locked prediction horizon on the follower arm."""
    policy_state.start_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.POLICY_CONTROLLED)

    locked_horizon = policy_state.get_locked_prediction_horizon()
    horizon_length = policy_state.get_locked_prediction_horizon_length()
    dt = 1.0 / frequency

    for i in range(horizon_length):
        start_time = time.time()

        body_deg = arm_joint_targets_deg_at_index(locked_horizon, i)
        if body_deg is not None and robot_controller.is_robot_enabled():
            robot_controller.set_target_joint_angles(body_deg)

        gripper = gripper_open_at_index(locked_horizon, i)
        if gripper is not None and robot_controller.is_robot_enabled():
            robot_controller.set_gripper_open_value(gripper)

        elapsed = time.time() - start_time
        time.sleep(max(0.0, dt - elapsed))

    policy_state.end_policy_execution()
    data_manager.set_robot_activity_state(RobotActivityState.ENABLED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SO101 policy rollout")
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--train-run-name", type=str, default=None)
    policy_group.add_argument("--model-path", type=str, default=None)
    policy_group.add_argument("--remote-endpoint-name", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default=DEFAULT_ROBOT_NAME)
    parser.add_argument("--follower-port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--follower-id", type=str, default="my_awesome_follower_arm")
    parser.add_argument(
        "--frequency",
        type=float,
        default=POLICY_EXECUTION_RATE,
        help="Policy execution rate (Hz)",
    )
    parser.add_argument(
        "--execution-ratio",
        type=float,
        default=PREDICTION_HORIZON_EXECUTION_RATIO,
        help="Fraction of the prediction horizon to execute",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SO101 POLICY ROLLOUT (MINIMAL)")
    print("=" * 60)

    print("\n🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name=args.robot_name,
        urdf_path=str(URDF_PATH),
        overwrite=False,
    )

    if args.remote_endpoint_name is not None:
        print(f"\n🤖 Remote policy endpoint: {args.remote_endpoint_name}...")
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
        print(f"\n🤖 Model file: {args.model_path}...")
        policy = nc.policy(
            model_file=args.model_path,
            device="cuda",
            robot_name=args.robot_name,
        )
    print("  ✓ Policy loaded")
    input_embodiment, output_embodiment = get_policy_embodiments(policy)
    print_policy_embodiments(input_embodiment, output_embodiment)

    data_manager = DataManager()
    policy_state = PolicyState()
    policy_state.set_execution_ratio(args.execution_ratio)

    print("\n🤖 Initializing SO101 follower...")
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

    camera_thread_obj = threading.Thread(
        target=camera_thread,
        args=(data_manager,),
        daemon=True,
    )
    camera_thread_obj.start()

    print("\n⏳ Waiting for initialization...")
    time.sleep(2.0)

    try:
        print("\n🟢 Enabling follower...")
        robot_controller.resume_robot()
        data_manager.set_robot_activity_state(RobotActivityState.ENABLED)
        print("✓ Follower enabled")

        print("\n🏠 Moving to home...")
        robot_controller.move_to_home()
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        start_time = time.time()
        while (
            data_manager.get_robot_activity_state() == RobotActivityState.HOMING
            and not robot_controller.is_robot_homed()
            and time.time() - start_time < 10.0
        ):
            time.sleep(0.1)
        print("✓ Follower at home")

        print("\n🚀 Policy loop running (Ctrl+C to stop)\n")
        while True:
            if not run_policy(data_manager, policy, policy_state, input_embodiment):
                print("⚠️  Policy run failed, retrying...")
                time.sleep(0.5)
                continue
            execute_horizon(
                data_manager, policy_state, robot_controller, args.frequency
            )

    except KeyboardInterrupt:
        print("\n\n👋 Interrupt received – shutting down...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()

    finally:
        print("\n🧹 Cleaning up...")
        print("\n🏠 Moving to home...")
        data_manager.set_robot_activity_state(RobotActivityState.HOMING)
        robot_controller.move_to_home()
        start_time = time.time()
        while (
            not robot_controller.is_robot_homed()
            and time.time() - start_time < 10.0
        ):
            time.sleep(0.1)

        policy.disconnect()
        data_manager.set_robot_activity_state(RobotActivityState.DISABLED)
        data_manager.request_shutdown()
        joint_state_thread_obj.join(timeout=2.0)
        camera_thread_obj.join(timeout=2.0)
        time.sleep(0.5)
        robot_controller.cleanup()
        nc.logout()
        print("\n👋 Done.")


if __name__ == "__main__":
    main()
