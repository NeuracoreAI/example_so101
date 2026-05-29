"""Replay a recorded dual-arm Neuracore dataset on the dual SO101 follower setup."""

import argparse
import sys
import time
from pathlib import Path
from typing import cast

import cv2
import neuracore as nc
import numpy as np
from neuracore_types import DataType, SynchronizedPoint
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

from common.configs import NEUTRAL_JOINT_ANGLES, ROBOT_RATE  # type: ignore  # noqa: E402
from common.so101_dual_controller import SO101DualController

_BODY_JOINT_SUFFIXES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_LEFT_JOINT_NAMES = [f"left_{s}" for s in _BODY_JOINT_SUFFIXES]
_RIGHT_JOINT_NAMES = [f"right_{s}" for s in _BODY_JOINT_SUFFIXES]
_DUAL_JOINT_NAMES = _LEFT_JOINT_NAMES + _RIGHT_JOINT_NAMES
_LEFT_GRIPPER_NAME = "left_gripper"
_RIGHT_GRIPPER_NAME = "right_gripper"


def main() -> None:
    """Replay a dual-arm Neuracore dataset on the dual SO101 follower setup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--frequency", type=int, required=True)
    parser.add_argument("--left-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--left-id", type=str, default="my_awesome_left_follower")
    parser.add_argument("--right-port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--right-id", type=str, default="my_awesome_right_follower")
    parser.add_argument("--episode-index", type=int, required=False, default=0)
    args = parser.parse_args()

    print("\n🤖 Initializing dual SO101 follower controller...")
    dual_ctrl = SO101DualController(
        left_port=args.left_port,
        left_follower_id=args.left_id,
        right_port=args.right_port,
        right_follower_id=args.right_id,
        robot_rate=ROBOT_RATE,
        neutral_joint_angles=np.asarray(NEUTRAL_JOINT_ANGLES, dtype=np.float64),
    )
    print("\n🚀 Starting robot control loop...")
    dual_ctrl.start_control_loop()

    print("\n🔑 Logging in to Neuracore...")
    nc.login()

    print("\n🔍 Getting dataset from Neuracore...")
    dataset = nc.get_dataset(args.dataset_name)

    print("\n🔁 Synchronizing dataset...")
    try:
        synced_dataset = dataset.synchronize(frequency=args.frequency)
    except TypeError:
        synced_dataset = dataset.synchronize()

    episode_indices: list[int] = []
    if args.episode_index == -1:
        episode_indices = list(range(len(synced_dataset)))
        print(f"\n📊 Found {len(synced_dataset)} episodes. Will play all episodes.")
    else:
        episode_indices = [args.episode_index]
        print(f"\n📊 Playing episode {args.episode_index} only.")

    try:
        for episode_idx in episode_indices:
            left_ok = dual_ctrl.left.resume_robot()
            right_ok = dual_ctrl.right.resume_robot()
            if not left_ok or not right_ok:
                print("✗ Failed to enable one or both arms; robot will not move.")

            print(f"\n{'='*60}")
            print(f"🎬 Playing Episode {episode_idx} / {len(synced_dataset) - 1}")
            print(f"{'='*60}")

            episode = synced_dataset[episode_idx]

            print(f"\n🚀 Collecting episode {episode_idx} data...")
            rgb_frames_per_step: list[dict[str, np.ndarray]] = []
            left_grippers: list[float] = []
            right_grippers: list[float] = []
            joint_positions: list[list[float]] = []

            for step in tqdm(episode, desc=f"Collecting episode {episode_idx}"):
                step = cast(SynchronizedPoint, step)

                # Extract 10 body joint target positions (radians)
                joint_dict: dict[str, float] = {}
                if DataType.JOINT_TARGET_POSITIONS in step.data:
                    joint_data = step.data[DataType.JOINT_TARGET_POSITIONS]
                    for jn in _DUAL_JOINT_NAMES:
                        if jn in joint_data:
                            joint_dict[jn] = joint_data[jn].value
                joint_positions.append([joint_dict.get(jn, 0.0) for jn in _DUAL_JOINT_NAMES])

                # Extract per-arm gripper open amounts
                left_g = 0.0
                right_g = 0.0
                if DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in step.data:
                    gripper_data = step.data[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]
                    if _LEFT_GRIPPER_NAME in gripper_data:
                        left_g = gripper_data[_LEFT_GRIPPER_NAME].open_amount
                    if _RIGHT_GRIPPER_NAME in gripper_data:
                        right_g = gripper_data[_RIGHT_GRIPPER_NAME].open_amount
                left_grippers.append(left_g)
                right_grippers.append(right_g)

                # Extract RGB frames for all cameras
                step_frames: dict[str, np.ndarray] = {}
                if DataType.RGB_IMAGES in step.data:
                    rgb_data = step.data[DataType.RGB_IMAGES]
                    for camera_name, img_value in rgb_data.items():
                        step_frames[camera_name] = img_value.frame
                rgb_frames_per_step.append(step_frames)

            # Neuracore stores joint positions in radians; SO101Controller expects degrees
            joint_positions_deg = np.degrees(np.array(joint_positions))  # (T, 10)
            left_grippers_arr = np.array(left_grippers)
            right_grippers_arr = np.array(right_grippers)

            print(f"\n🚀 Replaying episode {episode_idx} data...")
            for i in tqdm(range(len(joint_positions_deg)), desc=f"Replaying episode {episode_idx}"):
                start_time = time.time()

                dual_ctrl.left.set_target_joint_angles(joint_positions_deg[i, :5])
                dual_ctrl.right.set_target_joint_angles(joint_positions_deg[i, 5:])
                dual_ctrl.left.set_gripper_open_value(float(left_grippers_arr[i]))
                dual_ctrl.right.set_gripper_open_value(float(right_grippers_arr[i]))

                # Display camera frames (dataset stores RGB; OpenCV expects BGR)
                if i < len(rgb_frames_per_step):
                    for camera_name, frame_rgb in rgb_frames_per_step[i].items():
                        arr = np.asarray(frame_rgb, dtype=np.uint8)
                        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        cv2.imshow(f"Replay: {camera_name}", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n🛑 'q' pressed, stopping replay...")
                    break

                end_time = time.time()
                time.sleep(max(0, 1 / args.frequency - (end_time - start_time)))

            cv2.destroyAllWindows()
            print(f"🎉 Episode {episode_idx} replay completed.")

        if args.episode_index == -1:
            print(f"\n{'='*60}")
            print(f"🎉 All {len(synced_dataset)} episodes replay completed!")
            print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected, stopping...")
        cv2.destroyAllWindows()

    dual_ctrl.cleanup()


if __name__ == "__main__":
    main()
