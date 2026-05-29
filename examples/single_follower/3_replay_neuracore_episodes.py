"""Replay a recorded Neuracore dataset on the SO101 follower robot."""

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

from common.configs import (  # noqa: E402
    GRIPPER_LOGGING_NAME,
    JOINT_NAMES,
    NEUTRAL_JOINT_ANGLES,
    ROBOT_RATE,
)
from common.so101_controller import SO101Controller  # noqa: E402


def main() -> None:
    """Main function for replaying a Neuracore dataset on the SO101 follower robot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--frequency", type=int, required=True)
    parser.add_argument(
        "--follower-port",
        type=str,
        default="/dev/ttyACM1",
        help="Serial port for SO101 follower (default: /dev/ttyACM1)",
    )
    parser.add_argument(
        "--follower-id",
        type=str,
        default="my_awesome_follower_arm",
        help="Calibration id for SO101 follower",
    )
    parser.add_argument("--episode-index", type=int, required=False, default=0)
    args = parser.parse_args()

    # Initialize SO101 follower controller
    print("\n🤖 Initializing SO101 follower controller...")
    robot_controller = SO101Controller(
        port=args.follower_port,
        follower_id=args.follower_id,
        robot_rate=ROBOT_RATE,
        neutral_joint_angles=NEUTRAL_JOINT_ANGLES,
        debug_mode=False,
    )
    # Start robot control loop
    print("\n🚀 Starting robot control loop...")
    robot_controller.start_control_loop()

    # Login to Neuracore
    print("\n🔑 Logging in to Neuracore...")
    nc.login()

    # Get dataset from Neuracore
    print("\n🔍 Getting dataset from Neuracore...")
    dataset = nc.get_dataset(args.dataset_name)

    # Synchronize dataset. Different `neuracore` versions expose different
    # Dataset APIs (older versions may not have `get_full_data_spec`). Use a
    # simple synchronize call and let the library infer available channels.
    print("\n🔁 Synchronizing dataset...")
    try:
        synced_dataset = dataset.synchronize(frequency=args.frequency)
    except TypeError:
        # Fallback for older APIs that may require additional args — try
        # calling without frequency then filter later.
        synced_dataset = dataset.synchronize()

    # Determine which episodes to play
    episode_indices: list[int] = []
    if args.episode_index == -1:
        episode_indices = list(range(len(synced_dataset)))
        print(f"\n📊 Found {len(synced_dataset)} episodes. Will play all episodes.")
    else:
        episode_indices = [args.episode_index]
        print(f"\n📊 Playing episode {args.episode_index} only.")

    # Play episodes
    try:
        for episode_idx in episode_indices:

            # Ensure controller is enabled so the control loop sends commands.
            if not robot_controller.resume_robot():
                print("✗ Failed to enable SO101 controller; robot will not move.")

            print(f"\n{'='*60}")
            print(f"🎬 Playing Episode {episode_idx} / {len(synced_dataset) - 1}")
            print(f"{'='*60}")

            episode = synced_dataset[episode_idx]

            print(f"\n🚀 Collecting episode {episode_idx} data...")
            rgb_frames_per_step: list[dict[str, np.ndarray]] = []
            parallel_gripper_open_amounts = []
            joint_positions = []
            for step in tqdm(episode, desc=f"Collecting episode {episode_idx}"):
                step = cast(SynchronizedPoint, step)

                # Extract joint positions
                joint_positions_dict = {}
                if DataType.JOINT_TARGET_POSITIONS in step.data:
                    joint_data = step.data[DataType.JOINT_TARGET_POSITIONS]
                    for joint_name in JOINT_NAMES:
                        if joint_name in joint_data:
                            joint_positions_dict[joint_name] = joint_data[
                                joint_name
                            ].value
                joint_positions.append([joint_positions_dict[jn] for jn in JOINT_NAMES])

                # Extract gripper
                gripper_value = 0.0
                if DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in step.data:
                    gripper_data = step.data[
                        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS
                    ]
                    if GRIPPER_LOGGING_NAME in gripper_data:
                        gripper_value = gripper_data[GRIPPER_LOGGING_NAME].open_amount
                parallel_gripper_open_amounts.append(gripper_value)

                # Extract RGB for all cameras
                step_frames: dict[str, np.ndarray] = {}
                if DataType.RGB_IMAGES in step.data:
                    rgb_data = step.data[DataType.RGB_IMAGES]
                    for camera_name, img_value in rgb_data.items():
                        step_frames[camera_name] = img_value.frame
                rgb_frames_per_step.append(step_frames)

            joint_positions = np.degrees(np.array(joint_positions))
            parallel_gripper_open_amounts = np.array(parallel_gripper_open_amounts)

            print(f"\n🚀 Replaying episode {episode_idx} data...")
            for index in tqdm(
                range(len(joint_positions)), desc=f"Replaying episode {episode_idx}"
            ):
                start_time = time.time()
                robot_controller.set_target_joint_angles(joint_positions[index])
                robot_controller.set_gripper_open_value(
                    parallel_gripper_open_amounts[index]
                )

                # Display camera frames (dataset stores RGB; OpenCV expects BGR)
                if index < len(rgb_frames_per_step):
                    for camera_name, frame_rgb in rgb_frames_per_step[index].items():
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
        print("\n🛑 Keyboard interrupt detected, stopping robot control loop...")
        cv2.destroyAllWindows()

    robot_controller.stop_control_loop()
    robot_controller.cleanup()


if __name__ == "__main__":
    main()