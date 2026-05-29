#!/usr/bin/env python3
"""Visualize SO101 policy predictions from a Neuracore dataset.

Loads a policy and dataset, picks a random synchronized step, runs the policy,
and animates the predicted horizon on a Viser URDF model.
"""

import argparse
import random
import sys
import time
import traceback
from pathlib import Path

import neuracore as nc
import numpy as np
import viser
import yourdfpy
from neuracore.core.utils.robot_data_spec_utils import merge_cross_embodiment_description
from PIL import Image
from viser.extras import ViserUrdf

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

from common.configs import JOINT_NAMES, POLICY_EXECUTION_RATE, URDF_PATH  # noqa: E402
from common.policy_helpers import (  # noqa: E402
    DEFAULT_ROBOT_NAME,
    convert_predictions_to_horizon,
    get_policy_embodiments,
    gripper_open_at_index,
    horizon_length,
    joint_rad_ours_from_horizon,
    log_sync_step_for_policy,
    ours_joint_rad_to_urdf_cfg,
    print_policy_embodiments,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize SO101 policy predictions from a Neuracore dataset"
    )
    parser.add_argument("--dataset-name", type=str, required=True)
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--train-run-name", type=str, default=None)
    policy_group.add_argument("--model-path", type=str, default=None)
    policy_group.add_argument("--remote-endpoint-name", type=str, default=None)
    parser.add_argument(
        "--frequency",
        type=float,
        default=POLICY_EXECUTION_RATE,
        help="Visualization / dataset sync frequency (Hz)",
    )
    parser.add_argument("--robot-name", type=str, default=DEFAULT_ROBOT_NAME)
    args = parser.parse_args()

    print("🔧 Initializing Neuracore...")
    nc.login()
    nc.connect_robot(
        robot_name=args.robot_name, urdf_path=str(URDF_PATH), overwrite=False
    )

    if args.remote_endpoint_name:
        print(f"🤖 Remote endpoint: {args.remote_endpoint_name}...")
        try:
            policy = nc.policy_remote_server(args.remote_endpoint_name)
        except nc.EndpointError:
            print(
                f"❌ Endpoint '{args.remote_endpoint_name}' not available. "
                "Start it from the Neuracore dashboard."
            )
            sys.exit(1)
    elif args.train_run_name:
        print(f"🤖 Training run: {args.train_run_name}...")
        policy = nc.policy(
            train_run_name=args.train_run_name,
            device="cuda",
            robot_name=args.robot_name,
        )
    else:
        print(f"🤖 Model: {args.model_path}...")
        policy = nc.policy(
            model_file=args.model_path,
            device="cuda",
            robot_name=args.robot_name,
        )
    print("  ✓ Policy loaded")
    input_embodiment, output_embodiment = get_policy_embodiments(policy)
    print_policy_embodiments(input_embodiment, output_embodiment)

    print(f"🔍 Loading dataset: {args.dataset_name}...")
    dataset = nc.get_dataset(args.dataset_name)
    print(f"  ✓ {len(dataset)} episodes")

    print("🔁 Building cross_embodiment_union for synchronization...")
    input_cross_embodiment_description = {
        robot_id: input_embodiment for robot_id in dataset.robot_ids
    }
    output_cross_embodiment_description = (
        {robot_id: output_embodiment for robot_id in dataset.robot_ids}
        if output_embodiment is not None
        else {}
    )
    cross_embodiment_union = merge_cross_embodiment_description(
        input_cross_embodiment_description,
        output_cross_embodiment_description,
    )

    print("🔁 Synchronizing dataset...")
    synced_dataset = dataset.synchronize(
        frequency=args.frequency,
        cross_embodiment_union=cross_embodiment_union,
        prefetch_videos=True,
        max_prefetch_workers=2,
    )
    print(f"  ✓ {len(synced_dataset)} synchronized episodes")

    print("🖥️  Starting Viser...")
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    urdf = yourdfpy.URDF.load(str(URDF_PATH))
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(np.zeros(len(JOINT_NAMES)))

    current_horizon: dict[str, list[float]] | None = None
    current_action_idx = 0
    playing = False
    rgb_gui_handle = None

    def select_random_state() -> None:
        nonlocal current_horizon, current_action_idx, playing, rgb_gui_handle

        episode_idx = random.randint(0, len(synced_dataset) - 1)
        episode = synced_dataset[episode_idx]
        if len(episode) == 0:
            print(f"⚠️  Episode {episode_idx} is empty")
            return

        step_idx = random.randint(0, len(episode) - 1)
        step = episode[step_idx]
        print(f"📊 Episode {episode_idx}, step {step_idx}")

        if not log_sync_step_for_policy(step, input_embodiment):
            print("⚠️  Step has no data for policy input channels")
            return

        if hasattr(step, "data"):
            from neuracore_types import DataType

            rgb_data = step.data.get(DataType.RGB_IMAGES, {})
            for camera_name, frame in rgb_data.items():
                rgb_image = np.array(frame.frame)
                image_pil = Image.fromarray(rgb_image)
                image_pil.save("current_image.png")
                print("💾 Saved current_image.png")
                if rgb_gui_handle is None:
                    rgb_gui_handle = server.gui.add_image(
                        rgb_image,
                        label="RGB (dataset step)",
                        format="jpeg",
                        jpeg_quality=85,
                    )
                else:
                    rgb_gui_handle.image = rgb_image
                break

        print("🎯 Running policy...")
        start_time = time.time()
        try:
            predictions = policy.predict(timeout=60)
        except nc.EndpointError as e:
            print(f"✗ Prediction failed: {e}")
            traceback.print_exc()
            return

        duration = time.time() - start_time
        current_horizon = convert_predictions_to_horizon(predictions)
        current_action_idx = 0
        playing = True
        print(f"FINISHED PREDICTION in {duration:.3f} s")

        joint_rad_ours = joint_rad_ours_from_horizon(current_horizon or {}, 0)
        if joint_rad_ours is not None:
            urdf_vis.update_cfg(ours_joint_rad_to_urdf_cfg(joint_rad_ours))

        print(f"✅ Horizon length: {horizon_length(current_horizon or {})}")

    random_button = server.gui.add_button("Random Selection")
    random_button.on_click(lambda _: select_random_state())

    gripper_handle = server.gui.add_slider(
        "Gripper Open Amount",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=0.0,
        disabled=True,
    )

    frequency_handle = server.gui.add_number(
        "Visualization Frequency (Hz)",
        initial_value=args.frequency,
        min=1.0,
        max=500.0,
        step=1.0,
    )

    select_random_state()

    try:
        while True:
            loop_start = time.time()

            h_len = horizon_length(current_horizon or {})
            if playing and current_horizon and h_len > 0:
                if current_action_idx < h_len:
                    joint_rad_ours = joint_rad_ours_from_horizon(
                        current_horizon, current_action_idx
                    )
                    if joint_rad_ours is not None:
                        urdf_vis.update_cfg(ours_joint_rad_to_urdf_cfg(joint_rad_ours))
                        nc.log_joint_positions(
                            {
                                jn: float(joint_rad_ours[i])
                                for i, jn in enumerate(JOINT_NAMES)
                                if jn in current_horizon
                                or jn == "gripper"
                            }
                        )

                    gripper_value = gripper_open_at_index(
                        current_horizon, current_action_idx
                    )
                    if gripper_value is not None:
                        gripper_handle.value = round(gripper_value, 2)

                    current_action_idx = (current_action_idx + 1) % h_len

            elapsed = time.time() - loop_start
            frequency = max(frequency_handle.value, 0.1)
            time.sleep(max(0.0, 1.0 / frequency - elapsed))

    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        policy.disconnect()
        nc.logout()


if __name__ == "__main__":
    main()
