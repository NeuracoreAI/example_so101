"""Quest reader thread - reads controller data and manages teleop state."""

import time
import traceback

from common.configs import CONTROLLER_DATA_RATE, GRIP_THRESHOLD
from common.data_manager import DataManager, RobotActivityState
from meta_quest_teleop.reader import MetaQuestReader


def quest_reader_thread(
    data_manager: DataManager,
    quest_reader: MetaQuestReader,
    hand: str = "right",
) -> None:
    """Quest reader thread - reads controller data and manages teleop state.

    Handles:
    - Reading Meta Quest controller data
    - Processing grip button (dead man's switch)
    - Managing teleop activation/deactivation
    - Capturing initial poses when teleop activates

    Args:
        data_manager: DataManager object for thread-safe communication
        quest_reader: MetaQuestReader instance
        hand: Which controller to read — "right" (default) or "left"
    """
    print(f"🎮 Quest Controller thread started ({hand} hand)")

    dt: float = 1.0 / CONTROLLER_DATA_RATE
    prev_grip_active: bool = False

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            # Get controller data
            grip_value = quest_reader.get_grip_value(hand)
            trigger_value = quest_reader.get_trigger_value(hand)
            controller_transform = quest_reader.get_hand_controller_transform_ros(
                hand=hand
            )

            # Update shared state with controller data
            data_manager.set_controller_data(
                controller_transform, grip_value, trigger_value
            )

            # Grip button logic (dead man's switch)
            robot_activity_state = data_manager.get_robot_activity_state()
            grip_active = (
                grip_value >= GRIP_THRESHOLD
                and robot_activity_state == RobotActivityState.ENABLED
            )

            # Rising edge - grip just pressed AND robot is enabled
            if (
                grip_active
                and not prev_grip_active
                and controller_transform is not None
            ):
                controller_initial_transform = controller_transform.copy()
                robot_initial_transform = data_manager.get_current_end_effector_pose()

                data_manager.set_teleop_state(
                    True, controller_initial_transform, robot_initial_transform
                )
                # SO101 joint_state_thread gates on leader_teleop_engaged; grip acts as engagement
                data_manager.set_leader_teleop_engaged(True)

                print("✓ Teleop control activated")
                print(
                    f"  Controller initial position: {controller_initial_transform[:3, 3]}"
                )
                if robot_initial_transform is not None:
                    print(f"  Robot initial position: {robot_initial_transform[:3, 3]}")
                else:
                    print("  Robot initial position: None")

            # Falling edge - grip just released OR robot disabled
            elif not grip_active and prev_grip_active:
                data_manager.set_teleop_state(False, None, None)
                data_manager.set_leader_teleop_engaged(False)
                print("✗ Teleop control deactivated")

            prev_grip_active = grip_active

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Quest reader thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        data_manager.set_teleop_state(False, None, None)
        data_manager.set_leader_teleop_engaged(False)
        print(f"🎮 Quest Controller thread stopped ({hand} hand)")
