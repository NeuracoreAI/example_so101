"""Leader arm controller thread – reads leader arm and updates DataManager.

Mirrors the Meta Quest controller thread pattern: runs at a fixed rate, reads
the teleop device (SO101 leader arm), and writes mapped joint angles and gripper
into DataManager for use by the robot control and teleop loop.
"""

import time
import traceback

from common.configs import CONTROLLER_DATA_RATE
from common.data_manager import DataManager
from common.leader_arm import SO101LeaderArm


def leader_arm_controller_thread(
    data_manager: DataManager,
    leader_arm: SO101LeaderArm,
    rate_hz: float | None = None,
) -> None:
    """Leader arm controller thread – reads leader and updates DataManager.

    Runs at rate_hz (default CONTROLLER_DATA_RATE). Reads mapped joint angles
    and gripper from the leader arm and stores them in DataManager via
    set_leader_mapped_state(). The leader arm must have configure_follower()
    called before starting this thread.

    Args:
        data_manager: DataManager for thread-safe state.
        leader_arm: Connected SO101LeaderArm instance.
        rate_hz: Read rate in Hz; defaults to CONTROLLER_DATA_RATE.
    """
    dt = 1.0 / (rate_hz if rate_hz is not None else CONTROLLER_DATA_RATE)
    print("🎮 Leader arm controller thread started")

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            try:
                joint_angles, gripper_open = leader_arm.read_mapped()
                data_manager.set_leader_mapped_state(joint_angles, gripper_open)
            except RuntimeError as e:
                if "no calibration registered" in str(e):
                    print(
                        "❌ Leader has no calibration. Run lerobot-calibrate for the leader."
                    )
                    data_manager.request_shutdown()
                elif "configure_follower" in str(e):
                    print(
                        "❌ Leader arm has no follower config. "
                        "Call configure_follower() before starting the thread."
                    )
                    data_manager.request_shutdown()
                break

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Leader arm controller thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print("🎮 Leader arm controller thread stopped")
