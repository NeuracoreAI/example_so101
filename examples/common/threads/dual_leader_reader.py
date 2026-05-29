"""Dual-arm leader reader thread – reads one SO101 leader and writes to DualDataManager."""

import time
import traceback

from common.data_manager_dual import DualDataManager
from common.leader_arm import LerobotSO101LeaderArm


def dual_leader_reader_thread(
    data_manager: DualDataManager,
    leader_arm: LerobotSO101LeaderArm,
    arm_side: str,
    rate_hz: float,
) -> None:
    """Read one leader arm at rate_hz and store in data_manager under arm_side.

    Args:
        data_manager: Shared dual-arm DataManager.
        leader_arm: Configured and connected LerobotSO101LeaderArm for this side.
        arm_side: "left" or "right".
        rate_hz: Polling rate in Hz.
    """
    if arm_side not in ("left", "right"):
        raise ValueError("arm_side must be 'left' or 'right'")

    dt = 1.0 / rate_hz
    print(f"🦾 Dual leader reader thread started ({arm_side})")
    try:
        while not data_manager.is_shutdown_requested():
            t0 = time.perf_counter()
            try:
                joint_angles, gripper_open = leader_arm.read_mapped()
                data_manager.set_leader_mapped_state(arm_side, joint_angles, gripper_open)
            except OSError as e:
                print(f"⚠️  ({arm_side}) leader read error (retrying): {e}")
                continue
            except RuntimeError as e:
                if "no calibration registered" in str(e):
                    print(
                        f"❌ {arm_side} leader has no calibration. "
                        "Run lerobot-calibrate for the leader."
                    )
                    data_manager.request_shutdown()
                elif "configure_follower" in str(e):
                    print(
                        f"❌ {arm_side} leader has no follower config. "
                        "Call configure_follower() before starting the thread."
                    )
                    data_manager.request_shutdown()
                break
            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
    except Exception as e:
        print(f"❌ Dual leader reader error ({arm_side}): {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        print(f"🦾 Dual leader reader thread stopped ({arm_side})")
