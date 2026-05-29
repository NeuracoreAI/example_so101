#!/usr/bin/env python3
"""Dual-arm SO101 controller.

Wraps two SO101Controller instances (left and right arms) and exposes a
unified lifecycle interface. Each arm's control loop runs independently
in its own background thread via SO101Controller.
"""

import numpy as np

from .sts3215_bus import SO101FollowerDriver  # noqa: F401 (re-exported via SO101Controller)
from .so101_controller import SO101Controller


class SO101DualController:
    """Controller for a dual SO101 arm setup (left + right).

    Access each arm via .left and .right, which are full SO101Controller
    instances. All methods on SO101Controller are available directly.
    """

    def __init__(
        self,
        left_port: str,
        left_follower_id: str,
        right_port: str,
        right_follower_id: str,
        robot_rate: float = 100.0,
        neutral_joint_angles: np.ndarray | None = None,
        debug_mode: bool = False,
    ) -> None:
        print(f"Connecting left arm  → port={left_port}  id={left_follower_id}")
        self.left = SO101Controller(
            port=left_port,
            follower_id=left_follower_id,
            robot_rate=robot_rate,
            neutral_joint_angles=neutral_joint_angles,
            debug_mode=debug_mode,
        )

        print(f"Connecting right arm → port={right_port}  id={right_follower_id}")
        self.right = SO101Controller(
            port=right_port,
            follower_id=right_follower_id,
            robot_rate=robot_rate,
            neutral_joint_angles=neutral_joint_angles,
            debug_mode=debug_mode,
        )

    def start_control_loop(self) -> None:
        """Start background control loop threads for both arms."""
        self.left.start_control_loop()
        self.right.start_control_loop()

    def stop_control_loop(self) -> None:
        """Stop background control loop threads for both arms."""
        self.left.stop_control_loop()
        self.right.stop_control_loop()

    def cleanup(self) -> None:
        """Disconnect and release resources for both arms."""
        print("🧹 Cleaning up dual SO101 controller...")
        self.left.cleanup()
        self.right.cleanup()
        print("✓ Dual SO101 controller cleanup completed")
