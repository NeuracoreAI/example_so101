#!/usr/bin/env python3
"""SO101 follower arm controller using LeRobot SO101Follower.

Controls the physical SO101 (SO-ARM100) follower arm over USB via the
Feetech bus. Joint angles are in degrees; gripper is 0–1 (open).
No AgileX or Piper references; SO101 only.
"""

import threading
import time
from typing import Any

import numpy as np

# LeRobot SO101 follower (requires: pip install -e ".[feetech]" in lerobot)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# SO101 motor keys in order: 5 body joints + gripper
SO101_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]
SO101_GRIPPER_KEY = "gripper.pos"
NUM_BODY_JOINTS = 5


class SO101Controller:
    """Controller for the SO101 follower arm (LeRobot SO-100 / SO-ARM100).

    Presents the same logical interface as the previous robot controller:
    joint angles in degrees (5 body joints), gripper 0–1, enable/disable,
    home, and a background control loop that sends commands when enabled.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        follower_id: str = "my_awesome_follower_arm",
        robot_rate: float = 100.0,
        neutral_joint_angles: np.ndarray | None = None,
        debug_mode: bool = False,
    ) -> None:
        """Initialize the SO101 controller (does not connect yet).

        Args:
            port: Serial port for the SO101 follower (e.g. /dev/ttyACM0 or /dev/ttyUSB0).
            follower_id: Calibration id used for this follower (lerobot-setup-motors / calibration).
            robot_rate: Control loop rate in Hz.
            neutral_joint_angles: Home position [j1..j5] in degrees (default zeros).
            debug_mode: Enable debug logging.
        """
        self.port = port
        self.follower_id = follower_id
        self.robot_rate = robot_rate
        self.debug_mode = debug_mode

        self._config = SO101FollowerConfig(
            port=port,
            id=follower_id,
            use_degrees=True,
            cameras={},  # no cameras required for teleop
        )
        self._robot: SO101Follower | None = None

        self.position_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self._bus_lock = threading.Lock()  # Serialize all serial port access (read/write)
        self.running = threading.Event()
        self.running.set()

        self._robot_enabled = False

        if neutral_joint_angles is not None:
            self.HOME_JOINT_ANGLES = np.array(neutral_joint_angles, dtype=np.float64)
        else:
            self.HOME_JOINT_ANGLES = np.zeros(NUM_BODY_JOINTS, dtype=np.float64)

        # Gripper: 0 = closed, 1 = open (normalized). SO101 uses 0–100 in action.
        self._target_joint_angles = self.HOME_JOINT_ANGLES.copy()
        self._gripper_open_value = 0.5  # 0–1

        # Joint limits (degrees) – conservative defaults; can be updated from calibration
        self.JOINT_LIMITS = np.array(
            [
                (-150.0, 150.0),
                (-180.0, 180.0),
                (-150.0, 150.0),
                (-150.0, 150.0),
                (-180.0, 180.0),
            ],
            dtype=np.float64,
        )

        self._control_loop_thread = threading.Thread(
            target=self._control_loop, daemon=True
        )

        self._connect()

    def _connect(self) -> None:
        """Connect to the SO101 follower arm."""
        print(f"Connecting to SO101 follower on {self.port} (id={self.follower_id})...")
        self._robot = SO101Follower(self._config)
        self._robot.connect(calibrate=False)
        print("✓ SO101 follower connected")

    def start_control_loop(self) -> None:
        """Start the control loop thread."""
        self._control_loop_thread.start()

    def stop_control_loop(self) -> None:
        """Stop the control loop thread."""
        self.running.clear()
        if self._control_loop_thread.is_alive():
            self._control_loop_thread.join(timeout=2.0)
            print("✓ Control loop thread joined")
        else:
            print("Control loop thread is not running")

    def __del__(self) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Disconnect and release resources."""
        print("🧹 Cleaning up SO101 controller...")
        self.stop_control_loop()
        self._set_robot_enabled(False)
        if self._robot is not None and self._robot.is_connected:
            self._robot.disconnect()
            print("✓ SO101 follower disconnected")
        self._robot = None
        print("✓ SO101 controller cleanup completed")

    def _set_robot_enabled(self, enabled: bool) -> None:
        with self.state_lock:
            self._robot_enabled = enabled
            if self.debug_mode:
                print(f"SO101 enabled: {enabled}")

    def is_robot_enabled(self) -> bool:
        with self.state_lock:
            return self._robot_enabled

    def get_target_joint_angles(self) -> np.ndarray:
        """Target body joint angles [j1..j5] in degrees."""
        with self.position_lock:
            return self._target_joint_angles.copy()

    def set_target_joint_angles(self, joint_angles: np.ndarray) -> None:
        """Set target body joint angles [j1..j5] in degrees."""
        with self.position_lock:
            angles = np.array(joint_angles, dtype=np.float64)
            if angles.size >= NUM_BODY_JOINTS:
                angles = angles[:NUM_BODY_JOINTS]
            else:
                angles = np.resize(angles, NUM_BODY_JOINTS)
            self._target_joint_angles = np.clip(
                angles,
                self.JOINT_LIMITS[:, 0],
                self.JOINT_LIMITS[:, 1],
            )

    def get_gripper_open_value(self) -> float:
        """Gripper open value in [0, 1]."""
        with self.position_lock:
            return float(np.clip(self._gripper_open_value, 0.0, 1.0))

    def set_gripper_open_value(self, value: float) -> None:
        """Set gripper open value in [0, 1]."""
        with self.position_lock:
            self._gripper_open_value = float(np.clip(value, 0.0, 1.0))

    def get_current_joint_angles(self) -> np.ndarray | None:
        """Current body joint angles [j1..j5] in degrees, or None if not connected."""
        if self._robot is None or not self._robot.is_connected:
            return None
        try:
            with self._bus_lock:
                obs = self._robot.get_observation()
            angles = np.array(
                [obs[k] for k in SO101_JOINT_KEYS if k in obs],
                dtype=np.float64,
            )
            if angles.size >= NUM_BODY_JOINTS:
                return angles[:NUM_BODY_JOINTS]
            return None
        except Exception as e:
            if self.debug_mode:
                print(f"get_current_joint_angles: {e}")
            return None

    def get_current_gripper_open_value(self) -> float | None:
        """Current gripper open value in [0, 1], or None."""
        if self._robot is None or not self._robot.is_connected:
            return None
        try:
            with self._bus_lock:
                obs = self._robot.get_observation()
            raw = obs.get(SO101_GRIPPER_KEY, 50.0)
            return float(np.clip(raw / 100.0, 0.0, 1.0))
        except Exception as e:
            if self.debug_mode:
                print(f"get_current_gripper_open_value: {e}")
            return None

    def _action_from_targets(self) -> dict[str, float]:
        """Build LeRobot action dict from current targets (keys like shoulder_pan.pos, gripper.pos)."""
        with self.position_lock:
            j = self._target_joint_angles
            g = self._gripper_open_value * 100.0
        action = {}
        for i, key in enumerate(SO101_JOINT_KEYS):
            action[key] = float(j[i])
        action[SO101_GRIPPER_KEY] = float(g)
        return action

    def _control_loop(self) -> None:
        """Send target positions to the robot at robot_rate Hz when enabled."""
        period = 1.0 / self.robot_rate
        while self.running.is_set():
            try:
                if self.is_robot_enabled() and self._robot is not None and self._robot.is_connected:
                    action = self._action_from_targets()
                    with self._bus_lock:
                        self._robot.send_action(action)
                time.sleep(period)
            except Exception as e:
                print(f"SO101 control loop error: {e}")
                time.sleep(0.01)

    def move_to_home(self) -> bool:
        """Set target to home joint angles (and leave gripper as-is)."""
        try:
            with self.position_lock:
                self._target_joint_angles = self.HOME_JOINT_ANGLES.copy()
            print("✓ Home position set")
            return True
        except Exception as e:
            print(f"✗ Home error: {e}")
            return False

    def graceful_stop(self) -> bool:
        """Stop sending commands and mark robot disabled."""
        try:
            print("🛑 Graceful stop – disabling SO101 command stream")
            self._set_robot_enabled(False)
            return True
        except Exception as e:
            print(f"✗ Graceful stop error: {e}")
            return False

    def resume_robot(self) -> bool:
        """Re-enable and optionally sync targets to current state."""
        try:
            if self.is_robot_enabled():
                return True
            self._set_robot_enabled(True)
            # Optionally align targets to current to avoid jump
            cur = self.get_current_joint_angles()
            if cur is not None and len(cur) >= NUM_BODY_JOINTS:
                self.set_target_joint_angles(cur)
            g = self.get_current_gripper_open_value()
            if g is not None:
                self.set_gripper_open_value(g)
            print("✓ SO101 resumed")
            return True
        except Exception as e:
            print(f"✗ Resume error: {e}")
            return False

    def is_robot_homed(self, tolerance_degrees: float = 2.0) -> bool:
        """True if current joint angles are near home."""
        cur = self.get_current_joint_angles()
        if cur is None or len(cur) < NUM_BODY_JOINTS:
            return False
        return bool(np.all(np.abs(cur - self.HOME_JOINT_ANGLES) < tolerance_degrees))

    def get_robot_status(self) -> dict[str, Any]:
        """Status dict for debugging."""
        return {
            "enabled": self.is_robot_enabled(),
            "target_joint_angles": self.get_target_joint_angles(),
            "gripper_open_value": self.get_gripper_open_value(),
            "current_joint_angles": self.get_current_joint_angles(),
            "current_gripper_open_value": self.get_current_gripper_open_value(),
        }
