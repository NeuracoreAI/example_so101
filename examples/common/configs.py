"""Configuration parameters for SO101 robot demos."""

from pathlib import Path

import numpy as np

# SO101 URDF: so101_minimal.urdf is a placeholder; replace with so101.urdf from SO-ARM100 for accurate mesh (see so101_description/urdf/README.md)
URDF_PATH = str(
    Path(__file__).parent.parent.parent
    / "so101_description"
    / "robot.urdf"
)

GRIPPER_FRAME_NAME = "gripper_frame_link"

# Pink IK parameters (used if IK-based control is added later)
SOLVER_NAME = "quadprog"
POSITION_COST = 1.0
ORIENTATION_COST = 0.75
FRAME_TASK_GAIN = 0.4
LM_DAMPING = 0.0
DAMPING_COST = 0.25
SOLVER_DAMPING_VALUE = 1e-12

# Controller 1€ Filter parameters
CONTROLLER_MIN_CUTOFF = 0.8
CONTROLLER_BETA = 5.0
CONTROLLER_D_CUTOFF = 0.9

GRIP_THRESHOLD = 0.9

# Scaling factors for translation and rotation
TRANSLATION_SCALE = 3.0
ROTATION_SCALE = 2.0

# Thread rates (Hz)
CONTROLLER_DATA_RATE = 50.0
IK_SOLVER_RATE = 250.0
VISUALIZATION_RATE = 60.0
ROBOT_RATE = 100.0

JOINT_STATE_STREAMING_RATE = 100.0
CAMERA_FRAME_STREAMING_RATE = 30.0

# USB webcam (OpenCV)
CAMERA_DEVICE_INDEX = 1  # 0 = first camera, 1 = second, etc.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# SO101 neutral pose (degrees): 5 body joints [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
NEUTRAL_JOINT_ANGLES = [0.0, 90.0, -90.0, 0.0, 0.0]

# Posture task cost vector (one weight per joint)
POSTURE_COST_VECTOR = [0.0, 0.0, 0.0, 0.05, 0.0]

POLICY_EXECUTION_RATE = 100.0
PREDICTION_HORIZON_EXECUTION_RATIO = 0.8
MAX_SAFETY_THRESHOLD = 20.0
MAX_ACTION_ERROR_THRESHOLD = 3.0
TARGETING_POSE_TIME_THRESHOLD = 1.0

GRIPPER_LOGGING_NAME = "gripper"
CAMERA_LOGGING_NAME = "rgb"
# SO101: 5 body joints (same order as LeRobot so_follower)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    # NOTE: We append the gripper as a pseudo-joint in some demos so Neuracore
    # visualizers can show it alongside the arm. This couples joints and gripper
    # and is not generally recommended, but works for now.
    "gripper",
]

# Leader arm → SO101 follower mapping (used by leader_arm_controller and teleop examples)
# SO101 follower: 5 body joints. Leader 5 DOF + gripper → Follower 5 DOF + gripper (1:1).
SO101_JOINT_LIMITS_DEG = np.array(
    [
        (-150.0, 150.0),
        (-180.0, 180.0),
        (-150.0, 150.0),
        (-150.0, 150.0),
        (-180.0, 180.0),
    ],
    dtype=np.float64,
)
SO101_OFFSETS_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
SO101_DIRECTIONS = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
# Leader joint index -> Follower joint index (1:1)
LEADER_TO_SO101_JOINT = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
SO101_FIXED_JOINTS: dict = {}  # none

# robot.urdf actuated joint order (yourdfpy): gripper, wrist_roll, wrist_flex, elbow_flex, shoulder_lift, shoulder_pan
# Our order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper -> index [0..5]
# Reorder our 6-vector to URDF order: our[5], our[4], our[3], our[2], our[1], our[0]
URDF_JOINT_ORDER_FROM_OURS = np.array([5, 4, 3, 2, 1, 0], dtype=np.int32)
