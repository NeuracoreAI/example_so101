"""Utility functions for the examples."""

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def transform_from_position_wxyz(
    position: Sequence[float], wxyz: Sequence[float]
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from Viser position and wxyz quaternion."""
    quat_xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(position, dtype=float)
    transform[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    return transform


def position_wxyz_from_transform(
    transform: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert a 4x4 transform to Viser (position, wxyz quaternion)."""
    position = tuple(float(v) for v in transform[:3, 3])
    quat_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
    return position, wxyz


def se3_inverse(transform: np.ndarray) -> np.ndarray:
    """Invert a proper 4x4 rigid-body transform."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


_MIRROR_HEAD_FRAME_Y = np.diag([1.0, -1.0, 1.0, 1.0])


def mirror_head_frame_pose(transform: np.ndarray) -> np.ndarray:
    """Mirror a head-frame hand pose for face-to-face (mirror) teleoperation."""
    return _MIRROR_HEAD_FRAME_Y @ transform @ _MIRROR_HEAD_FRAME_Y


def map_quest_hands_to_robot_arms(
    left_hand_transform: np.ndarray,
    right_hand_transform: np.ndarray,
    *,
    mirror_control: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Map Quest left/right hands to robot left/right arms.

    Normal mode: Quest left → robot left, Quest right → robot right.
    Mirror mode: swap hands and reflect lateral motion.
    """
    if not mirror_control:
        return left_hand_transform, right_hand_transform
    return (
        mirror_head_frame_pose(right_hand_transform),
        mirror_head_frame_pose(left_hand_transform),
    )


def compute_hand_to_robot_calibration(
    robot_ee_pose: np.ndarray,
    hand_in_chest_frame: np.ndarray,
    hand_reference_at_calibration: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
) -> np.ndarray:
    """Return the robot EE pose at calibration time, used as the delta anchor."""
    return robot_ee_pose.copy()


def map_head_frame_hand_to_robot_target(
    hand_in_head_frame: np.ndarray,
    hand_to_robot_calibration: np.ndarray,
    hand_reference_at_calibration: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
) -> np.ndarray:
    """Map an absolute head-frame hand pose to a robot-base TCP target.

    Translation: world-frame delta from the calibration reference, scaled and
    added to the robot EE position at calibration — no rotation of the delta.

    Rotation: controller rotation delta from reference, scaled and composed
    onto the robot EE rotation at calibration.
    """
    # Translation — world-frame delta, added directly (matches single-arm ik_solver.py)
    delta_pos = hand_in_head_frame[:3, 3] - hand_reference_at_calibration[:3, 3]
    target_pos = hand_to_robot_calibration[:3, 3] + delta_pos * translation_scale

    # Rotation — delta from reference, scaled, composed onto EE rotation at calib
    rotation_hand = hand_in_head_frame[:3, :3]
    rotation_ref = hand_reference_at_calibration[:3, :3]
    rotation_delta = rotation_hand @ rotation_ref.T
    if rotation_scale != 1.0:
        rotation_delta = Rotation.from_rotvec(
            Rotation.from_matrix(rotation_delta).as_rotvec() * rotation_scale
        ).as_matrix()
    target_rot = hand_to_robot_calibration[:3, :3] @ rotation_delta

    target = np.eye(4)
    target[:3, 3] = target_pos
    target[:3, :3] = target_rot
    return target


def scale_and_add_delta_transform(
    delta_position: np.ndarray,
    delta_orientation: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
    initial_transform: np.ndarray,
) -> np.ndarray:
    """Scale and add delta position and orientation to an initial transform.

    Args:
        delta_position: 3D position vector
        delta_orientation: 3x3 rotation matrix
        translation_scale: translation scale
        rotation_scale: rotation scale
        initial_transform: 4x4 transformation matrix

    Returns:
        4x4 transformation matrix
    """
    scaled_delta_position = delta_position * translation_scale

    # Scale the delta rotation by converting to axis-angle, scaling the angle, then converting back
    delta_rotation = Rotation.from_matrix(delta_orientation)
    delta_axis_angle = delta_rotation.as_rotvec()
    scaled_delta_axis_angle = delta_axis_angle * rotation_scale
    scaled_delta_rotation = Rotation.from_rotvec(scaled_delta_axis_angle).as_matrix()

    # Compose rotations by matrix multiplication (correct way to combine rotations)
    initial_rotation = initial_transform[:3, :3]
    new_rotation = initial_rotation @ scaled_delta_rotation

    target_transform = np.eye(4)
    target_transform[:3, 3] = initial_transform[:3, 3] + scaled_delta_position
    target_transform[:3, :3] = new_rotation

    return target_transform
