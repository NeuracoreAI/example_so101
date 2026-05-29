"""Dual-arm camera thread — captures frames into DualDataManager.

Writes frames to DualDataManager only. NC logging is handled by a separate
Neuracore logging thread so this loop is never blocked by frame encoding.
"""

import time
import traceback

import cv2

from common.configs import CAMERA_FRAME_STREAMING_RATE
from common.data_manager_dual import DualDataManager


def dual_camera_thread(
    dm: DualDataManager,
    camera_name: str,
    device_index: int,
    width: int,
    height: int,
) -> None:
    """Capture frames from a USB camera and write them to DualDataManager."""
    print(f"📷 Camera thread started (device {device_index}, name='{camera_name}')")
    dt: float = 1.0 / CAMERA_FRAME_STREAMING_RATE
    cap: cv2.VideoCapture | None = None

    try:
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            print(
                f"❌ Could not open camera device {device_index} ('{camera_name}'). "
                "Check connection and device index."
            )
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FRAME_STREAMING_RATE)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  '{camera_name}' opened: {actual_w}x{actual_h} @ ~{CAMERA_FRAME_STREAMING_RATE} Hz")

        while not dm.is_shutdown_requested():
            iteration_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"⚠️  Camera '{camera_name}' read failed, skipping frame")
                time.sleep(dt)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dm.set_rgb_image(rgb, camera_name)

            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Camera thread error ('{camera_name}'): {e}")
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
            print(f"  ✓ Camera '{camera_name}' released")
        print(f"📷 Camera thread stopped ('{camera_name}')")
