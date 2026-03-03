#!/usr/bin/env python3
"""Minimal script to test the USB camera thread used in the SO100 examples.

This script:
- Sets up the same Python path layout as the other examples
- Creates a DataManager instance
- Starts the OpenCV-based camera_thread in a background thread
- Visualizes the latest RGB frame in an OpenCV window
"""

import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "examples"))

from common.configs import CAMERA_FRAME_STREAMING_RATE  # type: ignore  # noqa: E402
from common.data_manager import DataManager  # type: ignore  # noqa: E402
from common.threads.camera import camera_thread  # type: ignore  # noqa: E402

WINDOW_NAME = "SO100 camera test"


def _on_change(name: str, value: Any, timestamp: float) -> None:
    """Optional callback; visualization is done in the main loop."""
    if name != "log_rgb":
        return
    # Kept for optional verbose logging; can be removed if not needed
    pass


def main() -> None:
    """Start the camera thread and visualize frames in an OpenCV window."""
    data_manager = DataManager()
    data_manager.set_on_change_callback(_on_change)

    print("=" * 60)
    print("SO100 CAMERA THREAD TEST")
    print("=" * 60)
    print(f"Target frame rate: {CAMERA_FRAME_STREAMING_RATE} Hz")
    print("Opening USB webcam using common.threads.camera.camera_thread...")
    print("Close the camera window or press 'q' in the window to stop.\n")

    camera_thread_obj = threading.Thread(
        target=camera_thread,
        args=(data_manager,),
        daemon=True,
    )
    camera_thread_obj.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while not data_manager.is_shutdown_requested():
            frame = data_manager.get_rgb_image()
            if frame is not None:
                # DataManager stores RGB; OpenCV imshow expects BGR
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(WINDOW_NAME, bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("'q' pressed – stopping...")
                data_manager.request_shutdown()
                break
            time.sleep(1.0 / 30.0)  # ~30 Hz display refresh
    except KeyboardInterrupt:
        print("\nStopping camera test – requesting shutdown...")
        data_manager.request_shutdown()
    finally:
        cv2.destroyAllWindows()

    camera_thread_obj.join(timeout=2.0)
    print("Camera test finished.")


if __name__ == "__main__":
    main()

