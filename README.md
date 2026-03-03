# SO100 Leader → SO100 Follower Teleop (LeRobot / SO-ARM100)

This project is an example of **SO100-to-SO100 teleoperation**: you use one LeRobot SO100 arm as a **leader** and drive either the on-screen SO100 URDF or a second SO100 **follower** arm. Everything in this directory is for the SO100 (SO-ARM100) only; there are no AgileX or Piper references.

## Prerequisites

- Python 3.10+
- Conda (recommended)
- **Leader**: One SO100 arm (LeRobot SO100 leader) with calibration
- **Follower** (optional): Second SO100 arm for real-robot teleop (USB, motors configured)

## Installation

### 1. Create Conda environment

```bash
cd example_lerobot_so100
conda env create -f environment.yaml
conda activate so100-teleop
```

### 2. Install LeRobot with Feetech support

The SO100 uses Feetech STS3215 servos. Install LeRobot with the `feetech` extra (from the [lerobot](https://github.com/huggingface/lerobot) repo):

```bash
pip install -e ".[feetech]"
```

If you use a local clone of lerobot (e.g. in your workspace), install from that path:

```bash
cd /path/to/lerobot
pip install -e ".[feetech]"
```

## Getting your SO100 robot working

### Motor setup (follower arm, do before assembly if possible)

1. **Find the USB port** for the SO100 controller:
   ```bash
   lerobot-find-port
   ```
   Use the reported port (e.g. `/dev/ttyACM0` or `/dev/ttyUSB0`).

2. **Set motor IDs and baudrate** (1 Mbps standard). Do this **before** full assembly so you can access each motor:
   ```bash
   lerobot-setup-motors \
       --robot.type=so100_follower \
       --robot.port=/dev/ttyACM0
   ```
   Or set each motor manually; see [LeRobot SO100 docs](https://huggingface.co/docs/lerobot/so100).

3. **Linux**: grant access to the USB port:
   ```bash
   sudo chmod 666 /dev/ttyACM0
   ```
   Or add a udev rule so your user can access the device without sudo.

### Leader arm calibration

The **leader** arm must be calibrated so joint readings are correct:

```bash
lerobot-calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm
```

Use the same `--teleop.id` when running the example (`--leader-id`).

### Two arms (leader + follower)

- Connect **leader** and **follower** to **different** USB ports (e.g. leader on `/dev/ttyACM0`, follower on `/dev/ttyUSB0`).
- Leader uses the calibration id (`--leader-id`).
- Follower uses the id you gave when running `lerobot-setup-motors` or when calibrating the follower (`--follower-id`).

## Usage

### URDF only (no real follower)

Drive the SO100 URDF in the GUI with the leader arm:

```bash
cd example_lerobot_so100/examples
python X_leader_arm_teleop_so100.py --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm
```

### Real SO100 follower

Drive the physical follower arm with the leader:

```bash
python X_leader_arm_teleop_so100.py --real-robot \
  --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm \
  --follower-port /dev/ttyUSB0 --follower-id my_awesome_follower_arm
```

- **Enable robot** in the GUI before moving the leader; the follower will then follow.
- **Home** sends the follower to the neutral pose defined in `configs.py`.
- **Ctrl+C** shuts down cleanly.

## Configuration

- **URDF**: `examples/common/configs.py` sets `URDF_PATH` to `so101_description/urdf/so100_minimal.urdf`. For accurate mesh, use the official [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) URDF (see `so101_description/urdf/README.md`).
- **Neutral pose**: `NEUTRAL_JOINT_ANGLES` in `configs.py` (5 body joints in degrees).
- **Joint names**: SO100 uses `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` (+ gripper).
- **Camera (USB webcam)**: The camera thread in `examples/common/threads/camera.py` uses OpenCV and a basic USB webcam. In `configs.py` you can set `CAMERA_DEVICE_INDEX` (0 = first camera), `CAMERA_WIDTH`, `CAMERA_HEIGHT`, and `CAMERA_FRAME_STREAMING_RATE`. Start the camera thread from your script if you need RGB frames (e.g. for logging or visualization).

## Project structure

```
example_lerobot_so100/
├── examples/
│   ├── X_leader_arm_teleop_so100.py   # SO100 leader → SO100 follower teleop
│   └── common/                       # Config, data manager, visualizer, threads
├── so100_controller.py               # SO100 follower controller (LeRobot SO100Follower)
├── so101_description/urdf/            # SO100 URDF (minimal + README for official mesh)
├── environment.yaml
└── README.md
```

Legacy Piper/AgileX assets (e.g. `piper_controller.py`, `piper_description/`, `scripts/piper/`) are no longer used by the SO100 flow; you can remove them if you only need SO100.

## Troubleshooting

- **"No calibration registered"**: Run `lerobot-calibrate` for the leader with the same `--teleop.id` you pass as `--leader-id`.
- **Follower not moving**: Ensure the robot is **enabled** in the GUI and the follower is on the correct `--follower-port`.
- **Wrong port**: Use `lerobot-find-port` and/or `ls /dev/tty*` to see which port is which; leader and follower must be on different ports when using two arms.
- **Motor direction opposite**: Some setups need per-motor direction or recalibration; see [LeRobot SO100 issues](https://github.com/huggingface/lerobot/issues).

## Safety

- This software drives a physical robot. Keep a safe workspace and be ready to stop (disable in GUI or Ctrl+C).
- Start with the robot **disabled** and only enable after confirming the leader pose is safe.

## License

See LICENSE file.
