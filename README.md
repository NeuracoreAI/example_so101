# SO101 Leader → SO101 Follower Teleop (SO-ARM100)

This project is an example of **SO101-to-SO101 teleoperation**: you use one SO101 arm as a **leader** and drive either the on-screen SO101 URDF or a second SO101 **follower** arm. Everything in this directory is for the SO101 (SO-ARM100) only.

## Prerequisites

- Python 3.10+
- Conda (recommended)
- **Leader**: One SO101 arm (LeRobot SO101 leader) with calibration
- **Follower** (optional): Second SO101 arm for real-robot teleop (USB, motors configured)
- **Meta Quest** (optional): Meta Quest 2/3/Pro headset for Quest-based IK teleop (example 7)

## Installation

### 1. Create Conda environment

```bash
cd example_lerobot_so101
conda env create -f environment.yaml
conda activate so101-teleop
```

### 2. One-time hardware setup with LeRobot CLI tools

The teleoperation scripts **do not require lerobot at runtime** — `feetech-servo-sdk` (already in `environment.yaml`, imported as `scservo_sdk`) communicates with the motors directly.

However, the `lerobot` CLI tools are still needed **once** to set up motor IDs and calibrate the leader arm. Install lerobot with the feetech extra (from the [lerobot](https://github.com/huggingface/lerobot) repo):

```bash
pip install -e ".[feetech]"
```

You can uninstall it again after calibration is done.

If your LeRobot is already calibrated and you decide you do not need to use `lerobot` CLI tools, you can use a script local to this repo to discover the ports your leader and follower are connected to:

```bash
python scripts/lerobot_find_port.py
```

## Getting your SO101 robot working

### Motor setup (follower arm, do before assembly if possible)

1. **Find the USB port** for the SO101 controller:
   ```bash
   lerobot-find-port
   ```
   Use the reported port (e.g. `/dev/ttyACM0` or `/dev/ttyUSB0`).

2. **Set motor IDs and baudrate** (1 Mbps standard). Do this **before** full assembly so you can access each motor:
   ```bash
   lerobot-setup-motors \
       --robot.type=so101_follower \
       --robot.port=/dev/ttyACM0
   ```
   Or set each motor manually; see [LeRobot SO101 docs](https://huggingface.co/docs/lerobot/so101).

3. **Linux**: **Please remember to grant access to the USB port**:
   ```bash 
   sudo chmod 666 /dev/ttyACM0
   ```
   Or add a udev rule so your user can access the device without sudo.

### Leader arm calibration

The **leader** arm must be calibrated so joint readings are correct:

```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
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

Drive the SO101 URDF in the GUI with the leader arm:

```bash
cd example_so101/examples
python 1_leader_arm_teleop_so101.py --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm
```

### Real SO101 follower

Drive the physical follower arm with the leader:

```bash
python 1_leader_arm_teleop_so101.py --real-robot \
  --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm \
  --follower-port /dev/ttyUSB0 --follower-id my_awesome_follower_arm
```

- **Enable robot** in the GUI before moving the leader; the follower will then follow.
- **Home** sends the follower to the neutral pose defined in `configs.py`.
- **Ctrl+C** shuts down cleanly.

## Configuration

- **URDF**: `examples/common/configs.py` sets `URDF_PATH` to `so101_description/urdf/so101_minimal.urdf`. For accurate mesh, use the official [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) URDF (see `so101_description/urdf/README.md`).
- **Neutral pose**: `NEUTRAL_JOINT_ANGLES` in `configs.py` (5 body joints in degrees).
- **Joint names**: SO101 uses `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` (+ gripper).
- **Camera (USB webcam)**: The camera thread in `examples/common/threads/camera.py` uses OpenCV and a basic USB webcam. In `configs.py` you can set `CAMERA_DEVICE_INDEX` (0 = first camera), `CAMERA_WIDTH`, `CAMERA_HEIGHT`, and `CAMERA_FRAME_STREAMING_RATE`. Start the camera thread from your script if you need RGB frames (e.g. for logging or visualization).


### Collect teleop data with Neuracore

Stream and record teleoperation data (joint positions, gripper, RGB camera) to [Neuracore](https://neuracore.ai) for training:

```bash
python examples/2_collect_teleop_data_with_neuracore.py \
  --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm \
  --follower-port /dev/ttyACM1 --follower-id my_awesome_follower_arm \
  --dataset-name so101-demo
```

**Prerequisites:** A Neuracore account. The script calls `nc.login()` on startup — set your credentials beforehand (see [Neuracore docs](https://neuracore.ai/docs)).

**What it does:**
- Connects to Neuracore, creates (or reuses) the named dataset, and streams live data.
- The **real SO101 follower is always active** — the robot is auto-enabled at startup.
- Streams joint positions, gripper state, and RGB frames from a USB webcam simultaneously.
- Recording episodes is controlled from the **Neuracore web UI** (start / stop recording there).
- Press **Ctrl+C** to stop teleoperation and shut down cleanly; any active recording is stopped automatically.

**Optional arguments:**

| Flag | Default | Description |
|---|---|---|
| `--leader-port` | `/dev/ttyACM0` | Serial port for the leader arm |
| `--leader-id` | `my_awesome_leader_arm` | Calibration id (matches `--teleop.id` used with `lerobot-calibrate`) |
| `--follower-port` | `/dev/ttyUSB0` | Serial port for the follower arm |
| `--follower-id` | `my_awesome_follower_arm` | Follower arm id |
| `--dataset-name` | `so101-teleop-data-<timestamp>` | Dataset name in Neuracore |

### 3. Replay Neuracore Episodes

**Script**: `examples/3_replay_neuracore_episodes.py`

Replay recorded episodes from a Neuracore dataset on the physical robot.

```bash
python3 examples/3_replay_neuracore_episodes.py --dataset-name <dataset-name> --frequency <hz> --follower-port <follower-port> --follower-id my_awesome_follower_arm --episode-index <idx>
```

**Arguments**:
- `--dataset-name`: Name of the Neuracore dataset to replay
- `--frequency`: Playback frequency in Hz (default: 0). 0 plays the data aperiodically (not synchronized at a certain frequency as it was recorded). 
- `--follower-port`: The port on which your follower is connected (default: `/dev/ttyACM1`)
- `--follower-id`: The follower ID name you set for your follower arm
- `--episode-index`: Which episode to replay (default: 0). -1 will start replaying all the episodes one after the other.

**NOTE:** please be careful that the robot **will start moving** on the same trajectory that was recorded. Pressing `ctrl+C`
will gracefully disable the robot and it will cut power to the motors after 5 seconds.

### Policy rollout (examples 4–6)

After collecting data with example 2 and training a policy in Neuracore, use these scripts to test it on the SO101 follower. They use the same embodiment channels as data collection: six joint positions (including a pseudo gripper joint), parallel gripper open amount, and RGB camera `rgb`.

**Example 4 – interactive rollout with leader teleop and Viser**

```bash
python examples/4_rollout_neuracore_policy.py \
  --train-run-name YOUR_RUN_NAME \
  --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm \
  --follower-port /dev/ttyACM1 --follower-id my_awesome_follower_arm
```

- **SO101 leader arm** teleop (same mapping as example 2), not Meta Quest — Agilex example 4 uses Quest; SO101 uses a second SO101 arm as leader.
- Viser workflow: **Enable Robot** → **Engage Leader Teleop** (replaces Quest “hold grip”) → move leader → **Run Policy** / execute.
- Viser GUI (`http://localhost:8080`): enable/disable, engage leader teleop, home, run policy, execute horizon, continuous play, live camera + ghost preview.
- Policy source: `--train-run-name`, `--model-path`, or `--remote-endpoint-name`.

**Example 5 – minimal terminal-only rollout**

```bash
python examples/5_rollout_neuracore_policy_minimal.py \
  --train-run-name YOUR_RUN_NAME \
  --follower-port /dev/ttyACM1 --follower-id my_awesome_follower_arm
```

No leader arm or GUI: enables the follower, homes, then loops predict → execute. Ctrl+C homes and exits.

**Example 6 – visualize policy from a dataset**

```bash
python examples/6_visualize_policy_from_dataset.py \
  --dataset-name so101-demo \
  --train-run-name YOUR_RUN_NAME
```

Loads a random synchronized dataset step, runs the policy, and animates the predicted horizon in Viser (no real robot).

### Meta Quest Teleop

#### Additional installation

Install the Pink IK dependencies and the `meta_quest_teleop` package:

```bash
conda env update -f environment.yaml          # adds pin, pin-pink, qpsolvers[quadprog]
pip install -e meta_quest_teleop/             # installs the MetaQuestReader package
```

Ensure your Meta Quest headset is **on the same WiFi network** as your machine and has the companion app running (see the `meta_quest_teleop` README for the Android app setup).

#### Example 7 – IK parameter tuning (single arm)

Drive the SO101 follower arm using a **Meta Quest controller** (right hand). Unlike the leader-arm examples, this uses **Pink IK** to convert the Quest controller's cartesian pose into SO101 joint angles in real time. A Viser GUI lets you tune the 1€ filter and IK parameters live.

```bash
python examples/7_tune_quest_teleop_params.py \
  --port /dev/ttyACM0 \
  --follower-id my_awesome_follower_arm \
  --ip-address <quest-ip>         # omit for auto-discovery
```

**Optional arguments:**

| Flag | Default | Description |
|---|---|---|
| `--port` | `/dev/ttyACM0` | Serial port for the SO101 follower arm |
| `--follower-id` | `my_awesome_follower_arm` | Follower arm calibration ID |
| `--ip-address` | `None` | Meta Quest IP address (auto-discovered if omitted) |

**Controls:**

| Input | Action |
|---|---|
| **Button A** | Enable / disable robot |
| **Right grip (hold)** | Activate IK teleoperation (dead man's switch) |
| **Move controller** | Robot end-effector follows |
| **Right trigger (hold)** | Close gripper |
| **Button B** | Send robot to home position |
| **Release grip** | Pause teleoperation |

**Viser GUI** (`http://localhost:8080`):

- **Enable / Disable robot** toggle button
- **Home** button
- **Controller filter params** — tune the 1€ filter (`min_cutoff`, `beta`, `d_cutoff`) live to balance smoothness vs. latency
- **Scaling controls** — adjust `translation_scale` and `rotation_scale` to calibrate how much the robot moves relative to the controller
- **Pink IK parameters** — tune `position_cost`, `orientation_cost`, `frame_task_gain`, `lm_damping`, `damping_cost`, `solver_damping_value` in real time
- **Ghost robot** — translucent preview of the IK target (where the robot is aiming, not where it currently is)
- **IK solve time** — timing display to monitor solver performance

#### Example 8 – Dual-arm teleoperation

Drive **two SO101 arms simultaneously** using both Meta Quest controllers. The left controller maps to the left arm and the right controller maps to the right arm via a single **10-DOF IK solver** on the dual-arm URDF.

```bash
python examples/8_dual_arm_teleop.py \
  --left-port /dev/ttyACM0 --left-id L1 \
  --right-port /dev/ttyACM1 --right-id L1 \
  --ip-address <quest-ip>         # omit for auto-discovery
```

**Optional arguments:**

| Flag | Default | Description |
|---|---|---|
| `--left-port` | `/dev/ttyACM0` | Serial port for the left SO101 arm |
| `--left-id` | `L1` | Left arm follower ID |
| `--right-port` | `/dev/ttyACM1` | Serial port for the right SO101 arm |
| `--right-id` | `L1` | Right arm follower ID |
| `--ip-address` | `None` | Meta Quest IP address (auto-discovered if omitted) |

**Controls:**

| Input | Action |
|---|---|
| **Hold LEFT + RIGHT grip** | Activate dual-arm teleoperation (dead man's switch) |
| **Move left controller** | Left arm end-effector follows |
| **Move right controller** | Right arm end-effector follows |
| **Left / right trigger (hold)** | Close corresponding gripper |
| **Button A** | Enable / disable both arms |
| **Button B** | Send both arms to home position |
| **Ctrl+C** | Exit |

#### Example 9 – Dual-arm teleoperation with Neuracore data collection

Same as example 8 but streams joint positions, gripper state, and two RGB camera frames to [Neuracore](https://neuracore.ai) for dataset collection. Recording episodes is controlled directly from the Quest controller.

```bash
python examples/9_collect_dual_teleop_data.py \
  --left-port /dev/ttyACM0 --left-id L1 \
  --right-port /dev/ttyACM1 --right-id L1 \
  --ip-address <quest-ip> \
  --dataset-name so101-dual-demo
```

**Prerequisites:** A Neuracore account. The script calls `nc.login()` on startup — set your credentials beforehand (see [Neuracore docs](https://neuracore.ai/docs)).

**What it does:**
- Connects to Neuracore, creates (or reuses) the named dataset, and streams live data at 30 Hz.
- Logs left and right joint positions, gripper states, and two RGB camera frames (`rgb`, `rgb_2`) per episode.
- Recording is controlled via **Button X** on the Quest — no web UI required.
- Press **Ctrl+C** to shut down cleanly; any active recording is stopped automatically.

**Optional arguments:**

| Flag | Default | Description |
|---|---|---|
| `--left-port` | `/dev/ttyACM0` | Serial port for the left SO101 arm |
| `--left-id` | `L1` | Left arm follower ID |
| `--right-port` | `/dev/ttyACM1` | Serial port for the right SO101 arm |
| `--right-id` | `L1` | Right arm follower ID |
| `--ip-address` | `None` | Meta Quest IP address (auto-discovered if omitted) |
| `--dataset-name` | `so101-dual-teleop-<timestamp>` | Dataset name in Neuracore |

**Controls** (all example 8 controls, plus):

| Input | Action |
|---|---|
| **Button X** | Start / stop Neuracore recording |

## Project structure

```
example_so101/
├── examples/
│   ├── 1_leader_arm_teleop_so101.py              # SO101 leader → SO101 follower teleop (URDF or real robot)
│   ├── 2_collect_teleop_data_with_neuracore.py   # Teleop with Neuracore data collection
│   ├── 4_rollout_neuracore_policy.py             # Policy rollout + leader teleop + Viser
│   ├── 5_rollout_neuracore_policy_minimal.py     # Minimal policy rollout (terminal only)
│   ├── 6_visualize_policy_from_dataset.py        # Policy preview from dataset (Viser only)
│   ├── 7_tune_quest_teleop_params.py             # Meta Quest IK teleop + live parameter tuning (single arm)
│   ├── 8_dual_arm_teleop.py                      # Dual-arm Meta Quest teleoperation (10-DOF IK)
│   ├── 9_collect_dual_teleop_data.py             # Dual-arm teleop + Neuracore data collection
│   └── common/                                   # Config, data manager, visualizer, threads, STS3215 driver
├── tests/                                        # Unit tests (no hardware required)
├── meta_quest_teleop/                            # MetaQuestReader package (install with pip install -e)
├── pink_ik_solver.py                             # Generic Pink IK solver (used by examples 7–9)
├── vectorised_posture_task.py                    # Pink posture task helper (used by pink_ik_solver)
├── so101_controller.py                           # SO101 single-arm follower controller
├── so101_dual_controller.py                      # SO101 dual-arm follower controller (used by examples 8–9)
├── so101_description/urdf/                       # SO101 URDF (minimal + README for official mesh)
├── environment.yaml
└── README.md
```

## Troubleshooting

- **"Calibration file not found"**: Run `lerobot-calibrate` for the leader with the same `--teleop.id` you pass as `--leader-id`. The calibration JSON is saved to `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<id>.json`.
- **Follower not moving**: Ensure the robot is **enabled** in the GUI and the follower is on the correct `--follower-port`.
- **Wrong port**: Use `ls /dev/tty*` (or `lerobot-find-port` if lerobot is installed) to identify ports; leader and follower must be on different ports when using two arms.
- **Motor direction opposite**: Some setups need per-motor direction or recalibration; see [LeRobot SO101 docs](https://huggingface.co/docs/lerobot/so101).

## Safety

- This software drives a physical robot. Keep a safe workspace and be ready to stop (disable in GUI or Ctrl+C).
- Start with the robot **disabled** and only enable after confirming the leader pose is safe.

## License

See LICENSE file.
