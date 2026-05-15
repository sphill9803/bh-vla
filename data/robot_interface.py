"""
bh-VLA Robot Hardware Interface

Provides:
    - RobotConfig: configuration dataclass for SO-101 robot hardware
    - FeetechServoBus: communication layer for Feetech bus servos
    - CameraCapture: multi-camera capture abstraction
    - SO101Robot: complete robot control interface with teleoperation and
      policy execution loops
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ====================================================================
# Configuration
# ====================================================================

@dataclass
class RobotConfig:
    """Configuration for the SO-101 / ALOHA robot hardware interface.

    SO-101 robot specification:
        - 6 DOF leader arm (teleoperation input)
        - 6 DOF follower arm (policy output)
        - 3 cameras (1 main + 2 supplementary)
        - Feetech STS3215 bus servos

    Attributes:
        leader_port: USB serial port for the leader arm. Default /dev/ttyACM0.
        follower_port: USB serial port for the follower arm. Default /dev/ttyACM1.
        motor_baudrate: Servo communication baudrate. Default 1000000 (1 Mbps).
        camera_fps: Camera frame rate. Default 30.
        camera_width: Camera width in pixels. Default 224.
        camera_height: Camera height in pixels. Default 224.
        num_cameras: Number of cameras. Default 3 (main + 2 supplementary).
        teleop_frequency: Teleoperation loop frequency in Hz. Default 50.
        joint_limits: Per-joint angle limits in radians.
        motor_ids: Feetech servo motor IDs per joint (left arm first, then right).
    """
    # USB ports
    leader_port: str = "/dev/ttyACM0"
    follower_port: str = "/dev/ttyACM1"

    # Motor bus protocol
    motor_baudrate: int = 1000000  # 1 Mbps

    # Camera configuration
    camera_fps: int = 30
    camera_width: int = 224
    camera_height: int = 224
    num_cameras: int = 3  # ALOHA uses 3 cameras (main + 2 supplementary)

    # Teleoperation parameters
    teleop_frequency: int = 50  # Hz
    joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "base_pan": (-2.617, 2.617),     # ±150 degrees
        "shoulder_pan": (-2.094, 2.094),  # ±120 degrees
        "shoulder_lift": (-2.094, 2.094), # ±120 degrees
        "elbow_flex": (-2.268, 2.268),    # ±130 degrees
        "wrist_flex": (-1.745, 1.745),    # ±100 degrees
        "wrist_roll": (-2.617, 2.617),    # ±150 degrees
        "gripper": (0.0, 1.0),            # Open (0) to closed (1)
    })

    # Motor IDs for Feetech STS3215 servos
    # Format: [left arm joints, right arm joints]
    motor_ids: List[int] = field(default_factory=lambda: [
        11, 12, 13, 14, 15, 16,   # Left arm (6 joints)
        17, 18, 19, 20, 21, 22,   # Right arm (6 joints)
        23, 24,                      # Grippers
    ])

    # Data directory for collected episodes
    data_dir: str = "./data"


# ====================================================================
# Feetech Servo Bus Communication
# ====================================================================

class FeetechServoBus:
    """Low-level communication with Feetech STS3215 bus servos.

    The Feetech bus protocol uses a custom binary frame format over serial
    to read and write servo registers. This class abstracts the protocol
    details.

    Protocol summary:
        Frame: [Sync (0xFF, 0xFF)] [Length] [ID] [Instruction] [Parameters...] [CRC]

    Supported operations:
        - Read a register (position, velocity, current, etc.)
        - Write a register (position, velocity limit, torque enable)
        - Sync write (set positions for multiple servos at once)
        - Ping (verify servo connectivity)
        - Reset (factory reset a servo)

    Attributes:
        port: Serial port path.
        baudrate: Communication baudrate.
        servos: Dict of servo ID -> servo properties.
    """

    # Feetech protocol constants
    SYNC_WRITE = 0x31
    READ = 0x02
    WRITE = 0x03
    PING = 0x08
    RESET = 0x06
    SYNC_BROADCAST = 0xFE

    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 1000000):
        """
        Args:
            port: Serial port path (e.g., /dev/ttyACM0).
            baudrate: Communication baudrate.
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = None  # Serial connection (lazy init)
        self.servos: Dict[int, Dict[str, Any]] = {}
        self.connected = False

    def connect(self) -> bool:
        """Establish serial connection to the servo bus.

        Steps:
            1. Open the serial port.
            2. Ping all motor IDs to verify connectivity.
            3. Store servo information (model, firmware version, etc.).

        Returns:
            True if connection and ping successful, False otherwise.
        """
        try:
            import serial
        except ImportError:
            raise ImportError("pyserial is required for servo communication. Install with: pip install pyserial")

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
        )

        # Ping all known servo IDs
        for mot_id in self.servos.keys():
            if self.ping(mot_id):
                self.servos[mot_id]["connected"] = True
            else:
                self.servos[mot_id]["connected"] = False

        self.connected = True
        print(f"[ServoBus] Connected to {self.port} at {self.baudrate} baud")
        return True

    def disconnect(self) -> None:
        """Close the serial connection and disable torque on all servos."""
        if self.ser is not None:
            # Disable torque on all servos (safety)
            for mot_id in list(self.servos.keys()):
                self.disable_torque(mot_id)
            self.ser.close()
            self.ser = None
            self.connected = False
        print("[ServoBus] Disconnected")

    def ping(self, motor_id: int) -> bool:
        """Ping a servo to verify it is connected.

        Args:
            motor_id: Feetech servo ID to ping.

        Returns:
            True if the servo responds, False otherwise.
        """
        if self.ser is None:
            return False

        # Build ping frame
        frame = self._build_frame(motor_id, self.PING)
        self.ser.write(frame)
        response = self.ser.read(8)  # Ping response is 8 bytes

        return len(response) >= 4 and response[1] == len(response) - 1

    def read_register(self, motor_id: int, start_addr: int, length: int) -> np.ndarray:
        """Read bytes from a servo register.

        Args:
            motor_id: Servo ID.
            start_addr: Starting register address.
            length: Number of bytes to read.

        Returns:
            Array of register values.
        """
        if self.ser is None:
            raise RuntimeError("Servo bus not connected!")

        # Build read frame
        frame = bytearray([0xFF, 0xFF])
        frame.append(length + 2)  # Length
        frame.append(motor_id)
        frame.append(self.READ)
        frame.append(start_addr & 0xFF)
        frame.append(length & 0xFF)
        frame.append(self._crc(frame))
        frame = bytes(frame)

        self.ser.write(frame)
        response = self.ser.read(6 + length)

        if len(response) < 6:
            return np.zeros(length, dtype=np.int16)

        return np.frombuffer(response[5:5 + length], dtype=np.int16)

    def write_register(self, motor_id: int, start_addr: int, data: np.ndarray) -> None:
        """Write bytes to a servo register.

        Args:
            motor_id: Servo ID.
            start_addr: Starting register address.
            data: Data bytes to write.
        """
        if self.ser is None:
            raise RuntimeError("Servo bus not connected!")

        length = len(data)
        frame = bytearray([0xFF, 0xFF])
        frame.append(length + 3)
        frame.append(motor_id)
        frame.append(self.WRITE)
        frame.append(start_addr & 0xFF)
        frame.extend(data.astype(np.int8).tobytes())
        frame.append(self._crc(frame))
        frame = bytes(frame)

        self.ser.write(frame)

    def sync_write(self, motor_ids: List[int], start_addr: int, data_arrays: List[np.ndarray]) -> None:
        """Sync-write positions to multiple servos simultaneously.

        This is critical for real-time robot control — all servos receive
        their position commands in the same packet for precise synchronization.

        Args:
            motor_ids: List of servo IDs to control.
            start_addr: Starting register address (e.g., 64 for goal position).
            data_arrays: List of data arrays (one per motor) with position values.
        """
        if self.ser is None:
            raise RuntimeError("Servo bus not connected!")

        # Build sync-write frame
        frame = bytearray([0xFF, 0xFF])
        param_len = 4 * len(motor_ids) + 2  # ID(1) + Data(2) + Addr(1) per servo, plus 1 byte addr
        frame.append(param_len + 1)  # Length (1 byte for instruction)
        frame.append(self.SYNC_BROADCAST)
        frame.append(start_addr & 0xFF)

        for mot_id, data in zip(motor_ids, data_arrays):
            frame.append(mot_id & 0xFF)
            frame.extend(data.astype(np.int16).tobytes())

        frame.append(self._crc(frame))
        self.ser.write(bytes(frame))

    def set_position(self, motor_id: int, position: int) -> None:
        """Set the target position of a single servo.

        Args:
            motor_id: Servo ID.
            position: Goal position in servo counts (typically 0-4095 for STS3215).
        """
        # STS3215 goal position register starts at address 64
        pos_bytes = np.array([position & 0xFF, (position >> 8) & 0xFF], dtype=np.int16)
        self.write_register(motor_id, 64, pos_bytes)

    def set_positions(self, positions: np.ndarray) -> None:
        """Set positions for all servos in a sync-write packet.

        Args:
            positions: Array of goal positions (one per servo), dtype int16.
        """
        data_arrays = [np.array([p & 0xFF, (p >> 8) & 0xFF], dtype=np.int16) for p in positions]
        self.sync_write(list(self.servos.keys()), 64, data_arrays)

    def get_position(self, motor_id: int) -> int:
        """Read the current position of a servo.

        Args:
            motor_id: Servo ID.

        Returns:
            Current position in servo counts.
        """
        pos = self.read_register(motor_id, 140, 2)  # Present position at addr 140
        if len(pos) >= 2:
            return int(np.int16(pos[0]) | (pos[1] << 8))
        return 0

    def get_all_positions(self) -> np.ndarray:
        """Read positions from all connected servos.

        Returns:
            Array of current positions (one per servo).
        """
        positions = []
        for mot_id in sorted(self.servos.keys()):
            pos = self.get_position(mot_id)
            positions.append(pos)
        return np.array(positions, dtype=np.int16)

    def enable_torque(self, motor_id: int) -> None:
        """Enable torque (motor power) for a servo.

        Args:
            motor_id: Servo ID.
        """
        self.write_register(motor_id, 62, np.array([1], dtype=np.int16))  # Torque enable at addr 62

    def disable_torque(self, motor_id: int) -> None:
        """Disable torque (motor power) for safety.

        Args:
            motor_id: Servo ID.
        """
        self.write_register(motor_id, 62, np.array([0], dtype=np.int16))

    def _crc(self, data: bytearray) -> int:
        """Compute the CRC checksum for a Feetech frame.

        Args:
            data: Frame data without CRC byte.

        Returns:
            CRC byte.
        """
        crc = 0
        for byte in data:
            crc ^= byte
        for _ in range(7):
            if crc & 0x01:
                crc = (crc >> 1) ^ 0x5D
            else:
                crc >>= 1
        return (~crc) & 0xFF

    def _build_frame(self, motor_id: int, instruction: int) -> bytes:
        """Build a complete Feetech protocol frame.

        Args:
            motor_id: Servo ID.
            instruction: Instruction byte (PING, READ, WRITE, etc.).

        Returns:
            Complete frame as bytes.
        """
        frame = bytearray([0xFF, 0xFF])
        frame.append(2)  # Length
        frame.append(motor_id)
        frame.append(instruction)
        frame.append(self._crc(frame))
        return bytes(frame)


# ====================================================================
# Camera Capture
# ====================================================================

class CameraCapture:
    """Multi-camera capture abstraction for ALOHA robots.

    Manages the lifecycle of multiple USB cameras: opening, capturing,
    and closing camera streams. Supports both single and multi-camera
    configurations.

    Attributes:
        num_cameras: Number of cameras to manage.
        camera_width: Capture width in pixels.
        camera_height: Capture height in pixels.
        camera_fps: Target frame rate.
        video_ids: List of video device IDs (0-indexed for cv2.VideoCapture).
        cameras: List of cv2.VideoCapture objects.
    """

    def __init__(
        self,
        num_cameras: int = 3,
        camera_width: int = 224,
        camera_height: int = 224,
        camera_fps: int = 30,
        video_ids: Optional[List[int]] = None,
    ):
        """
        Args:
            num_cameras: Number of cameras.
            camera_width: Capture width.
            camera_height: Capture height.
            camera_fps: Target frame rate.
            video_ids: Explicit video device IDs (auto-detects if None).
        """
        self.num_cameras = num_cameras
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.video_ids = video_ids if video_ids is not None else list(range(num_cameras))

        self.cameras: List[Any] = []
        self.is_open: List[bool] = []

    def open(self) -> bool:
        """Open all cameras.

        Returns:
            True if all cameras opened successfully, False otherwise.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required for camera capture. Install with: pip install opencv-python")

        success = True
        for i, video_id in enumerate(self.video_ids):
            cap = cv2.VideoCapture(video_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                self.cameras.append(cap)
                self.is_open.append(True)
                print(f"[Camera {i}] Opened video device {video_id}")
            else:
                self.cameras.append(None)
                self.is_open.append(False)
                print(f"[Camera {i}] Failed to open video device {video_id}")
                success = False

        return success

    def capture(self) -> Dict[str, np.ndarray]:
        """Capture a frame from each camera.

        Returns:
            Dict mapping camera names to numpy arrays (H, W, C).
        """
        frames: Dict[str, np.ndarray] = {}
        for i, cap in enumerate(self.cameras):
            name = f"camera_{i}"
            if cap is not None and self.is_open[i]:
                ret, frame = cap.read()
                if ret:
                    frames[name] = frame
                else:
                    frames[name] = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            else:
                frames[name] = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        return frames

    def close(self) -> None:
        """Close all camera streams."""
        for i, cap in enumerate(self.cameras):
            if cap is not None and cap.isOpened():
                cap.release()
                self.is_open[i] = False
        print("[CameraCapture] All cameras closed")

    def __del__(self) -> None:
        self.close()


# ====================================================================
# SO-101 Robot Interface
# ====================================================================

class SO101Robot:
    """Complete robot interface for SO-101 / ALOHA hardware.

    This class orchestrates:
        - Servo bus communication (``FeetechServoBus``)
        - Multi-camera capture (``CameraCapture``)
        - Teleoperation data collection
        - Real-time policy execution

    The SO-101 robot has:
        - 6-DOF leader arm (teleoperation input via USB)
        - 6-DOF follower arm (policy output via USB)
        - 3 cameras (1 main + 2 supplementary)
        - Feetech STS3215 bus servos

    Attributes:
        config: RobotConfig instance.
        leader_servo_bus: FeetechServoBus for the leader arm.
        follower_servo_bus: FeetechServoBus for the follower arm.
        camera_capture: CameraCapture instance for multi-camera handling.
        connected: Whether the robot hardware is connected.
        teleop_enabled: Whether teleoperation is active.
        policy: The loaded policy for inference.
        action_history: List of executed actions for logging.
    """

    def __init__(self, config: Optional[RobotConfig] = None):
        """
        Args:
            config: Robot configuration. Creates defaults if None.
        """
        self.config = config or RobotConfig()
        self.connected = False
        self.teleop_enabled = False

        # Create hardware interfaces
        self.leader_servo_bus = FeetechServoBus(
            port=self.config.leader_port,
            baudrate=self.config.motor_baudrate,
        )
        self.leader_servo_bus.servos = {mid: {} for mid in self.config.motor_ids[:7]}

        self.follower_servo_bus = FeetechServoBus(
            port=self.config.follower_port,
            baudrate=self.config.motor_baudrate,
        )
        self.follower_servo_bus.servos = {mid: {} for mid in self.config.motor_ids[7:]}

        self.camera_capture = CameraCapture(
            num_cameras=self.config.num_cameras,
            camera_width=self.config.camera_width,
            camera_height=self.config.camera_height,
            camera_fps=self.config.camera_fps,
        )

        self.policy: Any = None
        self.teleop_loop = None
        self.inference_loop = None
        self.action_history: List[np.ndarray] = []

    def connect(self) -> bool:
        """Connect to all robot hardware.

        Steps:
            1. Initialize USB serial connections for both arms.
            2. Verify motor IDs and baudrates by pinging servos.
            3. Test camera connections.
            4. Calibrate both arms to zero position.

        Returns:
            True if all hardware connected successfully.
        """
        success = True

        # Connect leader arm
        print("Connecting to leader arm...")
        for mot_id in self.config.motor_ids[:7]:
            self.leader_servo_bus.servos[mot_id] = {"connected": False}
        if not self.leader_servo_bus.connect():
            print("Warning: Leader arm connection failed (may be disconnected during training)")
            success = False
        else:
            # Ping servos
            for mot_id in self.config.motor_ids[:7]:
                if self.leader_servo_bus.ping(mot_id):
                    self.leader_servo_bus.servos[mot_id]["connected"] = True
                    print(f"  Leader servo {mot_id}: OK")
                else:
                    print(f"  Leader servo {mot_id}: NOT FOUND")
                    success = False

        # Connect follower arm
        print("Connecting to follower arm...")
        for mot_id in self.config.motor_ids[7:]:
            self.follower_servo_bus.servos[mot_id] = {"connected": False}
        if not self.follower_servo_bus.connect():
            print("Warning: Follower arm connection failed")
            success = False
        else:
            for mot_id in self.config.motor_ids[7:]:
                if self.follower_servo_bus.ping(mot_id):
                    self.follower_servo_bus.servos[mot_id]["connected"] = True
                    print(f"  Follower servo {mot_id}: OK")
                else:
                    print(f"  Follower servo {mot_id}: NOT FOUND")
                    success = False

        # Initialize cameras
        print("Initializing cameras...")
        if not self.camera_capture.open():
            print("Warning: Some cameras failed to open")
            success = False
        else:
            open_count = sum(1 for s in self.camera_capture.is_open if s)
            print(f"  {open_count}/{self.config.num_cameras} cameras open")

        self.connected = success
        if success:
            print("Robot connected successfully!")
        else:
            print("Robot partially connected — some hardware failed.")

        return success

    def disconnect(self) -> None:
        """Disconnect from all robot hardware safely."""
        # Disable torque on all servos (safety)
        for bus in [self.leader_servo_bus, self.follower_servo_bus]:
            if bus.connected:
                for mot_id in list(bus.servos.keys()):
                    try:
                        bus.disable_torque(mot_id)
                    except Exception:
                        pass

        # Close cameras
        self.camera_capture.close()

        # Close servo buses
        self.leader_servo_bus.disconnect()
        self.follower_servo_bus.disconnect()

        self.connected = False
        print("Robot disconnected. All hardware safe.")

    def get_observation(self) -> Dict[str, Any]:
        """Capture a complete observation from the robot.

        Returns a dictionary containing:
            - images: Dict of camera name → numpy array (H, W, C) uint8
            - state: Robot joint positions (numpy array, 28-dim for ALOHA)
            - timestamps: Dict of camera timestamps
            - leader_positions: Leader arm servo positions (for teleop)

        Returns:
            Complete observation dictionary.
        """
        if not self.connected:
            raise RuntimeError("Robot not connected! Call connect() first.")

        # Capture images from all cameras
        images = self.camera_capture.capture()

        # Get follower arm positions (current state)
        follower_positions = self.follower_servo_bus.get_all_positions() if self.follower_servo_bus.connected else np.zeros(6)

        # Build 28-dim state vector:
        # [left_arm_6j, left_gripper, right_arm_6j, right_gripper, extra_8j]
        state = np.zeros(28, dtype=np.float32)
        state[0:6] = follower_positions[:6].astype(np.float32)
        state[6] = follower_positions[6] if len(follower_positions) > 6 else 0.0

        # Get leader arm positions for teleop
        leader_positions = self.leader_servo_bus.get_all_positions() if self.leader_servo_bus.connected else np.zeros(6)

        # Camera timestamps
        timestamps = {}
        for name in images.keys():
            timestamps[name] = time.time()

        return {
            "images": images,
            "state": state,
            "timestamp": time.time(),
            "leader_positions": leader_positions,
            "timestamps": timestamps,
        }

    def send_action(self, action: np.ndarray) -> None:
        """Send an action to the follower arm servos.

        The action vector is interpreted as:
            - First 7 values: left arm joint positions (normalized to servo range)
            - Last 7 values: right arm joint positions (including gripper)

        Args:
            action: Action vector of shape (14,) containing normalized positions.
        """
        if not self.follower_servo_bus.connected:
            raise RuntimeError("Follower arm not connected! Cannot send action.")

        # Convert normalized actions to servo counts (0-4095 for STS3215)
        positions = np.zeros(len(self.follower_servo_bus.servos), dtype=np.int16)
        for i in range(len(positions)):
            if i < 6:
                # Joint positions: map from [-1, 1] to [0, 4095]
                normalized = float(action[i]) if i < len(action) else 0.0
                positions[i] = int((normalized + 1.0) * 2047.5)
                # Clamp to servo range
                positions[i] = max(0, min(4095, positions[i]))
            elif i == 6:
                # Gripper: map from [0, 1] to [0, 4095]
                positions[i] = int(float(action[i]) * 4095) if i < len(action) else 2048

        # Send all positions via sync-write (atomic update)
        self.follower_servo_bus.set_positions(positions)

        # Record action history
        self.action_history.append(action.copy())

    def calibrate(self, arm: str = "both") -> None:
        """Calibrate the robot arms.

        Calibration aligns the joint angles so that:
            - Both arms report the same angles when in the same position
            - The policy trained on one robot works on another

        Steps:
            1. Move both arms to a known calibration pose (usually home/zero).
            2. Record the servo readings as the "zero" reference.
            3. Store the calibration offsets.

        Args:
            arm: Which arm to calibrate ("leader", "follower", or "both").
        """
        arms_to_cal = []
        if arm in ("leader", "both"):
            arms_to_cal.append(("leader", self.leader_servo_bus))
        if arm in ("follower", "both"):
            arms_to_cal.append(("follower", self.follower_servo_bus))

        for name, bus in arms_to_cal:
            if not bus.connected:
                print(f"  Skipping {name} arm calibration (not connected)")
                continue

            print(f"  Calibrating {name} arm...")
            # Move to home position (all joints at center)
            zero_positions = np.array([2048] * len(bus.servos), dtype=np.int16)
            bus.set_positions(zero_positions)
            time.sleep(1.0)  # Wait for servos to reach home

            # Record zero offsets
            zero_readings = bus.get_all_positions()
            bus.zero_offsets = zero_readings.astype(np.float32)

            print(f"  {name} arm calibrated with zero offsets: {zero_readings}")

    def collect_teleop_data(
        self,
        episode_name: str = "demo",
        max_episodes: int = 1,
        max_frames_per_episode: int = 10000,
    ) -> str:
        """Collect teleoperation data for training.

        The human operator moves the leader arm while the follower arm mirrors
        the movements. All sensor data is recorded and saved as numpy arrays.

        Data collected per frame:
            - Images from all cameras (at 30fps)
            - Joint positions of the follower arm
            - Timestamps

        The collected data is saved in the format:
            data/{task_name}/episode_{NNNN}/
            ├── images.npy       (N, H, W, 3) uint8
            ├── states.npy       (N, 28) float32
            ├── actions.npy      (N, 28) float32
            └── metadata.json

        Args:
            episode_name: Name prefix for the data episode.
            max_episodes: Maximum number of episodes to collect.
            max_frames_per_episode: Maximum frames per episode.

        Returns:
            Path to the saved dataset directory.
        """
        if not self.connected:
            raise RuntimeError("Robot not connected! Call connect() first.")

        import json

        data_dir = self.config.data_dir
        task_name = input("Enter task name: ").strip() or "demo"
        instruction = input("Enter language instruction: ").strip() or "pick up the object"

        episode_path = os.path.join(data_dir, task_name, episode_name)
        os.makedirs(os.path.join(episode_path, "images"), exist_ok=True)

        all_frames: List[np.ndarray] = []
        all_states: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []

        print(f"\n=== Collecting episode: {episode_name} ===")
        print("Move the leader arm to start.")
        print("Press Enter to START recording...")
        input()

        start_time = time.time()
        frame_count = 0

        print("Recording... (move the leader arm)")
        print("Press Enter to stop recording when done.")

        try:
            while frame_count < max_frames_per_episode:
                # Get observation
                obs = self.get_observation()

                # Extract frames
                for cam_name, frame in obs["images"].items():
                    all_frames.append(frame)

                # Record state and action (teleop: action = follower state)
                all_states.append(obs["state"].copy())
                all_actions.append(obs["state"].copy())  # Teleop: action mirrors state

                frame_count += 1

                # Record frame to disk (every 3 frames to save I/O)
                if frame_count % 3 == 0 and all_frames:
                    idx = frame_count // 3 - 1
                    frame_path = os.path.join(episode_path, "images", f"frame_{idx:06d}.npy")
                    np.save(frame_path, all_frames[-1])

                # Stop condition
                if input() == "":
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        duration = time.time() - start_time

        # Save as numpy arrays
        if all_frames:
            frames_array = np.array(all_frames, dtype=np.uint8)
        else:
            frames_array = np.zeros((0, self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8)

        states_array = np.array(all_states, dtype=np.float32) if all_states else np.zeros((0, 28), dtype=np.float32)
        actions_array = np.array(all_actions, dtype=np.float32) if all_actions else np.zeros((0, 28), dtype=np.float32)

        np.save(os.path.join(episode_path, "images.npy"), frames_array)
        np.save(os.path.join(episode_path, "states.npy"), states_array)
        np.save(os.path.join(episode_path, "actions.npy"), actions_array)

        # Save metadata
        metadata = {
            "task_name": task_name,
            "instruction": instruction,
            "num_frames": len(all_frames),
            "duration_seconds": duration,
            "fps": len(all_frames) / max(duration, 0.001),
            "timestamp": time.time(),
            "camera_config": {
                "num_cameras": self.config.num_cameras,
                "width": self.config.camera_width,
                "height": self.config.camera_height,
                "fps": self.config.camera_fps,
            },
            "robot_config": {
                "leader_port": self.config.leader_port,
                "follower_port": self.config.follower_port,
                "motor_baudrate": self.config.motor_baudrate,
            },
        }
        with open(os.path.join(episode_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nEpisode saved to {episode_path}")
        print(f"  Frames: {len(all_frames)}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  FPS: {len(all_frames) / max(duration, 0.001):.1f}")

        return episode_path

    def run_policy_loop(
        self,
        checkpoint_path: str,
        language: str = "pick up the object",
        flow_steps: int = 50,
    ) -> None:
        """Run the loaded policy on the robot in a real-time loop.

        This is the main inference loop:
            1. Load the policy checkpoint
            2. Continuously:
                a. Get observation from cameras
                b. Run policy forward pass
                c. Execute the predicted action on the follower arm
                d. Repeat at the configured teleop_frequency (default 50Hz)

        Args:
            checkpoint_path: Path to the trained policy checkpoint.
            language: Default language instruction.
            flow_steps: Number of flow matching steps (for pi0.5 policy).
        """
        if not self.connected:
            raise RuntimeError("Robot not connected! Call connect() first.")

        # Import the policy
        sys_path = os.path.dirname(os.path.abspath(__file__))
        bh_vla_dir = os.path.dirname(os.path.dirname(sys_path))
        if bh_vla_dir not in __import__("sys").path:
            __import__("sys").path.insert(0, bh_vla_dir)

        from policies.act import ACTPolicy, ACTConfig
        from policies.pi05 import Pi05Policy, Pi05Config

        # Detect policy mode from checkpoint filename
        if "pi05" in checkpoint_path.lower() or "pi_05" in checkpoint_path.lower():
            policy_config = Pi05Config(flow_steps=flow_steps)
            self.policy = Pi05Policy(policy_config)
        else:
            policy_config = ACTConfig()
            self.policy = ACTPolicy(policy_config)

        self.policy.load_checkpoint(checkpoint_path)
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.policy = self.policy.to(device)

        # Enable torque on follower arm
        for mot_id in self.follower_servo_bus.servos.keys():
            try:
                self.follower_servo_bus.enable_torque(mot_id)
            except Exception:
                pass

        print("\n" + "=" * 60)
        print("  Starting inference loop")
        print(f"  Policy: {type(self.policy).__name__}")
        print(f"  Language: {language}")
        print(f"  Device: {device}")
        print(f"  Target frequency: {self.config.teleop_frequency} Hz")
        print("  Press Ctrl+C to stop")
        print("=" * 60)

        step = 0
        action_history: List[np.ndarray] = []

        dt = 1.0 / self.config.teleop_frequency
        loop_start = time.time()

        try:
            while True:
                # Get observation
                obs = self.get_observation()

                # Process images for the policy
                images_list = []
                for cam_name in sorted(obs["images"].keys()):
                    frame = obs["images"][cam_name]
                    frame = frame.astype(np.float32) / 255.0
                    if frame.ndim == 3:
                        frame = np.transpose(frame, (2, 0, 1))
                    images_list.append(frame)

                images_tensor = np.stack(images_list, axis=0)  # (num_cam, C, H, W)

                # Convert to tensor and move to device
                import torch
                images_t = torch.tensor(images_tensor, dtype=torch.float32).unsqueeze(0).to(device)  # (1, num_cam, C, H, W)
                state_t = torch.tensor(obs["state"], dtype=torch.float32).unsqueeze(0).to(device)

                # Run policy forward pass
                action = self.policy.predict_action(images_t, language, state_t)
                action_np = action.cpu().numpy()

                # Send action to the robot
                self.send_action(action_np)

                # Track action history
                action_history.append(action_np)

                # Log every 20 steps
                step += 1
                if step % 20 == 0:
                    elapsed = time.time() - loop_start
                    fps = step / max(elapsed, 0.001)
                    print(f"  Step {step:5d} | FPS: {fps:6.1f} | "
                          f"Action: [{', '.join([f'{v:8.3f}' for v in action_np[:7]])}]")

                loop_start = time.time()

                # Maintain target frequency
                elapsed = time.time() - loop_start
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\nStopped after {step} steps.")
        finally:
            # Disable torque on all servos (safety)
            for mot_id in self.follower_servo_bus.servos.keys():
                try:
                    self.follower_servo_bus.disable_torque(mot_id)
                except Exception:
                    pass

            # Save action history
            if action_history:
                history_path = os.path.join(bh_vla_dir, "outputs", "action_history.npy")
                os.makedirs(os.path.dirname(history_path), exist_ok=True)
                np.save(history_path, np.array(action_history))
                print(f"Action history saved to {history_path}")

    def run_test_loop(self, num_steps: int = 100) -> None:
        """Run a test inference loop without real robot hardware.

        Creates synthetic observations and runs the policy forward pass.
        Useful for verifying the policy loads and runs correctly before
        connecting to the real robot.

        Args:
            num_steps: Number of inference steps to run.
        """
        if self.policy is None:
            raise RuntimeError("No policy loaded. Load a checkpoint first.")

        import torch

        print(f"\nRunning test inference for {num_steps} steps...")
        print("(Synthetic data — no real robot hardware)\n")

        device = self.policy.device
        batch_images = torch.randn(
            1, self.config.num_cameras, 3,
            self.config.camera_height, self.config.camera_width,
            device=device,
        )

        for step in range(num_steps):
            action = self.policy.predict_action(batch_images, "pick up the object")
            action_np = action.cpu().numpy()

            if step % 10 == 0:
                print(f"  Step {step + 1:3d}/{num_steps}: "
                      f"Action = [{', '.join([f'{v:.3f}' for v in action_np])}]")

            # Update synthetic images
            batch_images = torch.randn(
                1, self.config.num_cameras, 3,
                self.config.camera_height, self.config.camera_width,
                device=device,
            )
            time.sleep(0.015)  # Simulate 50Hz loop

        print(f"\nTest inference complete ({num_steps} steps).")
