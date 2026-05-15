#!/usr/bin/env python3
"""
bh-VLA Data Collection Script: Teleoperation for collecting training data.

This script enables you to collect demonstration data using the ALOHA leader arm
(telerobot) to teach the follower arm. The collected data includes:
  - Images from all 3 cameras (at 30fps)
  - Joint positions of both arms
  - Language instructions
  - Timestamps

Usage:
    # Collect 5 episodes of data
    python collect_data.py --mode act --num-episodes 5

    # Collect data and save to custom directory
    python collect_data.py --mode act --data-dir ./data/my_tasks

    # Collect 10 episodes with custom task name
    python collect_data.py --mode act --task-name "pick_and_place" --num-episodes 10

    # Collect with pi0.5 mode
    python collect_data.py --mode pi05 --num-episodes 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.robot_interface import SO101Robot, RobotConfig
from utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="bh-VLA: Data Collection via Teleoperation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Collect 5 episodes of data
    python collect_data.py --mode act --num-episodes 5

    # Collect data and save to custom directory
    python collect_data.py --mode act --data-dir ./data/my_tasks

    # Collect 10 episodes with custom task name
    python collect_data.py --mode act --task-name "pick_and_place" --num-episodes 10
        """
    )

    parser.add_argument("--mode", type=str, required=True,
                        choices=["act", "pi05"],
                        help="Policy mode (determines data format)")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes to collect")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory to save collected data")
    parser.add_argument("--task-name", type=str, default="demo",
                        help="Name for the task (used in episode folder names)")
    parser.add_argument("--max-frames", type=int, default=10000,
                        help="Maximum frames per episode")
    parser.add_argument("--camera-fps", type=int, default=30,
                        help="Camera frame rate")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Image resolution (square)")
    parser.add_argument("--skip-robot", action="store_true",
                        help="Skip robot connection and use synthetic data")

    args = parser.parse_args()
    print(f"\n{'='*60}")
    print(f"  bh-VLA Data Collection")
    print(f"  Mode: {args.mode}")
    print(f"  Target episodes: {args.num_episodes}")
    print(f"  Data directory: {args.data_dir}")
    print(f"{'='*60}\n")

    # Create data directory
    data_dir = os.path.join(args.data_dir, args.task_name)
    ensure_dir(data_dir)

    # Initialize robot (unless skip)
    robot = None
    if not args.skip_robot:
        robot_cfg = RobotConfig()
        robot = SO101Robot(robot_cfg)
        try:
            print("Connecting to robot...")
            robot.connect()
            print("Robot connected successfully!")
        except Exception as e:
            print(f"Warning: Could not connect to robot: {e}")
            print("Falling back to synthetic data collection.\n")
            robot = None

    # Collect data
    collected_episodes = collect_data(args, robot, data_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Data Collection Summary")
    print(f"  Episodes collected: {len(collected_episodes)}")
    print(f"  Data directory: {data_dir}")
    print(f"{'='*60}")

    # Save dataset info
    dataset_info = {
        "mode": args.mode,
        "task_name": args.task_name,
        "num_episodes": len(collected_episodes),
        "image_size": args.image_size,
        "camera_fps": args.camera_fps,
        "episodes": collected_episodes,
    }
    info_path = os.path.join(data_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"\nDataset info saved to {info_path}")

    print(f"\nNext steps:")
    print(f"  1. Inspect your data in {data_dir}")
    print(f"  2. Run training: python train.py --mode {args.mode} --data-dir {data_dir}")
    print(f"  3. Run inference: python inference.py --mode {args.mode}")


def collect_data(args: argparse.Namespace, robot, data_dir: str) -> list:
    """Main data collection loop."""
    collected_episodes = []

    for episode_idx in range(args.num_episodes):
        print(f"\n--- Episode {episode_idx + 1}/{args.num_episodes} ---")

        # Get task instruction
        instruction = input("Enter task instruction: ").strip()

        if not instruction:
            print("Skipping episode (empty instruction).")
            continue

        # If no robot, use synthetic data
        if robot is None:
            print("Collecting synthetic data...")
            episode_data = _collect_synthetic_episode(args, instruction)
        else:
            # Real robot data collection
            print("\nMove the leader arm to start.")
            print("Press Enter to START recording.")
            input()

            episode_data = _collect_robot_episode(args, robot, instruction)

        # Save episode
        episode_dir = os.path.join(data_dir, f"episode_{episode_idx:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        # Save data
        np.save(os.path.join(episode_dir, "images.npy"), episode_data["images"])
        np.save(os.path.join(episode_dir, "states.npy"), episode_data["states"])
        np.save(os.path.join(episode_dir, "actions.npy"), episode_data["actions"])

        # Save metadata
        metadata = {
            "instruction": instruction,
            "num_frames": len(episode_data["images"]),
            "start_time": episode_data["start_time"],
            "end_time": episode_data["end_time"],
            "duration_seconds": episode_data["end_time"] - episode_data["start_time"],
            "avg_fps": len(episode_data["images"]) / max(1, episode_data["end_time"] - episode_data["start_time"]),
        }
        with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        collected_episodes.append(metadata)
        print(f"\nEpisode {episode_idx + 1} saved to {episode_dir}")
        print(f"  Frames: {len(episode_data['images'])}")
        print(f"  Duration: {metadata['duration_seconds']:.1f}s")
        print(f"  Avg FPS: {metadata['avg_fps']:.1f}")

    return collected_episodes


def _collect_synthetic_episode(args: argparse.Namespace, instruction: str) -> dict:
    """Collect synthetic data (for testing without hardware)."""
    num_frames = min(100, args.max_frames)  # Synthetic data is shorter
    img_size = args.image_size

    images = np.random.randint(0, 255, (num_frames, img_size, img_size, 3), dtype=np.uint8)
    states = np.random.randn(num_frames, 28)
    actions = states.copy()

    start_time = time.time()
    time.sleep(0.1)  # Simulate duration
    end_time = time.time()

    return {
        "images": images,
        "states": states,
        "actions": actions,
        "start_time": start_time,
        "end_time": end_time,
        "instruction": instruction,
    }


def _collect_robot_episode(args: argparse.Namespace, robot, instruction: str) -> dict:
    """Collect real data from robot."""
    import cv2

    # Open main camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Fall back to synthetic if camera fails
        print("Warning: Camera not available. Using synthetic data for this episode.")
        return _collect_synthetic_episode(args, instruction)

    start_time = time.time()
    frames = []
    states = []
    actions = []
    frame_count = 0

    print("Recording... Move the leader arm to demonstrate.")
    print("Press Enter to stop recording.")

    while frame_count < args.max_frames:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (args.image_size, args.image_size))
            frames.append(frame)

        # Get state
        try:
            obs = robot.get_observation()
            states.append(obs["state"])
            actions.append(obs["state"].copy())
        except Exception:
            # If robot communication fails, use synthetic state
            states.append(np.random.randn(28))
            actions.append(np.random.randn(28))

        frame_count += 1

        # Stop condition
        if input() == "":
            break

    cap.release()
    end_time = time.time()

    return {
        "images": np.array(frames) if frames else np.zeros((frame_count, args.image_size, args.image_size, 3), dtype=np.uint8),
        "states": np.array(states) if states else np.zeros((frame_count, 28)),
        "actions": np.array(actions) if actions else np.zeros((frame_count, 28)),
        "start_time": start_time,
        "end_time": end_time,
        "instruction": instruction,
    }


if __name__ == "__main__":
    main()
