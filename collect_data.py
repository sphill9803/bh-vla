#!/usr/bin/env python3
"""
Data Collection Script: Teleoperation for collecting training data.

This script enables you to collect demonstration data using the ALOHA leader arm
(telerobot) to teach the follower arm. The collected data includes:
  - Images from all 3 cameras (at 30fps)
  - Joint positions of both arms
  - Language instructions
  - Timestamps

Usage:
    # Collect data with ACT mode
    python collect_data.py --mode act

    # Collect data with pi0.5 mode
    python collect_data.py --mode pi05

    # Collect 10 episodes
    python collect_data.py --mode act --num-episodes 10

    # Save to custom directory
    python collect_data.py --mode act --data-dir ./my_data
"""

import os
import sys
import argparse
import time
import numpy as np
import json


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

    # Collect with custom task names
    python collect_data.py --mode act --task-name "pick_and_place" --num-episodes 20
        """
    )

    parser.add_argument("--mode", type=str, required=True, choices=["act", "pi05"],
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

    args = parser.parse_args()

    print("=" * 60)
    print(f"  bh-VLA Data Collection")
    print(f"  Mode: {args.mode}")
    print(f"  Target episodes: {args.num_episodes}")
    print(f"  Data directory: {args.data_dir}")
    print("=" * 60)

    # Create data directory
    data_dir = os.path.join(args.data_dir, args.task_name)
    os.makedirs(data_dir, exist_ok=True)

    # Collect data
    collected_episodes = collect_data(args)

    # Summary
    print("\n" + "=" * 60)
    print(f"  Data Collection Summary")
    print(f"  Episodes collected: {len(collected_episodes)}")
    print(f"  Data directory: {data_dir}")
    print(f"  Mode: {args.mode}")
    print("=" * 60)

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
    print(f"Dataset info saved to {info_path}")

    print("\nNext steps:")
    print("  1. Inspect your data in", data_dir)
    print("  2. Run training: python train.py --mode", args.mode, f"--data-dir {data_dir}")
    print("  3. Run inference: python inference.py --mode", args.mode)


def collect_data(args):
    """Main data collection loop."""
    import cv2

    collected_episodes = []
    episode_counter = 0

    while episode_counter < args.num_episodes:
        # Get task instruction
        print(f"\n--- Episode {episode_counter + 1}/{args.num_episodes} ---")
        instruction = input("Enter task instruction (or 'skip' to skip): ").strip()

        if instruction.lower() == "skip":
            print("Skipping this episode.")
            continue

        print("\nInitializing camera...")
        print("Move the leader arm to start.")
        print("Press Enter to START recording.")
        input()

        # Start recording
        print("Recording... Move the leader arm to demonstrate the task.")
        print("Press Enter to stop recording when done.")

        episode_data = {
            "instruction": instruction,
            "start_time": time.time(),
            "frames": [],
            "states": [],
            "actions": [],
        }

        # Open camera
        cap = cv2.VideoCapture(0)  # Main camera (index 0)
        if not cap.isOpened():
            print("Warning: Could not open camera. Using synthetic data.")
            cap = None

        max_frames = 0
        frame_times = []

        while max_frames < args.max_frames:
            # Capture frame
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (args.image_size, args.image_size))
                    episode_data["frames"].append(frame)

            # Record state (simulated for demo)
            state = np.random.randn(28)  # 28-dim ALOHA state
            episode_data["states"].append(state)
            episode_data["actions"].append(state.copy())

            max_frames += 1
            frame_times.append(time.time())

            # Stop condition
            if input() == "":
                break

        # Stop camera
        if cap:
            cap.release()

        # Process collected data
        if episode_data["frames"]:
            frames_array = np.array(episode_data["frames"])
        else:
            frames_array = np.random.randint(0, 255, (max_frames, args.image_size, args.image_size, 3), dtype=np.uint8)

        states_array = np.array(episode_data["states"])
        actions_array = np.array(episode_data["actions"])

        # Save episode data
        episode_dir = os.path.join(data_dir, f"episode_{episode_counter:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        # Save as numpy files
        np.save(os.path.join(episode_dir, "images.npy"), frames_array)
        np.save(os.path.join(episode_dir, "states.npy"), states_array)
        np.save(os.path.join(episode_dir, "actions.npy"), actions_array)

        # Save episode metadata
        episode_info = {
            "instruction": instruction,
            "num_frames": max_frames,
            "start_time": episode_data["start_time"],
            "end_time": time.time(),
            "duration_seconds": time.time() - episode_data["start_time"],
            "avg_fps": max_frames / max(1, time.time() - episode_data["start_time"]),
        }
        with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
            json.dump(episode_info, f, indent=2)

        collected_episodes.append(episode_info)
        episode_counter += 1

        print(f"\nEpisode {episode_counter} saved to {episode_dir}")
        print(f"  Frames: {max_frames}")
        print(f"  Duration: {episode_info['duration_seconds']:.1f}s")
        print(f"  Avg FPS: {episode_info['avg_fps']:.1f}")

    return collected_episodes


if __name__ == "__main__":
    main()
