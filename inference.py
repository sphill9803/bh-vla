#!/usr/bin/env python3
"""
Inference Script: Run the trained VLA policy on real robot hardware.

Usage:
    # Inference with ACT policy
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt

    # Inference with pi0.5 policy
    python inference.py --mode pi05 --checkpoint ./checkpoints/pi05_last.pt

    # Run in headless mode (no robot, just test forward pass)
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt --test-mode

    # Run with specific language instruction
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt \
        --language "pick up the red cup"
"""

import os
import sys
import argparse
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="bh-VLA: Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on real robot
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt

    # Test with synthetic data
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt --test-mode

    # Run with custom language instruction
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt \\
        --language "fold the laundry"
        """
    )

    parser.add_argument("--mode", type=str, required=True, choices=["act", "pi05"],
                        help="Policy mode: 'act' or 'pi05'")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained checkpoint file")
    parser.add_argument("--language", type=str, default="pick up the object",
                        help="Language instruction for the policy")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in test mode with synthetic data (no robot needed)")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of inference steps (in test mode)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to run inference on")
    parser.add_argument("--flow-steps", type=int, default=50,
                        help="Number of flow matching steps (pi0.5 only)")

    args = parser.parse_args()

    print("=" * 60)
    print(f"  bh-VLA Inference")
    print(f"  Mode: {args.mode}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Language: {args.language}")
    print(f"  Device: {args.device}")
    print("=" * 60)

    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if args.test_mode:
        run_test_inference(args)
    else:
        run_robot_inference(args)


def run_test_inference(args):
    """Run inference with synthetic data (no robot hardware needed)."""
    import torch
    import sys

    # Import the policy
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import ACTPolicy, ACTConfig, Pi05Config

    # Create and load the policy
    if args.mode == "act":
        policy_config = ACTConfig(
            image_encoder="resnet18",
            image_size=224,
            action_dim=14,
            action_chunk_size=90,
        )
        policy = ACTPolicy(policy_config)
        policy.load_checkpoint(args.checkpoint)
    else:
        policy_config = Pi05Config(
            flow_steps=args.flow_steps,
            action_dim=14,
            action_chunk_size=32,
        )
        policy = Pi05Policy(policy_config)
        policy.load_checkpoint(args.checkpoint)

    print(f"\nPolicy loaded: {args.mode}")
    print(f"  Mode: {args.mode}")
    print(f"  Action dim: {policy.config.action_dim}")
    print(f"  Chunk size: {policy.config.action_chunk_size}")
    print("-" * 60)

    # Run synthetic inference
    print(f"\nRunning inference for {args.num_steps} steps...")
    print("(Using synthetic data - no robot hardware required)\n")

    # Create synthetic observation
    num_cameras = 3
    img_size = 224
    synthetic_images = torch.randn(
        num_cameras, 3, img_size, img_size,
        device=args.device
    )

    for step in range(args.num_steps):
        # Predict action
        if args.mode == "act":
            action = policy.predict_action(synthetic_images, args.language)
        else:
            action = policy.predict_action(synthetic_images, args.language)

        # Print the action
        action_np = action.cpu().numpy()
        action_str = " ".join([f"{v:.3f}" for v in action_np])

        # Print every 10 steps for readability
        if step % 10 == 0 or step == args.num_steps - 1:
            print(f"  Step {step+1:3d}/{args.num_steps}: action = [{action_str}]")

        # Simulate state update
        state_noise = np.random.randn(len(action_np)) * 0.01
        synthetic_images = torch.randn(num_cameras, 3, img_size, img_size,
                                       device=args.device)

        # Small delay to simulate real-time
        time.sleep(0.01)

    print("\nInference complete!")
    print("This was running in test mode with synthetic data.")
    print("For real robot execution, run without --test-mode.")


def run_robot_inference(args):
    """Run inference on real robot hardware."""
    import torch
    import sys
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import ACTPolicy, ACTConfig, Pi05Config, RobotConfig

    # Create and load the policy
    if args.mode == "act":
        policy_config = ACTConfig(
            image_encoder="resnet18",
            image_size=224,
            action_dim=14,
            action_chunk_size=90,
        )
        policy = ACTPolicy(policy_config)
    else:
        policy_config = Pi05Config(
            flow_steps=args.flow_steps,
            action_dim=14,
            action_chunk_size=32,
        )
        policy = Pi05Policy(policy_config)

    policy.load_checkpoint(args.checkpoint)
    policy = policy.to(args.device)

    # Initialize robot
    robot_config = RobotConfig()
    robot = RobotInterface(robot_config)

    print("\nInitializing robot hardware...")
    try:
        robot.connect()
    except Exception as e:
        print(f"Warning: Robot connection failed: {e}")
        print("Falling back to test mode...")
        run_test_inference(args)
        return

    print("\n" + "=" * 60)
    print("  Starting inference loop at 50Hz")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    step = 0
    action_history = []

    try:
        while True:
            # Get observation
            obs = robot.get_observation()

            # Process images
            images = np.stack([
                obs["images"][f"camera_{i}"]
                for i in range(robot.config.num_cameras)
            ], axis=0)

            # Convert to tensor
            images_tensor = torch.tensor(images, dtype=torch.float32).to(args.device)
            images_tensor = (images_tensor - 0.5) / 0.5  # Normalize

            # Predict action
            action = policy.predict_action(images_tensor, args.language, obs["state"])
            action_np = action.cpu().numpy()

            # Send action to robot
            robot.send_action(action_np)

            # Track action history
            action_history.append(action_np)

            # Log
            step += 1
            if step % 20 == 0:
                print(f"  Step {step}: action = [{', '.join([f'{v:.3f}' for v in action_np])}]")

            # Maintain 50Hz frequency
            time.sleep(0.015)  # ~66Hz target

    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")
    finally:
        robot.disconnect()

    # Save action history
    if action_history:
        history_path = "./outputs/action_history.npy"
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        np.save(history_path, np.array(action_history))
        print(f"Action history saved to {history_path}")


if __name__ == "__main__":
    main()
