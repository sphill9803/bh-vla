#!/usr/bin/env python3
"""
bh-VLA Inference Script: Run trained policy on robot or in test mode.

Usage:
    # On real robot
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt

    # Test mode (synthetic data, no robot needed)
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt --test-mode

    # pi0.5 inference with custom flow steps
    python inference.py --mode pi05 --checkpoint ./checkpoints/pi05_last.pt --flow-steps 100

    # With specific language instruction
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt \
        --language "pick up the red cup"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policies.act import ACTPolicy, ACTConfig
from policies.pi05 import Pi05Policy, Pi05Config
from data.robot_interface import SO101Robot, RobotConfig
from utils import get_device


def test_inference(args: argparse.Namespace) -> None:
    """Run inference with synthetic data (no robot needed)."""
    print(f"\n{'='*60}")
    print(f"  bh-VLA Inference — Test Mode")
    print(f"  Policy: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")

    # Load model with overrides
    overrides = {}
    if args.chunk_size:
        overrides["action_chunk_size"] = args.chunk_size
    if args.action_dim:
        overrides["action_dim"] = args.action_dim

    if args.mode == "act":
        policy_config = ACTConfig(**overrides)
        policy = ACTPolicy(policy_config)
        policy.load_checkpoint(args.checkpoint)
    else:
        policy_config = Pi05Config(**overrides, flow_steps=args.flow_steps)
        policy = Pi05Policy(policy_config)
        policy.load_checkpoint(args.checkpoint)

    policy = policy.to(args.device)
    device = torch.device(args.device)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Model loaded: {args.mode} ({total_params:,} parameters)\n")

    # Run synthetic inference
    print(f"Running inference for {args.num_steps} steps (synthetic data)...\n")
    num_cameras = 3
    img_size = 224
    action_history = []

    for step in range(args.num_steps):
        # Synthetic observation
        synthetic_images = torch.randn(num_cameras, 3, img_size, img_size, device=device)

        # Predict action
        with torch.no_grad():
            if args.mode == "act":
                from policies.act import CharacterTokenizer
                tokenizer = CharacterTokenizer()
                token_ids = torch.tensor(tokenizer.encode(args.language), dtype=torch.long).unsqueeze(0).to(device)
                state = torch.zeros(12, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy.predict_action(synthetic_images, token_ids, state)
            else:
                action = policy.predict_action(synthetic_images, args.language)

        action_np = action.cpu().numpy()
        action_history.append(action_np)

        if step % 10 == 0 or step == args.num_steps - 1:
            print(f"  Step {step+1:4d}/{args.num_steps}: "
                  f"action = [{', '.join([f'{v:+.4f}' for v in action_np[:5]])}...{len(action_np)-5} more]")

        # Small delay
        time.sleep(0.01)

    # Save action history
    history_dir = os.path.join("./outputs", "inference")
    os.makedirs(history_dir, exist_ok=True)
    np.save(os.path.join(history_dir, f"inference_{args.mode}.npy"),
            np.array(action_history))
    print(f"\nAction history saved to {os.path.join(history_dir, f'inference_{args.mode}.npy')}")


def robot_inference(args: argparse.Namespace) -> None:
    """Run inference on real SO-101 robot hardware."""
    print(f"\n{'='*60}")
    print(f"  bh-VLA Inference — Robot Mode")
    print(f"  Policy: {args.mode}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Language: {args.language}")
    print(f"{'='*60}\n")

    # Load model with overrides
    overrides = {}
    if args.chunk_size:
        overrides["action_chunk_size"] = args.chunk_size
    if args.action_dim:
        overrides["action_dim"] = args.action_dim

    if args.mode == "act":
        policy_config = ACTConfig(**overrides)
        policy = ACTPolicy(policy_config)
    else:
        policy_config = Pi05Config(**overrides, flow_steps=args.flow_steps)
        policy = Pi05Policy(policy_config)

    policy.load_checkpoint(args.checkpoint)
    policy = policy.to(args.device)
    device = torch.device(args.device)

    # Initialize robot
    robot_cfg = RobotConfig()
    robot = SO101Robot(robot_cfg)

    print("Connecting to robot hardware...")
    try:
        robot.connect()
    except Exception as e:
        print(f"Error connecting to robot: {e}")
        print("Falling back to test mode...")
        test_inference(args)
        return

    print(f"\n{'='*60}")
    print(f"  Starting inference loop at ~50Hz")
    print(f"  Language instruction: {args.language}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

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
            with torch.no_grad():
                if args.mode == "act":
                    from policies.act import CharacterTokenizer
                    tokenizer = CharacterTokenizer()
                    token_ids = torch.tensor(tokenizer.encode(args.language), dtype=torch.long).unsqueeze(0).to(device)
                    state_t = torch.tensor(obs["state"], dtype=torch.float32).unsqueeze(0).to(device)
                    action = policy.predict_action(images_tensor, token_ids, state_t)
                else:
                    action = policy.predict_action(images_tensor, args.language)
            action_np = action.cpu().numpy()

            # Send to robot
            robot.send_action(action_np)
            action_history.append(action_np)
            step += 1

            if step % 20 == 0:
                print(f"  Step {step}: action = [{', '.join([f'{v:+.4f}' for v in action_np[:5]])}...{len(action_np)-5} more]")

            time.sleep(0.015)

    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")
    finally:
        robot.disconnect()

    # Save action history
    if action_history:
        history_dir = os.path.join("./outputs", "inference")
        os.makedirs(history_dir, exist_ok=True)
        np.save(os.path.join(history_dir, f"inference_{args.mode}_robot.npy"),
                np.array(action_history))
        print(f"\nAction history saved to {os.path.join(history_dir, f'inference_{args.mode}_robot.npy')}")


def main():
    parser = argparse.ArgumentParser(
        description="bh-VLA: Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # On real robot
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt

    # Test mode (no robot needed)
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt --test-mode

    # pi0.5 with custom flow steps
    python inference.py --mode pi05 --checkpoint ./checkpoints/pi05_last.pt --flow-steps 100
        """
    )

    parser.add_argument("--mode", type=str, required=True,
                        choices=["act", "pi05"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--language", type=str, default="pick up the object",
                        help="Language instruction")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run with synthetic data (no robot)")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of inference steps (test mode)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--flow-steps", type=int, default=50,
                        help="Flow matching steps (pi0.5 only)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Override action chunk size")
    parser.add_argument("--action-dim", type=int, default=None,
                        help="Action dimension (overrides default)")

    args = parser.parse_args()

    if args.test_mode:
        test_inference(args)
    else:
        robot_inference(args)


if __name__ == "__main__":
    main()
