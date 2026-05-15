#!/usr/bin/env python3
"""
bh-VLA: Unified Vision-Language-Action Model Training Framework

Complete training script supporting two VLA policies:
    1. ACT (Action Chunking Transformers) — arXiv:2304.13705
    2. pi0.5 (Flow Matching VLM) — arXiv:2504.16054

Both policies can be trained and evaluated through this unified interface.

Usage:
    # Train ACT policy (single GPU)
    python train.py --mode act

    # Train pi0.5 policy
    python train.py --mode pi05

    # Train with custom hyperparameters
    python train.py --mode act --lr 1e-4 --batch-size 16 --epochs 50

    # Resume training from checkpoint
    python train.py --mode act --resume

    # Train with a YAML config file
    python train.py --config config.yaml

    # Multi-GPU training (accelerate auto-detects GPUs)
    accelerate launch train.py --mode act --use-accelerate

    # Multi-GPU with gradient accumulation
    accelerate launch train.py --mode act --use-accelerate --gradient-accumulation 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policies.config import PolicyFactory, validate_config, save_config_json, config_to_dict
from policies.act import ACTConfig, ACTPolicy, ACTLoss
from policies.pi05 import Pi05Config, Pi05Policy
from data.dataset import (
    LeRobotDataset,
    RLDSDataset,
    DirectoryDataset,
    ALOHADataset,
    DatasetConfig,
    collate_fn,
    compute_dataset_stats,
)
from data.transforms import get_train_transforms, get_val_transforms
from utils import (
    seed_everything,
    count_parameters,
    count_trainable_parameters,
    format_number,
    format_parameters,
    setup_logging,
    ensure_dir,
    save_checkpoint,
    load_checkpoint,
    resume_checkpoint,
    find_latest_checkpoint,
    Timer,
    ProgressTracker,
    get_device,
)


# =====================================================================
# Accelerate Setup
# =====================================================================

def setup_accelerate(args: argparse.Namespace):
    """Initialize accelerate settings if --use-accelerate is specified.

    Returns:
        (accelerator, model, optimizer, lr_scheduler, train_loader, val_loader) or
        (None, model, optimizer, lr_scheduler, train_loader, val_loader)
    """
    if not getattr(args, "use_accelerate", False):
        return None, None, None, None, None, None

    try:
        from accelerate import Accelerator
    except ImportError:
        raise ImportError("accelerate is required for --use-accelerate. "
                          "Install with: pip install accelerate")

    deepspeed_config = {}
    if getattr(args, "deepspeed_config", None):
        with open(args.deepspeed_config, "r") as f:
            import yaml
            deepspeed_config = yaml.safe_load(f)

    # Gradient accumulation
    accumulation_steps = getattr(args, "gradient_accumulation", 1)

    # Mixed precision: auto-select best dtype for this hardware
    mixed_precision = "no"
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            mixed_precision = "bf16"
        else:
            mixed_precision = "fp16"

    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps,
        mixed_precision=mixed_precision,
        deepspeed_plugin=deepspeed_config if deepspeed_config else None,
    )

    logging.info(f"Accelerator configured: {accelerator}")
    logging.info(f"  Device: {accelerator.device}")
    logging.info(f"  Mixed precision: {accelerator.mixed_precision}")
    logging.info(f"  Gradient accumulation: {accumulation_steps}")
    logging.info(f"  Number of processes: {accelerator.num_processes}")
    logging.info(f"  Gradient bits: {accelerator.scaler.loss_scale}")

    return accelerator


# =====================================================================
# Training State
# =====================================================================

class TrainingState:
    """Track training progress for checkpointing and resuming."""

    def __init__(self, best_val_loss: float = float("inf"), best_epoch: int = -1):
        self.epoch = 0
        self.best_val_loss = best_val_loss
        self.best_epoch = best_epoch
        self.patience_counter = 0
        self.metrics: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        state = cls(best_val_loss=d.get("best_val_loss", float("inf")),
                    best_epoch=d.get("best_epoch", -1))
        state.epoch = d.get("epoch", 0)
        state.patience_counter = d.get("patience_counter", 0)
        state.metrics = d.get("metrics", [])
        return state


# =====================================================================
# Training and Evaluation Loops
# =====================================================================

def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss_fn: Any,
    device: torch.device,
    mode: str,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    logger: Optional[logging.Logger] = None,
    accelerator: Optional[Any] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    if accelerator is not None:
        train_loader = accelerator.prepare(train_loader)

    for batch in train_loader:
        images = batch["images"].to(device, non_blocking=True)
        actions_gt = batch["actions"].to(device, non_blocking=True)
        states = batch["state"].to(device, non_blocking=True)
        language = batch["language"]

        optimizer.zero_grad()

        if accelerator is not None:
            # accelerator.prepare() handles AMP, DDP, etc. automatically
            if mode == "act":
                actions_pred = model(images, language[0], states)
                loss = loss_fn(actions_pred, actions_gt)
            elif mode == "pi05":
                text_ids = batch.get("language_ids",
                                     torch.zeros(actions_gt.size(0), 64,
                                                 dtype=torch.long, device=device))
                loss = model.compute_flow_matching_loss(images, text_ids, actions_gt)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            loss = loss.to(torch.float32)

            accelerator.backward(loss)
            if grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            # Standard AMP path
            if use_amp and torch.cuda.is_available():
                from torch.cuda.amp import GradScaler, autocast
                scaler = GradScaler(enabled=True)
                with autocast():
                    if mode == "act":
                        actions_pred = model(images, language[0], states)
                        loss = loss_fn(actions_pred, actions_gt)
                    elif mode == "pi05":
                        text_ids = batch.get("language_ids",
                                             torch.zeros(actions_gt.size(0), 64,
                                                         dtype=torch.long, device=device))
                        loss = model.compute_flow_matching_loss(images, text_ids, actions_gt)
                    else:
                        raise ValueError(f"Unknown mode: {mode}")
                    loss = loss.to(torch.float32)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if mode == "act":
                    actions_pred = model(images, language[0], states)
                    loss = loss_fn(actions_pred, actions_gt)
                elif mode == "pi05":
                    text_ids = batch.get("language_ids",
                                         torch.zeros(actions_gt.size(0), 64,
                                                     dtype=torch.long, device=device))
                    loss = model.compute_flow_matching_loss(images, text_ids, actions_gt)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        if accelerator is not None:
            epoch_loss += accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
        else:
            epoch_loss += loss.item()
        batch_count += 1

    if scheduler is not None and accelerator is None:
        scheduler.step()
    elif scheduler is not None and accelerator is not None:
        scheduler.step()

    return {"train_loss": epoch_loss / max(batch_count, 1)}


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    mode: str,
    use_amp: bool = True,
    accelerator: Optional[Any] = None,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    epoch_loss = 0.0
    batch_count = 0

    if accelerator is not None:
        val_loader = accelerator.prepare(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device, non_blocking=True)
            actions_gt = batch["actions"].to(device, non_blocking=True)
            states = batch["state"].to(device, non_blocking=True)
            language = batch["language"]

            if accelerator is not None:
                if mode == "act":
                    actions_pred = model(images, language[0], states)
                    loss = F.mse_loss(actions_pred, actions_gt)
                elif mode == "pi05":
                    text_ids = batch.get("language_ids",
                                         torch.zeros(actions_gt.size(0), 64,
                                                     dtype=torch.long, device=device))
                    loss = model.compute_flow_matching_loss(images, text_ids, actions_gt)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                loss = loss.to(torch.float32)
                epoch_loss += accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
            else:
                if use_amp and torch.cuda.is_available():
                    from torch.cuda.amp import autocast
                    with autocast():
                        if mode == "act":
                            actions_pred = model(images, language[0], states)
                            loss = F.mse_loss(actions_pred, actions_gt)
                        elif mode == "pi05":
                            text_ids = batch.get("language_ids",
                                                 torch.zeros(actions_gt.size(0), 64,
                                                             dtype=torch.long, device=device))
                            loss = model.compute_flow_matching_loss(images, text_ids, actions_gt)
                        else:
                            raise ValueError(f"Unknown mode: {mode}")
                        loss = loss.to(torch.float32)
                else:
                    if mode == "act":
                        actions_pred = model(images, language[0], states)
                        loss = F.mse_loss(actions_pred, actions_gt)
                    elif mode == "pi05":
                        text_ids = batch.get("language_ids",
                                             torch.zeros(actions_gt.size(0), 64,
                                                         dtype=torch.long, device=device))
                        loss = model.compute_flow_matching_loss(images, text_ids, actions_gt)
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                epoch_loss += loss.item()
            batch_count += 1

    return {"val_loss": epoch_loss / max(batch_count, 1)}


# =====================================================================
# Main Training Function
# =====================================================================

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    # 0. Accelerate setup (must come first for multi-GPU)
    accelerator = setup_accelerate(args)

    # 1. Setup
    device = accelerator.device if accelerator is not None else get_device(args.device)
    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.log_dir)
    logger = setup_logging(os.path.join(args.log_dir, "train.log"))
    logger.info(f"bh-VLA Training Framework")
    logger.info(f"Mode: {args.mode} | Device: {device}")

    # 2. Config loading
    if args.config:
        from policies.config import load_config_yaml, load_config_json
        if args.config.endswith((".yaml", ".yml")):
            config, loaded_mode = load_config_yaml(args.config)
        else:
            config, loaded_mode = load_config_json(args.config)
        if loaded_mode != args.mode:
            logger.warning(f"Config mode {loaded_mode} differs from --mode {args.mode}. Using --mode.")
        if args.lr:
            setattr(config, "lr", args.lr)
        if args.batch_size:
            setattr(config, "batch_size", args.batch_size)
        if args.epochs:
            setattr(config, "num_epochs", args.epochs)
    else:
        if args.mode == "act":
            config = ACTConfig(
                lr=args.lr or 1e-4,
                batch_size=args.batch_size or 32,
                num_epochs=args.epochs or 100,
            )
        else:
            config = Pi05Config(
                lr=args.lr or 1e-5,
                batch_size=args.batch_size or 8,
                num_epochs=args.epochs or 50,
            )
        validate_config(config, args.mode)

    # 3. Create policy
    factory = PolicyFactory()
    policy = factory.create(args.mode)

    total_params = count_parameters(policy)
    trainable_params = count_trainable_parameters(policy)
    logger.info(f"Total parameters: {format_parameters(total_params)}")
    logger.info(f"Trainable parameters: {format_parameters(trainable_params)}")

    policy = policy.to(device)

    # 4. Dataset loading
    ds_config = DatasetConfig(data_dir=args.data_dir or "./data")
    if args.data_format == "lerobot":
        dataset_cls = LeRobotDataset
    elif args.data_format == "rlds":
        dataset_cls = RLDSDataset
    elif args.data_format == "directory":
        dataset_cls = DirectoryDataset
    elif args.data_format == "aloha":
        dataset_cls = ALOHADataset
    else:
        if os.path.exists(os.path.join(ds_config.data_dir, "dataset_infos.json")):
            dataset_cls = LeRobotDataset
        elif os.path.exists(os.path.join(ds_config.data_dir, "data.json")):
            dataset_cls = DirectoryDataset
        else:
            dataset_cls = ALOHADataset

    # Dataset stats
    dataset_stats = None
    stats_path = os.path.join(args.data_dir or "./data", "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            dataset_stats = json.load(f)
        logger.info(f"Loaded dataset stats from {stats_path}")

    if dataset_stats is None:
        logger.info("Computing dataset statistics...")
        dataset_stats = compute_dataset_stats(ds_config.data_dir)
        ensure_dir(os.path.dirname(stats_path))
        with open(stats_path, "w") as f:
            json.dump(dataset_stats, f, indent=2)

    # Split dataset by index
    train_ds = dataset_cls(ds_config.data_dir, ds_config, split="train")
    val_ds = dataset_cls(ds_config.data_dir, ds_config, split="val")

    # DDP / distributed sampler
    if accelerator is not None and accelerator.num_processes > 1:
        train_sampler = DistributedSampler(train_ds, shuffle=True,
                                           seed=args.seed, drop_last=False)
        val_sampler = DistributedSampler(val_ds, shuffle=False,
                                         seed=args.seed, drop_last=False)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                  sampler=train_sampler,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn,
                                  pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                sampler=val_sampler,
                                num_workers=args.num_workers,
                                collate_fn=collate_fn,
                                pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn)

    # Prepare model/optimizer/ scheduler with accelerator
    if accelerator is not None:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay or 1e-4,
        )
        policy, optimizer, train_loader, val_loader = accelerator.prepare(
            policy, optimizer, train_loader, val_loader
        )
    else:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay or 1e-4,
        )

    # Scheduler
    warmup_steps = config.warmup_steps or 1000
    total_steps = len(train_loader) * config.num_epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=config.lr * 0.01
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Loss function
    loss_fn = ACTLoss() if args.mode == "act" else None

    # Training progress
    logger.info(f"\n{'='*60}")
    logger.info(f"  Training {args.mode.upper()} policy")
    logger.info(f"  Epochs: {config.num_epochs} | Batch size: {config.batch_size}")
    logger.info(f"  LR: {config.lr:.2e} | Grad clip: {config.gradient_clip}")
    if accelerator is not None:
        logger.info(f"  Multi-GPU: {accelerator.num_processes} GPUs")
        logger.info(f"  Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"  Gradient accumulation: {getattr(args, 'gradient_accumulation', 1)}")
    logger.info(f"  Early stopping patience: {args.patience} epochs")
    logger.info(f"{'='*60}\n")

    # Training loop
    train_state = TrainingState()

    if args.resume and accelerator is not None:
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path:
            logger.info(f"Resuming from {ckpt_path}")
            # accelerate loads checkpoints differently
            accelerator.load_state(ckpt_path)
            # Extract training state from checkpoint
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "training_state" in state_dict:
                train_state = TrainingState.from_dict(state_dict["training_state"])
            logger.info(f"Resumed from epoch {train_state.epoch}")
    elif args.resume and accelerator is None:
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path:
            extra_state = resume_checkpoint(ckpt_path, policy, optimizer, scheduler, device)
            train_state = TrainingState.from_dict(extra_state.get("training_state", {}))
        else:
            logger.info("No checkpoint found. Starting fresh.")
    else:
        logger.info("Starting fresh training.")

    best_val_loss = float("inf")
    train_state.best_val_loss = best_val_loss
    epoch = train_state.epoch
    progress = ProgressTracker(config.num_epochs, label="epoch")

    for epoch in range(train_state.epoch, config.num_epochs):
        if accelerator is not None and hasattr(train_loader, 'sampler'):
            train_loader.sampler.set_epoch(epoch)

        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=policy, train_loader=train_loader, optimizer=optimizer,
            scheduler=scheduler, loss_fn=loss_fn, device=device,
            mode=args.mode, grad_clip=config.gradient_clip,
            use_amp=args.amp, logger=logger,
            accelerator=accelerator,
        )

        val_metrics = evaluate(
            model=policy, val_loader=val_loader, device=device,
            mode=args.mode, use_amp=args.amp,
            accelerator=accelerator,
        )
        val_loss = val_metrics["val_loss"]

        # Sync loss across GPUs
        if accelerator is not None:
            val_loss = float(accelerator.gather(
                torch.tensor(val_loss, device=device)
            ).mean().item())

        if accelerator.is_main_process:
            # Only main process saves checkpoints
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                train_state.best_val_loss = best_val_loss
                ckpt_path = os.path.join(args.checkpoint_dir, f"{args.mode}_best.pt")
                save_checkpoint(policy, optimizer, scheduler, epoch + 1,
                                train_state.to_dict(), ckpt_path, args.mode)
                logger.info(f"  New best val_loss: {val_loss:.6f} (checkpoint saved)")

            elapsed = time.time() - epoch_start
            progress.update(
                train_loss=train_metrics["train_loss"],
                val_loss=val_loss,
                lr=scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr')
                   else scheduler.get_lr()[0],
            )
            progress.display(epoch + 1)

            if (epoch + 1) % args.save_interval == 0 or epoch + 1 == config.num_epochs:
                ckpt_path = os.path.join(args.checkpoint_dir,
                                         f"{args.mode}_epoch_{epoch+1}.pt")
                save_checkpoint(policy, optimizer, scheduler, epoch + 1,
                                train_state.to_dict(), ckpt_path, args.mode)

            config_path = os.path.join(args.output_dir, f"{args.mode}_config.json")
            if epoch == train_state.epoch:
                save_config_json(config, config_path, args.mode)

            if val_loss > train_state.best_val_loss * 1.02:
                train_state.patience_counter += 1
            else:
                train_state.best_val_loss = val_loss
                best_val_loss = val_loss
                train_state.patience_counter = 0

            if train_state.patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}. "
                            f"Best val_loss: {best_val_loss:.6f}")
                break

        # Gather metrics from all processes in DDP
        if accelerator is not None:
            gather_metrics = {
                "train_loss": torch.tensor(train_metrics["train_loss"], device=device),
                "val_loss": torch.tensor(val_loss, device=device),
            }
            gathered = accelerator.gather(gather_metrics)
            if logger:
                logger.info(f"Epoch {epoch+1}: "
                            f"train_loss={gathered['train_loss'].mean().item():.6f} "
                            f"val_loss={gathered['val_loss'].mean().item():.6f}")

    # Final checkpoint
    if accelerator is not None:
        # unwrap model for saving
        unwrapped = accelerator.unwrap_model(policy)
        save_path = os.path.join(args.checkpoint_dir, f"{args.mode}_last.pt")
        accelerator.save(unwrapped.state_dict(), save_path)
    else:
        final_ckpt = os.path.join(args.checkpoint_dir, f"{args.mode}_last.pt")
        save_checkpoint(policy, optimizer, scheduler, epoch + 1,
                        train_state.to_dict(), final_ckpt, args.mode)

    if accelerator.is_main_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Training complete!")
        logger.info(f"  Best val_loss: {best_val_loss:.6f} "
                    f"(epoch {train_state.best_epoch+1})")
        logger.info(f"  Total time: {time.time()-epoch_start:.0f}s")
        logger.info(f"{'='*60}")

    if accelerator is not None:
        accelerator.end_training()


# =====================================================================
# Main Entry Point
# =====================================================================

def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description="bh-VLA: Unified VLA Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train ACT policy (single GPU)
    python train.py --mode act

    # Train pi0.5 with custom settings
    python train.py --mode pi05 --lr 1e-5 --batch-size 8 --epochs 30

    # Resume from latest checkpoint
    python train.py --mode act --resume

    # Train with a YAML config file
    python train.py --config act_config.yaml

    # Use CPU (no GPU)
    python train.py --mode act --device cpu

    # Reduce early stopping patience
    python train.py --mode pi05 --patience 5

    # ===== Multi-GPU training with accelerate =====
    accelerate launch train.py --mode act --use-accelerate
    accelerate launch --num_processes=4 train.py --mode act --use-accelerate
    accelerate launch train.py --mode act --use-accelerate \
        --gradient-accumulation 4  # accumulate 4 batches per step
        """
    )

    parser.add_argument("--mode", type=str, required=True,
                        choices=["act", "pi05"],
                        help="Training mode: 'act' or 'pi05'")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config default)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides config default)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config default)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory containing training data")
    parser.add_argument("--data-format", type=str, default="auto",
                        choices=["auto", "lerobot", "rlds", "directory", "aloha"],
                        help="Dataset format (auto-detected if 'auto')")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory to save logs")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to train on")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML/JSON config file")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio (0.0-1.0)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                        help="Gradient clipping norm")

    # ===== Accelerate options =====
    parser.add_argument("--use-accelerate", action="store_true",
                        help="Enable multi-GPU training via accelerate")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="Number of gradient accumulation steps (with --use-accelerate)")
    parser.add_argument("--deepspeed-config", type=str, default=None,
                        help="DeepSpeed config file path (with --use-accelerate)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
