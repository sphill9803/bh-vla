#!/usr/bin/env python3
"""
bh-VLA Utility Functions

Provides common helper utilities used across the project:
    - Random seed management
    - Parameter counting
    - Number formatting
    - Logging setup
    - Directory creation
    - Checkpoint I/O (save/load/resume)
    - Timing utilities
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# ====================================================================
# Random Seed Management
# ====================================================================

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for all libraries used in the project.

    Ensures reproducibility across:
        - Python's built-in random module
        - NumPy random number generator
        - PyTorch CPU and CUDA random number generators

    Args:
        seed: Random seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        # Deterministic algorithms for reproducibility (may be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ====================================================================
# Parameter Counting
# ====================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of parameters in a model.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters (requires_grad=True).

    Args:
        model: PyTorch nn.Module.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ====================================================================
# Number Formatting
# ====================================================================

def format_number(n: int) -> str:
    """Format a number with comma separators.

    Args:
        n: Integer to format.

    Returns:
        String with comma separators (e.g., 1,234,567).
    """
    return f"{n:,}"


def format_parameters(n: int) -> str:
    """Format a parameter count in human-readable form.

    Args:
        n: Number of parameters.

    Returns:
        Formatted string (e.g., '1.23M', '45.6K').
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return f"{n}"


# ====================================================================
# Logging
# ====================================================================

def setup_logging(
    log_path: str,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Configure logging for the project.

    Creates a logger that outputs to:
        - A file (log_path) with detailed formatting
        - The console with concise formatting

    Args:
        log_path: Path to the log file.
        level: File logging level.
        console_level: Console logging level.

    Returns:
        Configured logger instance.
    """
    # Create directory if needed
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("bh_vla")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    logger.handlers.clear()

    # File handler (detailed)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (concise)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# ====================================================================
# Directory Utilities
# ====================================================================

def ensure_dir(path: str) -> str:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        The path (for chaining).
    """
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent_dir(path: str) -> str:
    """Ensure the parent directory of a file path exists.

    Args:
        path: File path.

    Returns:
        The path (for chaining).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


# ====================================================================
# Checkpoint Utilities
# ====================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    state: Dict[str, Any],
    path: str,
    mode: str = "act",
) -> None:
    """Save a complete training checkpoint.

    Saves the model state, optimizer state, scheduler state, and
    training metadata to a single file.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The LR scheduler to save.
        epoch: Current epoch number.
        state: Additional training state dict.
        path: Checkpoint file path.
        mode: Policy mode ('act' or 'pi05').
    """
    ensure_parent_dir(path)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "training_state": state,
        "mode": mode,
        "config": getattr(getattr(model, "config", None), "__dict__", None),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    weights_only: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Checkpoint file path.
        device: Device to load to.
        weights_only: Whether to use torch.load(..., weights_only=True).

    Returns:
        Loaded checkpoint dict.
    """
    return torch.load(path, map_location=device, weights_only=weights_only)


def resume_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Resume training from a checkpoint.

    Loads the model, optimizer, and scheduler states, and returns
    additional training metadata.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: LR scheduler to load state into.
        device: Device to load to.

    Returns:
        Dict with 'epoch' and 'training_state' keys.
    """
    checkpoint = load_checkpoint(path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    training_state = checkpoint.get("training_state", {})
    training_state["epoch"] = checkpoint.get("epoch", 0)

    print(f"Resumed from checkpoint: {path} (epoch {training_state['epoch']})")
    return training_state


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint file in a directory.

    Looks for files matching patterns: *_last.pt, *_final.pt, *_epoch_*.pt

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".pt"):
            checkpoint_files.append(os.path.join(checkpoint_dir, f))

    if not checkpoint_files:
        return None

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]


# ====================================================================
# Timing Utilities
# ====================================================================

class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer("model forward") as t:
            output = model(input)
        print(f"Forward pass took {t.elapsed:.3f}s")
    """

    def __init__(self, name: str = "block"):
        self.name = name
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.elapsed = time.perf_counter() - self.start_time


class ProgressTracker:
    """Track progress over multiple epochs or iterations.

    Usage:
        tracker = ProgressTracker(total_epochs=100)
        for epoch in range(100):
            loss = train_one_epoch()
            tracker.update(loss=loss)
            tracker.display(epoch)
    """

    def __init__(self, total: int, label: str = "epoch", width: int = 50):
        self.total = total
        self.label = label
        self.width = width
        self.current = 0
        self.history: list = []

    def update(self, **kwargs: Any) -> None:
        """Record a new data point."""
        self.history.append(kwargs)

    def display(self, current: int, **extra: Any) -> str:
        """Display the current progress bar with metrics.

        Args:
            current: Current iteration number.
            **extra: Additional metrics to display.

        Returns:
            Formatted progress string.
        """
        self.current = current
        fraction = current / max(self.total, 1)
        filled = int(self.width * fraction)
        bar = "█" * filled + "░" * (self.width - filled)

        metrics_str = ""
        if extra:
            parts = []
            for k, v in extra.items():
                if isinstance(v, float):
                    parts.append(f"{k}: {v:.4f}")
                else:
                    parts.append(f"{k}: {v}")
            metrics_str = " | " + " | ".join(parts)

        return f"\r[{bar}] {current}/{self.total} [{fraction*100:.1f}%]{metrics_str}"

    def finish(self) -> str:
        """Return the completed progress bar."""
        self.display(self.total)
        return "\n"


# ====================================================================
# Device Management
# ====================================================================

def get_device(device_str: str = "cuda") -> torch.device:
    """Get a PyTorch device, falling back to CPU if CUDA is unavailable.

    Args:
        device_str: Desired device string ('cuda' or 'cpu').

    Returns:
        torch.device object.
    """
    if device_str == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        print("CUDA not available. Using CPU.")
    return torch.device("cpu")


def get_available_gpus() -> list[int]:
    """Return a list of available GPU indices.

    Returns:
        List of GPU indices (e.g., [0, 1] for two GPUs).
    """
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def get_device_memory(device: torch.device | None = None) -> Dict[str, float]:
    """Get GPU memory usage information.

    Args:
        device: Target GPU device (None = current device).

    Returns:
        Dict with 'allocated', 'reserved', 'free', 'total' in MB.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}

    if device is None:
        device = torch.device("cuda")

    mem = torch.cuda.memory_stats(device)
    return {
        "allocated": mem["allocated_bytes.all.current"] / (1024 ** 2),
        "reserved": mem["reserved_bytes.all.current"] / (1024 ** 2),
        "free": (torch.cuda.get_device_properties(device).total_memory -
                 mem["reserved_bytes.all.current"]) / (1024 ** 2),
        "total": torch.cuda.get_device_properties(device).total_memory / (1024 ** 2),
    }
