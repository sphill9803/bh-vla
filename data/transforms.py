"""
bh-VLA Data Transformation Pipeline

Provides composable image, action, state, and language transforms
that can be chained together for data augmentation and preprocessing.

Transform categories:
    - Image transforms: resize, crop, flip, color jitter, normalise
    - Action transforms: normalise, denormalise, chunk, pad
    - State transforms: normalise, denormalise
    - Language transforms: tokenize, detokenize
    - Composable pipeline: compose transforms into a single callable
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ====================================================================
# Image Transforms
# ====================================================================

class ImageTransform:
    """Base class for image transforms.

    All image transforms should subclass this and implement ``__call__``.
    The input can be a PIL Image, numpy array, or torch tensor.

    Attributes:
        to_tensor: Whether to convert the output to a torch tensor. Default False.
    """

    def __init__(self, to_tensor: bool = False):
        self.to_tensor = to_tensor

    def __call__(self, image: Any) -> Any:
        """Apply the transform to an image."""
        raise NotImplementedError

    def _to_tensor(self, image: Any) -> torch.Tensor:
        """Convert the image to a torch tensor (C, H, W) of dtype float32."""
        if isinstance(image, torch.Tensor):
            return image
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
            if image.ndim == 3 and image.shape[-1] == 3:
                image = np.transpose(image, (2, 0, 1))
            return torch.from_numpy(image)
        if isinstance(image, Image.Image):
            return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        raise TypeError(f"Unsupported image type: {type(image)}")


class RandomCrop(ImageTransform):
    """Randomly crop an image to a given size.

    During training, crops a random sub-region of the image. During
    inference (eval mode), takes the center crop.

    Attributes:
        size: Target crop size (H, W) or just H if scalar.
        padding: Padding to add before cropping.
    """

    def __init__(self, size: Tuple[int, int] = (224, 224), padding: int = 0, to_tensor: bool = False):
        super().__init__(to_tensor=to_tensor)
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.padding = padding

    def __call__(self, image: Any) -> Any:
        if isinstance(image, torch.Tensor) and image.dim() == 3:
            # Torch tensor: (C, H, W)
            _, h, w = image.shape
            th, tw = self.size
            if h > th:
                start_h = random.randint(0, h - th) if self.training else (h - th) // 2
                image = image[:, start_h: start_h + th, :]
            if w > tw:
                start_w = random.randint(0, w - tw) if self.training else (w - tw) // 2
                image = image[:, :, start_w: start_w + tw]
            return image
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                h, w, _ = image.shape
                th, tw = self.size
                if h > th:
                    start_h = random.randint(0, h - th) if self.training else (h - th) // 2
                    image = image[start_h: start_h + th, :, :]
                if w > tw:
                    start_w = random.randint(0, w - tw) if self.training else (w - tw) // 2
                    image = image[:, start_w: start_w + tw, :]
                return image
            return image
        elif isinstance(image, Image.Image):
            w, h = image.size
            th, tw = self.size
            if h > th:
                left = random.randint(0, w - tw) if self.training else (w - tw) // 2
                top = random.randint(0, h - th) if self.training else (h - th) // 2
                image = image.crop((left, top, left + tw, top + th))
            return image
        return image


class RandomFlip(ImageTransform):
    """Randomly flip an image horizontally.

    With probability ``p``, flips the image left-to-right.

    Attributes:
        p: Probability of flipping. Default 0.5.
    """

    def __init__(self, p: float = 0.5, to_tensor: bool = False):
        super().__init__(to_tensor=to_tensor)
        self.p = p

    def __call__(self, image: Any) -> Any:
        if random.random() < self.p:
            if isinstance(image, torch.Tensor):
                image = image.flip(-1)
            elif isinstance(image, np.ndarray):
                image = np.flip(image, axis=-2 if image.ndim == 3 else -1)
            elif isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image


class ColorJitter(ImageTransform):
    """Randomly jitter image brightness, contrast, saturation, and hue.

    Useful for data augmentation to improve robustness to lighting changes.

    Attributes:
        brightness: Brightness jitter factor (0-1 range).
        contrast: Contrast jitter factor (0-1 range).
        saturation: Saturation jitter factor (0-1 range).
        hue: Hue jitter factor (0-1 range).
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        to_tensor: bool = False,
    ):
        super().__init__(to_tensor=to_tensor)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: Any) -> Any:
        if not isinstance(image, (torch.Tensor, np.ndarray)):
            return image

        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                arr = image.astype(np.float32) / 255.0
            else:
                arr = image.copy()

            # Brightness
            if self.brightness > 0:
                factor = 1 + random.uniform(-self.brightness, self.brightness)
                arr = arr * factor
            # Contrast
            if self.contrast > 0:
                mean = arr.mean()
                factor = 1 + random.uniform(-self.contrast, self.contrast)
                arr = (arr - mean) * factor + mean
            # Saturation (on last channel)
            if self.saturation > 0 and arr.shape[-1] >= 3:
                gray = arr.mean(axis=-1, keepdims=True)
                factor = 1 + random.uniform(-self.saturation, self.saturation)
                arr = (arr - gray) * factor + gray

            arr = np.clip(arr, 0, 1)
            return arr

        if isinstance(image, torch.Tensor):
            arr = image.clone()
            if self.brightness > 0:
                factor = 1 + random.uniform(-self.brightness, self.brightness)
                arr = arr * factor
            if self.contrast > 0:
                mean = arr.mean()
                factor = 1 + random.uniform(-self.contrast, self.contrast)
                arr = (arr - mean) * factor + mean
            if self.saturation > 0 and arr.size(0) >= 3:
                gray = arr.mean(dim=0, keepdim=True)
                factor = 1 + random.uniform(-self.saturation, self.saturation)
                arr = (arr - gray) * factor + gray
                factor = 1 + random.uniform(-self.saturation, self.saturation)
                arr = (arr - gray) * factor + gray
            return torch.clamp(arr, 0, 1)

        return image


class NormalizeImage(ImageTransform):
    """Normalize image pixel values.

    Applies channel-wise mean and std normalisation. Supports both
    ImageNet statistics (0.485, 0.456, 0.406) and custom statistics.

    Attributes:
        mean: Channel-wise mean values. Default ImageNet.
        std: Channel-wise std values. Default ImageNet.
    """

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        to_tensor: bool = False,
    ):
        super().__init__(to_tensor=to_tensor)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: Any) -> Any:
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                arr = image.astype(np.float32) / 255.0
            else:
                arr = image.astype(np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1))
            self.mean = self.mean.to(arr.device) if hasattr(arr, 'device') else self.mean
            self.std = self.std.to(arr.device) if hasattr(arr, 'device') else self.std
            return (arr - self.mean.numpy()) / self.std.numpy()
        elif isinstance(image, torch.Tensor):
            return (image - self.mean.to(image.device)) / self.std.to(image.device)
        return image


# ====================================================================
# Action Transforms
# ====================================================================

class ActionTransform:
    """Base class for action transforms.

    All action transforms should implement ``__call__`` which takes
    an action array or tensor and returns the transformed version.
    """

    def __call__(self, action: Any) -> Any:
        raise NotImplementedError


class NormalizeAction(ActionTransform):
    """Normalize action values using per-dimension mean and std.

    Transforms actions from their raw sensor values to a standardised
    distribution for training stability.

    Attributes:
        action_mean: Per-dimension mean for normalisation.
        action_std: Per-dimension std for normalisation.
    """

    def __init__(
        self,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        self.action_mean = torch.tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.tensor(action_std, dtype=torch.float32) + 1e-8

    def __call__(self, action: Any) -> Any:
        if isinstance(action, np.ndarray):
            return (action - self.action_mean.numpy()) / self.action_std.numpy()
        elif isinstance(action, torch.Tensor):
            return (action - self.action_mean.to(action.device)) / self.action_std.to(action.device)
        return action


class DenormalizeAction(ActionTransform):
    """Denormalize action values back to sensor space.

    Inverse of ``NormalizeAction``.  Useful for converting normalised
    predictions back to physical units for robot execution.

    Attributes:
        action_mean: Per-dimension mean used during normalisation.
        action_std: Per-dimension std used during normalisation.
    """

    def __init__(
        self,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        self.action_mean = torch.tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.tensor(action_std, dtype=torch.float32) + 1e-8

    def __call__(self, action: Any) -> Any:
        if isinstance(action, np.ndarray):
            return action * self.action_std.numpy() + self.action_mean.numpy()
        elif isinstance(action, torch.Tensor):
            return action * self.action_std.to(action.device) + self.action_mean.to(action.device)
        return action


class ChunkActions(ActionTransform):
    """Chunk actions into sliding windows for prediction.

    Breaks a long action sequence into overlapping chunks of a fixed
    size.  Each chunk represents a future action horizon.

    Attributes:
        chunk_size: Number of timesteps per chunk.
        stride: Step size between consecutive chunks. Default equals chunk_size (non-overlapping).
    """

    def __init__(self, chunk_size: int = 32, stride: Optional[int] = None):
        self.chunk_size = chunk_size
        self.stride = stride or chunk_size

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        """Chunk a single action sequence.

        Args:
            actions: Action sequence of shape ``(N, action_dim)``.

        Returns:
            Chunked actions of shape ``(num_chunks, chunk_size, action_dim)``.
        """
        N, dim = actions.shape
        chunks: List[np.ndarray] = []
        for start in range(0, N - self.chunk_size + 1, self.stride):
            chunks.append(actions[start: start + self.chunk_size])
        return np.stack(chunks, axis=0) if chunks else actions.reshape(1, 1, dim)


class PadActions(ActionTransform):
    """Pad action sequences to a fixed length.

    Pads shorter sequences with zeros and creates a boolean mask to
    indicate valid timesteps.

    Attributes:
        target_length: Target sequence length for padding.
    """

    def __init__(self, target_length: int = 90):
        self.target_length = target_length

    def __call__(self, actions: np.ndarray) -> Dict[str, Any]:
        """Pad actions to the target length.

        Args:
            actions: Action array of shape ``(N, action_dim)``.

        Returns:
            Dict with 'actions' (padded tensor) and 'mask' (valid timestep mask).
        """
        N, dim = actions.shape
        padded = np.zeros((self.target_length, dim), dtype=np.float32)
        mask = np.zeros(self.target_length, dtype=bool)
        padded[:N, :] = actions
        mask[:N] = True
        return {
            "actions": torch.tensor(padded, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }


# ====================================================================
# State Transforms
# ====================================================================

class StateTransform:
    """Base class for state transforms."""

    def __call__(self, state: Any) -> Any:
        raise NotImplementedError


class NormalizeState(StateTransform):
    """Normalise robot state (joint positions) using per-joint mean and std.

    Attributes:
        state_mean: Per-joint mean for normalisation.
        state_std: Per-joint std for normalisation.
    """

    def __init__(self, state_mean: np.ndarray, state_std: np.ndarray):
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.tensor(state_std, dtype=torch.float32) + 1e-8

    def __call__(self, state: Any) -> Any:
        if isinstance(state, np.ndarray):
            return (state - self.state_mean.numpy()) / self.state_std.numpy()
        elif isinstance(state, torch.Tensor):
            return (state - self.state_mean.to(state.device)) / self.state_std.to(state.device)
        return state


class DenormalizeState(StateTransform):
    """Denormalize state back to original units."""

    def __init__(self, state_mean: np.ndarray, state_std: np.ndarray):
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.tensor(state_std, dtype=torch.float32) + 1e-8

    def __call__(self, state: Any) -> Any:
        if isinstance(state, np.ndarray):
            return state * self.state_std.numpy() + self.state_mean.numpy()
        elif isinstance(state, torch.Tensor):
            return state * self.state_std.to(state.device) + self.state_mean.to(state.device)
        return state


# ====================================================================
# Language Transforms
# ====================================================================

class LanguageTransform:
    """Base class for language transforms."""

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError

    def detokenize(self, tokens: List[int]) -> str:
        raise NotImplementedError


class SimpleTokenizer(LanguageTransform):
    """A simple character-level tokenizer for language instructions.

    Maps characters to token IDs in the range [0, vocab_size-1].
    In production, replace with a proper tokenizer (e.g. BertTokenizer, CLIPTokenizer).

    Attributes:
        vocab_size: Vocabulary size. Default 32.
        max_len: Maximum sequence length. Default 64.
        pad_token: Token ID for padding. Default 0.
    """

    def __init__(self, vocab_size: int = 32, max_len: int = 64, pad_token: int = 0):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_token = pad_token

    def tokenize(self, text: str) -> List[int]:
        """Convert text to a list of token IDs.

        Args:
            text: Input text string.

        Returns:
            List of token IDs, padded to max_len.
        """
        tokens: List[int] = []
        for ch in text.lower():
            if "a" <= ch <= "z":
                tokens.append(ord(ch) - ord("a") + 1)
            elif "0" <= ch <= "9":
                tokens.append(ord(ch) - ord("0") + 27)
            elif ch == " ":
                tokens.append(0)
            else:
                tokens.append(0)  # pad
        # Pad to max_len
        while len(tokens) < self.max_len:
            tokens.append(self.pad_token)
        return tokens[: self.max_len]

    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs back to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Reconstructed text string.
        """
        text: List[str] = []
        for tok in tokens:
            if 1 <= tok <= 26:
                text.append(chr(ord("a") + tok - 1))
            elif 27 <= tok <= 36:
                text.append(chr(ord("0") + tok - 27))
            elif tok == 0:
                text.append(" ")
        return "".join(text).strip()


# ====================================================================
# Composable Transform Pipeline
# ====================================================================

class Compose(nn.Module):
    """Compose multiple transforms into a single callable pipeline.

    Transforms are applied sequentially in the order they are given.

    Usage example::

        transforms = Compose([
            RandomCrop(size=(224, 224)),
            RandomFlip(p=0.5),
            ColorJitter(brightness=0.2),
            NormalizeImage(),
        ])

        # Can be used as a Module (sets training/eval mode)
        transforms.train()  # Enable augmentations
        transforms.eval()   # Disable augmentations (center crop, etc.)

        processed = transforms(image)
    """

    def __init__(self, transforms: Sequence[Any]):
        super().__init__()
        self.transforms = nn.ModuleList()
        for t in transforms:
            if isinstance(t, nn.Module):
                self.transforms.append(t)
            else:
                # Wrap non-Module transforms (e.g., pure functions)
                self.transforms.append(_wrap_callable(t))

    def forward(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x


class _wrap_callable(nn.Module):
    """Wrap a plain callable as an nn.Module to support training/eval mode."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: Any) -> Any:
        return self.fn(x)


# ====================================================================
# Default Transform Pipelines
# ====================================================================

def get_train_transforms(image_size: int = 224) -> nn.Module:
    """Get the default training transform pipeline.

    Includes random crop, horizontal flip, and color jitter augmentations
    followed by ImageNet normalisation.

    Args:
        image_size: Target image size (H=W).

    Returns:
        Composed training transforms.
    """
    return Compose([
        RandomCrop(size=(image_size, image_size)),
        RandomFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        NormalizeImage(),
    ])


def get_val_transforms(image_size: int = 224) -> nn.Module:
    """Get the default validation transform pipeline.

    Only includes center crop and normalisation (no augmentations).

    Args:
        image_size: Target image size (H=W).

    Returns:
        Composed validation transforms.
    """
    return Compose([
        RandomCrop(size=(image_size, image_size)),  # Center crop by default
        NormalizeImage(),
    ])
