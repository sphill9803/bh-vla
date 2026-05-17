"""
bh-VLA Data Loading System

Provides:
    - LeRobotDataset: loads LeRobot/HDF5 format datasets
    - RLDSDataset: loads RLDS/tfds format datasets
    - DirectoryDataset: loads image+action directories
    - ALOHADataset: handles ALOHA-specific format
    - DatasetConfig: configuration dataclass
    - collate_fn: DataLoader collation
    - compute_dataset_stats: compute normalization stats
    - split_dataset: train/val/test splitting
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data
from PIL import Image


# ====================================================================
# Dataset Configuration
# ====================================================================

@dataclass
class DatasetConfig:
    """Configuration for data loading and preprocessing.

    Attributes:
        data_dir: Root directory of the dataset.
        image_keys: List of keys for image arrays in the dataset.
        action_keys: List of keys for action arrays.
        state_key: Key for the state vector.
        language_key: Key for language instructions.
        normalize_actions: Whether to normalize actions. Default True.
        normalize_observations: Whether to normalize observations. Default True.
        max_dataset_size: Maximum number of samples (-1 = all). Default -1.
        image_size: Target image size (H=W). Default 224.
        train_split: Fraction of data for training. Default 0.8.
        val_split: Fraction of data for validation. Default 0.1.
        test_split: Fraction of data for testing. Default 0.1.
        seed: Random seed for reproducibility. Default 42.
    """
    data_dir: str = "./data/aloha_datasets"
    image_keys: List[str] = field(default_factory=lambda: [
        "image", "image_supplementary_1", "image_supplementary_2",
    ])
    action_keys: List[str] = field(default_factory=lambda: ["actions"])
    state_key: str = "state"
    language_key: str = "language"
    normalize_actions: bool = True
    normalize_observations: bool = True
    max_dataset_size: int = -1
    image_size: int = 224
    num_cameras: int = 3
    action_chunk_size: int = 32
    action_dim: int = 14
    state_dim: int = 28
    mode: str = "act"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42

    @property
    def _split_ratios(self) -> List[float]:
        """Return normalised split ratios so they sum to 1.0."""
        total = self.train_split + self.val_split + self.test_split
        return [
            self.train_split / total,
            self.val_split / total,
            self.test_split / total,
        ]


def _split_items(items: List[str], config: DatasetConfig, split: str) -> List[str]:
    """Deterministically split episode paths into train/val/test partitions."""
    rng = random.Random(config.seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_ratio, val_ratio, _ = config._split_ratios
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if split == "train":
        return shuffled[:train_end] or shuffled[:1]
    if split == "val":
        return shuffled[train_end:val_end] or shuffled[:1]
    if split == "test":
        return shuffled[val_end:] or shuffled[-1:]
    return shuffled


def _pad_or_trim_vector(values: Any, target_dim: int) -> np.ndarray:
    """Return a 1D float32 vector with exactly ``target_dim`` elements."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_dim <= 0:
        return arr
    if arr.size >= target_dim:
        return arr[:target_dim]
    out = np.zeros(target_dim, dtype=np.float32)
    out[:arr.size] = arr
    return out


def _slice_state(states: Any, timestep: int, state_dim: int) -> np.ndarray:
    """Pick the current robot state at ``timestep`` and make it model-sized."""
    arr = np.asarray(states, dtype=np.float32)
    if arr.ndim <= 1:
        return _pad_or_trim_vector(arr, state_dim)
    t = min(max(timestep, 0), arr.shape[0] - 1)
    return _pad_or_trim_vector(arr[t], state_dim)


def _slice_action_chunk(
    actions: Any,
    start: int,
    chunk_size: int,
    action_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a fixed-size future action chunk and a True=ignore padding mask."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        arr = np.zeros((1, action_dim), dtype=np.float32)

    start = min(max(start, 0), max(arr.shape[0] - 1, 0))
    chunk = arr[start:start + chunk_size]

    valid = chunk.shape[0]
    if valid == 0:
        chunk = np.zeros((1, arr.shape[-1]), dtype=np.float32)
        valid = 1

    if valid < chunk_size:
        pad_value = chunk[-1:]
        pad = np.repeat(pad_value, chunk_size - valid, axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)

    if action_dim > 0:
        fixed = np.zeros((chunk_size, action_dim), dtype=np.float32)
        dim = min(action_dim, chunk.shape[1])
        fixed[:, :dim] = chunk[:, :dim]
        chunk = fixed

    ignore_mask = np.ones(chunk_size, dtype=bool)
    ignore_mask[:valid] = False
    return chunk.astype(np.float32), ignore_mask


def _ensure_rgb_image(image: Any, image_size: int) -> np.ndarray:
    """Convert an image-like object to HWC uint8 RGB with a fixed square size."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        arr = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
    if arr.shape[0] != image_size or arr.shape[1] != image_size:
        arr = np.asarray(Image.fromarray(arr).resize((image_size, image_size), Image.BILINEAR))
    return arr


def _load_image_array(path: str, timestep: int) -> Optional[np.ndarray]:
    """Load either a whole episode image array or a per-timestep image array."""
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.ndim >= 4:
        t = min(max(timestep, 0), arr.shape[0] - 1)
        return arr[t]
    return arr


def _load_episode_images(
    ep_dir: str,
    timestep: int,
    keys: Sequence[str],
    image_size: int,
) -> Dict[str, np.ndarray]:
    """Load multi-camera images, supporting both nested and simple layouts."""
    images: Dict[str, np.ndarray] = {}

    root_images = _load_image_array(os.path.join(ep_dir, "images.npy"), timestep)
    for idx, key in enumerate(keys):
        candidates = [
            os.path.join(ep_dir, "images", f"{key}.npy"),
            os.path.join(ep_dir, f"{key}.npy"),
        ]
        image = None
        for path in candidates:
            image = _load_image_array(path, timestep)
            if image is not None:
                break

        # The simple collector saves a single camera as images.npy. Duplicate it
        # across expected camera slots so the policy still receives N cameras.
        if image is None and root_images is not None:
            image = root_images

        if image is None:
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        images[key] = _ensure_rgb_image(image, image_size)

    return images


def _normalise_image_tensor(images: torch.Tensor, mode: str) -> torch.Tensor:
    """Normalize stacked images for ACT (ImageNet) or pi0.5 (SigLIP-style)."""
    images = images.float() / 255.0
    if mode == "pi05":
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (images - mean) / std


def _image_dict_to_tensor(images: Dict[str, Any], num_cameras: int, image_size: int, mode: str) -> torch.Tensor:
    """Convert a camera dict to (num_cameras, 3, H, W)."""
    ordered = list(images.values())
    if not ordered:
        ordered = [np.zeros((image_size, image_size, 3), dtype=np.uint8)]
    while len(ordered) < num_cameras:
        ordered.append(ordered[-1])
    ordered = ordered[:num_cameras]

    tensors = []
    for image in ordered:
        arr = _ensure_rgb_image(image, image_size)
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return _normalise_image_tensor(torch.stack(tensors, dim=0), mode)


def _tokenize_act(text: str, max_len: int = 128) -> torch.Tensor:
    """Character tokenizer matching the lightweight ACT language encoder."""
    chars = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        " .,!?;:'\"-()[]{}|/\\@#$%^&*~`"
        "\t\n"
    )
    vocab = ["<pad>", "<unk>", "<sos>", "<eos>"] + list(chars)
    table = {ch: i for i, ch in enumerate(vocab)}
    ids = [2]
    ids.extend(table.get(ch, 1) for ch in text.lower()[: max_len - 2])
    ids.append(3)
    ids.extend([0] * (max_len - len(ids)))
    return torch.tensor(ids[:max_len], dtype=torch.long)


def _stable_word_id(word: str, vocab_size: int) -> int:
    digest = hashlib.sha1(word.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(vocab_size - 2, 1) + 2


def _tokenize_pi05(text: str, max_len: int = 128, vocab_size: int = 32000) -> torch.Tensor:
    """Deterministic placeholder tokenizer for the pi0.5 scaffold."""
    ids = [0] * max_len
    words = text.lower().split()
    last_idx = 0
    for i, word in enumerate(words[: max_len - 2], start=1):
        ids[i] = _stable_word_id(word, vocab_size)
        last_idx = i
    ids[min(last_idx + 1, max_len - 1)] = 1
    return torch.tensor(ids, dtype=torch.long)


# ====================================================================
# LeRobot Dataset
# ====================================================================

class LeRobotDataset(torch.utils.data.Dataset):
    """Dataset loader for LeRobot/HDF5 formatted data.

    LeRobot datasets store episodes in an HDF5 file with the following structure:
        data/
        ├── dataset_infos.json   # Metadata
        ├── data.hdf5            # Main data file
        │   ├── data/
        │   │   ├── episode_0000/
        │   │   │   ├── images/
        │   │   │   │   ├── image        (N, H, W, 3) uint8
        │   │   │   │   ├── image_supplementary_1  (N, H, W, 3) uint8
        │   │   │   │   └── image_supplementary_2  (N, H, W, 3) uint8
        │   │   │   ├── actions        (N, action_dim) float32
        │   │   │   ├── states         (N, state_dim) float32
        │   │   │   └── language       (N,) list of text strings
        │   │   ├── episode_0001/
        │   │   └── ...

    The dataset supports:
        - Lazy loading via HDF5 references (memory efficient)
        - Per-dimension z-score action normalisation
        - Random crop / flip / color jitter augmentations
        - Episode-based sampling with configurable chunk sizes

    Attributes:
        data_dir: Path to the LeRobot dataset root.
        config: DatasetConfig instance.
        episode_refs: List of episode keys in the HDF5 file.
        normalization_stats: Dict of normalization parameters.
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[DatasetConfig] = None,
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.config = config or DatasetConfig(data_dir=data_dir)
        self.split = split

        # Internal state
        self.h5_file: Any = None
        self.episode_refs: List[str] = []
        self.normalization_stats: Dict[str, np.ndarray] = {}

        self.load_data()

    def load_data(self) -> None:
        """Load dataset from the LeRobot HDF5 file.

        Steps:
            1. Open the HDF5 file and discover episode keys.
            2. Compute per-dimension action mean/std across all episodes.
            3. Optionally limit to max_dataset_size samples.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for LeRobot format. Install with: pip install h5py")

        h5_path = os.path.join(self.data_dir, "data.hdf5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        self.h5_file = h5py.File(h5_path, "r")
        data_group = self.h5_file.get("data")
        if data_group is None:
            raise KeyError(f"No 'data' group found in {h5_path}")

        self.episode_refs = sorted(data_group.keys())
        if self.config.max_dataset_size > 0:
            self.episode_refs = self.episode_refs[: self.config.max_dataset_size]

        # Compute normalisation stats if requested
        if self.config.normalize_actions:
            self._compute_action_stats()

    def _compute_action_stats(self) -> None:
        """Compute per-dimension mean and std of actions across all episodes."""
        import h5py

        all_actions: List[np.ndarray] = []
        for ep_name in self.episode_refs:
            actions_key = self.config.action_keys[0]
            episode_group = self.h5_file[f"data/{ep_name}"]
            if actions_key in episode_group:
                actions = episode_group[actions_key][()]
                all_actions.append(actions)
            else:
                # Try alternate action key
                for key in self.config.action_keys:
                    if key in episode_group:
                        all_actions.append(episode_group[key][()])
                        break

        if all_actions:
            combined = np.concatenate(all_actions, axis=0)
            self.normalization_stats["action_mean"] = combined.mean(axis=0)
            self.normalization_stats["action_std"] = combined.std(axis=0) + 1e-8

    def __len__(self) -> int:
        # HDF5 episodes can have different lengths.  We keep the public length
        # episode-based for memory safety and choose a random timestep inside
        # each selected episode in __getitem__.
        return max(len(_split_items(self.episode_refs, self.config, self.split)), 1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample.

        Args:
            idx: Index of the episode (within the split).

        Returns:
            Dictionary with keys: images, actions, state, language.
        """
        split_refs = _split_items(self.episode_refs, self.config, self.split)
        ep_name = split_refs[idx % len(split_refs)]
        episode = self.h5_file[f"data/{ep_name}"]

        # Load actions
        actions = None
        for key in self.config.action_keys:
            if key in episode:
                actions = episode[key][()]
                break

        if actions is None:
            actions = np.zeros((1, self.config.action_dim), dtype=np.float32)

        # Select a random timestep within the episode
        seq_len = len(actions)
        t = random.randint(0, max(seq_len - 1, 0))

        # Load images for this timestep
        images = {}
        for key in self.config.image_keys:
            if key in episode:
                img_arr = episode[key][t]
                if img_arr.ndim == 2:
                    img_arr = np.stack([img_arr] * 3, axis=-1)
                images[key] = img_arr

        # Load state
        if self.config.state_key in episode:
            states = episode[self.config.state_key][()]
        else:
            states = np.zeros((1, self.config.state_dim), dtype=np.float32)

        # Load language instruction
        if self.config.language_key in episode:
            lang_list = episode[self.config.language_key][()]
            if isinstance(lang_list, np.ndarray) and lang_list.dtype.kind in ("U", "S"):
                instruction = lang_list[t].decode("utf-8") if isinstance(lang_list[t], bytes) else str(lang_list[t])
            else:
                instruction = str(lang_list[t])
        else:
            instruction = "pick up the object"

        # Normalize actions
        if self.config.normalize_actions and self.normalization_stats:
            mean = self.normalization_stats["action_mean"]
            std = self.normalization_stats["action_std"]
            actions = (actions - mean) / (std + 1e-8)

        action_chunk, action_mask = _slice_action_chunk(
            actions,
            t,
            self.config.action_chunk_size,
            self.config.action_dim,
        )

        return {
            "images": images,
            "actions": torch.tensor(action_chunk, dtype=torch.float32),
            "action_mask": torch.tensor(action_mask, dtype=torch.bool),
            "state": torch.tensor(_slice_state(states, t, self.config.state_dim), dtype=torch.float32),
            "language": instruction,
        }

    def close(self) -> None:
        """Close the HDF5 file handle."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self) -> None:
        self.close()


# ====================================================================
# RLDS Dataset
# ====================================================================

class RLDSDataset(torch.utils.data.Dataset):
    """Dataset loader for RLDS/tfds format data.

    RLDS datasets are typically stored as TFRecord files with tfds features.
    This loader parses the TFRecord format and yields individual samples.

    Expected directory structure:
        data/
        ├── dataset_infos.json
        ├── train/
        │   └── dataset_0.tfrecord
        ├── validation/
        │   └── dataset_0.tfrecord
        └── test/
            └── dataset_0.tfrecord

    Attributes:
        data_dir: Path to the RLDS dataset root.
        config: DatasetConfig instance.
        tf_records: List of TFRecord file paths.
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[DatasetConfig] = None,
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.config = config or DatasetConfig(data_dir=data_dir)
        self.split = split

        self.tf_records: List[str] = []
        self.normalization_stats: Dict[str, np.ndarray] = {}
        self._samples_per_record: List[int] = []

        self.load_data()

    def load_data(self) -> None:
        """Load TFRecord file paths and compute normalisation statistics."""
        split_dir = os.path.join(self.data_dir, self.split)
        if not os.path.isdir(split_dir):
            # Try without split subdirectory
            split_dir = self.data_dir

        self.tf_records = sorted(glob.glob(os.path.join(split_dir, "**/*.tfrecord"), recursive=True))
        if not self.tf_records:
            raise FileNotFoundError(f"No TFRecord files found in {split_dir}")

        # Compute normalisation stats from the first record
        if self.config.normalize_actions:
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("tensorflow_datasets is required for RLDS format. Install with: pip install tensorflow-datasets")

            # Load first record to compute stats
            first_path = self.tf_records[0]
            with tfds.load("rlds", split="train", data_dir=self.data_dir) as ds:
                all_actions = []
                for sample in ds.take(100):  # Sample 100 episodes for stats
                    if "action" in sample:
                        actions = sample["action"].numpy()
                        all_actions.append(actions)
                if all_actions:
                    combined = np.concatenate(all_actions, axis=0)
                    self.normalization_stats["action_mean"] = combined.mean(axis=0)
                    self.normalization_stats["action_std"] = combined.std(axis=0) + 1e-8

            # Estimate samples per record
            self._samples_per_record = [len(self.tf_records)]  # Rough estimate

    def __len__(self) -> int:
        # Estimate total samples
        total = sum(self._samples_per_record) if self._samples_per_record else 0
        return total

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample from the RLDS dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with keys: images, actions, state, language.
        """
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError("tensorflow_datasets is required for RLDS format.")

        # Compute which record and sample within that record
        record_idx = 0
        offset = idx
        for i, n in enumerate(self._samples_per_record):
            if offset < n:
                record_idx = i
                break
            offset -= n

        # Load the TFRecord
        ds_path = self.tf_records[record_idx]
        ds = tfds.core.DatasetBuilder("rlds", data_dir=self.data_dir)
        split_map = {"train": "train", "val": "validation", "test": "test"}
        split_name = split_map.get(self.split, "train")

        with tfds.load("rlds", split=split_name, data_dir=self.data_dir) as dataset:
            for i, sample in enumerate(dataset):
                if i == offset:
                    # Process the sample
                    images = {}
                    for key in self.config.image_keys:
                        if key in sample:
                            img = sample[key].numpy()
                            images[key] = img

                    actions = sample["action"].numpy()
                    state = sample.get("observation/state", np.zeros(28, dtype=np.float32)).numpy()

                    if isinstance(state, (list, tuple)):
                        state = np.array(state, dtype=np.float32)

                    lang = sample.get("language", "pick up the object")
                    if isinstance(lang, bytes):
                        lang = lang.decode("utf-8")
                    else:
                        lang = str(lang)

                    if self.config.normalize_actions and self.normalization_stats:
                        mean = self.normalization_stats["action_mean"]
                        std = self.normalization_stats["action_std"]
                        actions = (actions - mean) / (std + 1e-8)

                    return {
                        "images": images,
                        "actions": torch.tensor(actions, dtype=torch.float32),
                        "state": torch.tensor(state, dtype=torch.float32),
                        "language": lang,
                    }

        # Fallback if we can't find the sample
        return self._get_fallback_sample()

    def _get_fallback_sample(self) -> Dict[str, Any]:
        """Return a fallback sample when actual data is unavailable."""
        return {
            "images": {k: np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
                       for k in self.config.image_keys},
            "actions": torch.zeros(1, 14, dtype=torch.float32),
            "state": torch.zeros(28, dtype=torch.float32),
            "language": "pick up the object",
        }


# ====================================================================
# Directory Dataset
# ====================================================================

class DirectoryDataset(torch.utils.data.Dataset):
    """Dataset loader for image+action directory format.

    Expected directory structure:
        data/
        ├── episode_0000/
        │   ├── images/
        │   │   ├── image.npy           (N, H, W, 3) uint8
        │   │   ├── image_supplementary_1.npy
        │   │   └── image_supplementary_2.npy
        │   ├── actions.npy             (N, action_dim) float32
        │   ├── states.npy              (N, state_dim) float32
        │   └── metadata.json
        ├── episode_0001/
        │   └── ...

    This format is simple and doesn't require HDF5 — each episode is a
    self-contained directory with numpy arrays.

    Attributes:
        data_dir: Path to the dataset root.
        config: DatasetConfig instance.
        episode_dirs: List of episode directory paths.
        normalization_stats: Dict of normalisation parameters.
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[DatasetConfig] = None,
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.config = config or DatasetConfig(data_dir=data_dir)
        self.split = split

        self.episode_dirs: List[str] = []
        self.normalization_stats: Dict[str, np.ndarray] = {}
        self._episode_lengths: List[int] = []  # Length of each episode

        self.load_data()

    def load_data(self) -> None:
        """Load episode directories and compute normalisation statistics."""
        episode_dirs = sorted(glob.glob(os.path.join(self.data_dir, "episode_*")))
        if not episode_dirs:
            episode_dirs = sorted(glob.glob(os.path.join(self.data_dir, "episodes", "episode_*")))

        if not episode_dirs:
            raise FileNotFoundError(f"No episode directories found in {self.data_dir}")

        episode_dirs = episode_dirs[: self.config.max_dataset_size] if self.config.max_dataset_size > 0 else episode_dirs
        self.episode_dirs = _split_items(episode_dirs, self.config, self.split)

        # Compute normalisation stats from the first few episodes
        if self.config.normalize_actions:
            all_actions: List[np.ndarray] = []
            for ep_dir in self.episode_dirs[:10]:  # Sample first 10 episodes
                actions_path = os.path.join(ep_dir, "actions.npy")
                if os.path.exists(actions_path):
                    actions = np.load(actions_path)
                    all_actions.append(actions)

            if all_actions:
                combined = np.concatenate(all_actions, axis=0)
                self.normalization_stats["action_mean"] = combined.mean(axis=0)
                self.normalization_stats["action_std"] = combined.std(axis=0) + 1e-8

        # Store episode lengths
        for ep_dir in self.episode_dirs:
            states_path = os.path.join(ep_dir, "states.npy")
            if os.path.exists(states_path):
                self._episode_lengths.append(len(np.load(states_path)))
            else:
                self._episode_lengths.append(0)

    def __len__(self) -> int:
        # Total number of samples across all episodes
        total = sum(self._episode_lengths)
        return total

    def _get_episode_and_offset(self, idx: int) -> Tuple[int, int]:
        """Map a flat index to (episode_index, timestep_within_episode)."""
        for i, length in enumerate(self._episode_lengths):
            if idx < length:
                return i, idx
            idx -= length
        return len(self._episode_lengths) - 1, 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample.

        Args:
            idx: Flat index across all episodes.

        Returns:
            Dictionary with keys: images, actions, state, language.
        """
        ep_idx, t = self._get_episode_and_offset(idx)
        ep_dir = self.episode_dirs[ep_idx]

        # Load metadata for language instruction
        metadata_path = os.path.join(ep_dir, "metadata.json")
        language = "pick up the object"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                language = meta.get("instruction", "pick up the object")

        # Load actions
        actions_path = os.path.join(ep_dir, "actions.npy")
        actions = np.load(actions_path) if os.path.exists(actions_path) else np.zeros((1, self.config.action_dim), dtype=np.float32)

        # Load state
        state_path = os.path.join(ep_dir, "states.npy")
        states = np.load(state_path) if os.path.exists(state_path) else np.zeros((1, self.config.state_dim), dtype=np.float32)

        # Load images for this timestep.  Supports both the nested multi-camera
        # layout and the simple collector layout with root-level images.npy.
        images = _load_episode_images(
            ep_dir,
            t,
            self.config.image_keys,
            self.config.image_size,
        )

        # Normalize actions
        if self.config.normalize_actions and self.normalization_stats:
            mean = self.normalization_stats["action_mean"]
            std = self.normalization_stats["action_std"]
            actions = (actions - mean) / (std + 1e-8)

        action_chunk, action_mask = _slice_action_chunk(
            actions,
            t,
            self.config.action_chunk_size,
            self.config.action_dim,
        )

        return {
            "images": images,
            "actions": torch.tensor(action_chunk, dtype=torch.float32),
            "action_mask": torch.tensor(action_mask, dtype=torch.bool),
            "state": torch.tensor(_slice_state(states, t, self.config.state_dim), dtype=torch.float32),
            "language": language,
        }


# ====================================================================
# ALOHA Dataset
# ====================================================================

class ALOHADataset(torch.utils.data.Dataset):
    """Dataset loader for ALOHA-specific data format.

    ALOHA datasets store data in a slightly different directory structure:
        data/
        ├── dataset_info.json
        └── episodes/
            ├── episode_0000/
            │   ├── images/
            │   │   ├── image.npy              (N, H, W, 3)
            │   │   ├── image_manipulator_1.npy
            │   │   └── image_manipulator_2.npy
            │   ├── states.npy                 (N, 28)
            │   ├── actions.npy                (N, 28)
            │   └── metadata.json
            └── episode_0001/
            └── ...

    ALOHA-specific features:
        - Supports both leader and follower arm data
        - Gripper position normalisation (0-1 range)
        - Per-arm joint statistics for normalisation
        - Camera names specific to ALOHA (image_manipulator_*)

    Attributes:
        data_dir: Path to the ALOHA dataset root.
        config: DatasetConfig instance.
        episode_dirs: List of episode directory paths.
        normalization_stats: Dict of normalisation parameters (per-arm).
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[DatasetConfig] = None,
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.config = config or DatasetConfig(data_dir=data_dir)
        self.split = split

        self.episode_dirs: List[str] = []
        self.normalization_stats: Dict[str, np.ndarray] = {}
        self._episode_lengths: List[int] = []

        self.load_data()

    def load_data(self) -> None:
        """Load ALOHA episode directories and compute normalisation statistics."""
        episode_dirs = sorted(glob.glob(os.path.join(self.data_dir, "episode_*")))
        if not episode_dirs:
            # Try alternative directory structure
            episode_dirs = sorted(glob.glob(os.path.join(self.data_dir, "**/episode_*"), recursive=True))

        if not episode_dirs:
            raise FileNotFoundError(f"No ALOHA episodes found in {self.data_dir}")

        episode_dirs = episode_dirs[: self.config.max_dataset_size] if self.config.max_dataset_size > 0 else episode_dirs
        self.episode_dirs = _split_items(episode_dirs, self.config, self.split)

        # Compute per-arm normalisation stats
        if self.config.normalize_actions:
            left_actions: List[np.ndarray] = []
            right_actions: List[np.ndarray] = []
            for ep_dir in self.episode_dirs[:10]:
                actions_path = os.path.join(ep_dir, "actions.npy")
                if os.path.exists(actions_path):
                    actions = np.load(actions_path)
                    # ALOHA actions: first 7 = left arm, last 7 = right arm
                    if actions.shape[1] >= 14:
                        left_actions.append(actions[:, :7])
                        right_actions.append(actions[:, 7:])

            if left_actions:
                combined_left = np.concatenate(left_actions, axis=0)
                combined_right = np.concatenate(right_actions, axis=0)
                self.normalization_stats["left_action_mean"] = combined_left.mean(axis=0)
                self.normalization_stats["left_action_std"] = combined_left.std(axis=0) + 1e-8
                self.normalization_stats["right_action_mean"] = combined_right.mean(axis=0)
                self.normalization_stats["right_action_std"] = combined_right.std(axis=0) + 1e-8

        # Store episode lengths
        for ep_dir in self.episode_dirs:
            states_path = os.path.join(ep_dir, "states.npy")
            if os.path.exists(states_path):
                self._episode_lengths.append(len(np.load(states_path)))
            else:
                self._episode_lengths.append(0)

    def __len__(self) -> int:
        return sum(self._episode_lengths)

    def _get_episode_and_offset(self, idx: int) -> Tuple[int, int]:
        """Map flat index to (episode_index, timestep)."""
        for i, length in enumerate(self._episode_lengths):
            if idx < length:
                return i, idx
            idx -= length
        return len(self._episode_lengths) - 1, 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single ALOHA training sample.

        Args:
            idx: Flat index across all episodes.

        Returns:
            Dictionary with keys: images, actions, state, language.
        """
        ep_idx, t = self._get_episode_and_offset(idx)
        ep_dir = self.episode_dirs[ep_idx]

        # Load metadata
        metadata_path = os.path.join(ep_dir, "metadata.json")
        language = "pick up the object"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                language = meta.get("instruction", "pick up the object")

        # Load actions
        actions_path = os.path.join(ep_dir, "actions.npy")
        actions = np.load(actions_path) if os.path.exists(actions_path) else np.zeros((1, self.config.action_dim), dtype=np.float32)

        # Load state
        state_path = os.path.join(ep_dir, "states.npy")
        states = np.load(state_path) if os.path.exists(state_path) else np.zeros((1, self.config.state_dim), dtype=np.float32)

        # Load images (ALOHA-specific camera names)
        aloha_image_keys = [
            "image", "image_manipulator_1", "image_manipulator_2",
        ]
        images = _load_episode_images(
            ep_dir,
            t,
            aloha_image_keys,
            self.config.image_size,
        )

        # Normalize actions with per-arm statistics
        if self.config.normalize_actions and self.normalization_stats:
            if "left_action_mean" in self.normalization_stats:
                actions = np.array(actions, dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions.reshape(1, -1)
                if actions.shape[1] >= 14:
                    actions[:, :7] = (
                        actions[:, :7] - self.normalization_stats["left_action_mean"]
                    ) / self.normalization_stats["left_action_std"]
                    actions[:, 7:14] = (
                        actions[:, 7:14] - self.normalization_stats["right_action_mean"]
                    ) / self.normalization_stats["right_action_std"]
            else:
                # Fallback: global normalisation
                mean = self.normalization_stats.get("action_mean", np.zeros(28))
                std = self.normalization_stats.get("action_std", np.ones(28) + 1e-8)
                actions = (actions - mean) / (std + 1e-8)

        action_chunk, action_mask = _slice_action_chunk(
            actions,
            t,
            self.config.action_chunk_size,
            self.config.action_dim,
        )

        return {
            "images": images,
            "actions": torch.tensor(action_chunk, dtype=torch.float32),
            "action_mask": torch.tensor(action_mask, dtype=torch.bool),
            "state": torch.tensor(_slice_state(states, t, self.config.state_dim), dtype=torch.float32),
            "language": language,
        }


# ====================================================================
# Collate Function
# ====================================================================

def collate_fn(
    batch: List[Dict[str, Any]],
    mode: str = "act",
    image_size: int = 224,
    num_cameras: int = 3,
    action_chunk_size: int = 32,
    action_dim: int = 14,
    state_dim: int = 28,
    max_language_len: int = 128,
) -> Dict[str, Any]:
    """Collate a batch of samples for training.

    Handles variable-length sequences by padding actions to the longest
    sequence in the batch. Images and states are already fixed-size.

    Args:
        batch: List of sample dictionaries from the dataset.

    Returns:
        Collated batch dictionary with batched tensors.
    """
    # Collect all fields
    images_list: List[Dict[str, Any]] = []
    actions_list: List[torch.Tensor] = []
    states_list: List[torch.Tensor] = []
    languages: List[str] = []

    for sample in batch:
        images_list.append(sample["images"])
        actions_list.append(sample["actions"])
        states_list.append(sample["state"])
        languages.append(sample["language"])

    # Batch images into the model-ready shape:
    # (B, num_cameras, 3, image_size, image_size)
    batched_images = torch.stack([
        _image_dict_to_tensor(img, num_cameras=num_cameras, image_size=image_size, mode=mode)
        for img in images_list
    ], dim=0)

    # Pad actions to the requested chunk length.  action_mask follows the
    # convention used by ACTLoss: True means "ignore this padded timestep".
    max_len = max(action_chunk_size, max(a.size(0) for a in actions_list))
    batched_actions = torch.zeros(len(batch), max_len, action_dim, dtype=torch.float32)
    action_mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, act in enumerate(actions_list):
        act = act.float()
        L = min(act.size(0), max_len)
        D = min(act.size(1), action_dim)
        batched_actions[i, :L, :D] = act[:L, :D]
        sample_mask = batch[i].get("action_mask")
        if sample_mask is not None:
            sample_mask = sample_mask.bool()[:L]
            action_mask[i, :L] = sample_mask
        else:
            action_mask[i, :L] = False

    # Stack states (already fixed size)
    batched_states = torch.stack([
        torch.tensor(_pad_or_trim_vector(state, state_dim), dtype=torch.float32)
        for state in states_list
    ], dim=0)

    if mode == "pi05":
        language_ids = torch.stack([
            _tokenize_pi05(lang, max_len=max_language_len)
            for lang in languages
        ], dim=0)
    else:
        language_ids = torch.stack([
            _tokenize_act(lang, max_len=max_language_len)
            for lang in languages
        ], dim=0)

    return {
        "images": batched_images,
        "actions": batched_actions,
        "state": batched_states,
        "action_mask": action_mask,
        "language": languages,
        "language_ids": language_ids,
    }


# ====================================================================
# Dataset Statistics
# ====================================================================

def compute_dataset_stats(
    data_dir: str,
    dataset_type: str = "directory",
    max_episodes: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute per-dimension mean and std statistics across the dataset.

    Useful for computing normalisation parameters before training.

    Args:
        data_dir: Path to the dataset root.
        dataset_type: One of 'directory', 'lerobot', 'rlds', 'aloha'.
        max_episodes: Maximum number of episodes to process.

    Returns:
        Dict with 'action_mean', 'action_std', 'state_mean', 'state_std'.
    """
    all_actions: List[np.ndarray] = []
    all_states: List[np.ndarray] = []

    if dataset_type == "directory":
        episode_dirs = sorted(glob.glob(os.path.join(data_dir, "episode_*")))[:max_episodes]
        for ep_dir in episode_dirs:
            actions_path = os.path.join(ep_dir, "actions.npy")
            states_path = os.path.join(ep_dir, "states.npy")
            if os.path.exists(actions_path):
                all_actions.append(np.load(actions_path))
            if os.path.exists(states_path):
                all_states.append(np.load(states_path))

    elif dataset_type == "lerobot":
        import h5py
        h5_path = os.path.join(data_dir, "data.hdf5")
        if os.path.exists(h5_path):
            with h5py.File(h5_path, "r") as f:
                data_group = f.get("data")
                if data_group:
                    episodes = sorted(data_group.keys())[:max_episodes]
                    for ep_name in episodes:
                        ep = data_group[ep_name]
                        if "actions" in ep:
                            all_actions.append(ep["actions"][()])
                        if "states" in ep:
                            all_states.append(ep["states"][()])

    elif dataset_type == "aloha":
        episode_dirs = sorted(glob.glob(os.path.join(data_dir, "episode_*")))[:max_episodes]
        for ep_dir in episode_dirs:
            actions_path = os.path.join(ep_dir, "actions.npy")
            states_path = os.path.join(ep_dir, "states.npy")
            if os.path.exists(actions_path):
                all_actions.append(np.load(actions_path))
            if os.path.exists(states_path):
                all_states.append(np.load(states_path))

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    if not all_actions:
        return {}

    combined_actions = np.concatenate(all_actions, axis=0)
    combined_states = np.concatenate(all_states, axis=0) if all_states else None

    stats = {
        "action_mean": combined_actions.mean(axis=0),
        "action_std": combined_actions.std(axis=0) + 1e-8,
        "state_mean": combined_states.mean(axis=0) if combined_states is not None else np.zeros(28),
        "state_std": combined_states.std(axis=0) + 1e-8 if combined_states is not None else np.ones(28),
    }
    return stats


# ====================================================================
# Dataset Splitting
# ====================================================================

def split_dataset(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Split a dataset into train/val/test subsets.

    Creates separate subdirectories with the appropriate episode splits.

    Args:
        data_dir: Path to the source dataset directory.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for shuffling.
        output_dir: Directory to write split datasets (default: ./splits/).

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to output paths.
    """
    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    # Discover episodes
    episode_dirs = sorted(glob.glob(os.path.join(data_dir, "episode_*")))
    if not episode_dirs:
        episode_dirs = sorted(glob.glob(os.path.join(data_dir, "episodes", "episode_*")))

    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found in {data_dir}")

    # Shuffle
    random.seed(seed)
    shuffled = episode_dirs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_episodes = shuffled[:train_end]
    val_episodes = shuffled[train_end:val_end]
    test_episodes = shuffled[val_end:]

    # Create output directories
    output_dir = output_dir or os.path.join(data_dir, "..", "splits")
    output_dir = os.path.abspath(output_dir)
    splits = {}
    for name, episodes in [("train", train_episodes), ("val", val_episodes), ("test", test_episodes)]:
        split_dir = os.path.join(output_dir, name)
        os.makedirs(split_dir, exist_ok=True)
        splits[name] = split_dir

        # Copy episode directories
        for ep_dir in episodes:
            ep_name = os.path.basename(ep_dir)
            dest = os.path.join(split_dir, ep_name)
            if not os.path.exists(dest):
                import shutil
                shutil.copytree(ep_dir, dest)

    # Save split info
    split_info = {
        "train": train_episodes,
        "val": val_episodes,
        "test": test_episodes,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_episodes": n,
    }
    info_path = os.path.join(output_dir, "split_info.json")
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2, default=str)

    return splits
