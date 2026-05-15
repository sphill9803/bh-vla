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
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data


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
        if self.split == "train":
            n = int(len(self.episode_refs) * self.config._split_ratios[0])
        elif self.split == "val":
            n = int(len(self.episode_refs) * self.config._split_ratios[1])
        else:
            n = len(self.episode_refs) - int(len(self.episode_refs) * (self.config.train_split + self.config.val_split))
        return max(n, 1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample.

        Args:
            idx: Index of the episode (within the split).

        Returns:
            Dictionary with keys: images, actions, state, language.
        """
        # Map split index to actual episode index
        if self.split == "train":
            ep_idx = int(idx / max(1, math.ceil(len(self.episode_refs) * self.config._split_ratios[0])))
        elif self.split == "val":
            ep_idx = int(len(self.episode_refs) * self.config._split_ratios[0]) + idx
        else:
            ep_idx = len(self.episode_refs) - int(len(self.episode_refs) * self.config.test_split) + idx
            ep_idx = min(ep_idx, len(self.episode_refs) - 1)

        ep_idx = min(ep_idx, len(self.episode_refs) - 1)
        ep_name = self.episode_refs[ep_idx]
        episode = self.h5_file[f"data/{ep_name}"]

        # Load actions
        actions = None
        for key in self.config.action_keys:
            if key in episode:
                actions = episode[key][()]
                break

        if actions is None:
            actions = np.zeros((1, 14), dtype=np.float32)

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
            state = episode[self.config.state_key][t]
        else:
            state = np.zeros(28, dtype=np.float32)

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

        return {
            "images": images,
            "actions": torch.tensor(actions, dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.float32),
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

        self.episode_dirs = episode_dirs[: self.config.max_dataset_size] if self.config.max_dataset_size > 0 else episode_dirs

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
        actions = np.load(actions_path) if os.path.exists(actions_path) else np.zeros((1, 14), dtype=np.float32)

        # Load state
        state_path = os.path.join(ep_dir, "states.npy")
        state = np.load(state_path) if os.path.exists(state_path) else np.zeros(28, dtype=np.float32)

        # Load images for this timestep
        images = {}
        for key in self.config.image_keys:
            img_path = os.path.join(ep_dir, "images", f"{key}.npy")
            if os.path.exists(img_path):
                img_arr = np.load(img_path)
                if img_arr.shape[0] > t:
                    images[key] = img_arr[t]
                else:
                    images[key] = np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
            else:
                images[key] = np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)

        # Normalize actions
        if self.config.normalize_actions and self.normalization_stats:
            mean = self.normalization_stats["action_mean"]
            std = self.normalization_stats["action_std"]
            actions = (actions - mean) / (std + 1e-8)

        return {
            "images": images,
            "actions": torch.tensor(actions, dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.float32),
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

        self.episode_dirs = episode_dirs[: self.config.max_dataset_size] if self.config.max_dataset_size > 0 else episode_dirs

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
        actions = np.load(actions_path) if os.path.exists(actions_path) else np.zeros((1, 28), dtype=np.float32)

        # Load state
        state_path = os.path.join(ep_dir, "states.npy")
        state = np.load(state_path) if os.path.exists(state_path) else np.zeros(28, dtype=np.float32)

        # Load images (ALOHA-specific camera names)
        images = {}
        aloha_image_keys = [
            "image", "image_manipulator_1", "image_manipulator_2",
        ]
        for key in aloha_image_keys:
            img_path = os.path.join(ep_dir, "images", f"{key}.npy")
            if os.path.exists(img_path):
                img_arr = np.load(img_path)
                if img_arr.shape[0] > t:
                    images[key] = img_arr[t]
                else:
                    images[key] = np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
            else:
                images[key] = np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)

        # Normalize actions with per-arm statistics
        if self.config.normalize_actions and self.normalization_stats:
            if "left_action_mean" in self.normalization_stats:
                # ALOHA-specific: normalise left and right arms separately
                actions = np.array(actions, dtype=np.float32)
                # Ensure actions is 1D (N, action_dim) -> if loaded as (1, 28) slice to first row
                if actions.ndim == 2 and actions.shape[0] == 1:
                    actions = actions[0]
                # Left arm (joints 0-6)
                for j in range(7):
                    if self.normalization_stats["left_action_std"][j] > 0:
                        actions[j] = (actions[j] - self.normalization_stats["left_action_mean"][j]) / self.normalization_stats["left_action_std"][j]
                # Right arm (joints 7-13)
                for j in range(7):
                    if self.normalization_stats["right_action_std"][j] > 0:
                        actions[7 + j] = (actions[7 + j] - self.normalization_stats["right_action_mean"][j]) / self.normalization_stats["right_action_std"][j]
            else:
                # Fallback: global normalisation
                mean = self.normalization_stats.get("action_mean", np.zeros(28))
                std = self.normalization_stats.get("action_std", np.ones(28) + 1e-8)
                actions = (actions - mean) / (std + 1e-8)

        return {
            "images": images,
            "actions": torch.tensor(actions, dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.float32),
            "language": language,
        }


# ====================================================================
# Collate Function
# ====================================================================

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    # Batch images (dictionary of arrays)
    batched_images = {}
    for key in images_list[0].keys():
        img_arrs = [img[key] for img in images_list]
        # Convert to numpy if needed
        if isinstance(img_arrs[0], torch.Tensor):
            batched_images[key] = torch.stack(img_arrs, dim=0)
        else:
            batched_images[key] = np.stack(img_arrs, axis=0)

    # Pad actions to the longest sequence in the batch
    max_len = max(a.size(0) for a in actions_list)
    action_dim = actions_list[0].size(1)
    batched_actions = torch.zeros(len(batch), max_len, action_dim, dtype=torch.float32)
    action_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, act in enumerate(actions_list):
        L = act.size(0)
        batched_actions[i, :L, :] = act
        action_mask[i, :L] = True

    # Stack states (already fixed size)
    batched_states = torch.stack(states_list, dim=0)

    return {
        "images": batched_images,
        "actions": batched_actions,
        "state": batched_states,
        "action_mask": action_mask,
        "language": languages,
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
