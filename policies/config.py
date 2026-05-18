#!/usr/bin/env python3
"""
Policy Configuration System

Provides:
    - ACTConfig / Pi05Config dataclasses (re-exported from submodules)
    - PolicyFactory for creating policies from config strings / dicts / YAML / JSON
    - Config validation, serialization, and YAML/JSON I/O utilities
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Type, Union

import yaml

from .act import ACTConfig, ACTPolicy, ACTLoss, ACTDataPreprocessor, ResNetImageEncoder, LanguageEncoder
from .pi05 import Pi05Config, Pi05Policy, FlowMatchingSampler, ActionExpert, SigLIPVisionEncoder, MultimodalProjector, LLaMADecoder

__all__ = [
    "ACTConfig",
    "ACTPolicy",
    "ACTLoss",
    "ACTDataPreprocessor",
    "ResNetImageEncoder",
    "LanguageEncoder",
    "Pi05Config",
    "Pi05Policy",
    "FlowMatchingSampler",
    "ActionExpert",
    "SigLIPVisionEncoder",
    "MultimodalProjector",
    "LLaMADecoder",
    "PolicyFactory",
    "validate_config",
    "config_to_dict",
    "config_from_dict",
    "save_config_yaml",
    "save_config_json",
    "load_config_yaml",
    "load_config_json",
]


# ====================================================================
# Configuration Validation
# ====================================================================

def validate_config(config: Any, mode: str = "act") -> None:
    """Validate a policy configuration object.

    Checks that all required attributes exist and that values fall within
    reasonable ranges.

    Args:
        config: A config dataclass (``ACTConfig`` or ``Pi05Config``).
        mode: Expected mode ('act' or 'pi05').

    Raises:
        ValueError: If any validation check fails.
    """
    if mode == "act":
        assert isinstance(config, ACTConfig), f"Expected ACTConfig, got {type(config).__name__}"
        if config.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be > 0")
        if getattr(config, "n_action_steps", 1) <= 0:
            raise ValueError("n_action_steps must be > 0")
        if getattr(config, "n_action_steps", 1) > config.action_chunk_size:
            raise ValueError("n_action_steps must be <= action_chunk_size")
        if config.action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if getattr(config, "use_vae", False) and getattr(config, "kl_weight", 0.0) < 0:
            raise ValueError("kl_weight must be >= 0")
        if getattr(config, "temporal_ensemble_coeff", None) is not None and getattr(config, "n_action_steps", 1) > 1:
            raise ValueError("temporal_ensemble_coeff requires n_action_steps == 1")
        if config.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
    elif mode == "pi05":
        assert isinstance(config, Pi05Config), f"Expected Pi05Config, got {type(config).__name__}"
        if config.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be > 0")
        if config.action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if config.lr <= 0 or config.lr > 1:
            raise ValueError(f"lr must be in (0, 1], got {config.lr}")
        if config.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if config.flow_steps <= 0:
            raise ValueError("flow_steps must be > 0")
        if config.flow_sigma <= 0:
            raise ValueError("flow_sigma must be > 0")
        if config.vision_width <= 0:
            raise ValueError("vision_width must be > 0")
        if config.llm_width <= 0:
            raise ValueError("llm_width must be > 0")
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ====================================================================
# Config Serialization Utilities
# ====================================================================

def config_to_dict(config: Any) -> Dict[str, Any]:
    """Convert a config dataclass to a serialisable dict.

    Args:
        config: ``ACTConfig`` or ``Pi05Config`` instance.

    Returns:
        Dict representation.
    """
    if isinstance(config, (ACTConfig, Pi05Config)):
        return asdict(config)
    raise TypeError(f"Expected ACTConfig or Pi05Config, got {type(config).__name__}")


def config_from_dict(d: Dict[str, Any], mode: str = "act") -> Union[ACTConfig, Pi05Config]:
    """Create a config dataclass from a dict.

    Args:
        d: Dict with configuration keys.
        mode: 'act' or 'pi05'.

    Returns:
        Config dataclass instance.
    """
    if mode == "act":
        return ACTConfig.from_dict(d)
    elif mode == "pi05":
        return Pi05Config.from_dict(d)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def save_config_yaml(config: Any, path: str, mode: str = "act") -> None:
    """Save a config to a YAML file.

    Args:
        config: Config dataclass.
        path: Output YAML file path.
        mode: 'act' or 'pi05'.
    """
    validate_config(config, mode)
    d = config_to_dict(config)
    d["_mode"] = mode
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False)


def save_config_json(config: Any, path: str, mode: str = "act") -> None:
    """Save a config to a JSON file.

    Args:
        config: Config dataclass.
        path: Output JSON file path.
        mode: 'act' or 'pi05'.
    """
    validate_config(config, mode)
    d = config_to_dict(config)
    d["_mode"] = mode
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_config_yaml(path: str) -> tuple[Union[ACTConfig, Pi05Config], str]:
    """Load a config from a YAML file.

    Args:
        path: YAML file path.

    Returns:
        Tuple of (config_instance, mode_string).
    """
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    mode = d.pop("_mode", "act")
    return config_from_dict(d, mode=mode), mode


def load_config_json(path: str) -> tuple[Union[ACTConfig, Pi05Config], str]:
    """Load a config from a JSON file.

    Args:
        path: JSON file path.

    Returns:
        Tuple of (config_instance, mode_string).
    """
    with open(path, "r") as f:
        d = json.load(f)
    mode = d.pop("_mode", "act")
    return config_from_dict(d, mode=mode), mode


# ====================================================================
# PolicyFactory
# ====================================================================

class PolicyFactory:
    """Creates policy instances from config strings, dicts, or file paths.

    Supports three creation interfaces:

    1. From a config string (e.g. "act" or "pi05") with optional overrides:
       ``factory.create("act", lr=1e-4, batch_size=16)``

    2. From a config file (YAML or JSON):
       ``factory.create_from_file("config.yaml")``

    3. From a pre-built config dataclass:
       ``factory.create_from_config(act_config)``

    Usage example::

        factory = PolicyFactory()
        policy = factory.create("act", lr=5e-5, batch_size=16)
        policy = factory.create("pi05", lr=1e-5, freeze_backbone=True)
    """

    # Mapping from mode string to (ConfigClass, PolicyClass)
    _MODE_MAP: Dict[str, tuple] = {
        "act": (ACTConfig, ACTPolicy),
        "pi05": (Pi05Config, Pi05Policy),
    }

    def create(
        self,
        mode: str,
        **overrides: Any,
    ) -> Union[ACTPolicy, Pi05Policy]:
        """Create a policy from a mode string with optional parameter overrides.

        Args:
            mode: 'act' or 'pi05'.
            **overrides: Keyword arguments to override default config values.

        Returns:
            Instantiated policy (``ACTPolicy`` or ``Pi05Policy``).

        Raises:
            ValueError: If *mode* is unknown.
        """
        if mode not in self._MODE_MAP:
            raise ValueError(f"Unknown policy mode: {mode}. Choose from {list(self._MODE_MAP.keys())}")

        config_cls, policy_cls = self._MODE_MAP[mode]
        config = config_cls(**overrides)
        validate_config(config, mode)
        return policy_cls(config)

    def create_from_file(self, path: str, **overrides: Any) -> Union[ACTConfig, ACTPolicy, Pi05Config, Pi05Policy]:
        """Create a policy or config from a YAML / JSON file.

        The file should contain either a ``_mode`` key or no mode key (defaults to "act").
        Additional keys should match the config dataclass fields.

        Args:
            path: Path to a YAML or JSON config file.
            **overrides: Keyword arguments to override loaded values.

        Returns:
            Instantiated policy.
        """
        if path.endswith(".yaml") or path.endswith(".yml"):
            config, mode = load_config_yaml(path)
        elif path.endswith(".json"):
            config, mode = load_config_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path}. Use .yaml or .json.")

        # Apply overrides
        for key, val in overrides.items():
            if hasattr(config, key):
                setattr(config, key, val)

        validate_config(config, mode)

        _, policy_cls = self._MODE_MAP[mode]
        return policy_cls(config)

    @classmethod
    def from_config(
        cls,
        config: Union[ACTConfig, Pi05Config],
    ) -> Union[ACTPolicy, Pi05Policy]:
        """Create a policy from a pre-built config dataclass.

        Args:
            config: ACTConfig or Pi05Config instance.

        Returns:
            Instantiated policy.
        """
        validate_config(config, "act" if isinstance(config, ACTConfig) else "pi05")
        if isinstance(config, ACTConfig):
            return ACTPolicy(config)
        elif isinstance(config, Pi05Config):
            return Pi05Policy(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

    @classmethod
    def create_config(cls, mode: str, **overrides: Any) -> Union[ACTConfig, Pi05Config]:
        """Create a config dataclass from a mode string with overrides.

        Args:
            mode: 'act' or 'pi05'.
            **overrides: Keyword arguments to override defaults.

        Returns:
            Config dataclass instance.
        """
        if mode not in cls._MODE_MAP:
            raise ValueError(f"Unknown mode: {mode}")
        config_cls, _ = cls._MODE_MAP[mode]
        config = config_cls(**overrides)
        validate_config(config, mode)
        return config

    @staticmethod
    def list_supported_modes() -> list[str]:
        """Return the list of supported policy modes."""
        return list(PolicyFactory._MODE_MAP.keys())
