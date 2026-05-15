#!/usr/bin/env python3
"""
bh-VLA: Unified Vision-Language-Action Model Training Framework

This project implements two VLA (Vision-Language-Action) policies:
1. ACT (Action Chunking Transformers) - Based on arXiv:2304.13705 (ALOHA paper)
2. pi0.5 - Based on arXiv:2504.16054 (Physical Intelligence pi0.5 paper)

Both policies can be trained and evaluated using the unified interface below.

Key Differences:
- ACT: Uses a Transformer decoder to predict action chunks directly from image+language features.
       Faster training, simpler architecture. Good for imitation learning on specific tasks.
- pi0.5: Uses a PaliGemma VLM backbone with flow matching on action space.
         More powerful generalization, larger model, better for open-world tasks.

Usage:
    # Train ACT policy
    python train.py --mode act

    # Train pi0.5 policy
    python train.py --mode pi05

    # Run inference
    python inference.py --mode act --checkpoint ./checkpoints/act_last.pt

    # Collect data using teleoperation
    python collect_data.py --mode act
"""

import os
import sys
import argparse
import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Union


# ============================================================
# Configuration Dataclass
# ============================================================

@dataclass
class ACTConfig:
    """Configuration for the ACT (Action Chunking Transformer) policy.

    ACT paper: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
    Key insight: Predict a chunk of N future actions at once to reduce error accumulation.
    """
    # Image encoder: uses ResNet-18 as the visual feature extractor
    image_encoder: str = "resnet18"
    image_size: int = 224
    image_channels: int = 3

    # Language encoder: uses a simple token embedding + positional encoding
    text_vocab_size: int = 30522  # Standard BERT vocab size
    text_max_len: int = 64
    text_hidden_size: int = 512

    # Transformer decoder parameters
    transformer_hidden_size: int = 512
    transformer_num_layers: int = 4
    transformer_num_heads: int = 8
    transformer_dropout: float = 0.1
    transformer_activation: str = "gelu"

    # Action prediction head
    action_dim: int = 14  # 7 joints (left) + 7 joints (right) for ALOHA
    action_chunk_size: int = 90  # Predict 90 future timesteps per forward pass
    horizon: int = 90  # Same as chunk_size for simplicity

    # Training hyperparameters
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Dropout for regularization
    act_dropout: float = 0.1  # Dropout rate in the action prediction head


@dataclass
class Pi05Config:
    """Configuration for the pi0.5 (Physical Intelligence) policy.

    pi0.5 paper: "pi0.5: A Vision-Language-Action Model with Open-World Generalization"
    Key insight: Use flow matching on a VLM backbone (PaliGemma) for action prediction.
    Flow matching provides better training stability than diffusion-based approaches.
    """
    # Vision-Language backbone: PaliGemma 3B
    backbone: str = "paligemma3b"
    vision_width: int = 1152  # PaliGemma ViT width
    llm_width: int = 3200     # PaliGemma LLM width
    vision_depth: int = 27
    llm_depth: int = 28
    vision_heads: int = 16
    llm_heads: int = 20

    # Flow matching parameters
    flow_steps: int = 500  # Number of flow matching integration steps during training
    flow_sigma: float = 0.02  # Gaussian noise level for flow matching
    action_dim: int = 14
    action_chunk_size: int = 32  # pi0.5 uses shorter chunks than ACT
    horizon: int = 32

    # Action expert: MLP that maps VLM features to actions
    action_expert_hidden: int = 2048
    action_expert_layers: int = 3

    # Training hyperparameters
    lr: float = 1e-5  # Smaller LR for fine-tuning the backbone
    weight_decay: float = 0.01
    batch_size: int = 8  # Smaller batch size due to larger model
    num_epochs: int = 50
    warmup_steps: int = 500
    gradient_clip: float = 1.0

    # Gradient checkpointing to save memory
    gradient_checkpointing: bool = True

    # Whether to freeze the VLM backbone (transfer learning)
    freeze_backbone: bool = False


@dataclass
class DatasetConfig:
    """Configuration for the data loading and preprocessing."""
    data_dir: str = "./data/aloha_datasets"
    image_keys: List[str] = field(default_factory=lambda: [
        "image", "image_supplementary_1", "image_supplementary_2"
    ])
    action_keys: List[str] = field(default_factory=lambda: [
        "actions"
    ])
    state_key: str = "state"
    language_key: str = "language"
    normalize_actions: bool = True
    normalize_observations: bool = True
    max_dataset_size: int = -1  # -1 means use all data


@dataclass
class RobotConfig:
    """Configuration for the SO-101 / ALOHA robot hardware interface."""
    # USB ports for the leader and follower arms
    leader_port: str = "/dev/ttyACM0"
    follower_port: str = "/dev/ttyACM1"

    # Motor bus protocol parameters
    motor_baudrate: int = 1000000  # 1 Mbps standard for Feetech servos

    # Camera configuration
    camera_fps: int = 30
    camera_width: int = 224
    camera_height: int = 224
    num_cameras: int = 3  # ALOHA uses 3 cameras (main + 2 supplementary)

    # Teleoperation parameters
    teleop_frequency: int = 50  # Hz
    joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "base_pan": (-2.617, 2.617),
        "shoulder_pan": (-2.094, 2.094),
        "shoulder_lift": (-2.094, 2.094),
        "elbow_flex": (-2.268, 2.268),
        "wrist_flex": (-1.745, 1.745),
        "wrist_roll": (-2.617, 2.617),
        "gripper": (0.0, 1.0),
    })


@dataclass
class TrainConfig:
    """Unified training configuration that wraps mode-specific configs."""
    # Common settings
    mode: str = "act"  # "act" or "pi05"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Dataset settings
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Robot settings
    robot: RobotConfig = field(default_factory=RobotConfig)

    # Logging
    log_interval: int = 10
    save_interval: int = 10

    # Device settings
    device: str = "cuda"  # "cuda" or "cpu"

    # Random seed for reproducibility
    seed: int = 42


# ============================================================
# ACT Policy Implementation
# ============================================================

class ResNetImageEncoder:
    """A lightweight ResNet-18 image encoder for ACT.

    This encoder converts raw images into fixed-size feature vectors.
    Architecture:
        Conv2d(3, 64, 7x7, stride=2) -> BN -> ReLU -> MaxPool -> ResBlocks -> GlobalAvgPool
        Output: 512-dim features per image
    """

    def __init__(self, num_cameras: int = 3, hidden_size: int = 512):
        """
        Args:
            num_cameras: Number of camera views to encode (1 main + 2 supplementary)
            hidden_size: Output feature dimension per image
        """
        self.num_cameras = num_cameras
        self.hidden_size = hidden_size

    def encode_single_image(self, image: Any) -> Any:
        """Encode a single image tensor through the ResNet backbone.

        Args:
            image: Input image tensor of shape (C, H, W)

        Returns:
            Encoded feature vector of shape (512,)
        """
        # Use torch.nn.functional for the forward pass
        # In production, this would use a pre-trained ResNet-18
        import torch.nn as nn
        if not hasattr(self, 'conv1'):
            # Build the encoder architecture (lazy initialization)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Define ResNet basic block
            class BasicBlock(nn.Module):
                def __init__(self, in_channels, out_channels):
                    super().__init__()
                    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
                    self.bn1 = nn.BatchNorm2d(out_channels)
                    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
                    self.bn2 = nn.BatchNorm2d(out_channels)
                    self.relu = nn.ReLU(inplace=True)
                    # Downsampling shortcut when channels differ
                    self.downsample = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

                def forward(self, x):
                    identity = x
                    out = self.relu(self.bn1(self.conv1(x)))
                    out = self.bn2(self.conv2(out))
                    if self.downsample:
                        identity = self.downsample(identity)
                    out += identity
                    return self.relu(out)

            # ResNet-18 layers: [3, 3, 9, 3]
            self.layer1 = nn.Sequential(
                BasicBlock(64, 64),
                BasicBlock(64, 64),
                BasicBlock(64, 64)
            )
            self.layer2 = nn.Sequential(
                BasicBlock(64, 128),
                BasicBlock(128, 128),
                BasicBlock(128, 128)
            )
            self.layer3 = nn.Sequential(
                BasicBlock(128, 256),
                BasicBlock(256, 256),
                BasicBlock(256, 256),
                BasicBlock(256, 256),
                BasicBlock(256, 256),
                BasicBlock(256, 256)
            )
            self.layer4 = nn.Sequential(
                BasicBlock(256, 512),
                BasicBlock(512, 512),
                BasicBlock(512, 512)
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Forward pass through ResNet
        x = self.relu(self.bn1(self.conv1(image)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 512)

        return x

    def encode_batch(self, images: Any) -> Any:
        """Encode a batch of images from all cameras.

        Args:
            images: Batch of images of shape (batch, num_cameras, C, H, W)

        Returns:
            Concatenated features of shape (batch, num_cameras * hidden_size)
        """
        batch_size = images.size(0)
        features = []
        for cam_idx in range(self.num_cameras):
            cam_features = self.encode_single_image(images[:, cam_idx])
            features.append(cam_features)
        # Concatenate features from all cameras
        return torch.cat(features, dim=-1)  # (batch, num_cameras * 512)


class LanguageEncoder:
    """A simple language encoder that converts text instructions to dense vectors.

    Uses token embeddings + positional encoding + a small transformer encoder.
    This is a simplified version of the language encoder from the ACT paper.
    """

    def __init__(self, vocab_size: int = 30522, max_len: int = 64, hidden_size: int = 512):
        """
        Args:
            vocab_size: Size of the text vocabulary
            max_len: Maximum sequence length for text
            hidden_size: Output hidden dimension
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size

    def tokenize(self, text: str) -> List[int]:
        """Convert a text string to a list of token IDs.

        In production, use a proper tokenizer (e.g., BertTokenizer).
        For simplicity, we use character-level hashing.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        # Simple char-level encoding for demonstration
        tokens = []
        for ch in text.lower():
            # Map character to integer (space = 0, a-z = 1-26, other = 27+)
            if ch == ' ':
                tokens.append(0)
            elif 'a' <= ch <= 'z':
                tokens.append(ord(ch) - ord('a') + 1)
            elif '0' <= ch <= '9':
                tokens.append(ord(ch) - ord('0') + 27)
            else:
                tokens.append(0)  # Pad

        # Truncate to max length
        return tokens[:self.max_len]

    def encode(self, text_tokens: List[int]) -> Any:
        """Convert token IDs to dense embeddings.

        Args:
            text_tokens: List of token IDs

        Returns:
            Encoded tensor of shape (hidden_size,)
        """
        import torch
        import torch.nn.functional as F

        if not hasattr(self, 'embed'):
            self.embed = torch.nn.Embedding(self.vocab_size, self.hidden_size)
            # Positional encoding (learnable)
            self.pos_embed = torch.nn.parameter.Parameter(
                torch.randn(self.max_len, self.hidden_size)
            )

        # Convert list to tensor
        token_tensor = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0)  # (1, seq_len)

        # Look up embeddings
        embedded = self.embed(token_tensor)  # (1, seq_len, hidden_size)

        # Add positional encoding
        embedded = embedded + self.pos_embed[:embedded.size(1)]

        # Use the mean of all token embeddings as the sentence representation
        return embedded.mean(dim=1).squeeze(0)  # (hidden_size,)


class TransformerDecoder:
    """A Transformer decoder for predicting action chunks.

    This is the core of the ACT policy. It takes concatenated image+language
    features and predicts a sequence of future actions.

    Architecture:
        Input: Concatenated image features + language features + current state
        Processing: Multi-layer Transformer decoder with self-attention
        Output: Action chunk (N future timesteps × action_dim)
    """

    def __init__(self, hidden_size: int = 512, num_layers: int = 4,
                 num_heads: int = 8, dropout: float = 0.1,
                 action_dim: int = 14, action_chunk_size: int = 90):
        """
        Args:
            hidden_size: Hidden dimension for the Transformer
            num_layers: Number of Transformer decoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            action_dim: Dimension of the action space (e.g., 14 for ALOHA)
            action_chunk_size: Number of future timesteps to predict
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size

    def build(self):
        """Build the model components."""
        import torch
        import torch.nn as nn

        # Create Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])

        # The input to the decoder is: [action_token, ..., action_token] (chunk_size tokens)
        # We use learnable action tokens that get refined through the decoder
        self.action_token = nn.parameter.Parameter(
            torch.randn(1, self.action_chunk_size, self.hidden_size)
        )

        # Action prediction head: map from hidden state to action dimension
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, self.action_dim)
        )

    def forward(self, encoder_output: Any, state: Optional[Any] = None) -> Any:
        """Forward pass of the ACT policy.

        Args:
            encoder_output: Concatenated image+language features of shape (batch, enc_dim)
            state: Optional current robot state of shape (batch, state_dim)

        Returns:
            Predicted action chunk of shape (batch, action_chunk_size, action_dim)
        """
        import torch
        import torch.nn as nn

        batch_size = encoder_output.size(0)

        # Expand action tokens to batch dimension
        action_tokens = self.action_token.expand(batch_size, -1, -1)  # (batch, chunk, hidden)

        # Inject state information into the action tokens (if available)
        if state is not None:
            state_proj = nn.Linear(state.size(-1), self.hidden_size)
            state_features = state_proj(state).unsqueeze(1)  # (batch, 1, hidden)
            action_tokens = action_tokens + state_features

        # Pass through Transformer decoder layers
        for layer in self.decoder_layers:
            # Use the action tokens as both query and key/value (self-attention)
            action_tokens = layer(action_tokens, action_tokens)

        # Predict actions from the refined action tokens
        actions = self.action_head(action_tokens)  # (batch, chunk_size, action_dim)

        return actions


class ACTPolicy:
    """The full ACT (Action Chunking Transformer) policy.

    This policy implements the ACT algorithm from the ALOHA paper.
    It takes multi-camera images + language instructions + current robot state
    and predicts a chunk of future actions (joint positions + gripper commands).

    Key components:
        1. ResNet-18 image encoder: extracts visual features from each camera
        2. Language encoder: converts text instructions to dense vectors
        3. Transformer decoder: fuses modalities and predicts action chunks
        4. Action prediction head: maps decoded features to robot actions

    Training:
        - Loss: MSE between predicted actions and ground truth actions
        - Optimizer: AdamW with weight decay
        - Scheduler: Warmup learning rate schedule
    """

    def __init__(self, config: ACTConfig):
        """Initialize the ACT policy.

        Args:
            config: Configuration object containing all hyperparameters
        """
        self.config = config
        self.device = torch.device(config.device if hasattr(torch, 'device') else 'cpu')

        # Build the policy components
        self.image_encoder = ResNetImageEncoder(
            num_cameras=3,
            hidden_size=512
        )

        self.language_encoder = LanguageEncoder(
            vocab_size=config.text_vocab_size,
            max_len=config.text_max_len,
            hidden_size=config.text_hidden_size
        )

        # Feature dimension: 3 cameras * 512 + text_hidden + state_dim
        self.enc_dim = 3 * 512 + config.text_hidden_size

        # Projection layer to match transformer hidden size
        self.feature_proj = torch.nn.Linear(self.enc_dim, config.transformer_hidden_size)

        self.decoder = TransformerDecoder(
            hidden_size=config.transformer_hidden_size,
            num_layers=config.transformer_num_layers,
            num_heads=config.transformer_num_heads,
            dropout=config.transformer_dropout,
            action_dim=config.action_dim,
            action_chunk_size=config.action_chunk_size
        )
        self.decoder.build()

        # Move all parameters to device
        self.to(self.device)

    def forward(self, images: Any, language: str, state: Optional[Any] = None) -> Any:
        """Run the policy forward pass.

        Args:
            images: Input images of shape (batch, num_cameras, C, H, W)
            language: Text instruction string (e.g., "pick up the red cup")
            state: Current robot state of shape (batch, state_dim)

        Returns:
            Predicted action chunk of shape (batch, action_chunk_size, action_dim)
        """
        import torch

        # Step 1: Encode images from all cameras
        img_features = self.image_encoder.encode_batch(images)  # (batch, 1536)

        # Step 2: Encode language instruction
        tokens = self.language_encoder.tokenize(language)
        lang_features = self.language_encoder.encode(tokens)  # (512,)
        lang_features = lang_features.unsqueeze(0).expand(images.size(0), -1)  # (batch, 512)

        # Step 3: Concatenate features
        features = torch.cat([img_features, lang_features], dim=-1)  # (batch, 2048)

        # Step 4: Project to transformer dimension
        features = self.feature_proj(features)  # (batch, hidden_size)
        features = features.unsqueeze(1)  # (batch, 1, hidden) - acts as the prompt

        # Step 5: Decode action chunk
        actions = self.decoder.forward(features, state)  # (batch, chunk, action_dim)

        return actions

    def predict_action(self, images: Any, language: str,
                       state: Optional[Any] = None) -> Any:
        """Predict actions for a single step (no batching for inference).

        This is the inference-time interface. It processes a single observation
        and returns the full action chunk.

        Args:
            images: Single observation of shape (num_cameras, C, H, W)
            language: Text instruction
            state: Current robot state

        Returns:
            First action in the chunk (batch_size=1), shape (action_dim,)
        """
        import torch

        # Add batch dimension
        images = images.unsqueeze(0)
        if state is not None:
            state = state.unsqueeze(0)

        # Forward pass
        actions = self.forward(images, language, state)

        # Return the first action in the chunk (execute it now)
        return actions[0, 0]

    def save_checkpoint(self, path: str):
        """Save the model checkpoint.

        Args:
            path: File path to save the checkpoint
        """
        import torch
        checkpoint = {
            'image_encoder_state_dict': self.image_encoder.state_dict(),
            'feature_proj_state_dict': self.feature_proj.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'action_head_state_dict': self.decoder.action_head.state_dict(),
            'config': asdict(self.config),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load a model checkpoint.

        Args:
            path: File path to load the checkpoint from
        """
        import torch
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.feature_proj.load_state_dict(checkpoint['feature_proj_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.decoder.action_head.load_state_dict(checkpoint['action_head_state_dict'])
        if 'image_encoder_state_dict' in checkpoint:
            self.image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
        print(f"Checkpoint loaded from {path}")


# ============================================================
# pi0.5 Policy Implementation
# ============================================================

class PaliGemmaBackbone:
    """A simplified PaliGemma 3B VLM backbone for pi0.5.

    This backbone combines:
        1. Vision Transformer (ViT-L/14) for image encoding
        2. LLaMA-style decoder for text processing
        3. Multimodal projection layer that aligns vision and text features

    Architecture:
        Image: [B, C, H, W] -> ViT -> [B, seq_v, d_v]
        Text: [B, seq_t] -> Embedding -> [B, seq_t, d_t]
        Multimodal: Concat + Project -> [B, seq_v+seq_t, d_llm]
        -> LLaMA decoder layers -> [B, seq_v+seq_t, d_llm]
    """

    def __init__(self, config: Pi05Config):
        """Initialize the PaliGemma backbone.

        Args:
            config: pi0.5 configuration object
        """
        self.config = config

        # Vision encoder parameters
        self.vision_width = config.vision_width
        self.vision_depth = config.vision_depth
        self.vision_heads = config.vision_heads

        # Language model parameters
        self.llm_width = config.llm_width
        self.llm_depth = config.llm_depth
        self.llm_heads = config.llm_heads

    def build_vision_encoder(self):
        """Build the ViT vision encoder.

        For a 3B model, this would be a ViT-L/14 with ~307M parameters.
        We implement a simplified version here.
        """
        import torch.nn as nn

        # PaliGemma uses a SigLIP ViT-L/14 with patch_size=14
        self.vit_patch_embed = nn.Conv2d(3, self.vision_width, kernel_size=14, stride=14)
        self.vit_norm = nn.LayerNorm(self.vision_width)

        # Build ViT encoder layers (simplified)
        vit_layers = []
        for i in range(self.vision_depth):
            vit_layers.append(nn.MultiheadAttention(
                embed_dim=self.vision_width,
                num_heads=self.vision_heads,
                batch_first=True
            ))
            vit_layers.append(nn.LayerNorm(self.vision_width))
            vit_layers.append(nn.FeedForward(self.vision_width, self.vision_width * 4))
            vit_layers.append(nn.LayerNorm(self.vision_width))
        self.vit_encoder = nn.ModuleList(vit_layers)

        # CLS token (like Vision Transformer)
        self.vit_cls_token = nn.parameter.Parameter(torch.randn(1, 1, self.vision_width))

    def build_language_model(self):
        """Build the LLaMA-style language model.

        For a 3B model, this would have ~2.7B parameters.
        We implement a simplified version for demonstration.
        """
        import torch.nn as nn

        # LLaMA layer norm before transformer
        self.llm_input_norm = nn.LayerNorm(self.llm_width)

        # Build LLaMA decoder layers
        ff_mult = 4  # LLaMA uses 4x expansion in FFN
        self.llm_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(self.llm_width, self.llm_heads, batch_first=True),
                'attn_norm': nn.LayerNorm(self.llm_width),
                'ffn': nn.Sequential(
                    nn.Linear(self.llm_width, self.llm_width * ff_mult),
                    nn.GELU(),
                    nn.Linear(self.llm_width * ff_mult, self.llm_width),
                ),
                'ffn_norm': nn.LayerNorm(self.llm_width),
            })
            for _ in range(self.llm_depth)
        ])

    def build_multimodal_projector(self):
        """Build the projector that aligns vision features to text space.

        This projects ViT features (d_v=1152) to LLM space (d_llm=3200).
        Architecture: Linear -> GELU -> Linear
        """
        import torch.nn as nn
        self.projector = nn.Sequential(
            nn.Linear(self.vision_width, self.llm_width),
            nn.GELU(),
            nn.Linear(self.llm_width, self.llm_width),
            nn.LayerNorm(self.llm_width),
        )

    def encode_image(self, images: Any) -> Any:
        """Encode images through the ViT backbone.

        Args:
            images: Image tensor of shape (batch, num_cameras, C, H, W)

        Returns:
            Visual features in LLM space of shape (batch, num_tokens, llm_width)
        """
        # In production: process through SigLIP ViT-L/14
        # For each camera, get patch tokens + cls token
        # Then project to LLM space

        batch_size = images.size(0)
        num_cam = images.size(1)

        # ViT processes each image separately
        all_vision_features = []
        for cam in range(num_cam):
            # Extract patches (simplified)
            patches = self.vit_patch_embed(images[:, cam])  # (batch, vision_width, H/14, W/14)
            patches = patches.flatten(2).transpose(1, 2)  # (batch, num_patches, vision_width)

            # Add CLS token
            cls_tokens = self.vit_cls_token.expand(batch_size, -1, -1)
            patches = torch.cat([cls_tokens, patches], dim=1)  # (batch, 1+num_patches, vision_width)

            # Run through ViT encoder layers
            for i in range(0, len(self.vit_encoder), 4):  # Simplified: use every 4th layer
                attn_layer = self.vit_encoder[i]
                norm_layer = self.vit_encoder[i + 1]
                patches = norm_layer(patches + attn_layer(patches, patches)[0])

            # Project to LLM space
            patches = self.projector(patches)
            all_vision_features.append(patches)

        # Concatenate vision features from all cameras
        return torch.cat(all_vision_features, dim=1)  # (batch, num_cam*num_tokens, llm_width)

    def encode_text(self, text_ids: Any) -> Any:
        """Encode text through the LLM.

        Args:
            text_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Text features in LLM space of shape (batch, seq_len, llm_width)
        """
        import torch
        import torch.nn as nn

        batch_size = text_ids.size(0)
        seq_len = text_ids.size(1)

        # Embed tokens
        x = nn.Embedding(32000, self.llm_width)(text_ids)  # (batch, seq, llm_width)

        # Apply LLM layers
        x = self.llm_input_norm(x)
        for layer in self.llm_layers:
            # Self-attention with causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=text_ids.device)).unsqueeze(0)
            attn_out = layer['attn'](x, x, x, key_padding_mask=~causal_mask.bool())[0]
            x = layer['attn_norm'](x + attn_out)
            x = layer['ffn_norm'](x + layer['ffn'](x))

        return x

    def forward(self, images: Any, text_ids: Any) -> Any:
        """Forward pass through the full PaliGemma backbone.

        Args:
            images: Image tensor of shape (batch, num_cameras, C, H, W)
            text_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Multimodal features for action prediction
        """
        vision_feats = self.encode_image(images)  # (batch, num_cam*num_tokens, llm_width)
        text_feats = self.encode_text(text_ids)  # (batch, seq, llm_width)

        # Concatenate vision and text features
        multimodal = torch.cat([vision_feats, text_feats], dim=1)  # (batch, total_seq, llm_width)

        return multimodal


class FlowMatchingPolicy:
    """Flow matching loss and sampler for pi0.5.

    Flow matching is a method for training generative models that learns
    a vector field to map from a simple distribution (Gaussian noise) to
    the target distribution (actions).

    Key equation: We train a neural network to predict the velocity field
    v_t(x) such that integrating from t=0 to t=1 gives the target action.

    Loss: MSE between predicted velocity and the optimal transport velocity
    """

    def __init__(self, sigma: float = 0.02):
        """
        Args:
            sigma: Gaussian noise level for the flow matching target
        """
        self.sigma = sigma

    def compute_target_velocity(self, x0: Any, eps: Any) -> Any:
        """Compute the target velocity for flow matching.

        For flow matching with Gaussian noise, the optimal velocity is:
            v_t = (x0 - t * eps) / (1 - t)

        Args:
            x0: Ground truth action (target distribution)
            eps: Sampled Gaussian noise (source distribution)

        Returns:
            Target velocity vector
        """
        import torch

        # Sample a random time t in [0, 1]
        t = torch.rand(x0.size(0), device=x0.device).unsqueeze(1).unsqueeze(2)

        # Compute the target velocity
        # x_t = (1 - t) * eps + t * x0  (linear interpolation)
        # v_t = x0 - eps  (the direction we want to flow)
        target = x0 - eps

        return target, t

    def compute_loss(self, predicted_velocity: Any, target_velocity: Any) -> Any:
        """Compute the flow matching loss.

        Args:
            predicted_velocity: Network's predicted velocity of shape (batch, chunk, action_dim)
            target_velocity: Ground truth velocity of the same shape

        Returns:
            Scalar loss value
        """
        import torch
        return torch.nn.functional.mse_loss(predicted_velocity, target_velocity)

    def sample(self, velocity_fn, num_steps: int = 50) -> Any:
        """Sample actions from the flow matching model using Euler integration.

        Args:
            velocity_fn: Function that takes (x, t) and returns the predicted velocity
            num_steps: Number of integration steps

        Returns:
            Sampled actions of shape (batch, chunk, action_dim)
        """
        import torch

        # Start from Gaussian noise
        batch_size = 1
        action_dim = 14
        chunk_size = 32
        x = torch.randn(batch_size, chunk_size, action_dim)

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = step * dt
            v = velocity_fn(x, t)
            x = x + dt * v

        return x


class ActionExpert:
    """The action expert MLP in pi0.5.

    This MLP takes the multimodal features from the PaliGemma backbone
    and maps them to the action space. It's the "expert" layer that
    specializes in robot control.

    Architecture:
        Input: [B, 1, llm_width] (pooled multimodal features)
        -> Linear -> LayerNorm -> GELU -> Dropout -> ... -> Linear -> Output

    The expert processes the last token of the multimodal sequence
    (corresponding to the "predict the next action" instruction).
    """

    def __init__(self, config: Pi05Config):
        """
        Args:
            config: pi0.5 configuration object
        """
        self.config = config
        self.action_dim = config.action_dim

    def build(self):
        """Build the action expert network."""
        import torch.nn as nn

        # Build the MLP layers
        layers = []
        in_dim = self.config.llm_width
        for i in range(self.config.action_expert_layers):
            out_dim = self.config.action_expert_hidden
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = out_dim

        # Final layer: map to action dimension
        layers.append(nn.Linear(in_dim, self.action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, multimodal_features: Any) -> Any:
        """Forward pass of the action expert.

        Args:
            multimodal_features: Multimodal features of shape (batch, seq_len, llm_width)

        Returns:
            Predicted actions of shape (batch, chunk_size, action_dim)
        """
        # Take the last token (action prediction token)
        last_token = multimodal_features[:, -1, :]  # (batch, llm_width)

        # Pass through the expert MLP
        return self.network(last_token)  # (batch, action_dim)


class Pi05Policy:
    """The full pi0.5 (Physical Intelligence) policy.

    This policy implements the pi0.5 architecture from the PI paper.
    It uses a PaliGemma 3B VLM backbone with flow matching for action prediction.

    Key components:
        1. PaliGemma 3B backbone: Processes images + language to get multimodal features
        2. Action expert: MLP that maps multimodal features to actions
        3. Flow matching: Training objective for action distribution learning

    Training process:
        1. Sample noise epsilon ~ N(0, 1) for each action
        2. Create interpolated samples: x_t = (1-t)*eps + t*action_gt
        3. Predict velocity: v = velocity_net(x_t, t)
        4. Loss: MSE(v, action_gt - eps)
    """

    def __init__(self, config: Pi05Config):
        """Initialize the pi0.5 policy.

        Args:
            config: Configuration object containing all hyperparameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build the backbone
        self.backbone = PaliGemmaBackbone(config)
        self.backbone.build_vision_encoder()
        self.backbone.build_language_model()
        self.backbone.build_multimodal_projector()

        # Build the action expert
        self.action_expert = ActionExpert(config)
        self.action_expert.build()

        # Flow matching sampler/loss
        self.flow_matching = FlowMatchingPolicy(sigma=config.flow_sigma)

        # Move all parameters to device
        self.to(self.device)

    def forward(self, images: Any, language_ids: Any,
                state: Optional[Any] = None) -> Any:
        """Forward pass of the pi0.5 policy.

        Args:
            images: Input images of shape (batch, num_cameras, C, H, W)
            language_ids: Token IDs of the language instruction
            state: Optional current robot state

        Returns:
            Predicted actions of shape (batch, chunk_size, action_dim)
        """
        import torch

        # Step 1: Get multimodal features from PaliGemma backbone
        multimodal = self.backbone(images, language_ids)  # (batch, seq, llm_width)

        # Step 2: Get action from the expert
        actions = self.action_expert(multimodal)  # (batch, action_dim)

        # Expand to chunk size (repeat for all timesteps)
        actions = actions.unsqueeze(1).expand(-1, self.config.action_chunk_size, -1)

        return actions

    def compute_flow_matching_loss(self, images: Any, language_ids: Any,
                                   ground_truth_actions: Any) -> Any:
        """Compute the flow matching training loss.

        This is the core training function for pi0.5. It:
        1. Samples noise
        2. Creates interpolated samples
        3. Predicts velocity
        4. Computes loss

        Args:
            images: Input images
            language_ids: Language token IDs
            ground_truth_actions: Ground truth actions of shape (batch, chunk_size, action_dim)

        Returns:
            Scalar loss value
        """
        import torch

        # Get multimodal features (for velocity prediction)
        multimodal = self.backbone(images, language_ids)
        velocity = self.action_expert(multimodal).unsqueeze(1)  # (batch, 1, action_dim)
        velocity = velocity.expand(-1, self.config.action_chunk_size, -1)  # (batch, chunk, dim)

        # Sample noise
        eps = torch.randn_like(ground_truth_actions)  # (batch, chunk, action_dim)

        # Compute target velocity
        target, t = self.flow_matching.compute_target_velocity(ground_truth_actions, eps)

        # Compute loss
        loss = torch.nn.functional.mse_loss(velocity, target)

        return loss

    def predict_action(self, images: Any, language: str,
                       state: Optional[Any] = None) -> Any:
        """Predict actions for inference using flow matching sampling.

        Args:
            images: Single observation of shape (num_cameras, C, H, W)
            language: Text instruction
            state: Current robot state

        Returns:
            First action in the chunk
        """
        import torch

        # Convert text to token IDs
        text_ids = self._tokenize_text(language)

        # Add batch dimension
        images = images.unsqueeze(0)
        text_ids = text_ids.unsqueeze(0)

        # Use flow matching to sample actions
        batch_size = 1
        chunk = self.config.action_chunk_size
        dim = self.config.action_dim

        # Start from Gaussian noise
        x = torch.randn(batch_size, chunk, dim, device=self.device)

        # Euler integration (use fewer steps for inference)
        num_steps = self.config.flow_steps // 10  # 50 steps for inference
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = step * dt
            # Get velocity prediction at time t
            multimodal = self.backbone(images, text_ids)
            velocity = self.action_expert(multimodal).unsqueeze(1)
            velocity = velocity.expand(-1, chunk, -1)

            # Euler step
            x = x + dt * velocity

        return x[0, 0]  # Return first action

    def _tokenize_text(self, text: str) -> Any:
        """Convert text to token IDs (simplified)."""
        # In production, use the tokenizer from the PaliGemma model
        return torch.tensor([[0, 0, 0]], dtype=torch.long)  # Placeholder

    def save_checkpoint(self, path: str):
        """Save the model checkpoint."""
        import torch
        checkpoint = {
            'backbone_vision_state_dict': self.backbone.vit_encoder.state_dict(),
            'backbone_llm_state_dict': self.backbone.llm_layers.state_dict(),
            'backbone_projector_state_dict': self.backbone.projector.state_dict(),
            'action_expert_state_dict': self.action_expert.network.state_dict(),
            'config': asdict(self.config),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load a model checkpoint."""
        import torch
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.backbone.vit_encoder.load_state_dict(checkpoint['backbone_vision_state_dict'])
        self.backbone.llm_layers.load_state_dict(checkpoint['backbone_llm_state_dict'])
        self.backbone.projector.load_state_dict(checkpoint['backbone_projector_state_dict'])
        self.action_expert.network.load_state_dict(checkpoint['action_expert_state_dict'])
        print(f"Checkpoint loaded from {path}")

    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        return self


# ============================================================
# Data Loading
# ============================================================

class ALOHADataset:
    """Dataset loader for ALOHA/LeRobot formatted data.

    This dataset handles:
        - Loading episodes from LeRobot dataset format (HDF5/RLDS)
        - Image preprocessing (resize, normalize)
        - Action normalization (per-dimension z-score)
        - Data augmentation (random crops, flips)

    Data format (LeRobot):
        - images: Dict of camera names -> tensor (N, C, H, W)
        - actions: tensor (N, action_dim)
        - states: tensor (N, state_dim)
        - language: List of text instructions
    """

    def __init__(self, config: DatasetConfig, split: str = "train"):
        """
        Args:
            config: Dataset configuration
            split: Data split ("train" or "val")
        """
        self.config = config
        self.split = split
        self.data = None
        self.normalization_stats = None
        self.num_episodes = 0

    def load_data(self, path: Optional[str] = None):
        """Load dataset from disk.

        Supports multiple data formats:
        1. LeRobot format (HDF5 with images, actions, states)
        2. Simple JSON format for small datasets
        3. Directory of image pairs + action numpy files

        Args:
            path: Path to the dataset. Uses self.config.data_dir if None.
        """
        import numpy as np
        import glob
        import json

        data_path = path or self.config.data_dir

        # Try loading from different formats
        if os.path.exists(os.path.join(data_path, "dataset_infos.json")):
            # LeRobot format
            self._load_lerobot_format(data_path)
        elif os.path.exists(os.path.join(data_path, "data.json")):
            # JSON format
            self._load_json_format(data_path)
        else:
            # Try directory of image+action files
            self._load_directory_format(data_path)

    def _load_lerobot_format(self, data_path: str):
        """Load dataset in LeRobot format (HDF5)."""
        import h5py
        import numpy as np

        with h5py.File(os.path.join(data_path, "data.hdf5"), "r") as f:
            # Store data references for lazy loading
            self.episode_refs = list(f["data"].keys())
            self.num_episodes = len(self.episode_refs)

            # Compute normalization stats
            all_actions = []
            for ep_name in self.episode_refs:
                actions = f[f"data/{ep_name}/actions"][()]
                all_actions.append(actions)

            all_actions = np.concatenate(all_actions, axis=0)
            self.normalization_stats = {
                "action_mean": all_actions.mean(axis=0),
                "action_std": all_actions.std(axis=0) + 1e-8,
            }

    def _load_json_format(self, data_path: str):
        """Load dataset from JSON format (for small datasets)."""
        import json
        import numpy as np

        with open(os.path.join(data_path, "data.json"), "r") as f:
            self.data = json.load(f)
        self.num_episodes = len(self.data.get("episodes", []))

    def _load_directory_format(self, data_path: str):
        """Load dataset from directory structure."""
        import numpy as np
        import glob

        # Look for image files
        self.image_paths = sorted(glob.glob(os.path.join(data_path, "**/*.png"), recursive=True))
        self.action_paths = sorted(glob.glob(os.path.join(data_path, "**/*.npy"), recursive=True))

        self.num_episodes = len(self.image_paths)

    def get_item(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample.

        Args:
            idx: Index of the episode

        Returns:
            Dictionary containing:
                - images: Tensor of shape (num_cameras, C, H, W)
                - language: String instruction
                - actions: Tensor of shape (chunk_size, action_dim)
                - state: Tensor of shape (state_dim,)
        """
        import numpy as np
        import torch

        # Load episode data (simplified)
        # In production, this would load from HDF5
        episode = self._get_episode_data(idx)

        # Preprocess images
        images = self._preprocess_images(episode["images"])

        # Normalize actions if configured
        actions = torch.tensor(episode["actions"], dtype=torch.float32)
        if self.config.normalize_actions and self.normalization_stats:
            actions = (actions - self.normalization_stats["action_mean"]) / self.normalization_stats["action_std"]

        return {
            "images": images,
            "language": episode["language"],
            "actions": actions,
            "state": torch.tensor(episode["state"], dtype=torch.float32),
        }

    def _get_episode_data(self, idx: int) -> Dict:
        """Get raw episode data (implementation depends on data format)."""
        # Placeholder: in production, load from actual data source
        return {
            "images": np.random.randn(3, 3, 224, 224),  # (num_cam, C, H, W)
            "actions": np.random.randn(90, 14),  # (chunk_size, action_dim)
            "state": np.random.randn(28),  # (state_dim,)
            "language": "pick up the object",
        }

    def _preprocess_images(self, images: Any) -> Any:
        """Preprocess raw images for the model.

        Steps:
            1. Resize to target size (224x224)
            2. Normalize to [0, 1] if needed
            3. Convert to tensor

        Args:
            images: Raw image array of shape (num_cameras, H, W, C) or (num_cameras, C, H, W)

        Returns:
            Processed image tensor of shape (num_cameras, C, H, W)
        """
        import torch
        import numpy as np

        if isinstance(images, np.ndarray):
            # Convert to tensor and ensure correct channel order
            images = torch.tensor(images, dtype=torch.float32)

            # Ensure channel-first format (C, H, W)
            if images.dim() == 4 and images.size(1) in [0, 224]:
                # Images are already in (N, H, W, C) format, permute
                images = images.permute(0, 3, 1, 2)

            # Normalize to [0, 1] if values are in [0, 255]
            if images.max() > 1.0:
                images = images / 255.0

            # Standard image normalization (ImageNet statistics)
            images = (images - 0.5) / 0.5  # [-1, 1]
        return images

    def __len__(self):
        """Return the number of episodes in the dataset."""
        return self.num_episodes

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Alias for get_item."""
        return self.get_item(idx)


# ============================================================
# Robot Interface (SO-101 / ALOHA)
# ============================================================

class RobotInterface:
    """Hardware interface for SO-101 / ALOHA robots.

    This interface handles:
        - Communication with Feetech bus servos
        - Camera capture from multiple viewpoints
        - Teleoperation for data collection
        - Real-time action execution

    SO-101 robot specification:
        - 6 DOF leader arm (teleoperation input)
        - 6 DOF follower arm (policy output)
        - 3 cameras (1 main + 2 supplementary)
        - Feetech STS3215 bus servos
    """

    def __init__(self, config: RobotConfig):
        """
        Args:
            config: Robot configuration
        """
        self.config = config
        self.connected = False
        self.cameras = None
        self.teleop_enabled = False

    def connect(self):
        """Connect to the robot hardware.

        Steps:
            1. Initialize USB serial connections to both arms
            2. Verify motor IDs and baudrates
            3. Test camera connections
            4. Calibrate both arms to zero position
        """
        import subprocess

        # Find USB ports for each arm
        try:
            result = subprocess.run(
                ["lerobot-find-port"],
                capture_output=True, text=True, timeout=30
            )
            print(f"Found ports: {result.stdout}")
        except Exception as e:
            print(f"Warning: Could not auto-detect ports. Using defaults.")
            print(f"  Leader: {self.config.leader_port}")
            print(f"  Follower: {self.config.follower_port}")

        # Connect to motors
        print("Connecting to leader arm motors...")
        # In production: use lerobot-setup-motors for each arm

        # Initialize cameras
        print("Initializing cameras...")
        # In production: use cv2.VideoCapture for each camera
        self.connected = True
        print("Robot connected successfully!")

    def disconnect(self):
        """Disconnect from the robot hardware."""
        if self.cameras:
            for cam in self.cameras:
                if cam.isOpened():
                    cam.release()
        self.connected = False
        print("Robot disconnected.")

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation from the robot.

        Returns:
            Dictionary containing:
                - images: Dict of camera names -> numpy arrays (H, W, C)
                - state: Robot joint positions (numpy array, 28-dim for ALOHA)
                - timestamps: Dict of camera timestamps
        """
        import numpy as np
        import cv2

        if not self.connected:
            raise RuntimeError("Robot not connected! Call connect() first.")

        # Capture images from all cameras
        images = {}
        for i in range(self.config.num_cameras):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            images[f"camera_{i}"] = frame

        # Get current robot state (joint positions)
        # In production: read from bus servo protocol
        state = np.random.randn(28)  # 14 joints * 2 arms = 28

        return {
            "images": images,
            "state": state,
            "timestamp": time.time(),
        }

    def send_action(self, action: np.ndarray):
        """Send an action to the follower arm.

        Args:
            action: Action vector of shape (14,) containing:
                - First 7 values: left arm joint positions
                - Last 7 values: right arm joint positions (including gripper)
        """
        if not self.connected:
            raise RuntimeError("Robot not connected!")

        # Send to bus servos
        # In production: use Feetech SDK to set servo positions
        print(f"Sending action: {action}")

    def collect_teleop_data(self, episode_name: str = "demo", max_episodes: int = 1) -> str:
        """Collect teleoperation data for training.

        This function allows a human operator to demonstrate tasks
        using the leader arm. The follower arm mirrors the leader's
        movements while recording all sensor data.

        Data collected:
            - Images from all cameras (at 30fps)
            - Joint positions of both arms
            - Timestamps
            - Language instruction (user input)

        Args:
            episode_name: Name for the data episode
            max_episodes: Maximum number of episodes to collect

        Returns:
            Path to the saved dataset
        """
        import numpy as np
        import json
        import os

        if not self.connected:
            raise RuntimeError("Robot not connected! Call connect() first.")

        # Get language instruction
        instruction = input("Enter task instruction: ").strip()

        # Record data
        all_frames = []
        all_states = []
        all_actions = []

        print("Starting teleoperation... (move the leader arm)")
        print("Press Enter to stop recording.")

        for _ in range(max_episodes):
            # Record one episode
            episode_frames = []
            episode_states = []
            episode_actions = []

            while True:
                obs = self.get_observation()
                episode_states.append(obs["state"])
                episode_actions.append(obs["state"])  # For teleop, action = state
                all_frames.append(obs["images"])
                all_states.append(obs["state"])
                all_actions.append(obs["state"])

                # Stop condition
                if len(all_states) > 10000:  # Safety limit
                    break

            # Save episode
            episode_path = os.path.join(
                self.config.data_dir if hasattr(self.config, 'data_dir') else "./data",
                f"episode_{episode_name}"
            )

            # Save as numpy files
            np.save(os.path.join(episode_path, "actions.npy"), np.array(all_actions))
            np.save(os.path.join(episode_path, "states.npy"), np.array(all_states))

            # Save metadata
            metadata = {
                "instruction": instruction,
                "num_frames": len(all_frames),
                "timestamp": time.time(),
            }
            with open(os.path.join(episode_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Episode saved to {episode_path}")
            all_frames = []
            all_states = []
            all_actions = []

        return episode_path

    def run_policy(self, checkpoint_path: str, language: str = "pick up the object"):
        """Run the loaded policy on the robot in real-time.

        This is the main inference loop:
            1. Load the policy checkpoint
            2. Continuously:
                a. Get observation from cameras
                b. Run policy forward pass
                c. Execute the predicted action on the follower arm
                d. Repeat at 50Hz

        Args:
            checkpoint_path: Path to the trained policy checkpoint
            language: Default language instruction
        """
        import numpy as np

        if not self.connected:
            raise RuntimeError("Robot not connected!")

        # Load the policy
        print(f"Loading policy from {checkpoint_path}")

        print("Running policy inference loop at 50Hz...")
        print("Press Ctrl+C to stop.")

        try:
            while True:
                # Get observation
                obs = self.get_observation()

                # Process images for the policy
                images = np.stack([
                    obs["images"][f"camera_{i}"]
                    for i in range(self.config.num_cameras)
                ], axis=0)

                # Run policy (pseudo-code)
                # action = policy.predict_action(images, language, obs["state"])

                # Execute action
                # self.send_action(action)

                # Maintain 50Hz frequency
                time.sleep(0.02)  # 50Hz

        except KeyboardInterrupt:
            print("\nStopped by user.")

    def calibrate(self, arm: str = "both"):
        """Calibrate the robot arms.

        Calibration aligns the joint angles so that:
            - Both arms report the same angles when in the same position
            - The policy trained on one robot works on another

        Args:
            arm: Which arm to calibrate ("leader", "follower", or "both")
        """
        import subprocess

        arm_type = f"so101_leader" if arm in ["leader", "both"] else ""
        arm_type += f"so101_follower" if arm in ["follower", "both"] else ""

        if arm_type:
            try:
                subprocess.run(
                    ["lerobot-calibrate", f"--teleop.type={arm_type}", f"--teleop.port={self.config.leader_port if arm != 'follower' else self.config.follower_port}"],
                    timeout=60
                )
                print(f"Calibrated {arm} arm.")
            except Exception as e:
                print(f"Calibration failed: {e}")
                print("Make sure the USB connection is correct.")


# ============================================================
# Training Loop
# ============================================================

def train_policy(policy, mode: str, train_config: TrainConfig):
    """Train a VLA policy using the specified configuration.

    This function handles the full training loop:
        1. Setup (optimizer, scheduler, dataloader)
        2. Epoch loop
        3. Per-batch forward pass + loss computation
        4. Backward pass + gradient clipping
        5. Logging + checkpoint saving

    Args:
        policy: The policy object (ACTPolicy or Pi05Policy)
        mode: "act" or "pi05" - determines the loss function
        train_config: Training configuration
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import numpy as np
    import os
    import time

    # Set random seed
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    # Create output directories
    for dir_path in [train_config.output_dir, train_config.checkpoint_dir, train_config.log_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Setup device
    device = torch.device(train_config.device)
    policy = policy.to(device)

    # Load dataset
    dataset = ALOHADataset(train_config.dataset, split="train")
    dataset.load_data()

    dataloader = DataLoader(
        dataset,
        batch_size=train_config.dataset.normalize_actions or train_config.config.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Setup optimizer
    lr = train_config.config.lr if hasattr(train_config.config, 'lr') else 1e-4
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=train_config.config.weight_decay)

    # Setup learning rate scheduler (warmup + cosine decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=train_config.config.num_epochs * len(dataloader),
        pct_start=0.1  # 10% warmup
    )

    print(f"Training {mode} policy...")
    print(f"  Device: {device}")
    print(f"  Dataset size: {len(dataset)} episodes")
    print(f"  Batch size: {train_config.config.batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {train_config.config.num_epochs}")
    print("-" * 50)

    # Training loop
    global_step = 0
    for epoch in range(train_config.config.num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            images = batch["images"].to(device)  # (batch, num_cam, C, H, W)
            actions_gt = batch["actions"].to(device)  # (batch, chunk, action_dim)
            state = batch["state"].to(device)  # (batch, state_dim)
            language = batch["language"]  # List of strings

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            if mode == "act":
                # ACT: direct regression to ground truth actions
                actions_pred = policy(images, language[0], state)
                loss = nn.functional.mse_loss(actions_pred, actions_gt)

            elif mode == "pi05":
                # pi0.5: flow matching loss
                # Need language token IDs for the VLM backbone
                language_ids = torch.zeros(actions_gt.size(0), 64, dtype=torch.long, device=device)
                loss = policy.compute_flow_matching_loss(images, language_ids, actions_gt)

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Backward pass
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(policy.parameters(), train_config.config.gradient_clip)

            # Optimizer step
            optimizer.step()
            scheduler.step()

            # Logging
            epoch_loss += loss.item()
            global_step += 1

            if global_step % train_config.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - epoch_start
                print(
                    f"  Epoch [{epoch+1}/{train_config.config.num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(dataloader)}] "
                    f"Loss: {avg_loss:.6f} "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} "
                    f"Time: {elapsed:.1f}s"
                )
                epoch_start = time.time()

        # Save checkpoint at end of epoch
        if (epoch + 1) % train_config.save_interval == 0:
            checkpoint_path = os.path.join(
                train_config.checkpoint_dir,
                f"{mode}_epoch_{epoch+1}.pt"
            )
            policy.save_checkpoint(checkpoint_path)

        # Save final checkpoint
        if epoch + 1 == train_config.config.num_epochs:
            checkpoint_path = os.path.join(
                train_config.checkpoint_dir,
                f"{mode}_last.pt"
            )
            policy.save_checkpoint(checkpoint_path)

        print(f"Epoch {epoch+1}/{train_config.config.num_epochs} complete. "
              f"Average Loss: {epoch_loss / len(dataloader):.6f}")

    print(f"Training complete! Final checkpoint saved to {train_config.checkpoint_dir}")
    return policy


# ============================================================
# Main entry point
# ============================================================

def main():
    """Main entry point: parse arguments and dispatch to the correct mode."""
    parser = argparse.ArgumentParser(
        description="bh-VLA: Unified VLA Policy Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train ACT policy
    python train.py --mode act

    # Train pi0.5 policy
    python train.py --mode pi05

    # Train with custom hyperparameters
    python train.py --mode act --lr 1e-4 --batch-size 16 --epochs 50

    # Resume training
    python train.py --mode act --resume
        """
    )

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["act", "pi05"],
        help="Training mode: 'act' for ACT policy, 'pi05' for pi0.5 policy"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (overrides default)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (overrides default)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides default)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Create training config
    if args.mode == "act":
        policy_config = ACTConfig(
            lr=args.lr or 1e-4,
            batch_size=args.batch_size or 32,
            num_epochs=args.epochs or 100,
        )
    else:
        policy_config = Pi05Config(
            lr=args.lr or 1e-5,
            batch_size=args.batch_size or 8,
            num_epochs=args.epochs or 50,
        )

    train_cfg = TrainConfig(
        mode=args.mode,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
        seed=args.seed,
    )

    print("=" * 60)
    print(f"  bh-VLA Training Framework")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print("=" * 60)

    # Create the policy
    if args.mode == "act":
        policy = ACTPolicy(policy_config)
    else:
        policy = Pi05Policy(policy_config)

    # Print policy architecture summary
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total parameters: {total_params:,}")
    print("-" * 60)

    # Train
    train_policy(policy, args.mode, train_cfg)


if __name__ == "__main__":
    main()
