"""
ACT Policy - Action Chunking Transformer (from ALOHA paper, arXiv:2304.13705)

The ACT policy predicts a sequence (chunk) of future joint positions instead of
one action at a time. This reduces error accumulation by leveraging the temporal
structure of actions via a Transformer decoder.

Architecture overview:
  1. Image encoder (ResNet backbone)      -> visual features per camera
  2. Language encoder (Transformer encoder) -> text instruction embedding
  3. Concatenate visual + language features + state -> projected embedding
  4. Transformer decoder with action tokens -> action chunk prediction
  5. Action prediction head (MLP)           -> per-dimension action values
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet18, resnet50
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.transforms import functional as TF


# ============================================================================
# Helpers
# ============================================================================

def _get_default_vocab() -> List[str]:
    """Default character-level vocabulary for tokenization."""
    chars = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        " .,!?;:'\"-()[]{}|/\\@#$%^&*~`"
        "\t\n"
    )
    specials = ["<pad>", "<unk>", "<sos>", "<eos>"]
    return specials + list(chars)


# ============================================================================
# CharacterTokenizer
# ============================================================================

class CharacterTokenizer:
    """Simple character-level tokenizer with special tokens.

    Every byte in the string maps to a token id. Position encoding is handled
    separately by the language encoder.

    Attributes:
        vocab: List of token strings indexed by token id.
        pad_id: Id of the padding token (0).
        unk_id: Id of the unknown-token fallback (1).
        sos_id: Start-of-sequence token id (2).
        eos_id: End-of-sequence token id (3).
    """

    def __init__(self, vocab: Optional[List[str]] = None) -> None:
        if vocab is None:
            vocab = _get_default_vocab()
        self.vocab = vocab
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3
        self.str_to_id: Dict[str, int] = {ch: i for i, ch in enumerate(vocab)}

    def encode(self, text: str) -> List[int]:
        """Encode *text* into a list of token ids: [<sos>, chars..., <eos>]."""
        ids = [self.sos_id]
        for ch in text.lower():
            token = self.str_to_id.get(ch)
            if token is None:
                token = self.unk_id
            ids.append(token)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to a string (excluding special tokens)."""
        chars = []
        for i in ids:
            if i in (self.pad_id, self.sos_id, self.eos_id, self.unk_id):
                continue
            chars.append(self.vocab[i])
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.vocab)


# ============================================================================
# 1. ResNetImageEncoder
# ============================================================================

class ResNetImageEncoder(nn.Module):
    """ResNet-based image encoder for camera views.

    Strips the final classification layer of a pre-trained ResNet and optionally
    further removes the global average pooling so we can use adaptive pooling to
    get a fixed-width vector regardless of input resolution.

    Args:
        backbone_name: "resnet18" or "resnet50".
        num_cameras: Number of distinct camera views (default 3 for ALOHA).
        pretrained: Whether to load ImageNet weights (default True).
        hidden_dim: Width of the output feature vector per camera.
        drop_rate: Dropout rate applied to the pooled features (default 0.25).
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_cameras: int = 3,
        pretrained: bool = True,
        hidden_dim: int = 512,
        drop_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.hidden_dim = hidden_dim

        # Build per-camera backbone
        if backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            in_planes = 512  # ResNet-18 final stage channels
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
            in_planes = 2048  # ResNet-50 final stage channels
        else:
            raise ValueError(f"Unsupported ResNet variant: {backbone_name}")

        # Remove classification head (fc layer)
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-2],  # keeps conv5_x, strips avgpool+fc
        )

        # Adaptive average pooling -> (B, in_planes, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projector: in_planes -> hidden_dim
        self.projection = nn.Sequential(
            nn.Conv2d(in_planes, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
        )

        # Optional dropout after pooling
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of multi-camera images.

        Args:
            images: Tensor of shape (B, num_cameras, C, H, W).

        Returns:
            Encoded features of shape (B, hidden_dim * num_cameras) after
            concatenation across camera views.
        """
        b, n_cam = images.shape[:2]

        # Flatten (B, N, C, H, W) -> (B*N, C, H, W) for batched encoding
        images_flat = images.view(-1, *images.shape[2:])  # (B*N, 3, H, W)

        # Backbone -> (B*N, in_planes, h, w)
        feats = self.backbone(images_flat)

        # Pool -> (B*N, in_planes, 1, 1)
        feats = self.pool(feats)

        # Project & flatten -> (B*N, hidden_dim)
        feats = self.projection(feats)

        # Dropout
        feats = self.dropout(feats)

        # Reshape back to (B, N*hidden_dim)
        feats = feats.view(b, n_cam * self.hidden_dim)
        return feats


# ============================================================================
# 2. LanguageEncoder
# ============================================================================

class LanguageEncoder(nn.Module):
    """Transformer-encoder based language encoder with character-level tokenization.

    Converts text instructions into dense vectors via:
      char-token -> embedding -> positional encoding -> Transformer encoder

    Args:
        vocab_size: Size of the tokenizer vocabulary.
        hidden_dim: Width of the language embedding (must match the overall model
            hidden_dim).
        num_layers: Number of Transformer encoder layers (default 2).
        num_heads: Number of attention heads per layer (default 4).
        dropout: Dropout rate (default 0.1).
        max_seq_len: Maximum instruction length (default 128).
    """

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Token embedding (learnable, includes pad/unk/sos/eos)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        # Learnable positional encoding (sinusoidal)
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_enc", pe)  # (max_seq_len, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm after the encoder
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Final projection
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of token ids into language vectors.

        Args:
            token_ids: LongTensor of shape (B, seq_len). Token ids from the
                character tokenizer.

        Returns:
            Encoded language vector of shape (B, hidden_dim).
        """
        b, seq_len = token_ids.shape

        # Embed -> (B, seq_len, hidden_dim)
        x = self.token_embed(token_ids)  # (B, seq_len, d)

        # Add positional encoding
        x = x + self.pos_enc[:seq_len, :]  # broadcast

        # Transformer encoder
        mask = self._generate_padding_mask(token_ids)  # (B, seq_len)
        x = self.encoder(x, src_key_padding_mask=mask)  # (B, seq_len, d)

        # Average pooling over non-padding tokens
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        num_tokens = (~mask).sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        x = x.sum(dim=1, keepdim=True) / num_tokens  # (B, 1, d)
        x = x.squeeze(1)  # (B, d)

        # Final projection + GELU
        x = F.gelu(self.final_proj(x))

        return x

    def _generate_padding_mask(self, token_ids: torch.Tensor) -> torch.BoolTensor:
        """Create a boolean mask: True where the token is padding (id==0)."""
        return token_ids == 0


# ============================================================================
# 3. ACTPolicy - the main policy class
# ============================================================================

@dataclass
class ACTConfig:
    """Configuration for the ACT policy.

    Attributes:
        num_cameras: Number of camera views (default 3).
        image_size: Input image size for each camera (H, W), default (224, 224).
        backbone: ResNet variant, "resnet18" or "resnet50" (default "resnet18").
        hidden_dim: Model hidden dimension (default 256).
        action_chunk_size: Number of future actions to predict per step (default 32).
        action_dim: Number of action dimensions (robot joint positions, default 14 for ALOHA/SO-101).
        num_decoder_layers: Number of Transformer decoder layers (default 4).
        num_decoder_heads: Number of attention heads in decoder (default 8).
        num_layers_lang: Number of Transformer encoder layers for language (default 2).
        vocab_size: Tokenizer vocabulary size (default 128).
        state_dim: Dimension of robot state input (default 28).
        dropout: Dropout rate for all layers (default 0.1).
        max_instr_len: Maximum instruction length (default 128).
        clip_norm: Gradient clipping norm (default 1.0).
        lr: Learning rate (default 1e-4).
        weight_decay: Weight decay for optimizer (default 1e-4).
        warmup_steps: Warmup steps for LR scheduler (default 1000).
        gradient_clip: Gradient clipping norm (default 1.0).
        batch_size: Training batch size (default 32).
        num_epochs: Number of training epochs (default 100).
    """

    num_cameras: int = 3
    image_size: Tuple[int, int] = (224, 224)
    backbone: str = "resnet18"
    pretrained_backbone: bool = False
    hidden_dim: int = 256
    action_chunk_size: int = 32
    action_dim: int = 14
    num_decoder_layers: int = 4
    num_decoder_heads: int = 8
    num_layers_lang: int = 2
    vocab_size: int = 128
    state_dim: int = 28
    dropout: float = 0.1
    max_instr_len: int = 128
    clip_norm: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    batch_size: int = 32
    num_epochs: int = 100

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ACTConfig":
        """Create config from a dict, ignoring unknown keys for robustness."""
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in values.items() if k in allowed})


class ACTPolicy(nn.Module):
    """Action Chunking Transformer (ACT) policy.

    Predicts a chunk of ``action_chunk_size`` future joint positions given:
      - multi-camera images (B, num_cameras, C, H, W)
      - language instruction (B, seq_len) token ids
      - robot state (B, state_dim)

    The architecture follows arXiv:2304.13705 Sec. 3.1-3.3:
      1. Encode each camera view with a shared ResNet -> visual features.
      2. Encode the language instruction with a Transformer encoder.
      3. Concatenate visual + language features and project to hidden_dim.
      4. Use this as the initial hidden state for a Transformer decoder whose
         inputs are learnable "action tokens" (one per chunk position).
      5. Inject robot state via FiLM-style modulation at each decoder layer.
      6. Project decoder outputs to action dimensions.
    """

    def __init__(self, config: Optional[ACTConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = ACTConfig()

        self.config = config
        self.action_chunk_size = config.action_chunk_size
        self.action_dim = config.action_dim

        # ---- Image encoder (shared across cameras) ------
        self.image_encoder = ResNetImageEncoder(
            backbone_name=config.backbone,
            num_cameras=config.num_cameras,
            pretrained=config.pretrained_backbone,
            hidden_dim=config.hidden_dim,
        )

        # ---- Language encoder ------
        self.lang_encoder = LanguageEncoder(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers_lang,
            num_heads=config.num_decoder_heads,
            dropout=config.dropout,
            max_seq_len=config.max_instr_len,
        )

        # ---- Feature projection (visual + lang -> hidden_dim) ------
        # Visual features: (B, num_cameras * hidden_dim)
        # Language features: (B, hidden_dim)
        # Concatenated: (B, (num_cameras+1) * hidden_dim) -> hidden_dim
        visual_dim = config.num_cameras * config.hidden_dim
        self.feat_proj = nn.Sequential(
            nn.Linear(visual_dim + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # ---- Action token initialization ------
        # Learnable "action tokens" serve as the decoder input (one per chunk
        # position). They are analogous to the object query in DETR.
        self.action_tokens = nn.Parameter(
            torch.zeros(1, config.action_chunk_size, config.hidden_dim)
        )
        nn.init.normal_(self.action_tokens, std=0.02)

        # ---- State injection via FiLM modulation ------
        # The robot state is projected to (gamma, beta) per decoder layer.
        self.state_film_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.state_dim, config.hidden_dim * 2),
            )
            for _ in range(config.num_decoder_layers)
        ])

        # ---- Transformer decoder ------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_decoder_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)

        # ---- Action prediction head ------
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.action_dim),
        )

        # ---- Position embedding for action tokens ------
        # The decoder already uses positional encoding internally, but since our
        # "queries" are learnable tokens, we add a small positional bias for each
        # chunk position to give the model an absolute-time signal.
        self.chunk_pos_embed = nn.Parameter(
            torch.zeros(1, config.action_chunk_size, config.hidden_dim)
        )
        nn.init.normal_(self.chunk_pos_embed, std=0.01)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-initialize the action head and FiLM layers."""
        for module in self.action_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for film in self.state_film_proj:
            for sub in film:
                if isinstance(sub, nn.Linear):
                    nn.init.xavier_uniform_(sub.weight)
                    nn.init.zeros_(sub.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,          # (B, num_cameras, C, H, W)
        token_ids: torch.Tensor,       # (B, seq_len)
        state: torch.Tensor,           # (B, state_dim)
        return_features: bool = False, # If True, also return intermediate features
    ) -> torch.Tensor:
        """Full forward pass of the ACT policy.

        Args:
            images: Multi-camera images.
            token_ids: Token ids of the language instruction.
            state: Robot joint state (current configuration).
            return_features: If True, also return visual/lang features for analysis.

        Returns:
            Predicted action chunk of shape (B, action_chunk_size, action_dim).
        """
        b = images.shape[0]

        # 1. Image encoding -> (B, hidden_dim * num_cameras)
        visual_feats = self.image_encoder(images)  # (B, num_cameras*hidden_dim)

        # 2. Language encoding -> (B, hidden_dim)
        lang_feats = self.lang_encoder(token_ids)  # (B, hidden_dim)

        # 3. Concatenate and project -> (B, hidden_dim)
        fused = torch.cat([visual_feats, lang_feats], dim=1)  # (B, (N+1)*hidden_dim)
        fused = self.feat_proj(fused)  # (B, hidden_dim)

        # 4. Expand action tokens to batch and add positional bias
        tokens = self.action_tokens.expand(b, -1, -1) + self.chunk_pos_embed.expand(b, -1, -1)

        # 5. FiLM modulation parameters per decoder layer
        film_params: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(self.config.num_decoder_layers):
            gamma_beta = self.state_film_proj[layer_idx](state)  # (B, hidden_dim*2)
            gamma = gamma_beta[:, :self.config.hidden_dim]        # (B, hidden_dim)
            beta = gamma_beta[:, self.config.hidden_dim:]          # (B, hidden_dim)
            film_params.append((gamma, beta))

        # 6. Transformer decoder with FiLM modulation
        output = self._forward_with_film(tokens, fused, film_params)  # (B, chunk, hidden_dim)

        # 7. Project to action space
        action_chunk = self.action_head(output)  # (B, chunk, action_dim)

        if return_features:
            return action_chunk, {
                "visual_feats": visual_feats,
                "lang_feats": lang_feats,
                "fused_feats": fused,
            }
        return action_chunk

    def _forward_with_film(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        film_params: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Run the decoder with FiLM modulation applied before each layer.

        Args:
            tgt: Action tokens of shape (B, chunk_size, hidden_dim).
            memory: Fused visual+lang features of shape (B, hidden_dim) or
                (B, 1, hidden_dim). Expanded internally if needed.
            film_params: List of (gamma, beta) per decoder layer.

        Returns:
            Decoder output of shape (B, chunk_size, hidden_dim).
        """
        # Expand memory to (B, 1, hidden_dim) if needed
        if memory.ndim == 2:
            memory = memory.unsqueeze(1)

        output = tgt
        for layer_idx in range(len(self.decoder.layers)):
            layer = self.decoder.layers[layer_idx]
            gamma, beta = film_params[layer_idx]

            # FiLM on target (applied to the current layer input)
            tgt_film = output * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

            # Multi-head self-attention on target
            attn_out = layer.self_attn(tgt_film, tgt_film, tgt_film)[0]
            output = output + F.dropout(attn_out, p=self.config.dropout, training=self.training)
            output = layer.norm1(output)

            # FiLM on memory (computed fresh each iteration with the same gamma/beta)
            mem_film = memory * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

            # Cross-attention with FiLM-modulated memory
            cross_attn = layer.cross_attn(query=output, key=mem_film, value=mem_film)[0]
            output = output + F.dropout(cross_attn, p=self.config.dropout, training=self.training)
            output = layer.norm2(output)

            # Feed-forward
            ff = layer.linear2(F.gelu(F.dropout(
                layer.linear1(output), p=self.config.dropout, training=self.training
            )))
            output = output + F.dropout(ff, p=self.config.dropout, training=self.training)
            output = layer.norm3(output)

        return output

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: return the **first** action in the predicted chunk.

        In ALOHA, the robot executes only the first action of the chunk and
        replans at the next timestep (rolling horizon).

        Args:
            images: (B, num_cameras, C, H, W) or (num_cameras, C, H, W).
            token_ids: (B, seq_len) or (seq_len,).
            state: (B, state_dim) or (state_dim,).

        Returns:
            Action tensor of shape (B, action_dim) -- only the first timestep.
        """
        # Handle both batched and single-sample inputs
        singleton = False
        if images.ndim == 4:
            images = images.unsqueeze(0)
            singleton = True
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
            singleton = True
        if state.ndim == 1:
            state = state.unsqueeze(0)
            singleton = True

        chunk = self.forward(images, token_ids, state)  # (B, chunk, dim)
        action = chunk[:, 0, :]  # (B, dim) -- first action of the chunk

        if singleton:
            action = action.squeeze(0)  # back to (dim,) for single sample
        return action

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save the full model state dict to *path*.

        Args:
            path: File path (will be created/overwritten).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: str | torch.device = "cpu",
        config: Optional[ACTConfig] = None,
    ) -> ACTPolicy:
        """Load a checkpoint and return an ACTPolicy instance.

        Args:
            path: Checkpoint file path.
            device: Device to load the model onto.

        Returns:
            Loaded ACTPolicy instance.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if config is None and isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
            config = ACTConfig.from_dict(checkpoint["config"])
        policy = cls(config)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        policy.load_state_dict(state_dict)
        policy = policy.to(device)
        policy.eval()
        return policy


# ============================================================================
# 4. ACTLoss
# ============================================================================

class ACTLoss(nn.Module):
    """Loss function for the ACT policy.

    Uses MSE loss between the predicted action chunk and the ground-truth
    action chunk. Supports sample-level masking for variable-length sequences.

    Args:
        reduction: "mean" or "sum" -- how to aggregate the loss.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        pred_chunk: torch.Tensor,  # (B, chunk_size, action_dim)
        gt_chunk: torch.Tensor,    # (B, chunk_size, action_dim)
        mask: Optional[torch.Tensor] = None,  # (B, chunk_size), True=ignore
    ) -> torch.Tensor:
        """Compute the MSE loss.

        Args:
            pred_chunk: Predicted action chunk.
            gt_chunk: Ground-truth action chunk.
            mask: Optional boolean mask of shape (B, chunk_size). Masked
                positions contribute zero to the loss.

        Returns:
            Scalar loss tensor.
        """
        if mask is None:
            return self.mse_loss(pred_chunk, gt_chunk)

        valid_mask = (~mask).unsqueeze(-1).to(pred_chunk.dtype)
        squared_error = (pred_chunk - gt_chunk).pow(2) * valid_mask
        if self.reduction == "sum":
            return squared_error.sum()
        denom = valid_mask.sum().clamp(min=1.0) * pred_chunk.shape[-1]
        return squared_error.sum() / denom


# ============================================================================
# 5. ACTDataPreprocessor
# ============================================================================

class ACTDataPreprocessor:
    """Data preprocessor for ACT policy training and inference.

    Handles:
      - Image normalization (ImageNet stats)
      - Action normalization (per-dimension z-score)
      - State normalization (per-dimension z-score)
      - Data augmentation (random crop, color jitter, horizontal flip)

    Attributes:
        action_means: Per-dimension mean of action space (computed from data).
        action_stds: Per-dimension std of action space.
        state_means: Per-dimension mean of state space.
        state_stds: Per-dimension std of state space.
        imagenet_mean: ImageNet RGB mean (default).
        imagenet_std: ImageNet RGB std (default).
        augment: Whether to apply data augmentation (default True for training).
    """

    # ImageNet normalization constants (RGB)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        action_dim: int = 12,
        state_dim: int = 12,
        augment: bool = True,
    ) -> None:
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.augment = augment

        # Normalization statistics -- set via ``compute_stats()`` before training
        self.action_means: Optional[np.ndarray] = None
        self.action_stds: Optional[np.ndarray] = None
        self.state_means: Optional[np.ndarray] = None
        self.state_stds: Optional[np.ndarray] = None

        # ImageNet normalization is applied per-channel
        self.imagenet_mean = torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.imagenet_std = torch.tensor(self.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def compute_stats(
        self,
        actions: np.ndarray,   # (N, action_dim)
        states: np.ndarray,    # (N, state_dim)
    ) -> None:
        """Compute normalization statistics from training data.

        Args:
            actions: Array of all action vectors.
            states: Array of all robot state vectors.
        """
        self.action_means = actions.mean(axis=0)
        self.action_stds = actions.std(axis=0) + 1e-8  # avoid division by zero
        self.state_means = states.mean(axis=0)
        self.state_stds = states.std(axis=0) + 1e-8

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Z-score normalize an action vector."""
        assert self.action_means is not None, "Call compute_stats() first"
        return (action - self.action_means) / self.action_stds

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Inverse z-score normalize an action vector."""
        assert self.action_stds is not None, "Call compute_stats() first"
        return action * self.action_stds + self.action_means

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Z-score normalize a robot state vector."""
        assert self.state_means is not None, "Call compute_stats() first"
        return (state - self.state_means) / self.state_stds

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        """Inverse z-score normalize a robot state vector."""
        assert self.state_stds is not None, "Call compute_stats() first"
        return state * self.state_stds + self.state_means

    def normalize_images(self, images: np.ndarray) -> np.ndarray:
        """Normalize images to ImageNet stats.

        Args:
            images: Array of shape (N, num_cameras, 3, H, W) or
                (num_cameras, 3, H, W). Values should be in [0, 255].

        Returns:
            Normalized array with the same shape.
        """
        images = images.astype(np.float32) / 255.0  # -> [0, 1]
        mean = np.asarray(self.IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3, 1, 1)
        std = np.asarray(self.IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3, 1, 1)
        return (images - mean) / std

    def preprocess_sample(
        self,
        sample: Dict[str, Any],
        augmentation: bool = False,
    ) -> Dict[str, Any]:
        """Preprocess a single dataset sample.

        Args:
            sample: Dictionary with keys:
                - "images": (N_cam, 3, H, W) uint8 images
                - "actions": (chunk_size, action_dim) float actions
                - "states": (state_dim,) float state
                - "instruction": str text instruction
                - "actions_mask": optional (chunk_size,) bool mask

        Returns:
            Processed dictionary with normalized tensors and tokenized text.
        """
        images = sample["images"]  # (N_cam, 3, H, W)
        actions = sample["actions"]
        state = sample["states"]
        instruction = sample["instruction"]

        # Normalize images
        images_normalized = self.normalize_images(images[np.newaxis])[0]  # (N, 3, H, W)

        # Normalize action and state
        actions_norm = self.normalize_action(actions)
        state_norm = self.normalize_state(state)

        # Tokenize instruction
        tokenizer = CharacterTokenizer()
        token_ids = tokenizer.encode(instruction)

        result: Dict[str, Any] = {
            "images": torch.from_numpy(images_normalized).float(),
            "actions": torch.from_numpy(actions_norm).float(),
            "state": torch.from_numpy(state_norm).float(),
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
        }

        if "actions_mask" in sample and sample["actions_mask"] is not None:
            result["actions_mask"] = torch.tensor(sample["actions_mask"], dtype=torch.bool)

        # Apply augmentation if requested (training)
        if augmentation and self.augment:
            result["images"] = self._augment_images(result["images"])

        return result

    def _augment_images(self, images: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to images.

        Args:
            images: (N_cam, 3, H, W) float tensor in normalized range.

        Returns:
            Augmented images with the same shape.
        """
        # Random crop & flip (applied per-camera independently for variety)
        n_cam = images.shape[0]
        aug_images: List[torch.Tensor] = []
        for i in range(n_cam):
            img = images[i]  # (3, H, W)
            # Random horizontal flip
            if random.random() > 0.5:
                img = img.flip(dims=[2])
            # Random color jitter
            if random.random() > 0.5:
                img = TF.adjust_brightness(img, 0.8 + 0.4 * random.random())
                img = TF.adjust_contrast(img, 0.8 + 0.4 * random.random())
                img = TF.adjust_saturation(img, 0.8 + 0.4 * random.random())
            aug_images.append(img)
        return torch.stack(aug_images)


# ============================================================================
# 6. ACTDataset -- Dataset wrapper for DataLoader support
# ============================================================================

class ACTDataset(Dataset):
    """PyTorch Dataset for ACT training data.

    Expects a list of raw sample dicts with the following keys:
      - "images": (N_cam, C, H, W) uint8 numpy array
      - "actions": (chunk_size, action_dim) float numpy array
      - "states": (state_dim,) float numpy array
      - "instruction": str
      - "actions_mask": optional (chunk_size,) bool numpy array

    Usage:
        dataset = ACTDataset(samples, preprocessor, augment=True)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        preprocessor: Optional[ACTDataPreprocessor] = None,
        augment: bool = True,
    ) -> None:
        self.samples = samples
        self.preprocessor = preprocessor or ACTDataPreprocessor(augment=augment)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return self.preprocessor.preprocess_sample(sample, augmentation=self.augment)


# ============================================================================
# Helper: create a default policy instance with convenient factory method
# ============================================================================

def create_act_policy(
    device: str | torch.device = "cpu",
    config: Optional[ACTConfig] = None,
) -> ACTPolicy:
    """Create and return an ACTPolicy instance on the given device.

    Convenience factory: instantiates with the given config (defaults to
    ALOHA-style settings) and moves the model to *device*.
    """
    if config is None:
        config = ACTConfig()
    policy = ACTPolicy(config)
    policy = policy.to(device)
    policy.eval()
    return policy
