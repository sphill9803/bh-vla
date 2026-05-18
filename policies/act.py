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
from collections import deque
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
        )

        # Optional dropout after pooling
        self.dropout = nn.Dropout(drop_rate)

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Encode multi-camera images as spatial tokens.

        Args:
            images: Tensor of shape (B, num_cameras, C, H, W).

        Returns:
            Encoded features of shape (B, num_cameras * h * w, hidden_dim).
        """
        b, n_cam = images.shape[:2]

        # Flatten (B, N, C, H, W) -> (B*N, C, H, W) for batched encoding
        images_flat = images.view(-1, *images.shape[2:])  # (B*N, 3, H, W)

        # Backbone -> (B*N, in_planes, h, w)
        feats = self.backbone(images_flat)

        # Project -> (B*N, hidden_dim, h, w)
        feats = self.projection(feats)
        feats = self.dropout(feats)
        _, d, h, w = feats.shape

        # Reshape back to camera-aware spatial tokens.
        feats = feats.view(b, n_cam, d, h, w)
        feats = feats.permute(0, 1, 3, 4, 2).reshape(b, n_cam * h * w, d)
        return feats

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of multi-camera images as pooled camera vectors."""
        b, n_cam = images.shape[:2]
        images_flat = images.view(-1, *images.shape[2:])
        feats = self.backbone(images_flat)
        feats = self.pool(feats)
        feats = self.projection(feats)
        feats = feats.flatten(1)
        feats = self.dropout(feats)
        feats = feats.view(b, n_cam * self.hidden_dim)
        return feats

def _sinusoidal_1d(num_positions: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(num_positions, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(num_positions, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


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
        hidden_dim: Model hidden dimension (default 512).
        action_chunk_size: Number of future actions to predict per step (default 100).
        n_action_steps: Number of queued actions to execute per model call.
        action_dim: Number of action dimensions (robot joint positions, default 14 for ALOHA/SO-101).
        num_encoder_layers: Number of Transformer encoder layers (default 4).
        num_decoder_layers: Number of Transformer decoder layers (default 1).
        num_decoder_heads: Number of attention heads in decoder (default 8).
        num_layers_lang: Number of Transformer encoder layers for language (default 2).
        vocab_size: Tokenizer vocabulary size (default 128).
        state_dim: Dimension of robot state input (default 28).
        use_vae: Whether to train with the ACT CVAE latent objective.
        latent_dim: CVAE latent width.
        num_vae_encoder_layers: Number of VAE encoder layers.
        kl_weight: KL divergence loss weight.
        temporal_ensemble_coeff: Optional exponential temporal ensemble coefficient.
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
    pretrained_backbone: bool = True
    hidden_dim: int = 512
    action_chunk_size: int = 100
    n_action_steps: int = 100
    action_dim: int = 14
    num_encoder_layers: int = 4
    num_decoder_layers: int = 1
    num_decoder_heads: int = 8
    num_layers_lang: int = 2
    vocab_size: int = 128
    state_dim: int = 28
    use_vae: bool = True
    latent_dim: int = 32
    num_vae_encoder_layers: int = 4
    kl_weight: float = 10.0
    temporal_ensemble_coeff: Optional[float] = None
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
        if isinstance(values.get("image_size"), list):
            values = dict(values)
            values["image_size"] = tuple(values["image_size"])
        return cls(**{k: v for k, v in values.items() if k in allowed})


class ACTPolicy(nn.Module):
    """Action Chunking Transformer (ACT) policy.

    Predicts a chunk of ``action_chunk_size`` future joint positions given:
      - multi-camera images (B, num_cameras, C, H, W)
      - language instruction (B, seq_len) token ids
      - robot state (B, state_dim)

    The architecture follows the ACT/LeRobot shape:
      1. Encode each camera view with a shared ResNet -> spatial feature tokens.
      2. Encode the language instruction with a Transformer encoder.
      3. Encode state, language, image tokens, and an ACT latent token.
      4. During training, infer the latent from state + target action chunk.
      5. Decode learnable action queries against the encoded context.
      6. Train with masked L1 reconstruction plus optional KL divergence.
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

        # ---- Conditional tokens for the ACT transformer ------
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)
        self.lang_type_embed = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.state_type_embed = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.latent_type_embed = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.camera_embed = nn.Embedding(config.num_cameras, config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_decoder_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        # ---- VAE encoder used only during training ------
        if config.use_vae:
            self.vae_cls = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
            self.vae_state_proj = nn.Linear(config.state_dim, config.hidden_dim)
            self.vae_action_proj = nn.Linear(config.action_dim, config.hidden_dim)
            vae_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_decoder_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.vae_encoder = nn.TransformerEncoder(
                vae_layer, num_layers=config.num_vae_encoder_layers
            )
            self.vae_latent_head = nn.Linear(config.hidden_dim, config.latent_dim * 2)

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
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

        # ---- Learnable decoder queries, DETR-style ------
        self.action_query = nn.Embedding(config.action_chunk_size, config.hidden_dim)

        self._init_weights()
        self.reset()

    def _init_weights(self) -> None:
        """Xavier-initialize projection and transformer parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.lang_type_embed, std=0.02)
        nn.init.normal_(self.state_type_embed, std=0.02)
        nn.init.normal_(self.latent_type_embed, std=0.02)
        if hasattr(self, "vae_cls"):
            nn.init.normal_(self.vae_cls, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,          # (B, num_cameras, C, H, W)
        token_ids: torch.Tensor,       # (B, seq_len)
        state: torch.Tensor,           # (B, state_dim)
        actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        return_features: bool = False, # If True, also return intermediate features
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        # 1. Image encoding -> spatial tokens (B, num_cameras*h*w, hidden_dim)
        image_tokens = self.image_encoder.forward_tokens(images)
        image_tokens = image_tokens + self._image_pos_embed(image_tokens, images.shape[1])

        # 2. Language and state tokens.
        lang_feats = self.lang_encoder(token_ids).unsqueeze(1) + self.lang_type_embed
        state_feats = self.state_proj(state).unsqueeze(1) + self.state_type_embed

        latent, mu, logvar = self._encode_latent(state, actions, action_mask)
        latent_feats = self.latent_proj(latent).unsqueeze(1) + self.latent_type_embed

        encoder_input = torch.cat([latent_feats, state_feats, lang_feats, image_tokens], dim=1)
        memory = self.encoder(encoder_input)

        # 3. Decode fixed action queries against encoded observation/context tokens.
        query = self.action_query.weight.unsqueeze(0).expand(b, -1, -1)
        output = self.decoder(query, memory)
        action_chunk = self.action_head(output)

        aux: Dict[str, torch.Tensor] = {}
        if mu is not None and logvar is not None:
            aux = {"mu": mu, "logvar": logvar}

        if return_features:
            aux.update({
                "image_tokens": image_tokens,
                "lang_feats": lang_feats.squeeze(1),
                "state_feats": state_feats.squeeze(1),
                "memory": memory,
            })
            return action_chunk, aux
        return action_chunk

    def _encode_latent(
        self,
        state: torch.Tensor,
        actions: Optional[torch.Tensor],
        action_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        b = state.shape[0]
        if not (self.config.use_vae and self.training and actions is not None):
            latent = torch.zeros(
                b, self.config.latent_dim, dtype=state.dtype, device=state.device
            )
            return latent, None, None

        cls = self.vae_cls.expand(b, -1, -1)
        state_tok = self.vae_state_proj(state).unsqueeze(1)
        action_tok = self.vae_action_proj(actions)
        vae_input = torch.cat([cls, state_tok, action_tok], dim=1)

        pos = _sinusoidal_1d(vae_input.shape[1], self.config.hidden_dim, vae_input.device, vae_input.dtype)
        vae_input = vae_input + pos.unsqueeze(0)

        key_padding_mask = None
        if action_mask is not None:
            prefix = torch.zeros(b, 2, dtype=torch.bool, device=action_mask.device)
            key_padding_mask = torch.cat([prefix, action_mask.bool()], dim=1)

        cls_out = self.vae_encoder(vae_input, src_key_padding_mask=key_padding_mask)[:, 0]
        params = self.vae_latent_head(cls_out)
        mu, logvar = params.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        latent = mu + std * torch.randn_like(std)
        return latent, mu, logvar

    def _image_pos_embed(self, image_tokens: torch.Tensor, num_cameras: int) -> torch.Tensor:
        """Create camera-aware 1D spatial position embeddings for image tokens."""
        b, seq, d = image_tokens.shape
        tokens_per_camera = max(seq // max(num_cameras, 1), 1)
        spatial = _sinusoidal_1d(tokens_per_camera, d, image_tokens.device, image_tokens.dtype)
        spatial = spatial.unsqueeze(0).repeat(num_cameras, 1, 1)
        cam_ids = torch.arange(num_cameras, device=image_tokens.device)
        cam = self.camera_embed(cam_ids).to(dtype=image_tokens.dtype).unsqueeze(1)
        pos = (spatial + cam).reshape(1, num_cameras * tokens_per_camera, d)
        return pos[:, :seq, :]

    def compute_loss(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred, aux = self.forward(
            images,
            token_ids,
            state,
            actions=actions,
            action_mask=action_mask,
            return_features=True,
        )
        valid_mask = torch.ones_like(actions[..., :1], dtype=pred.dtype)
        if action_mask is not None:
            valid_mask = (~action_mask).unsqueeze(-1).to(pred.dtype)

        l1 = (torch.abs(pred - actions) * valid_mask).sum()
        denom = (valid_mask.sum() * pred.shape[-1]).clamp(min=1.0)
        l1 = l1 / denom

        loss = l1
        metrics = {"l1_loss": float(l1.detach().cpu())}
        if self.config.use_vae and "mu" in aux and "logvar" in aux:
            mu, logvar = aux["mu"], aux["logvar"]
            kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
            loss = loss + self.config.kl_weight * kld
            metrics["kld_loss"] = float(kld.detach().cpu())
        metrics["loss"] = float(loss.detach().cpu())
        return loss, metrics

    def reset(self) -> None:
        """Reset rollout state between episodes."""
        self._action_queue: deque[torch.Tensor] = deque(maxlen=max(self.config.n_action_steps, 1))
        self._ensembled_actions: Optional[torch.Tensor] = None
        self._ensembled_count: Optional[torch.Tensor] = None

    def select_action(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Select one executable action using queueing or temporal ensemble."""
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            chunk = self._predict_chunk_batched(images, token_ids, state)
            return self._temporal_ensemble_update(chunk)

        if not self._action_queue:
            chunk = self._predict_chunk_batched(images, token_ids, state)
            steps = min(self.config.n_action_steps, self.config.action_chunk_size)
            for action in chunk[:, :steps].transpose(0, 1):
                self._action_queue.append(action)
        return self._action_queue.popleft()

    def _predict_chunk_batched(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.forward(images, token_ids, state)

    def _temporal_ensemble_update(self, actions: torch.Tensor) -> torch.Tensor:
        coeff = float(self.config.temporal_ensemble_coeff or 0.0)
        chunk = actions.shape[1]
        weights = torch.exp(-coeff * torch.arange(chunk, device=actions.device, dtype=actions.dtype))
        cumsum = torch.cumsum(weights, dim=0)
        if self._ensembled_actions is None:
            self._ensembled_actions = actions.clone()
            self._ensembled_count = torch.ones((chunk, 1), dtype=torch.long, device=actions.device)
        else:
            count = self._ensembled_count
            prev = self._ensembled_actions
            prev *= cumsum[count - 1].view(1, -1, 1)
            next_count = torch.clamp(count, max=chunk - 1)
            prev += actions[:, :-1] * weights[next_count].view(1, -1, 1)
            prev /= cumsum[next_count].view(1, -1, 1)
            count = torch.clamp(count + 1, max=chunk)
            self._ensembled_actions = torch.cat([prev, actions[:, -1:]], dim=1)
            self._ensembled_count = torch.cat([count, torch.ones_like(count[-1:])])
        action = self._ensembled_actions[:, 0]
        self._ensembled_actions = self._ensembled_actions[:, 1:]
        self._ensembled_count = self._ensembled_count[1:]
        return action

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
        """Inference: return one executable action.

        By default this uses an ACT-style action queue and only queries the
        network when the queue is empty. If temporal ensembling is enabled, the
        policy instead queries every step and returns the ensembled action.

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

        action = self.select_action(images, token_ids, state)

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
    """Masked L1 reconstruction loss for ACT action chunks.

    LeRobot and the original ACT implementation train the action decoder with
    L1 reconstruction, optionally adding a KL term when using the VAE path.

    Args:
        reduction: "mean" or "sum" -- how to aggregate the loss.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(
        self,
        pred_chunk: torch.Tensor,  # (B, chunk_size, action_dim)
        gt_chunk: torch.Tensor,    # (B, chunk_size, action_dim)
        mask: Optional[torch.Tensor] = None,  # (B, chunk_size), True=ignore
    ) -> torch.Tensor:
        """Compute masked L1 loss.

        Args:
            pred_chunk: Predicted action chunk.
            gt_chunk: Ground-truth action chunk.
            mask: Optional boolean mask of shape (B, chunk_size). Masked
                positions contribute zero to the loss.

        Returns:
            Scalar loss tensor.
        """
        if mask is None:
            return self.l1_loss(pred_chunk, gt_chunk)

        valid_mask = (~mask).unsqueeze(-1).to(pred_chunk.dtype)
        abs_error = torch.abs(pred_chunk - gt_chunk) * valid_mask
        if self.reduction == "sum":
            return abs_error.sum()
        denom = valid_mask.sum().clamp(min=1.0) * pred_chunk.shape[-1]
        return abs_error.sum() / denom


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
