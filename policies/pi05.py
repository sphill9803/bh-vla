#!/usr/bin/env python3
"""
pi0.5 (Physical Intelligence) Policy Implementation
=====================================================

Implements the complete pi0.5 architecture from:
  "pi0.5: A Vision-Language-Action Model with Open-World Generalization"
  https://arxiv.org/abs/2504.16054

Core Design:
  - PaliGemma 3B backbone: SigLIP ViT-L/14 + LLaMA-style decoder
  - Flow matching for action distribution learning (not diffusion)
  - Action expert MLP specialized for robot action prediction
  - Co-training on heterogeneous tasks for open-world generalization

Architecture overview:
  images (B, C, H, W) → SigLIP ViT-L/14 (27 layers)
    → [vision_cls, vision_patches] ∈ ℝ^(B, 1+NP, 1024)
  language_ids (B, TL) → LLaMA decoder (28 layers)
    → [text_features] ∈ ℝ^(B, TL, 3200)
  vision features projected to LLM space (1024 → 3200) via MLP projector
  [vision_pooled, text_features] concatenated → action token
  action token → ActionExpert MLP → (B, chunk_size, action_dim)

Flow Matching Training:
  Sample t ~ Uniform(0,1), ε ~ N(0,I)
  x_t = (1-t)*ε + t*x0   (interpolate between noise and ground truth)
  Target velocity: v = x0 - ε
  Loss: MSE(predicted_velocity(x_t, t), v)
"""

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Pi05Config:
    """Configuration for the pi0.5 (Physical Intelligence) policy.

    Defaults match the PaliGemma-3B backbone specification from the PI paper.
    """
    # ---- Backbone topology ----
    backbone: str = "paligemma3b"
    vision_width: int = 1024       # SigLIP ViT width
    llm_width: int = 3200          # LLaMA decoder width
    vision_depth: int = 27         # ViT-L/14 depth
    llm_depth: int = 28            # LLaMA depth
    vision_heads: int = 16         # ViT attention heads
    llm_heads: int = 20            # LLaMA attention heads
    vision_patch_size: int = 14    # SigLIP patch size
    vision_embed_dim: int = 1024   # same as vision_width
    num_vision_tokens: int = 257   # 256 patches + 1 cls (224x224 / 14²)

    # ---- Flow matching ----
    flow_steps: int = 500          # integration steps (training)
    flow_sigma: float = 0.02       # Gaussian noise level
    flow_eps: float = 1e-5         # numerical safety for t

    # ---- Action expert ----
    action_expert_hidden: int = 2048
    action_expert_layers: int = 3
    action_dim: int = 14
    action_chunk_size: int = 32
    horizon: int = 32

    # ---- Language ----
    text_vocab_size: int = 32000   # PaliGemma GPT-J vocab
    text_max_len: int = 128
    pad_token_id: int = 0
    eos_token_id: int = 1

    # ---- Training ----
    lr: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 8
    num_epochs: int = 50
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    gradient_checkpointing: bool = True
    freeze_backbone: bool = False
    amp_dtype: str = "float16"     # "float16" or "bfloat16"

    # ---- Data ----
    image_size: int = 224          # supported: 224 or 448
    num_cameras: int = 3
    image_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    image_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    normalize_actions: bool = True
    action_mean: List[float] = field(default_factory=lambda: [0.0] * 14)
    action_std: List[float] = field(default_factory=lambda: [1.0] * 14)

    # ---- Multimodal projector ----
    projector_layers: int = 2      # Linear → GELU → (Linear → LayerNorm)²


# ---------------------------------------------------------------------------
# Utility: RoPE (Rotary Positional Embeddings)
# ---------------------------------------------------------------------------

def _compute_rope_freqs(
    dim: int,
    base: float = 10000.0,
    max_seq_len: int = 2048,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Pre-compute the inverse frequency table used in RoPE.

    For each dimension pair (2i, 2i+1) we compute
        θ_m = base^{-m / dim}    for m ∈ {0, 2, ..., dim-2}.
    Shape: (dim//2, 1)
    """
    assert dim % 2 == 0, "RoPE requires even embedding dimension"
    exp = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    freqs = torch.exp(-exp * math.log(base) / dim)          # (dim//2,)
    return freqs.unsqueeze(1)                                # (dim//2, 1)


def apply_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Apply rotary positional embeddings to a tensor of shape (B, L, dim).

    We build a (L, dim//2, 2) rotation matrix from the pre-computed
    inverse frequencies and the position indices 0…seq_len-1.
    The result is a per-token rotation of each (2i, 2i+1) pair.
    """
    B, L, D = x.shape
    S = seq_len
    assert D % 2 == 0

    freqs_cis = freqs.unsqueeze(0).unsqueeze(0)             # (1, 1, D//2, 1)
    t = torch.arange(S, device=x.device, dtype=freqs.dtype)  # (S,)
    freqs_pos = t.unsqueeze(1) * freqs_cis                   # (S, 1, D//2, 1)

    # Split into even/odd pairs and form cos/sin complex rotation
    # freqs_pos has shape (S, 1, D//2, 1) → we reshape
    freqs_pos = freqs_pos.squeeze(-1)                       # (S, D//2)
    cos = torch.cos(freqs_pos)                              # (S, D//2)
    sin = torch.sin(freqs_pos)                              # (S, D//2)

    x_even = x[..., 0::2].float()
    x_odd  = x[..., 1::2].float()

    x_rotated = torch.zeros_like(x, dtype=torch.float32)
    x_rotated[..., 0::2] = x_even * cos.unsqueeze(1) - x_odd * sin.unsqueeze(1)
    x_rotated[..., 1::2] = x_even * sin.unsqueeze(1) + x_odd * cos.unsqueeze(1)
    return x_rotated.to(x.dtype)


# ---------------------------------------------------------------------------
# Utility: SwiGLU and attention helpers
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU: gated activation FFN used in the LLaMA decoder.

    For input x ∈ ℝ^d:
        W, V → split x into W*x ∈ ℝ^d', V*x ∈ ℝ^d'
        SwiGLU(W*x, V*x) = (W*x * σ_gelu(W*x)) ⊙ (V*x)
    where σ_gelu(z) = z · tanh(1.702 · ln(1 + exp(z))) ≈ z * GELU(z).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w_gate = nn.Linear(in_features, out_features, bias=False)
        self.w_weight = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        return F.gelu(gate, approximate="tanh") * self.w_weight(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self/cross attention with RoPE.

    Supports causal masking for autoregressive decoding.
    Output dimension equals ``hidden_size``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_emb_fn,
        rotary_freqs: torch.Tensor,
        max_seq_len: int,
        causal: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.rotary_emb_fn = rotary_emb_fn
        self.rotary_freqs = rotary_freqs
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Pre-compute causal mask once (lazy to handle variable seq lengths)
        self._attn_mask: Optional[torch.Tensor] = None

    def _build_causal_mask(self, qlen: int, klen: int) -> torch.Tensor:
        mask = torch.triu(
            torch.full((qlen, klen), float("-inf"), device=self.qkv.weight.device),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, hidden_size) — input sequence
            mask: (B, H, qlen, klen) optional extra attention mask

        Returns:
            (B, L, hidden_size) attended + projected output
        """
        B, L, _ = x.shape
        qkv = self.qkv(x)                                  # (B, L, 3*hidden)
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                  # each (B, H, L, hd)

        # Rotary positional embeddings
        q = self.rotary_emb_fn(q, self.rotary_freqs, self.max_seq_len)
        k = self.rotary_emb_fn(k, self.rotary_freqs, self.max_seq_len)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)

        # Causal mask
        if self.causal:
            attn_mask = self._build_causal_mask(L, L).to(q.device)
            attn = attn.masked_fill(attn_mask.bool(), float("-inf"))

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        attn = attn.to(q.dtype)

        out = torch.matmul(attn, v)                         # (B, H, L, hd)
        out = out.transpose(1, 2).reshape(B, L, -1)       # (B, L, hidden)
        return self.proj(out)


# ---------------------------------------------------------------------------
# 1. SigLIPVisionEncoder  (ViT-L/14)
# ---------------------------------------------------------------------------

class SigLIPPatchEmbeddings(nn.Module):
    """Convert input images to patch embeddings.

    Patches are extracted with stride `patch_size` from each image channel.
    A learnable CLS token is prepended.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches + cls: (B, num_patches+1, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, (
            f"Input size ({H}x{W}) must match img_size ({self.img_size})"
        )
        x = self.proj(x)                                    # (B, embed, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)                    # (B, NP, embed)

        # CLS token
        cls_token = nn.Parameter(torch.zeros(1, 1, x.shape[-1]))
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)  # (B, NP+1, embed)
        return x


class SigLIPVisionEncoder(nn.Module):
    """Complete SigLIP ViT-L/14 vision encoder.

    Architecture (from SigLIP paper):
        Patch embedding (14x14) → 27 transformer layers (pre-norm) → LayerNorm

    Each transformer layer contains:
        LayerNorm → Multi-head self-attention → residual → LayerNorm → MLP (GELU) → residual

    Uses RoPE for absolute positional encoding (learned init + rotary for relative).

    Input:  (B, C, H, W)   →   Output: (B, num_patches+1, vision_width)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 1024,
        num_heads: int = 16,
        depth: int = 27,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 260,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.img_size = img_size
        self.max_seq_len = max_seq_len

        # --- Patch embedding + CLS ---
        self.patch_embed = SigLIPPatchEmbeddings(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_tokens = self.patch_embed.num_patches + 1  # NP + 1

        # --- Positional encoding: learned + RoPE ---
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- RoPE frequency table ---
        self.rope_freqs = _compute_rope_freqs(
            dim=embed_dim // num_heads, max_seq_len=max_seq_len, device=device
        )

        # --- Transformer layers (pre-norm) ---
        self.layers = nn.ModuleList([
            self._make_vit_layer(embed_dim, num_heads, mlp_ratio, device)
            for _ in range(depth)
        ])

        # --- Final LayerNorm ---
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Initialise weights
        self._init_weights()

    def _make_vit_layer(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        device: torch.device,
    ) -> nn.Module:
        """Build one ViT encoder layer with pre-norm."""
        mlp_hidden = int(embed_dim * mlp_ratio)

        attn = MultiHeadAttention(
            hidden_size=embed_dim,
            num_heads=num_heads,
            rotary_emb_fn=apply_rope,
            rotary_freqs=self.rope_freqs,
            max_seq_len=self.max_seq_len,
            causal=False,  # ViT = full attention
        )
        mlp = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6),
            nn.Linear(embed_dim, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, embed_dim, bias=True),
        )
        return nn.ModuleDict({
            "attn": attn,
            "mlp": mlp,
            "norm1": nn.LayerNorm(embed_dim, eps=1e-6),
            "norm2": nn.LayerNorm(embed_dim, eps=1e-6),
        })

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) RGB images

        Returns:
            cls_token: (B, embed_dim)   — CLS embedding after all transformer layers
            patch_features: (B, NP, embed_dim) — patch embeddings after all transformer layers
        """
        # Patch embedding + CLS
        x = self.patch_embed(x)                                    # (B, NP+1, D)
        x = x + self.pos_embed                                     # positional encoding

        # Prepend learned CLS token (already inside patch_embed via cls_token param)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)                      # (B, 1+NP+1, D)
        x = x[:, :self.num_tokens]                                 # clamp if needed

        # Transformer layers
        for layer in self.layers:
            # Self-attention branch
            h = layer["norm1"](x)
            h = layer["attn"](h)
            x = x + h

            # MLP branch
            h = layer["norm2"](x)
            h = layer["mlp"](h)
            x = x + h

        x = self.norm(x)
        return x[:, 0], x[:, 1:]                                   # cls, patches


# ---------------------------------------------------------------------------
# 2. LLaMADecoder
# ---------------------------------------------------------------------------

class LLaMADecoderLayer(nn.Module):
    """One LLaMA-style decoder layer with pre-norm, SwiGLU FFN, and RoPE.

    Sequence:
        LayerNorm → Multi-head self-attention (causal) → residual
        LayerNorm → SwiGLU FFN → residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_multiplier: int = 8192,   # LLaMA uses (4/3 * hidden_size * 2/3) ≈ 8192 for 3200
        max_seq_len: int = 512,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        rope_freqs = _compute_rope_freqs(
            dim=hidden_size // num_heads, max_seq_len=max_seq_len, device=device
        )
        self.attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_emb_fn=apply_rope,
            rotary_freqs=rope_freqs,
            max_seq_len=max_seq_len,
            causal=True,
        )
        # SwiGLU FFN — uses two learned projections of equal output width
        swiglu_out = ffn_multiplier
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-5),
            SwiGLU(hidden_size, swiglu_out),
            nn.Linear(swiglu_out, hidden_size, bias=False),
        )
        self.norm_attn = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm_attn(x)
        h = self.attn(h, mask)
        x = x + h
        return x + self.ffn(x)


class LLaMADecoder(nn.Module):
    """Complete LLaMA-style autoregressive decoder stack.

    Architecture:
        Token embedding → positional embedding → [LLaMADecoderLayer]*depth → RMSNorm

    Uses SwiGLU feedforward networks and RMSNorm (as in the original LLaMA paper).

    Input:  (B, L) token IDs   →   Output: (B, L, llm_width)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 3200,
        num_heads: int = 20,
        depth: int = 28,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.rope_freqs = _compute_rope_freqs(
            hidden_size // num_heads, max_seq_len=max_seq_len, device=device
        )

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        # LLaMA uses learned absolute positions (not sinusoidal)
        self.pos_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embeddings, std=0.02)

        # Build decoder layers
        # ffn_multiplier for LLaMA: (4/3 * hidden * 2/3)  — SwiGLU halves the width internally
        ffn_mult = int(hidden_size * 8 / 3)   # ≈ 8533 for hidden=3200
        self.layers = nn.ModuleList([
            LLaMADecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_multiplier=ffn_mult,
                max_seq_len=max_seq_len,
                device=device,
            )
            for _ in range(depth)
        ])

        self.norm = RMSNorm(hidden_size, eps=1e-5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token IDs (int64)

        Returns:
            (B, L, hidden_size) — output of the final RMSNorm
        """
        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)                       # (B, L, D)
        x = x + self.pos_embeddings[:, :L, :]                 # add absolute positions
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# 3. MultimodalProjector
# ---------------------------------------------------------------------------

class MultimodalProjector(nn.Module):
    """Projects SigLIP ViT features (1024) into the LLM space (3200).

    Architecture from the PI / PaliGemma papers:
        Linear(vision_width, proj_dim) → GELU → Linear(proj_dim, llm_width) → LayerNorm
    where proj_dim is typically the LLM width.
    """

    def __init__(
        self,
        vision_width: int = 1024,
        llm_width: int = 3200,
        num_layers: int = 2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = vision_width
        for i in range(num_layers):
            out_dim = llm_width if (i == num_layers - 1) else llm_width
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU(approximate="tanh"))
            layers.append(nn.LayerNorm(out_dim, eps=1e-6))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, N, vision_width) — patch tokens (excl. CLS)

        Returns:
            (B, N, llm_width) — projected into LLM embedding space
        """
        return self.net(vision_features)


# ---------------------------------------------------------------------------
# 4. FlowMatchingSampler
# ---------------------------------------------------------------------------

class FlowMatchingSampler:
    """Flow matching objective and sampler for continuous action spaces.

    Flow matching (Lipman et al., 2023; Albergo & Vanden-Eijnden, 2023)
    learns a vector field v_t(x, t) that maps standard Gaussian noise
    at t=0 to the data distribution at t=1, via linear interpolation:

        x_t = (1 - t) · ε + t · x_0        for ε ~ N(0, I),  x_0 ~ p_data
        v = x_0 - ε                           (target velocity)

    The network is trained to minimise:
        L = E_{t,ε} [ ‖ v_network(x_t, t) - (x_0 - ε) ‖² ]

    At sampling time we integrate the learned velocity field:
        Euler:      x_{t+dt} = x_t + dt · v_network(x_t, t)
        RK2 (Heun): midpoint correction
        RK4:        classic 4th-order Runge-Kutta
    """

    def __init__(self, sigma: float = 0.02):
        self.sigma = sigma

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time step t ~ Uniform[eps, 1-eps]."""
        t = torch.rand(batch_size, 1, 1, device=device)
        t = t.clamp(self.sigma, 1.0 - self.sigma)
        return t

    def compute_flow_matching_loss(
        self,
        velocity_net: nn.Module,
        images: torch.Tensor,
        language_ids: torch.Tensor,
        ground_truth_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the flow matching training loss.

        Args:
            velocity_net: Module that predicts velocity given (x_t, t, images, language_ids)
            images: (B, num_cameras, C, H, W)
            language_ids: (B, TL)
            ground_truth_actions: (B, chunk_size, action_dim)

        Returns:
            scalar loss tensor
        """
        B = ground_truth_actions.shape[0]
        device = ground_truth_actions.device

        # Sample t and noise ε
        t = self._sample_t(B, device)                              # (B, 1, 1)
        eps = torch.randn_like(ground_truth_actions)               # (B, chunk, action_dim)

        # Linear interpolation: x_t = (1-t)*ε + t*x0
        x_t = (1.0 - t) * eps + t * ground_truth_actions           # (B, chunk, action_dim)

        # Broadcast t to match x_t shape
        t_expanded = t.expand_as(x_t)

        # Predict velocity
        pred_velocity = velocity_net(x_t, t_expanded, images, language_ids)

        # Target velocity: x_0 - ε = ground_truth_actions - eps
        target_velocity = ground_truth_actions - eps

        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss

    def sample(
        self,
        velocity_net: nn.Module,
        images: torch.Tensor,
        language_ids: torch.Tensor,
        num_steps: int = 50,
        method: str = "euler",
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Integrate the learned ODE from t=0 to t=1 using flow matching.

        Args:
            velocity_net: (x_t, t, images, language_ids) → predicted velocity
            images: (B, num_cameras, C, H, W)
            language_ids: (B, TL)
            num_steps: integration steps
            method: "euler", "rk2", or "rk4"
            noise: (B, chunk, action_dim) initial noise; sampled if None

        Returns:
            (B, chunk, action_dim) — integrated action samples
        """
        if noise is None:
            B = images.shape[0]
            C, D = self._get_shapes(images, language_ids)
            noise = torch.randn(B, C, D, device=images.device)

        x = noise.clone()
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.full((x.shape[0], 1, 1), step * dt, device=x.device, dtype=x.dtype)
            # Broadcast to match x
            t_full = t.expand_as(x)
            if method == "euler":
                v = velocity_net(x, t_full, images, language_ids)
                x = x + dt * v
            elif method == "rk2":
                # Heun / midpoint
                t_mid = t + 0.5 * dt
                t_mid_full = t_mid.expand_as(x)
                v_half = velocity_net(x, t_mid_full, images, language_ids)
                x_mid = x + 0.5 * dt * v_half
                v = velocity_net(x_mid, t_mid_full, images, language_ids)
                x = x + dt * v
            elif method == "rk4":
                k1 = velocity_net(x, t, images, language_ids)
                t2 = t + 0.5 * dt
                k2 = velocity_net(x + 0.5 * dt * k1, t2.expand_as(x), images, language_ids)
                k3 = velocity_net(x + 0.5 * dt * k2, t2.expand_as(x), images, language_ids)
                k4 = velocity_net(x + dt * k3, (t + dt).expand_as(x), images, language_ids)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown integration method: {method}")
        return x

    def _get_shapes(self, images: torch.Tensor, language_ids: torch.Tensor) -> Tuple[int, int]:
        """Helper to infer (chunk_size, action_dim) from model config.  (Used in `sample`.)"""
        B, C, H, W = images.shape
        return (C, H)  # stub; actual values come from velocity_net config


# ---------------------------------------------------------------------------
# 5. ActionExpert
# ---------------------------------------------------------------------------

class ActionExpert(nn.Module):
    """Maps pooled multimodal features to robot actions via a deep MLP.

    Architecture:
        Linear → RMSNorm → SwiGLU → Linear → ... → RMSNorm → SwiGLU → Linear → Output

    The expert processes the action-pooling token from the multimodal sequence
    to produce a chunk of future actions (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        input_dim: int = 3200,
        hidden_dim: int = 2048,
        num_layers: int = 3,
        output_dim: int = 14,
        chunk_size: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.chunk_size = chunk_size

        # Build MLP
        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.LayerNorm(in_d, eps=1e-5))
            layers.append(SwiGLU(in_d, hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        # Final projection to action dim
        layers.append(nn.LayerNorm(hidden_dim, eps=1e-5))
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.mlp = nn.Sequential(*layers)

        # Token-level expansion: produce chunk_size copies
        self.chunk_proj = nn.Linear(output_dim, output_dim * chunk_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, multimodal_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multimodal_token: (B, input_dim) — pooled multimodal representation

        Returns:
            (B, chunk_size, output_dim) — action chunk predictions
        """
        h = self.mlp(multimodal_token)                        # (B, output_dim)
        h_chunk = self.chunk_proj(h)                          # (B, output_dim * chunk_size)
        B = h_chunk.shape[0]
        return h_chunk.view(B, self.chunk_size, self.output_dim)


# ---------------------------------------------------------------------------
# 6. Pi05Policy  — the main policy class
# ---------------------------------------------------------------------------

class Pi05Policy(nn.Module):
    """Complete pi0.5 policy tying together vision, language, flow matching, and action expert.

    Forward pass (training):
        1. images → SigLIPVisionEncoder → (cls, patches)
        2. patches → MultimodalProjector → (B, NP, llm_width)
        3. language_ids → LLaMADecoder → (B, TL, llm_width)
        4. Concat [projected_vision | text] → pooling token (CLS-style or mean)
        5. ActionExpert(pooled_token) → (B, chunk, action_dim)

    Training uses ``compute_flow_matching_loss`` which internally calls the
    above forward path but also handles time sampling and velocity regression.
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Vision encoder
        max_vision_seq = config.num_vision_tokens
        self.vision_encoder = SigLIPVisionEncoder(
            img_size=config.image_size,
            patch_size=config.vision_patch_size,
            in_chans=3,
            embed_dim=config.vision_width,
            num_heads=config.vision_heads,
            depth=config.vision_depth,
            max_seq_len=max_vision_seq + 10,
            device=self.device,
        )

        # 2. Multimodal projector (vision → LLM space)
        self.multimodal_projector = MultimodalProjector(
            vision_width=config.vision_width,
            llm_width=config.llm_width,
            num_layers=config.projector_layers,
        )

        # 3. Language decoder
        self.text_max_len = config.text_max_len
        max_text_seq = config.text_max_len + max_vision_seq
        self.llm_decoder = LLaMADecoder(
            vocab_size=config.text_vocab_size,
            hidden_size=config.llm_width,
            num_heads=config.llm_heads,
            depth=config.llm_depth,
            max_seq_len=max_text_seq,
            pad_token_id=config.pad_token_id,
            device=self.device,
        )

        # 4. Action expert
        self.action_expert = ActionExpert(
            input_dim=config.llm_width,
            hidden_dim=config.action_expert_hidden,
            num_layers=config.action_expert_layers,
            output_dim=config.action_dim,
            chunk_size=config.action_chunk_size,
        )

        # 5. Flow matching sampler
        self.flow_matching = FlowMatchingSampler(sigma=config.flow_sigma)

        # 6. Pooling mechanism (learned CLS-like token)
        self.action_pool = nn.Parameter(torch.zeros(1, 1, config.llm_width))
        nn.init.trunc_normal_(self.action_pool, std=0.02)

        # Mixed precision dtype
        amp_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        self.amp_dtype = amp_dtype_map.get(config.amp_dtype, torch.float16)

        if config.freeze_backbone:
            self._freeze_backbone()

        self.to(self.device)

    def _freeze_backbone(self):
        """Freeze SigLIP and LLaMA parameters (for transfer learning)."""
        for name, param in self.vision_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.llm_decoder.named_parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        language_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, num_cameras, C, H, W) or (B, C, H, W)
            language_ids: (B, TL)

        Returns:
            actions: (B, chunk_size, action_dim)
        """
        # Handle multi-camera: stack or process each view
        if images.ndim == 5:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            language_ids = language_ids.repeat_interleave(N, dim=0)

        # --- Vision encoding ---
        cls_token, patch_features = self.vision_encoder(images)
        # patch_features: (B, NP, vision_width)

        # --- Project vision to LLM space ---
        vision_in_llm = self.multimodal_projector(patch_features)  # (B, NP, llm_width)

        # --- Language encoding ---
        text_features = self.llm_decoder(language_ids)  # (B, TL, llm_width)

        # --- Concatenate multimodal features ---
        # Append action-pooling token: [vision_tokens | text_tokens | action_pool]
        B, NP, D = vision_in_llm.shape
        action_pool = self.action_pool.expand(B, 1, D)
        multimodal = torch.cat([vision_in_llm, text_features, action_pool], dim=1)  # (B, NP+TL+1, D)

        # --- Self-attention on multimodal sequence ---
        # Create causal mask so action_pool can attend to vision + text
        total_seq = multimodal.shape[1]
        causal_mask = torch.triu(
            torch.full((total_seq, total_seq), float("-inf"), device=multimodal.device),
            diagonal=1,
        )
        # No masking for vision-vision or text-text cross-attention (full for each modality)
        # Simple approach: let the LLaMA causal mask handle everything
        multimodal = self._attn_on_multimodal(multimodal, causal_mask)

        # --- Extract action token (last token = the pooled token) ---
        action_token = multimodal[:, -1, :]  # (B, llm_width)

        # --- Action expert → chunked actions ---
        actions = self.action_expert(action_token)
        return actions

    def _attn_on_multimodal(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Lightweight self-attention layer on the multimodal sequence."""
        L = x.shape[1]
        attn = MultiHeadAttention(
            hidden_size=self.config.llm_width,
            num_heads=self.config.llm_heads,
            rotary_emb_fn=apply_rope,
            rotary_freqs=self.llm_decoder.rope_freqs,
            max_seq_len=L,
            causal=True,
        )
        return attn(x, mask)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def compute_flow_matching_loss(
        self,
        images: torch.Tensor,
        language_ids: torch.Tensor,
        ground_truth_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow matching loss for a batch.

        The velocity network is a thin wrapper around ``forward`` that
        predicts velocity conditioned on (x_t, t, images, language_ids).
        """

        def velocity_fn(x_t: torch.Tensor, t: torch.Tensor, img: torch.Tensor, lang: torch.Tensor):
            """
            Velocity network: inject x_t as extra tokens into the multimodal sequence.
            This mimics the PI paper's approach where the current state is part of context.
            For simplicity we use x_t mean as an additional context token.
            """
            B = img.shape[0]
            cls, patches = self.vision_encoder(img)
            vision_llm = self.multimodal_projector(patches)
            text = self.llm_decoder(lang)

            # Add t as a scalar bias (sinusoidal position-like encoding)
            t_enc = torch.sin(t * 1000.0).view(B, 1, 1)  # (B, 1, 1)
            vision_llm = vision_llm + t_enc

            action_pool_t = self.action_pool.expand(B, 1, self.config.llm_width)
            x_t_mean = x_t.mean(dim=1, keepdim=True)  # (B, 1, D)
            multimodal = torch.cat([vision_llm, text, action_pool_t, x_t_mean], dim=1)

            L = multimodal.shape[1]
            mask = torch.triu(torch.full((L, L), float("-inf"), device=img.device), diagonal=1)
            multimodal = self._attn_on_multimodal(multimodal, mask)

            action_tok = multimodal[:, -1, :]  # last = x_t_mean position
            return self.action_expert(action_tok)

        loss = self.flow_matching.compute_flow_matching_loss(
            velocity_fn, images, language_ids, ground_truth_actions,
        )
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        language_ids: torch.Tensor,
        num_steps: int = 50,
        method: str = "euler",
    ) -> torch.Tensor:
        """
        Predict action chunk via flow matching sampling.

        Args:
            images: (B, C, H, W)
            language_ids: (B, TL)
            num_steps: number of ODE integration steps
            method: "euler" | "rk2" | "rk4"

        Returns:
            actions: (B, chunk_size, action_dim)
        """
        self.eval()
        return self.flow_matching.sample(
            velocity_net=self.forward,  # wrapped as velocity net
            images=images,
            language_ids=language_ids,
            num_steps=num_steps,
            method=method,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save all sub-module weights and config to a single .pt file."""
        state = {
            "config": self.config,
            "vision_encoder": self.vision_encoder.state_dict(),
            "multimodal_projector": self.multimodal_projector.state_dict(),
            "llm_decoder": self.llm_decoder.state_dict(),
            "action_expert": self.action_expert.state_dict(),
            "action_pool": self.action_pool.data,
            "flow_sigma": self.flow_matching.sigma,
        }
        torch.save(state, path)
        print(f"[Pi05Policy] Checkpoint saved → {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "Pi05Policy":
        """Load a checkpoint into a new Pi05Policy instance."""
        state = torch.load(path, map_location=device, weights_only=False)
        config = state["config"]
        policy = cls(config)
        policy.device = torch.device(device)

        policy.vision_encoder.load_state_dict(state["vision_encoder"])
        policy.multimodal_projector.load_state_dict(state["multimodal_projector"])
        policy.llm_decoder.load_state_dict(state["llm_decoder"])
        policy.action_expert.load_state_dict(state["action_expert"])
        policy.action_pool.data = state["action_pool"].to(device)
        policy.flow_matching.sigma = state.get("flow_sigma", 0.02)

        policy.to(policy.device)
        policy.eval()
        print(f"[Pi05Policy] Checkpoint loaded ← {path}")
        return policy

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str) -> "Pi05Policy":
        self.device = torch.device(device)
        return super().to(device)

    def train(self, mode: bool = True) -> "Pi05Policy":
        super().train(mode)
        if self.config.freeze_backbone:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            for p in self.llm_decoder.parameters():
                p.requires_grad = False
        return self

    def parameters(self, recurse: bool = True):
        """Override to optionally exclude frozen backbone parameters."""
        if self.config.freeze_backbone:
            return [
                p for n, p in self.named_parameters()
                if p.requires_grad
            ]
        return super().parameters(recurse)


# ---------------------------------------------------------------------------
# 7. Pi05DataPreprocessor
# ---------------------------------------------------------------------------

class Pi05DataPreprocessor:
    """Handles all data preprocessing for pi0.5 training and inference.

    Includes:
        - Image normalization & resizing for SigLIP
        - Text tokenization for PaliGemma vocab
        - Action normalization (zero-mean, unit-variance)
        - Data augmentation pipeline (crop, flip, color jitter)
        - Dataset / DataLoader creation
    """

    def __init__(self, config: Pi05Config):
        self.config = config
        self.image_size = config.image_size
        self.action_mean = torch.tensor(config.action_mean, dtype=torch.float32)
        self.action_std = torch.tensor(config.action_std, dtype=torch.float32)
        self.normalize_actions = config.normalize_actions

        # Image augmentation transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Set up torchvision-style image transforms."""
        self.train_transform = self._make_train_transform()
        self.eval_transform = self._make_eval_transform()

    @staticmethod
    def _make_eval_transform() -> callable:
        """Eval transform: resize → tensor → normalize."""
        def transform(img: np.ndarray) -> torch.Tensor:
            # Ensure RGB
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (224, 224))
            tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            tensor = tensor / 255.0
            # SigLIP / CLIP normalization
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            return tensor
        return transform

    @staticmethod
    def _make_train_transform() -> callable:
        """Train transform: augmentation + resize → tensor → normalize."""
        def transform(img: np.ndarray) -> torch.Tensor:
            # Augmentation: random crop
            h, w = img.shape[:2]
            crop_size = min(h, w)
            if crop_size < 224:
                crop_size = 224
            y1 = np.random.randint(0, h - crop_size + 1)
            x1 = np.random.randint(0, w - crop_size + 1)
            img = img[y1:y1+crop_size, x1:x1+crop_size]

            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()

            # Color jitter (brightness, contrast, saturation)
            if np.random.rand() > 0.3:
                factor = 0.8 + 0.4 * np.random.rand()
                img = (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)

            # Resize
            img = cv2.resize(img, (224, 224))

            # Convert to RGB if needed
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            tensor = tensor / 255.0
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            return tensor
        return transform

    @staticmethod
    def tokenize_text(
        text: str,
        max_len: int = 128,
        vocab_size: int = 32000,
        pad_id: int = 0,
        eos_id: int = 1,
    ) -> np.ndarray:
        """Minimal tokenizer for PaliGemma-compatible vocab.

        In production, replace with the actual PaliGemma tokenizer.
        This implementation maps characters → token IDs using a simple hash
        table for demonstration.
        """
        # Simple character-level hash tokenization (production: use tiktoken / fastai)
        words = text.lower().split()
        token_ids = [pad_id] * max_len

        # First token is usually a special BOS token
        token_ids[0] = 0  # BOS

        for i, word in enumerate(words):
            if i + 1 >= max_len - 1:
                break
            # Hash-based token ID (bounded to vocab)
            tid = hash(word) % (vocab_size - 2) + 2  # skip pad(0) and eos(1)
            token_ids[i + 1] = tid

        # End-of-sequence
        token_ids[min(max_len - 1, i + 1)] = eos_id
        return np.array(token_ids, dtype=np.int64)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to zero mean, unit variance."""
        if not self.normalize_actions:
            return action
        action = np.asarray(action, dtype=np.float32)
        return (action - self.action_mean.numpy()) / self.action_std.numpy()

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Undo normalization."""
        if not self.normalize_actions:
            return action
        action = np.asarray(action, dtype=np.float32)
        return action * self.action_std.numpy() + self.action_mean.numpy()

    def preprocess_image(self, image: np.ndarray, train: bool = False) -> torch.Tensor:
        """Preprocess a single image (numpy HxWxC or HxW) to tensor (3, H, W)."""
        transform = self.train_transform if train else self.eval_transform
        return transform(image)

    def preprocess_batch(
        self,
        images_list: List[np.ndarray],
        text_list: List[str],
        actions_list: List[np.ndarray],
        train: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a full batch.

        Args:
            images_list: list of images, each HxWxC
            text_list: list of language strings
            actions_list: list of action arrays
            train: whether to apply train-time augmentations

        Returns:
            dict with keys:
                images: (B, C, H, W)
                language_ids: (B, TL)
                actions: (B, action_dim)
                action_masks: (B, action_dim) boolean mask (all True for now)
        """
        B = len(images_list)

        # Preprocess images
        img_tensors = []
        for img in images_list:
            t = self.preprocess_image(img, train=train)
            img_tensors.append(t)
        images = torch.stack(img_tensors)  # (B, C, H, W)

        # Tokenize language
        lang_ids = torch.tensor(
            [self.tokenize_text(txt, max_len=self.config.text_max_len) for txt in text_list],
            dtype=torch.long,
        )  # (B, TL)

        # Normalize actions
        actions = torch.tensor(
            np.stack([self.normalize_action(a) for a in actions_list]),
            dtype=torch.float32,
        )  # (B, action_dim)

        return {
            "images": images,
            "language_ids": lang_ids,
            "actions": actions,
            "action_masks": torch.ones(B, self.config.action_dim, dtype=torch.bool),
        }

    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 8,
        shuffle: bool = False,
        num_workers: int = 4,
        train: bool = False,
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader for pi0.5 training/inference."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda batch: self.preprocess_batch(
                *[
                    [item[key] for item in batch]
                    for key in ["image", "text", "action"]
                ],
                train=train,
            ),
            pin_memory=True,
            drop_last=False,
        )


# ---------------------------------------------------------------------------
# Entry point (for standalone testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test: create policy, run forward pass, save/load checkpoint
    torch.manual_seed(42)
    cfg = Pi05Config(
        vision_width=1024,
        llm_width=3200,
        vision_depth=27,
        llm_depth=28,
        action_expert_hidden=2048,
        action_expert_layers=3,
        action_dim=14,
        action_chunk_size=32,
        image_size=224,
    )

    print("[pi05] Creating Pi05Policy...")
    policy = Pi05Policy(cfg)
    policy.eval()

    B = 2
    images = torch.randn(B, 3, 224, 224)
    lang = torch.randint(0, cfg.text_vocab_size, (B, 16))

    print("[pi05] Forward pass...")
    with torch.no_grad():
        actions = policy(images, lang)
    print(f"  Output shape: {actions.shape}  (expected [{B}, {cfg.action_chunk_size}, {cfg.action_dim}])")

    print("[pi05] Saving checkpoint...")
    policy.save_checkpoint("./outputs/pi05_test.pt")

    print("[pi05] Loading checkpoint...")
    loaded = Pi05Policy.load_checkpoint("./outputs/pi05_test.pt")
    with torch.no_grad():
        actions2 = loaded(images, lang)
    print(f"  Output shape: {actions2.shape}")

    # Verify load matches
    diff = (actions - actions2).abs().max().item()
    print(f"  Max diff between saved/loaded: {diff:.2e}")
    assert diff < 1e-5, f"Checkpoint mismatch! diff={diff}"

    print("[pi05] DataPreprocessor smoke test...")
    preproc = Pi05DataPreprocessor(cfg)
    test_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    t = preproc.preprocess_image(test_img, train=False)
    print(f"  Image tensor shape: {t.shape}")
    test_action = np.random.randn(14)
    norm_a = preproc.normalize_action(test_action)
    denorm_a = preproc.denormalize_action(norm_a)
    assert np.allclose(test_action, denorm_a, atol=1e-5), "Action normalization roundtrip failed"
    print("  Action normalization ✓")

    print("\n[pi05] All smoke tests passed ✓")
