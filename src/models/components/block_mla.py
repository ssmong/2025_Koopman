from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm  
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


class MLABlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_kv_heads: int | None = None,  
        q_lora_rank: int | None = None,   
        kv_lora_rank: int = 16,          
        qk_norm: bool = True,             
        window_size: int | None = None,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        # --- Query Projections ---
        if self.q_lora_rank is not None:
            # Query Compression (Down -> Norm -> Up)
            self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank) if self.qk_norm else nn.Identity()
            self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        else:
            # Standard Linear Query
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            if self.qk_norm:
                self.q_norm = RMSNorm(self.head_dim)

        # --- Key-Value Projections (MLA) ---
        # 1. Down Projection to Latent Space (c_KV)
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        # 2. Norm on Latent Vector
        self.kv_norm = RMSNorm(self.kv_lora_rank) if self.qk_norm else nn.Identity()
        # 3. Up Projection to Full Heads (K and V)
        # c_KV를 받아 K와 V 헤드로 각각 복원
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank, 
            self.num_kv_heads * self.head_dim * 2, # K head dim + V head dim
            bias=False
        )

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len]"

        batch_size, q_len, _ = hidden_states.size()

        # 1. Query Processing
        if self.q_lora_rank is not None:
            # Compressed Query: x -> Down -> Norm -> Up
            q = self.q_down_proj(hidden_states)
            q = self.q_norm(q)
            q = self.q_up_proj(q)
        else:
            q = self.q_proj(hidden_states)
            if self.qk_norm:
                q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
                q = self.q_norm(q)
                q = rearrange(q, '... h d -> ... (h d)')
        
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)

        # 2. Key-Value Processing (MLA)
        # Generate Latent Vector c_KV
        c_kv = self.kv_down_proj(hidden_states)
        c_kv = self.kv_norm(c_kv)

        # 3. KV Cache Update (Caching Compressed Latent Vector c_KV)
        if past_key_values is not None:            
            # Update cache with latent vector
            c_kv_cached_tuple = past_key_values.update(
                attn_state=(c_kv, c_kv),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            
            # Retrieve full history of latent vectors
            c_kv = c_kv_cached_tuple[0] # Cached c_kv (history + current)

        # 4. Up-projection to generate K, V heads
        kv = self.kv_up_proj(c_kv)
        k, v = torch.chunk(kv, 2, dim=-1)
        
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # 5. Flash Attention
        # Contains at least one padding token in the sequence
        cu_seqlens = kwargs.get('cu_seqlens')
        
        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            # varlen without padding mask tensor (packed sequence)
            seqlen_offset = 0
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            o = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
            
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None
        else:
            attentions = None 

        return o, attentions, past_key_values
