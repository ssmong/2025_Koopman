import torch
import torch.nn as nn
from typing import Dict, Any

from fla.models.kda.configuration_kda import KDAConfig
from fla.models.kda.modeling_kda import KDAPreTrainedModel, KDABlock

from src.models.components.block_mla import MLABlock

class KimiBlock(KDAPreTrainedModel):
    config_class = KDAConfig

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        state_dim: int,
        control_dim: int,
        kda_params: Dict[str, Any],
        mla_params: Dict[str, Any],
    ):
        kda_cfg = self._build_kda_cfg(hidden_size, num_layers, num_heads, kda_params)
        super().__init__(kda_cfg)

        input_dim = state_dim + control_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, kda_cfg.hidden_size),
            nn.SiLU()
        )

        kv_lora_rank = mla_params['kv_lora_rank']
        window_size = mla_params['window_size']
        qk_norm = mla_params['qk_norm']
        q_lora_rank = mla_params['q_lora_rank']
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if (i + 1) % 4 == 0:
                self.layers.append(MLABlock(hidden_size=hidden_size, 
                                            num_heads=num_heads, 
                                            num_kv_heads=num_heads, 
                                            kv_lora_rank=kv_lora_rank, 
                                            qk_norm=qk_norm, 
                                            q_lora_rank=q_lora_rank, 
                                            window_size=window_size,
                                            layer_idx=i))
            else:
                self.layers.append(KDABlock(config=kda_cfg, layer_idx=i))
        

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs
    ):
        hidden_states = self.input_proj(hidden_states)
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                
            if use_cache:
                past_key_values = layer_outputs[2]
        
        return hidden_states[:, -1, :], past_key_values, all_attentions

    def _build_kda_cfg(
        self, 
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        kda_params: Dict[str, Any]
    ) -> KDAConfig:
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"model_dim must be divisible by num_heads: {hidden_size} % {num_heads} != 0")
        head_dim = hidden_size // num_heads
        
        init_kwargs = dict(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        optional_keys = ["norm_eps", "allow_neg_eigval", "expand_v", "use_short_conv", "conv_size"]
        for key in optional_keys:
            if key in kda_params:
                init_kwargs[key] = kda_params[key]

        return KDAConfig(**init_kwargs)