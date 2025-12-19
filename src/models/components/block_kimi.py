import torch
import torch.nn as nn
from typing import Dict, Any

from fla.models.kda.configuration_kda import KDAConfig
from fla.models.kda.modeling_kda import KDAPreTrainedModel, KDABlock

from src.models.components.block_mla import MLABlock

class KimiBlock(KDAPreTrainedModel):
    config_class = KDAConfig

    def __init__(self, model_cfg: Dict[str, Any]):
        ctxt_cfg = model_cfg.context
        hidden_size = ctxt_cfg.hidden_size
        num_layers = ctxt_cfg.num_layers
        num_heads = ctxt_cfg.num_heads

        kda_cfg = self._build_kda_cfg(ctxt_cfg)
        super().__init__(kda_cfg)

        input_dim = model_cfg["state_dim"] + model_cfg["control_dim"]
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, kda_cfg.hidden_size),
            nn.SiLU()
        )

        mla_cfg = ctxt_cfg.mla
        kv_lora_rank = mla_cfg.kv_lora_rank
        window_size = mla_cfg.window_size
        qk_norm = mla_cfg.qk_norm
        q_lora_rank = mla_cfg.q_lora_rank
        

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

    def _build_kda_cfg(self, ctxt_cfg: Dict[str, Any]) -> KDAConfig:
        required_keys = ["num_layers", "num_heads", "hidden_size"]
        if not all(key in ctxt_cfg for key in required_keys):
            missing_keys = [key for key in required_keys if key not in ctxt_cfg]
            raise ValueError(f"Missing required keys: {missing_keys}")

        if ctxt_cfg["hidden_size"] % ctxt_cfg["num_heads"] != 0:
            raise ValueError(f"model_dim must be divisible by num_heads: {ctxt_cfg['model_dim']} % {ctxt_cfg['num_heads']} != 0")
        head_dim = ctxt_cfg["hidden_size"] // ctxt_cfg["num_heads"]
        
        init_kwargs = dict(
            hidden_size=ctxt_cfg["hidden_size"],
            num_hidden_layers=ctxt_cfg["num_layers"],
            num_heads=ctxt_cfg["num_heads"],
            head_dim=head_dim,
        )

        _kda_cfg = ctxt_cfg.kda
        optional_keys = ["norm_eps", "allow_neg_eigval", "expand_v", "use_short_conv", "conv_size"]
        for key in optional_keys:
            if key in _kda_cfg:
                init_kwargs[key] = _kda_cfg[key]

        return KDAConfig(**init_kwargs)