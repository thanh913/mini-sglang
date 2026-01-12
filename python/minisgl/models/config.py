from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from transformers import LlamaConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, float] | None


@dataclass(frozen=True)
class MoEConfig:
    """Configuration for Mixture of Experts layers."""

    num_experts: int
    num_experts_per_tok: int
    hidden_size: int
    moe_intermediate_size: int
    num_shared_experts: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    scoring_func: str = "softmax"
    routed_scaling_factor: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    moe_config: MoEConfig | None = None

    @classmethod
    def from_hf(cls, config: LlamaConfig) -> ModelConfig:
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        # Check for MoE configuration
        moe_config = None
        num_experts = getattr(config, "num_experts", None)
        if num_experts is not None and num_experts > 1:
            num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
            moe_intermediate_size = getattr(
                config, "moe_intermediate_size", config.intermediate_size
            )
            num_shared_experts = getattr(config, "num_shared_experts", 0)
            shared_expert_intermediate_size = getattr(
                config, "shared_expert_intermediate_size", config.intermediate_size
            )
            norm_topk_prob = getattr(config, "norm_topk_prob", True)
            scoring_func = getattr(config, "scoring_func", "softmax")
            routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

            moe_config = MoEConfig(
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=config.hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                num_shared_experts=num_shared_experts,
                shared_expert_intermediate_size=shared_expert_intermediate_size,
                norm_topk_prob=norm_topk_prob,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
            )

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            moe_config=moe_config,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
        )
