from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_even

from .base import BaseOP

if TYPE_CHECKING:
    from minisgl.models.config import MoEConfig


class MoEGate(BaseOP):
    """
    Router/gate network for Mixture of Experts.

    Computes routing weights for each token to decide which experts to use.
    Supports top-k routing with optional normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        *,
        norm_topk_prob: bool = True,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor

        # Gate weight: [num_experts, hidden_size]
        self.weight = torch.empty(num_experts, hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights and expert indices for each token.

        Args:
            x: Input tensor of shape [num_tokens, hidden_size]

        Returns:
            topk_weights: Routing weights of shape [num_tokens, num_experts_per_tok]
            topk_indices: Expert indices of shape [num_tokens, num_experts_per_tok]
        """
        # Compute router logits: [num_tokens, num_experts]
        router_logits = F.linear(x, self.weight, bias=None)

        # Apply scoring function
        if self.scoring_func == "softmax":
            scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits.float())
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

        # Select top-k experts for each token
        topk_weights, topk_indices = torch.topk(
            scores, k=self.num_experts_per_tok, dim=-1, sorted=False
        )

        # Normalize weights if requested (so they sum to 1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Apply scaling factor
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights.to(x.dtype), topk_indices


class MoEExpertLinear(BaseOP):
    """
    Grouped linear layer for MoE experts.

    Implements efficient batched matrix multiplication for multiple experts.
    Supports expert parallelism (EP) across devices.
    """

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        has_bias: bool = False,
        tp_split_dim: int | None = None,
    ):
        tp_info = get_tp_info()
        self.num_experts = num_experts
        self.tp_size = tp_info.size
        self.tp_rank = tp_info.rank

        self.full_input_size = input_size
        self.full_output_size = output_size

        # Apply tensor parallelism
        if tp_split_dim == 0:
            # Column parallel: split output dimension
            self.local_input_size = input_size
            self.local_output_size = divide_even(output_size, tp_info.size)
        elif tp_split_dim == 1:
            # Row parallel: split input dimension
            self.local_input_size = divide_even(input_size, tp_info.size)
            self.local_output_size = output_size
        else:
            self.local_input_size = input_size
            self.local_output_size = output_size

        # Weight: [num_experts, output_size, input_size]
        self.weight = torch.empty(
            num_experts, self.local_output_size, self.local_input_size
        )
        self.bias = (
            torch.empty(num_experts, self.local_output_size) if has_bias else None
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply expert-specific linear transformations.

        Args:
            x: Sorted input tensor of shape [num_tokens * num_experts_per_tok, input_size]
            expert_indices: Which expert each token-slot is assigned to
            num_tokens_per_expert: Number of tokens assigned to each expert

        Returns:
            Output tensor of shape [num_tokens * num_experts_per_tok, output_size]
        """
        # Use grouped GEMM for efficient batched computation
        return _grouped_gemm(
            x, self.weight, self.bias, expert_indices, num_tokens_per_expert
        )


def _grouped_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    expert_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """
    Efficient grouped GEMM implementation for MoE.

    Uses sequential expert processing when grouped GEMM kernels aren't available.
    For production, this should be replaced with optimized kernels (e.g., from flashinfer).
    """
    num_experts = weight.shape[0]
    output_size = weight.shape[1]
    total_tokens = x.shape[0]

    # Allocate output
    output = torch.empty(total_tokens, output_size, dtype=x.dtype, device=x.device)

    # Process each expert
    token_offset = 0
    for expert_idx in range(num_experts):
        num_tokens = num_tokens_per_expert[expert_idx].item()
        if num_tokens == 0:
            continue

        # Get tokens assigned to this expert
        expert_input = x[token_offset : token_offset + num_tokens]
        expert_weight = weight[expert_idx]

        # Apply linear transformation
        expert_output = F.linear(expert_input, expert_weight)
        if bias is not None:
            expert_output = expert_output + bias[expert_idx]

        output[token_offset : token_offset + num_tokens] = expert_output
        token_offset += num_tokens

    return output


class MoEGateUpProj(BaseOP):
    """
    Merged gate and up projection for MoE experts (SwiGLU-style).

    Each expert has its own gate_proj and up_proj merged together.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        has_bias: bool = False,
    ):
        tp_info = get_tp_info()
        self.num_experts = num_experts
        self.tp_size = tp_info.size

        # Apply column parallelism
        local_intermediate_size = divide_even(intermediate_size, tp_info.size)

        # Weight: [num_experts, 2 * intermediate_size, hidden_size] (gate and up merged)
        self.weight = torch.empty(
            num_experts, 2 * local_intermediate_size, hidden_size
        )
        self.bias = (
            torch.empty(num_experts, 2 * local_intermediate_size) if has_bias else None
        )

        self.full_hidden_size = hidden_size
        self.full_intermediate_size = intermediate_size
        self.local_intermediate_size = local_intermediate_size

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Apply merged gate+up projection per expert."""
        return _grouped_gemm(
            x, self.weight, self.bias, expert_indices, num_tokens_per_expert
        )


class MoEDownProj(BaseOP):
    """
    Down projection for MoE experts with all-reduce for tensor parallelism.
    """

    def __init__(
        self,
        num_experts: int,
        intermediate_size: int,
        hidden_size: int,
        *,
        has_bias: bool = False,
    ):
        tp_info = get_tp_info()
        self.num_experts = num_experts
        self.tp_size = tp_info.size
        self._comm = DistributedCommunicator()

        # Apply row parallelism
        local_intermediate_size = divide_even(intermediate_size, tp_info.size)

        # Weight: [num_experts, hidden_size, intermediate_size]
        self.weight = torch.empty(num_experts, hidden_size, local_intermediate_size)
        self.bias = torch.empty(num_experts, hidden_size) if has_bias else None

        self.full_hidden_size = hidden_size
        self.full_intermediate_size = intermediate_size
        self.local_intermediate_size = local_intermediate_size

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Apply down projection per expert with all-reduce."""
        y = _grouped_gemm(
            x, self.weight, self.bias, expert_indices, num_tokens_per_expert
        )
        if self.tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class SparseMoELayer(BaseOP):
    """
    Complete Sparse Mixture of Experts layer.

    Replaces the dense MLP with a sparse MoE that routes each token
    to a subset of expert networks.
    """

    def __init__(self, config: MoEConfig, hidden_act: str = "silu"):
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        # Router/gate
        self.gate = MoEGate(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
            scoring_func=config.scoring_func,
            routed_scaling_factor=config.routed_scaling_factor,
        )

        # Expert networks (merged gate+up and down projections)
        self.experts_gate_up = MoEGateUpProj(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            has_bias=False,
        )
        self.experts_down = MoEDownProj(
            num_experts=config.num_experts,
            intermediate_size=config.moe_intermediate_size,
            hidden_size=config.hidden_size,
            has_bias=False,
        )

        # Shared expert (optional, for Qwen3-MoE style)
        if config.num_shared_experts > 0:
            from .linear import LinearColParallelMerged, LinearRowParallel

            self.shared_expert_gate_up = LinearColParallelMerged(
                config.hidden_size,
                [
                    config.shared_expert_intermediate_size,
                    config.shared_expert_intermediate_size,
                ],
                has_bias=False,
            )
            self.shared_expert_down = LinearRowParallel(
                config.shared_expert_intermediate_size,
                config.hidden_size,
                has_bias=False,
            )
            self.num_shared_experts = config.num_shared_experts
        else:
            self.shared_expert_gate_up = None
            self.shared_expert_down = None
            self.num_shared_experts = 0

        # Activation function
        match hidden_act:
            case "silu":
                from .activation import silu_and_mul

                self.act_fn = silu_and_mul
            case act_fn:
                raise ValueError(f"Unsupported activation function: {act_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE layer.

        Args:
            x: Input tensor of shape [num_tokens, hidden_size]

        Returns:
            Output tensor of shape [num_tokens, hidden_size]
        """
        num_tokens, hidden_size = x.shape

        # Compute routing weights and indices
        topk_weights, topk_indices = self.gate.forward(x)
        # topk_weights: [num_tokens, num_experts_per_tok]
        # topk_indices: [num_tokens, num_experts_per_tok]

        # Flatten for processing
        flat_topk_indices = topk_indices.view(-1)  # [num_tokens * num_experts_per_tok]
        flat_topk_weights = topk_weights.view(-1)  # [num_tokens * num_experts_per_tok]

        # Sort tokens by expert assignment for efficient processing
        sorted_expert_indices, sort_indices = torch.sort(flat_topk_indices, stable=True)
        unsort_indices = torch.argsort(sort_indices)

        # Count tokens per expert
        num_tokens_per_expert = torch.bincount(
            sorted_expert_indices, minlength=self.num_experts
        )

        # Expand input for each expert assignment
        # [num_tokens, hidden] -> [num_tokens * num_experts_per_tok, hidden]
        expanded_x = x.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1)
        expanded_x = expanded_x.reshape(-1, hidden_size)

        # Sort inputs according to expert assignment
        sorted_x = expanded_x[sort_indices]

        # Apply expert gate+up projection
        gate_up_out = self.experts_gate_up.forward(
            sorted_x, sorted_expert_indices, num_tokens_per_expert
        )

        # Apply activation
        activated = self.act_fn(gate_up_out)

        # Apply expert down projection
        expert_out = self.experts_down.forward(
            activated, sorted_expert_indices, num_tokens_per_expert
        )

        # Unsort to original order
        expert_out = expert_out[unsort_indices]

        # Weight by routing probability
        expert_out = expert_out * flat_topk_weights.unsqueeze(-1)

        # Combine contributions from different experts
        # Reshape and sum: [num_tokens * k, hidden] -> [num_tokens, k, hidden] -> [num_tokens, hidden]
        expert_out = expert_out.view(num_tokens, self.num_experts_per_tok, hidden_size)
        output = expert_out.sum(dim=1)

        # Add shared expert contribution if present
        if self.shared_expert_gate_up is not None:
            shared_gate_up = self.shared_expert_gate_up.forward(x)
            shared_activated = self.act_fn(shared_gate_up)
            shared_out = self.shared_expert_down.forward(shared_activated)
            output = output + shared_out

        return output


class FusedSparseMoELayer(BaseOP):
    """
    High-performance Sparse MoE layer with Triton kernel optimizations.

    This variant uses:
    - Fused token permutation kernels
    - Optimized grouped GEMM when available
    - Fused SiLU activation
    - Efficient unpermutation and reduction
    """

    def __init__(self, config: MoEConfig, hidden_act: str = "silu"):
        from minisgl.kernel import MoEKernelDispatcher

        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.norm_topk_prob = config.norm_topk_prob

        # Kernel dispatcher for optimized operations
        self._dispatcher = MoEKernelDispatcher(use_triton=True)

        # Router gate weight
        self.gate_weight = torch.empty(config.num_experts, config.hidden_size)

        # Expert networks
        self.experts_gate_up = MoEGateUpProj(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            has_bias=False,
        )
        self.experts_down = MoEDownProj(
            num_experts=config.num_experts,
            intermediate_size=config.moe_intermediate_size,
            hidden_size=config.hidden_size,
            has_bias=False,
        )

        # Shared expert (optional)
        if config.num_shared_experts > 0:
            from .linear import LinearColParallelMerged, LinearRowParallel

            self.shared_expert_gate_up = LinearColParallelMerged(
                config.hidden_size,
                [
                    config.shared_expert_intermediate_size,
                    config.shared_expert_intermediate_size,
                ],
                has_bias=False,
            )
            self.shared_expert_down = LinearRowParallel(
                config.shared_expert_intermediate_size,
                config.hidden_size,
                has_bias=False,
            )
            self.num_shared_experts = config.num_shared_experts
        else:
            self.shared_expert_gate_up = None
            self.shared_expert_down = None
            self.num_shared_experts = 0

        # Activation
        match hidden_act:
            case "silu":
                from .activation import silu_and_mul

                self.act_fn = silu_and_mul
            case act_fn:
                raise ValueError(f"Unsupported activation function: {act_fn}")

        # Local intermediate size for TP
        tp_info = get_tp_info()
        self._local_intermediate_size = divide_even(
            config.moe_intermediate_size, tp_info.size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimized kernel dispatch.

        Args:
            x: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        num_tokens = x.shape[0]

        # Compute routing and permute tokens (fused operation)
        (
            sorted_hidden,
            sorted_expert_indices,
            unsort_indices,
            flat_weights,
            num_tokens_per_expert,
        ) = self._dispatcher.compute_routing(
            x,
            self.gate_weight,
            self.num_experts_per_tok,
            self.norm_topk_prob,
        )

        # Expert gate+up projection
        gate_up_out = self.experts_gate_up.forward(
            sorted_hidden, sorted_expert_indices, num_tokens_per_expert
        )

        # Activation (SiLU and multiply)
        activated = self.act_fn(gate_up_out)

        # Expert down projection
        expert_out = self.experts_down.forward(
            activated, sorted_expert_indices, num_tokens_per_expert
        )

        # Unpermute and reduce with routing weights (fused operation)
        output = self._dispatcher.reduce_expert_outputs(
            expert_out, unsort_indices, flat_weights, num_tokens, self.num_experts_per_tok
        )

        # Add shared expert contribution
        if self.shared_expert_gate_up is not None:
            shared_gate_up = self.shared_expert_gate_up.forward(x)
            shared_activated = self.act_fn(shared_gate_up)
            shared_out = self.shared_expert_down.forward(shared_activated)
            output = output + shared_out

        return output


def create_moe_layer(config: MoEConfig, hidden_act: str = "silu", use_fused: bool = True) -> BaseOP:
    """
    Factory function to create an MoE layer.

    Args:
        config: MoE configuration
        hidden_act: Activation function name
        use_fused: Whether to use fused Triton kernels (recommended)

    Returns:
        MoE layer instance
    """
    if use_fused and torch.cuda.is_available():
        return FusedSparseMoELayer(config, hidden_act)
    return SparseMoELayer(config, hidden_act)


__all__ = [
    "MoEGate",
    "MoEExpertLinear",
    "MoEGateUpProj",
    "MoEDownProj",
    "SparseMoELayer",
    "FusedSparseMoELayer",
    "create_moe_layer",
]
