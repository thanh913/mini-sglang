"""
High-performance Triton kernels for Mixture of Experts.

This module provides optimized kernels for MoE operations:
- Permutation/unpermutation for expert routing
- Grouped GEMM for batched expert computation
- Fused gating and activation functions
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _permute_tokens_kernel(
    # Input pointers
    input_ptr,
    sorted_indices_ptr,
    # Output pointer
    output_ptr,
    # Dimensions
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr,
):
    """Permute tokens according to sorted expert indices."""
    token_idx = tl.program_id(0)

    # Load the source index for this output position
    src_idx = tl.load(sorted_indices_ptr + token_idx)

    # Copy hidden states
    for h_start in range(0, hidden_size, BLOCK_SIZE_H):
        h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
        mask = h_offsets < hidden_size

        # Load from source position
        src_ptr = input_ptr + src_idx * hidden_size + h_offsets
        values = tl.load(src_ptr, mask=mask, other=0.0)

        # Store to destination position
        dst_ptr = output_ptr + token_idx * hidden_size + h_offsets
        tl.store(dst_ptr, values, mask=mask)


@triton.jit
def _unpermute_and_reduce_kernel(
    # Input pointers
    expert_output_ptr,
    unsort_indices_ptr,
    weights_ptr,
    # Output pointer
    output_ptr,
    # Dimensions
    num_token_slots: tl.constexpr,
    num_tokens: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    hidden_size: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr,
):
    """Unpermute expert outputs and reduce with routing weights."""
    token_idx = tl.program_id(0)

    for h_start in range(0, hidden_size, BLOCK_SIZE_H):
        h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
        mask = h_offsets < hidden_size

        # Accumulate weighted outputs from all expert slots for this token
        acc = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)

        for k in range(num_experts_per_tok):
            slot_idx = token_idx * num_experts_per_tok + k

            # Get the position in sorted expert outputs
            sorted_pos = tl.load(unsort_indices_ptr + slot_idx)

            # Load expert output
            src_ptr = expert_output_ptr + sorted_pos * hidden_size + h_offsets
            expert_out = tl.load(src_ptr, mask=mask, other=0.0)

            # Load routing weight
            weight = tl.load(weights_ptr + slot_idx)

            # Accumulate
            acc += expert_out.to(tl.float32) * weight

        # Store accumulated result
        dst_ptr = output_ptr + token_idx * hidden_size + h_offsets
        tl.store(dst_ptr, acc.to(expert_output_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_silu_mul_kernel(
    # Input/output pointer (in-place for gate output)
    gate_up_ptr,
    # Dimensions
    num_elements: tl.constexpr,
    intermediate_size: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU activation and element-wise multiplication."""
    pid = tl.program_id(0)
    token_idx = pid

    for i_start in range(0, intermediate_size, BLOCK_SIZE):
        offsets = i_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < intermediate_size

        # Compute pointers for gate and up projections
        gate_offsets = token_idx * (2 * intermediate_size) + offsets
        up_offsets = token_idx * (2 * intermediate_size) + intermediate_size + offsets

        # Load gate and up values
        gate_val = tl.load(gate_up_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
        up_val = tl.load(gate_up_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)

        # SiLU: x * sigmoid(x)
        gate_sigmoid = tl.sigmoid(gate_val)
        gate_activated = gate_val * gate_sigmoid

        # Multiply with up projection
        result = gate_activated * up_val

        # Store result in the gate position (reuse memory)
        tl.store(gate_up_ptr + gate_offsets, result.to(gate_up_ptr.dtype.element_ty), mask=mask)


def permute_tokens(
    input: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Permute tokens according to expert assignment order.

    Args:
        input: Input tensor [num_tokens, hidden_size]
        sorted_indices: Indices that sort tokens by expert [num_tokens]

    Returns:
        Permuted tensor [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = input.shape
    output = torch.empty_like(input)

    BLOCK_SIZE_H = min(1024, triton.next_power_of_2(hidden_size))

    grid = (num_tokens,)
    _permute_tokens_kernel[grid](
        input,
        sorted_indices,
        output,
        num_tokens,
        hidden_size,
        BLOCK_SIZE_H,
    )

    return output


def unpermute_and_reduce(
    expert_output: torch.Tensor,
    unsort_indices: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """
    Unpermute expert outputs and reduce with routing weights.

    Args:
        expert_output: Expert outputs [num_tokens * k, hidden_size]
        unsort_indices: Indices to unsort [num_tokens * k]
        weights: Routing weights [num_tokens * k]
        num_tokens: Number of original tokens
        num_experts_per_tok: Number of experts per token (k)

    Returns:
        Reduced output [num_tokens, hidden_size]
    """
    num_token_slots, hidden_size = expert_output.shape
    output = torch.empty(num_tokens, hidden_size, dtype=expert_output.dtype, device=expert_output.device)

    BLOCK_SIZE_H = min(1024, triton.next_power_of_2(hidden_size))

    grid = (num_tokens,)
    _unpermute_and_reduce_kernel[grid](
        expert_output,
        unsort_indices,
        weights,
        output,
        num_token_slots,
        num_tokens,
        num_experts_per_tok,
        hidden_size,
        BLOCK_SIZE_H,
    )

    return output


def fused_silu_and_mul_inplace(gate_up: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    """
    Fused SiLU activation and multiplication in-place.

    Args:
        gate_up: Concatenated gate and up projections [num_tokens, 2 * intermediate_size]
        intermediate_size: Size of intermediate dimension

    Returns:
        Activated output [num_tokens, intermediate_size] (stored in first half of input)
    """
    num_tokens = gate_up.shape[0]

    BLOCK_SIZE = min(1024, triton.next_power_of_2(intermediate_size))

    grid = (num_tokens,)
    _fused_silu_mul_kernel[grid](
        gate_up,
        num_tokens * 2 * intermediate_size,
        intermediate_size,
        BLOCK_SIZE,
    )

    # Return only the first half which contains the result
    return gate_up[:, :intermediate_size].contiguous()


class MoEKernelDispatcher:
    """
    Dispatcher for MoE kernel operations.

    Provides a unified interface that selects between:
    - Triton kernels (when available and beneficial)
    - PyTorch fallback (for debugging or unsupported configurations)
    """

    def __init__(self, use_triton: bool = True):
        self.use_triton = use_triton and torch.cuda.is_available()

    def compute_routing(
        self,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        num_experts_per_tok: int,
        norm_topk_prob: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute expert routing for tokens.

        Returns:
            sorted_hidden: Hidden states sorted by expert
            sorted_expert_indices: Expert indices in sorted order
            unsort_indices: Indices to restore original order
            weights: Routing weights (flat)
            num_tokens_per_expert: Count of tokens per expert
        """
        num_tokens, hidden_size = hidden_states.shape
        num_experts = gate_weight.shape[0]

        # Compute router logits and apply softmax
        router_logits = torch.nn.functional.linear(hidden_states, gate_weight)
        scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Top-k selection
        topk_weights, topk_indices = torch.topk(
            scores, k=num_experts_per_tok, dim=-1, sorted=False
        )

        if norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        topk_weights = topk_weights.to(hidden_states.dtype)

        # Flatten for processing
        flat_indices = topk_indices.view(-1)
        flat_weights = topk_weights.view(-1)

        # Sort by expert
        sorted_expert_indices, sort_indices = torch.sort(flat_indices, stable=True)
        unsort_indices = torch.argsort(sort_indices)

        # Count tokens per expert
        num_tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=num_experts)

        # Expand and permute hidden states
        expanded_hidden = hidden_states.unsqueeze(1).expand(-1, num_experts_per_tok, -1)
        expanded_hidden = expanded_hidden.reshape(-1, hidden_size)

        if self.use_triton:
            sorted_hidden = permute_tokens(expanded_hidden, sort_indices)
        else:
            sorted_hidden = expanded_hidden[sort_indices]

        return sorted_hidden, sorted_expert_indices, unsort_indices, flat_weights, num_tokens_per_expert

    def reduce_expert_outputs(
        self,
        expert_output: torch.Tensor,
        unsort_indices: torch.Tensor,
        weights: torch.Tensor,
        num_tokens: int,
        num_experts_per_tok: int,
    ) -> torch.Tensor:
        """Reduce expert outputs with routing weights."""
        if self.use_triton:
            return unpermute_and_reduce(
                expert_output, unsort_indices, weights, num_tokens, num_experts_per_tok
            )
        else:
            # PyTorch fallback
            hidden_size = expert_output.shape[-1]
            unsorted = expert_output[unsort_indices]
            weighted = unsorted * weights.unsqueeze(-1)
            weighted = weighted.view(num_tokens, num_experts_per_tok, hidden_size)
            return weighted.sum(dim=1)


__all__ = [
    "permute_tokens",
    "unpermute_and_reduce",
    "fused_silu_and_mul_inplace",
    "MoEKernelDispatcher",
]
