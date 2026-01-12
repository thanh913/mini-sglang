from __future__ import annotations

import glob
import os
import re
from typing import Dict

import safetensors
import torch
from huggingface_hub import snapshot_download
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_up
from tqdm.asyncio import tqdm


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def _is_moe_expert_weight(key: str) -> bool:
    """Check if the weight key belongs to an MoE expert."""
    return ".experts." in key


def _is_moe_shared_expert_weight(key: str) -> bool:
    """Check if the weight key belongs to an MoE shared expert."""
    return ".shared_expert." in key


def _is_moe_gate_weight(key: str) -> bool:
    """Check if the weight key belongs to an MoE router/gate."""
    # Match patterns like "mlp.gate.weight" but not "mlp.gate_proj"
    return ".mlp.gate.weight" in key


def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size

    # Standard dense layer patterns
    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
    ]
    SPLIT_DIM_1_LIST = [
        ".o_proj",
    ]

    # Dense MLP patterns (non-MoE)
    DENSE_MLP_SPLIT_DIM_0 = [".gate_proj", ".up_proj"]
    DENSE_MLP_SPLIT_DIM_1 = [".down_proj"]

    for key, value in state_dict.items():
        # Handle MoE expert weights - apply TP to each expert
        if _is_moe_expert_weight(key):
            if any(sub in key for sub in [".gate_proj", ".up_proj"]):
                # Expert gate/up: split output dim (dim 0 for 2D weight)
                shard_state_dict[key] = value.chunk(n, dim=0)[r]
            elif ".down_proj" in key:
                # Expert down: split input dim (dim 1 for 2D weight)
                shard_state_dict[key] = value.chunk(n, dim=1)[r]
            else:
                shard_state_dict[key] = value

        # Handle MoE shared expert weights
        elif _is_moe_shared_expert_weight(key):
            if any(sub in key for sub in [".gate_proj", ".up_proj"]):
                shard_state_dict[key] = value.chunk(n, dim=0)[r]
            elif ".down_proj" in key:
                shard_state_dict[key] = value.chunk(n, dim=1)[r]
            else:
                shard_state_dict[key] = value

        # Handle MoE gate/router weights - not sharded
        elif _is_moe_gate_weight(key):
            shard_state_dict[key] = value

        # Standard attention projections
        elif any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]

        # Dense MLP (non-MoE)
        elif any(key.count(sub) for sub in DENSE_MLP_SPLIT_DIM_0):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in DENSE_MLP_SPLIT_DIM_1):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]

        # Embeddings
        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = divide_up(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value

    return shard_state_dict


def _merge_moe_expert_weights(
    state_dict: Dict[str, torch.Tensor],
    filtered_state_dict: Dict[str, torch.Tensor],
) -> None:
    """
    Merge MoE expert weights into grouped format.

    HuggingFace format: model.layers.N.mlp.experts.M.{gate,up,down}_proj.weight
    Our format: model.layers.N.mlp.experts_{gate_up,down}.weight [num_experts, ...]
    """
    # Find all expert weight keys and group by layer
    expert_pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
    )

    layer_experts: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

    keys_to_remove = []
    for key in list(state_dict.keys()):
        match = expert_pattern.match(key)
        if match:
            layer_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            proj_type = match.group(3)

            if layer_idx not in layer_experts:
                layer_experts[layer_idx] = {}
            if expert_idx not in layer_experts[layer_idx]:
                layer_experts[layer_idx][expert_idx] = {}

            layer_experts[layer_idx][expert_idx][proj_type] = state_dict[key]
            keys_to_remove.append(key)

    # Remove processed keys
    for key in keys_to_remove:
        del state_dict[key]

    # Merge experts per layer
    for layer_idx, experts in layer_experts.items():
        num_experts = len(experts)
        if num_experts == 0:
            continue

        # Get a sample weight to determine shapes
        sample_expert = experts[0]

        # Stack gate and up projections: [num_experts, 2*intermediate, hidden]
        gate_up_list = []
        down_list = []

        for expert_idx in range(num_experts):
            expert_weights = experts[expert_idx]
            gate = expert_weights["gate_proj"]
            up = expert_weights["up_proj"]
            down = expert_weights["down_proj"]

            # Merge gate and up
            gate_up = torch.cat([gate, up], dim=0)
            gate_up_list.append(gate_up)
            down_list.append(down)

        # Stack all experts: [num_experts, out_dim, in_dim]
        stacked_gate_up = torch.stack(gate_up_list, dim=0)
        stacked_down = torch.stack(down_list, dim=0)

        # Add to filtered dict with our naming convention
        base_key = f"model.layers.{layer_idx}.mlp.experts_gate_up.weight"
        filtered_state_dict[base_key] = stacked_gate_up

        base_key = f"model.layers.{layer_idx}.mlp.experts_down.weight"
        filtered_state_dict[base_key] = stacked_down


def _merge_moe_shared_expert_weights(
    state_dict: Dict[str, torch.Tensor],
    filtered_state_dict: Dict[str, torch.Tensor],
) -> None:
    """Merge shared expert weights (gate_proj + up_proj)."""
    shared_pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight"
    )

    layer_shared: Dict[int, Dict[str, torch.Tensor]] = {}
    keys_to_remove = []

    for key in list(state_dict.keys()):
        match = shared_pattern.match(key)
        if match:
            layer_idx = int(match.group(1))
            proj_type = match.group(2)

            if layer_idx not in layer_shared:
                layer_shared[layer_idx] = {}
            layer_shared[layer_idx][proj_type] = state_dict[key]
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del state_dict[key]

    for layer_idx, weights in layer_shared.items():
        if "gate_proj" in weights and "up_proj" in weights:
            gate_up = torch.cat([weights["gate_proj"], weights["up_proj"]], dim=0)
            new_key = f"model.layers.{layer_idx}.mlp.shared_expert_gate_up.weight"
            filtered_state_dict[new_key] = gate_up

        if "down_proj" in weights:
            new_key = f"model.layers.{layer_idx}.mlp.shared_expert_down.weight"
            filtered_state_dict[new_key] = weights["down_proj"]


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    filtered_state_dict: Dict[str, torch.Tensor] = {}

    # First handle MoE expert weights
    _merge_moe_expert_weights(state_dict, filtered_state_dict)

    # Handle shared expert weights
    _merge_moe_shared_expert_weights(state_dict, filtered_state_dict)

    # Handle remaining standard weights
    for key in list(state_dict.keys()):
        if key.count(".q_proj"):
            q_proj = state_dict[key]
            k_key = key.replace(".q_proj", ".k_proj")
            v_key = key.replace(".q_proj", ".v_proj")
            if k_key in state_dict and v_key in state_dict:
                k_proj = state_dict[k_key]
                v_proj = state_dict[v_key]
                new_key = key.replace(".q_proj", ".qkv_proj")
                filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
                del state_dict[key]
                del state_dict[k_key]
                del state_dict[v_key]
            else:
                filtered_state_dict[key] = state_dict[key]
        elif key.count(".gate_proj") and ".mlp.experts" not in key and ".shared_expert" not in key:
            # Dense MLP gate+up merge (non-MoE)
            gate_proj = state_dict[key]
            up_key = key.replace(".gate_proj", ".up_proj")
            if up_key in state_dict:
                up_proj = state_dict[up_key]
                new_key = key.replace(".gate_proj", ".gate_up_proj")
                filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
                del state_dict[key]
                del state_dict[up_key]
            else:
                filtered_state_dict[key] = state_dict[key]
        elif key.count(".k_proj") or key.count(".v_proj"):
            # Skip - handled with q_proj
            continue
        elif key.count(".up_proj") and ".mlp.experts" not in key and ".shared_expert" not in key:
            # Skip - handled with gate_proj for dense MLP
            continue
        else:
            filtered_state_dict[key] = state_dict[key]

    return filtered_state_dict


def load_hf_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    if os.path.isdir(model_path):
        hf_folder = model_path
    else:
        try:
            hf_folder = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors"],
                tqdm_class=DisabledTqdm,
            )
        except Exception:
            raise ValueError(
                f"Model path '{model_path}' is neither a local directory nor a valid HuggingFace repository ID"
            )

    # find the all *.pt files in the hf_folder
    files = glob.glob(f"{hf_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict)

    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    return _merge_state_dict(state_dict)
