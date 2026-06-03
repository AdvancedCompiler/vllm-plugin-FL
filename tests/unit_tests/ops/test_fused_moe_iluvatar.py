# Copyright (c) 2026 BAAI. All rights reserved.

"""
Tests for fused MoE operator implementations.

Compares Iluvatar PyTorch reference implementations against FlagGems
Triton implementations as the correctness baseline.
"""

import itertools

import flag_gems
import pytest
import torch

from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
    moe_align_block_size_flaggems,
    moe_sum_flaggems,
    topk_softmax_flaggems,
)
from vllm_fl.dispatch.backends.vendor.iluvatar.impl.fused_moe import (
    moe_align_block_size_iluvatar,
    moe_sum_iluvatar,
    topk_softmax_iluvatar,
)

# Adapted from: https://github.com/flagos-ai/FlagGems/blob/master/tests/accuracy_utils.py

bf16_is_supported = flag_gems.runtime.device.support_bf16
PRIMARY_FLOAT_DTYPES = [torch.float16, torch.float32]
FLOAT_DTYPES = (
    PRIMARY_FLOAT_DTYPES + [torch.bfloat16]
    if bf16_is_supported
    else PRIMARY_FLOAT_DTYPES
)


# =============================================================================
# moe_sum
# =============================================================================
# Adapted from: https://github.com/flagos-ai/FlagGems/blob/master/tests/test_moe_sum.py


M_VALUES = [1, 33, 64, 222]
TOP_KS = [2, 6]
K_VALUES = [128, 511, 1024]
MOE_SHAPES = list(itertools.product(M_VALUES, TOP_KS, K_VALUES))


@pytest.mark.moe_sum
@pytest.mark.parametrize("shape", MOE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_moe_sum(shape, dtype):
    m, topk, k = shape
    inp = torch.randn((m, topk, k), dtype=dtype, device=flag_gems.device)

    gems_out = torch.empty((m, k), dtype=dtype, device=flag_gems.device)
    iluvatar_out = torch.empty((m, k), dtype=dtype, device=flag_gems.device)

    moe_sum_iluvatar(inp, iluvatar_out)
    moe_sum_flaggems(inp, gems_out)

    assert torch.allclose(gems_out, iluvatar_out, atol=1e-5), (
        "moe_sum: iluvatar out mismatch"
    )


# =============================================================================
# topk_softmax
# =============================================================================
# Adapted from: https://github.com/flagos-ai/FlagGems/blob/master/tests/test_topk_softmax.py


@pytest.mark.topk_softmax
@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    [
        (1, 4, 2),
        (4, 8, 2),
        (8, 16, 4),
        (32, 64, 8),
        (128, 128, 16),
        (500, 255, 30),
        (512, 256, 32),
        (1024, 512, 32),
    ],
)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32])
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax(
    num_tokens, num_experts, topk, input_dtype, index_dtype, renormalize
):
    torch.manual_seed(42)
    device = flag_gems.device

    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    # weights are always float32 regardless of input_dtype (same as flaggems test)
    gems_weights = torch.empty(num_tokens, topk, device=device, dtype=torch.float32)
    gems_indices = torch.empty(num_tokens, topk, device=device, dtype=index_dtype)
    gems_token_expert = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)

    topk_softmax_flaggems(
        gems_weights,
        gems_indices,
        gems_token_expert,
        gating_output,
        renormalize,
    )

    iluvatar_weights = torch.empty_like(gems_weights)
    iluvatar_indices = torch.empty_like(gems_indices)
    iluvatar_token_expert = torch.empty_like(gems_token_expert)

    topk_softmax_iluvatar(
        iluvatar_weights,
        iluvatar_indices,
        iluvatar_token_expert,
        gating_output,
        renormalize,
    )

    assert torch.allclose(iluvatar_weights, gems_weights, atol=1e-5), (
        "topk_softmax: weights mismatch"
    )
    assert torch.equal(iluvatar_indices.cpu(), gems_indices.cpu()), (
        "topk_softmax: indices mismatch"
    )
    assert torch.equal(iluvatar_token_expert.cpu(), gems_token_expert.cpu()), (
        "topk_softmax: token_expert_indices mismatch"
    )

    if renormalize:
        sums = gems_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            "topk_softmax: renormalized weights should sum to 1"
        )


# =============================================================================
# moe_align_block_size
# =============================================================================
# Adapted from: https://github.com/flagos-ai/FlagGems/blob/master/tests/test_moe_align_block_size.py


def _group_tokens_by_expert(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
) -> dict:
    """Group tokens in sorted_ids by expert_id, returning {expert_id: [token_ids]}."""
    num_blocks = valid_length // block_size
    expert_tokens: dict[int, list[int]] = {}

    for block_idx in range(num_blocks):
        expert_id = expert_ids[block_idx].item()
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, valid_length)

        block_tokens = sorted_ids[block_start:block_end]
        valid_tokens = block_tokens[block_tokens < total_tokens]

        if expert_id not in expert_tokens:
            expert_tokens[expert_id] = []
        expert_tokens[expert_id].extend(valid_tokens.tolist())
    return expert_tokens


def _verify_expert_level_sorting(
    actual_sorted_ids: torch.Tensor,
    expected_sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
):
    """
    Compare sorted_ids by grouping tokens per expert and comparing sets.

    The kernel may not preserve the original token order within each expert,
    which does not affect correctness.  This comparison verifies that the
    same tokens are assigned to each expert in both implementations.
    """
    expected_expert_tokens = _group_tokens_by_expert(
        expected_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )
    actual_expert_tokens = _group_tokens_by_expert(
        actual_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    assert set(expected_expert_tokens.keys()) == set(actual_expert_tokens.keys()), (
        f"Expert IDs mismatch: expected={set(expected_expert_tokens.keys())}, "
        f"actual={set(actual_expert_tokens.keys())}"
    )

    for expert_id in expected_expert_tokens:
        expected_tokens = torch.tensor(
            expected_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        actual_tokens = torch.tensor(
            actual_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        assert torch.equal(
            torch.sort(expected_tokens)[0], torch.sort(actual_tokens)[0]
        ), (
            f"Expert {expert_id} token mismatch: "
            f"expected={expected_expert_tokens[expert_id]}, "
            f"actual={actual_expert_tokens[expert_id]}"
        )


@pytest.mark.moe_align
@pytest.mark.parametrize("num_experts", [10, 128, 256, 512])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize(
    "topk_ids_shape",
    [
        (1024, 10),
        (6152, 10),
        (11575, 10),
        (16384, 10),
    ],
)
def test_moe_align_block_size(num_experts, block_size, topk_ids_shape):
    torch.manual_seed(42)
    device = flag_gems.device
    topk_ids = torch.randint(
        0, num_experts, topk_ids_shape, dtype=torch.int32, device=device
    )
    s_i, e_i, n_i = moe_align_block_size_iluvatar(
        topk_ids.clone(), block_size, num_experts
    )
    s_g, e_g, n_g = moe_align_block_size_flaggems(
        topk_ids.clone(), block_size, num_experts
    )

    assert n_i.item() == n_g.item(), (
        f"num_tokens_post_pad mismatch: ilu={n_i.item()}, gems={n_g.item()}"
    )

    # sorted_ids: verify expert-level grouping (order within expert may differ)
    _verify_expert_level_sorting(
        s_i, s_g, e_g, block_size, n_g.item(), topk_ids.numel()
    )

    # expert_ids: compare valid blocks only
    # flaggems fills all max_num_m_blocks positions; iluvatar uses -1 sentinel
    # for unused blocks. Both are correct — downstream kernel only reads
    # up to ceil(num_tokens_post_pad / block_size).
    n_blocks = (n_i.item() + block_size - 1) // block_size
    assert torch.equal(e_i[:n_blocks].cpu(), e_g[:n_blocks].cpu()), (
        "expert_ids mismatch in valid blocks"
    )
