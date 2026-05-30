from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class TilingAnalysis:
    block_shape: tuple[int, ...]
    grid_shape: tuple[int, ...]
    input_shape: tuple[int, ...]
    block_utilization: float
    grid_efficiency: float
    memory_access_pattern: str
    num_blocks: int


def compute_block_utilization(
    block_shape: tuple[int, ...], dim_size: int, dim_index: int = 0
) -> float:
    if dim_size == 0:
        return 1.0
    valid = min(block_shape[dim_index], dim_size)
    return valid / block_shape[dim_index]


def compute_grid_efficiency(
    dim_size: int, block_size: int
) -> float:
    if block_size == 0:
        return 0.0
    blocks_needed = (dim_size + block_size - 1) // block_size
    total_elements = blocks_needed * block_size
    return dim_size / total_elements


def analyze_tiling(
    block_shape: tuple[int, ...],
    grid_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
) -> TilingAnalysis:
    utils = []
    effs = []
    for i in range(min(len(block_shape), len(input_shape))):
        util = compute_block_utilization(block_shape, input_shape[i], i)
        eff = compute_grid_efficiency(input_shape[i], block_shape[i])
        utils.append(util)
        effs.append(eff)

    avg_util = sum(utils) / len(utils) if utils else 1.0
    avg_eff = sum(effs) / len(effs) if effs else 1.0

    num_blocks = 1
    for g in grid_shape:
        num_blocks *= g

    access_pattern = _classify_access_pattern(block_shape, input_shape)

    return TilingAnalysis(
        block_shape=block_shape,
        grid_shape=grid_shape,
        input_shape=input_shape,
        block_utilization=avg_util,
        grid_efficiency=avg_eff,
        memory_access_pattern=access_pattern,
        num_blocks=num_blocks,
    )


def _classify_access_pattern(
    block_shape: tuple[int, ...], input_shape: tuple[int, ...]
) -> str:
    if not block_shape or not input_shape:
        return "unknown"
    if len(block_shape) == 1:
        return "contiguous_1d"
    ndim = min(len(block_shape), len(input_shape))
    inner_dim = block_shape[-1]
    last_dim_size = input_shape[-1] if input_shape else 0
    if inner_dim == last_dim_size:
        return "contiguous_full_inner"
    return "tiled_inner"
