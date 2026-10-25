"""Colour previews of the part mask for the app."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib import colormaps

from .data_inspection import PartMapping


def _palette(values: Iterable[int]) -> dict[int, tuple[int, int, int]]:
    nonzero = sorted(value for value in set(values) if value != 0)
    count = max(len(nonzero), 1)
    color_map = colormaps["hsv"]
    return {
        value: tuple(int(channel * 255) for channel in color_map(index / count)[:3])
        for index, value in enumerate(nonzero)
    }


def colorize_part_mask(
    mask: np.ndarray,
    mappings: Iterable[PartMapping],
    channel_index: int = 0,
) -> np.ndarray:
    """Return an RGB visualization from exact mapped values on one mask channel."""
    array = np.asarray(mask)
    if array.ndim == 2:
        ids = array
    elif array.ndim == 3 and 0 <= channel_index < array.shape[2]:
        ids = array[..., channel_index]
    else:
        raise ValueError(f"Unsupported mask shape or channel index: {array.shape}/{channel_index}")
    parts = list(mappings)
    colors = _palette(part.stencil_value for part in parts)
    colorized = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for value, color in colors.items():
        colorized[ids == value] = color
    return colorized


def part_mask_legend(mappings: Iterable[PartMapping]) -> list[dict[str, object]]:
    """Return the mapping and display colors used by ``colorize_part_mask``."""
    parts = list(mappings)
    colors = _palette(part.stencil_value for part in parts)
    return [
        {
            "pixel_value": part.stencil_value,
            "part_name": part.name,
            "rgb": colors.get(part.stencil_value),
        }
        for part in parts
    ]
