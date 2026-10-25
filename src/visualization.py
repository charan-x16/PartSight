"""Non-destructive diagnostic visualizations for PartSight masks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib import colormaps
from PIL import Image

from .data_inspection import PartMapping, inspect_mask, load_image


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


def save_mask_visualizations(
    mask_path: str | Path,
    mappings: Iterable[PartMapping],
    output_dir: str | Path,
    reference_path: str | Path | None = None,
) -> list[Path]:
    """Save colorized mask, optional CAD overlay, and a text legend."""

    loaded = load_image(mask_path)
    stats = inspect_mask(loaded)
    channel_index = stats["selected_channel_index"]
    if channel_index is None:
        raise ValueError(f"Cannot visualize ambiguous multi-channel mask: {mask_path}")
    ids = loaded.array if loaded.array.ndim == 2 else loaded.array[..., channel_index]
    colors = _palette(int(value) for value in stats["selected_value_counts"])
    colorized = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for value, color in colors.items():
        colorized[ids == value] = color

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(mask_path).stem
    color_path = destination / f"{stem}_colorized.png"
    Image.fromarray(colorized).save(color_path)
    saved = [color_path]

    mapping_by_value = {entry.stencil_value: entry.name for entry in mappings}
    legend_path = destination / f"{stem}_legend.txt"
    legend_lines = ["value\tpixels\tRGB\tpart_name"]
    for value, count in sorted(stats["selected_value_counts"].items()):
        color = colors.get(int(value), (0, 0, 0))
        legend_lines.append(
            f"{value}\t{count}\t{color}\t{mapping_by_value.get(int(value), '<unmapped/background>')}"
        )
    legend_path.write_text("\n".join(legend_lines) + "\n", encoding="utf-8")
    saved.append(legend_path)

    if reference_path is not None:
        reference = load_image(reference_path)
        if (reference.height, reference.width) != ids.shape:
            raise ValueError("Reference image and mask dimensions are not compatible")
        base = (
            reference.array[..., :3]
            if reference.array.ndim == 3
            else np.repeat(reference.array[..., None], 3, axis=2)
        )
        base = base.astype(np.float32)
        foreground = ids != 0
        overlay = base.copy()
        overlay[foreground] = 0.55 * base[foreground] + 0.45 * colorized[foreground]
        overlay_path = destination / f"{stem}_cad_overlay.png"
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
            overlay_path
        )
        saved.append(overlay_path)
    return saved


def save_part_geometry_diagnostics(
    mask_path: str | Path,
    reference_path: str | Path,
    mappings: Iterable[PartMapping],
    output_dir: str | Path,
) -> list[Path]:
    """Write a labeled mask, CAD contour overlay, and one ID view per mapping.

    Whole-ID geometry is calculated from exact equality on the safely selected
    mask channel. Per-ID views deliberately retain disconnected components so
    fragmented encodings are visible rather than hidden by a largest-component
    assumption.
    """
    import cv2
    from PIL import ImageDraw, ImageFont

    from .detector import extract_part_regions

    loaded_mask = load_image(mask_path)
    stats = inspect_mask(loaded_mask)
    channel = stats["selected_channel_index"]
    if channel is None:
        raise ValueError(f"Cannot infer an unambiguous part-ID channel in {mask_path}")
    ids = loaded_mask.array if loaded_mask.array.ndim == 2 else loaded_mask.array[..., channel]
    mappings = list(mappings)
    regions = extract_part_regions(ids, mappings)
    reference = load_image(reference_path)
    if (reference.height, reference.width) != ids.shape:
        raise ValueError("Reference image and mask dimensions are not compatible")
    cad_array = reference.array
    if cad_array.ndim == 2:
        cad_array = np.repeat(cad_array[..., None], 3, axis=2)
    elif cad_array.shape[2] == 4:
        cad_array = cad_array[..., :3]
    cad = Image.fromarray(cad_array.astype(np.uint8))

    colors = _palette(region.stencil_value for region in regions)
    destination = Path(output_dir)
    parts_dir = destination / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    font = ImageFont.load_default(size=18)
    title_font = ImageFont.load_default(size=22)

    # Full-resolution color mask plus a real, readable legend in the same PNG.
    colorized = np.zeros((*ids.shape, 3), dtype=np.uint8)
    mapping_names = {part.stencil_value: part.name for part in mappings}
    for value, color in colors.items():
        colorized[ids == value] = color
    legend_width = 760
    full = Image.new("RGB", (ids.shape[1] + legend_width, ids.shape[0]), "#15191f")
    full.paste(Image.fromarray(colorized), (0, 0))
    draw = ImageDraw.Draw(full)
    draw.text((ids.shape[1] + 18, 16), "Part mask values", font=title_font, fill="white")
    row_height = 78 if len(regions) <= 36 else 48
    for index, region in enumerate(regions):
        y = 60 + index * row_height
        if y + row_height > ids.shape[0]:
            break
        color = colors.get(region.stencil_value, (120, 120, 120))
        draw.rectangle((ids.shape[1] + 18, y + 4, ids.shape[1] + 46, y + 32), fill=color)
        label = (
            f"{region.stencil_value}: {region.name}\n"
            f"area={region.area:,} px | components={region.component_count}"
        )
        draw.multiline_text((ids.shape[1] + 58, y), label, font=font, fill="white", spacing=3)
    color_path = destination / "colorized_mask_with_legend.png"
    full.save(color_path)
    saved.append(color_path)

    # Filled overlay and contours for all mapped IDs, labeled at each mask
    # centroid. The full-ID aggregate bbox is not drawn as a part rectangle.
    overlay = np.asarray(cad).copy()
    for region in regions:
        color = np.asarray(colors.get(region.stencil_value, (255, 255, 255)), dtype=np.float32)
        inside = region.binary_mask
        overlay[inside] = np.clip(0.62 * overlay[inside] + 0.38 * color, 0, 255).astype(np.uint8)
    contour_image = Image.fromarray(overlay)
    contour_array = np.array(contour_image)
    for region in regions:
        color = colors.get(region.stencil_value, (255, 255, 255))
        contours, _ = cv2.findContours(region.binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(contour_array, contours, -1, color, 2)
    contour_image = Image.fromarray(contour_array)
    contour_draw = ImageDraw.Draw(contour_image)
    for region in regions:
        color = colors.get(region.stencil_value, (255, 255, 255))
        if region.centroid is not None:
            cx, cy = (int(value) for value in region.centroid)
            label = str(region.stencil_value)
            contour_draw.text((cx + 5, cy + 5), label, font=font, fill="white", stroke_width=2, stroke_fill="black")
    contour_path = destination / "cad_part_region_contours.png"
    contour_image.save(contour_path)
    saved.append(contour_path)

    # Individual views keep the source CAD context and show exact-ID pixels.
    for region in regions:
        if region.bbox is None:
            crop = cad.copy()
            note = "NO PIXELS IN MASK — geometry unavailable"
        else:
            x0, y0, x1, y1 = region.bbox
            pad = max(18, int(round(max(x1 - x0, y1 - y0) * 0.04)))
            crop_box = (max(0, x0 - pad), max(0, y0 - pad), min(cad.width, x1 + pad), min(cad.height, y1 + pad))
            crop = cad.crop(crop_box)
            px0, py0, px1, py1 = crop_box
            local_mask = region.binary_mask[py0:py1, px0:px1]
            local_color = np.zeros((py1 - py0, px1 - px0, 3), dtype=np.uint8)
            local_color[local_mask] = colors.get(region.stencil_value, (255, 255, 255))
            local_array = np.asarray(crop).astype(np.float32)
            selected = local_mask
            local_array[selected] = 0.35 * local_array[selected] + 0.65 * local_color[selected]
            crop_array = np.clip(local_array, 0, 255).astype(np.uint8)
            local_contours, _ = cv2.findContours(local_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(crop_array, local_contours, -1, colors.get(region.stencil_value, (255, 255, 255)), 2)
            crop = Image.fromarray(crop_array)
            note = (
                f"area={region.area:,} px | aggregate bbox={region.bbox} | "
                f"centroid={region.centroid}\ncomponents={region.component_count} | "
                f"largest={region.largest_component_area:,} px"
            )
        view = Image.new("RGB", (1500, 1050), "#15191f")
        view_draw = ImageDraw.Draw(view)
        color = colors.get(region.stencil_value, (255, 255, 255))
        view_draw.text((24, 16), f"ID {region.stencil_value}: {mapping_names.get(region.stencil_value, region.name)}", font=title_font, fill=color)
        crop.thumbnail((1450, 900))
        if region.bbox is not None and max(crop.size) < 600:
            scale = min(4, 600 // max(crop.size))
            if scale > 1:
                crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
        view.paste(crop, ((1500 - crop.width) // 2, 60))
        view_draw.multiline_text((24, 970), note, font=font, fill="white", spacing=4)
        safe_name = "_".join("".join(ch for ch in region.name if ch.isalnum() or ch in "-_ ").split())[:64]
        part_path = parts_dir / f"id_{region.stencil_value:02d}_{safe_name}.png"
        view.save(part_path)
        saved.append(part_path)
    return saved
