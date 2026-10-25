"""Loading and resolving real photos, CAD images, part masks, and part lists."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# "<name>_Real.jpeg" with "<name>_CAD.png", "<name>_Mask.png", "<name>_PartNamesToPixelMap.json".
_REAL_PHOTO = re.compile(r"^(?P<prefix>.+?)_real(?:[_\- ].*)?$", re.IGNORECASE)
_NAME_KEYS = ("Name", "name", "part_name", "PartName", "part")
_VALUE_KEYS = ("StencilValue", "stencil_value", "pixel_value", "PixelValue", "pixel", "value", "color", "colour")


@dataclass(frozen=True)
class ImageInfo:
    """Metadata and pixels loaded from an image file."""

    path: Path
    format: str
    mode: str
    width: int
    height: int
    channels: int
    dtype: str
    array: np.ndarray

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "mode": self.mode,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class PartMapping:
    """One JSON part-name to mask-value association."""

    name: str
    stencil_value: int
    node_id: int | str | None = None
    # Set when the JSON gives the part's mask colour instead of a scalar ID;
    # ``stencil_value`` is then an index assigned in JSON order.
    color: tuple[int, int, int] | None = None


def load_image(path: str | Path) -> ImageInfo:
    """Load an image without changing its native mode or data type."""

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.load()
            array = np.asarray(image).copy()
            image_format = image.format or image_path.suffix.lstrip(".").upper()
            mode = image.mode
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image '{image_path}': {exc}") from exc

    channels = 1 if array.ndim == 2 else int(array.shape[2])
    return ImageInfo(
        path=image_path,
        format=image_format,
        mode=mode,
        width=width,
        height=height,
        channels=channels,
        dtype=str(array.dtype),
        array=array,
    )


def _part_value(value: Any, where: str) -> int | tuple[int, int, int]:
    if isinstance(value, bool):
        raise ValueError(f"{where} has an invalid pixel value {value!r}")
    if isinstance(value, int) and value >= 0:
        return value
    if (
        isinstance(value, (list, tuple)) and len(value) in (3, 4)
        and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255 for v in value)
    ):
        return tuple(value[:3])
    raise ValueError(f"{where} has an invalid pixel value {value!r}")


def _parse_parts(entries: Any, where: str) -> list[PartMapping]:
    """Parse ``[{name, value}, ...]`` or ``{name: value}`` into part mappings."""
    if isinstance(entries, dict):
        pairs = [(name, value, None) for name, value in entries.items()]
    elif isinstance(entries, list):
        pairs = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {index} in {where} must be an object")
            name = next((entry[key] for key in _NAME_KEYS if key in entry), None)
            value = next((entry[key] for key in _VALUE_KEYS if key in entry), None)
            pairs.append((name, value, entry.get("NodeId", entry.get("node_id"))))
    else:
        raise ValueError(f"{where} must be a list of parts or a name-to-value object")
    values = []
    for index, (name, value, _) in enumerate(pairs):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Entry {index} in {where} has an invalid part name")
        values.append(_part_value(value, f"Entry {index} ('{name}') in {where}"))
    colours = [isinstance(value, tuple) for value in values]
    if any(colours) and not all(colours):
        raise ValueError(f"{where} mixes scalar IDs and RGB colours")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate pixel value in {where}")
    if values and all(colours):
        return [
            PartMapping(name.strip(), index + 1, node_id, value)
            for index, ((name, _, node_id), value) in enumerate(zip(pairs, values))
        ]
    return [PartMapping(name.strip(), value, node_id) for (name, _, node_id), value in zip(pairs, values)]


def load_part_mapping(path: str | Path) -> dict[str, list[PartMapping]]:
    """Load a part list JSON as ``{mask file name or "": [PartMapping, ...]}``.

    Accepted layouts: ``{"<mask file>": [{"Name": ..., "StencilValue": ...}]}``
    (the sample data), ``{"<part name>": <value>}``, a plain list of part
    objects, or ``{"parts": [...]}``. A value is a scalar mask ID or an RGB
    colour ``[r, g, b]``.
    """

    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON mapping file does not exist: {json_path}")
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{json_path}': {exc}") from exc
    if isinstance(raw, dict) and isinstance(raw.get("parts"), (list, dict)) and len(raw) == 1:
        raw = raw["parts"]
    if not raw:
        raise ValueError(f"No parts in mapping JSON '{json_path}'")
    if isinstance(raw, dict) and all(isinstance(entries, list) and all(isinstance(e, dict) for e in entries) for entries in raw.values()):
        return {str(mask_name): _parse_parts(entries, f"'{mask_name}'") for mask_name, entries in raw.items()}
    return {"": _parse_parts(raw, f"'{json_path.name}'")}


@dataclass(frozen=True)
class CaseFiles:
    """Files that belong to one real photo: CAD image, part mask, and part list."""

    photo: Path
    cad: Path
    mask: Path
    mapping: Path
    mapping_group: str


def is_real_photo(path: Path) -> bool:
    """``BG_*`` photos (sample data layout) or ``<name>_Real*`` photos."""
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return False
    return path.stem.upper().startswith("BG_") or bool(_REAL_PHOTO.match(path.stem))


def discover_photos(case_dir: str | Path) -> list[Path]:
    return sorted(path for path in Path(case_dir).iterdir() if is_real_photo(path))


def resolve_case_files(case_dir: str | Path, photo_name: str | Path) -> CaseFiles:
    """Find the CAD image, mask, and part list for a real photo.

    ``<name>_Real*.<ext>`` uses ``<name>_CAD.*``, ``<name>_Mask.*`` and
    ``<name>_PartNamesToPixelMap.json`` from the same folder. Otherwise the
    folder layout of the sample data applies: the folder's JSON, the mask it
    names (or the folder's only ``*mask*`` file), and ``FG_*`` as the CAD image.
    """
    directory = Path(case_dir).resolve()
    photo = directory / photo_name
    match = _REAL_PHOTO.match(photo.stem)
    if match and not photo.stem.upper().startswith("BG_"):
        prefix = match["prefix"].lower()
        files = [path for path in directory.iterdir() if path.is_file()]

        def named(kind: str, suffixes: set[str]) -> Path:
            found = [p for p in files if p.stem.lower() == f"{prefix}_{kind.lower()}" and p.suffix.lower() in suffixes]
            if len(found) != 1:
                raise FileNotFoundError(f"Expected one '{match['prefix']}_{kind}' file next to {photo.name} in {directory}")
            return found[0]

        return CaseFiles(
            photo, named("CAD", SUPPORTED_IMAGE_SUFFIXES), named("Mask", SUPPORTED_IMAGE_SUFFIXES),
            named("PartNamesToPixelMap", {".json"}), "",
        )
    json_paths = sorted(directory.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON part list found in {directory}")
    groups = load_part_mapping(json_paths[0])
    group = next(iter(groups))
    mask = directory / group if group else None
    if mask is None or not mask.is_file():
        candidates = sorted(p for p in directory.iterdir() if p.is_file() and "mask" in p.stem.lower())
        if len(candidates) != 1:
            raise FileNotFoundError(f"Cannot uniquely resolve the part mask in {directory}")
        mask = candidates[0]
    cad = next((p for p in sorted(directory.glob("FG_*.*")) if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES), None)
    if cad is None:
        raise FileNotFoundError(f"No FG_* CAD image in {directory}")
    return CaseFiles(photo, cad, mask, json_paths[0], group)


def mask_part_ids(mask: ImageInfo, mappings: list[PartMapping]) -> np.ndarray:
    """Scalar part-ID image from a mask: one ID channel, or exact RGB colours."""
    array = mask.array
    if mappings and mappings[0].color is not None:
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("The part list gives RGB colours, but the mask is not a colour image")
        ids = np.zeros(array.shape[:2], np.uint16 if len(mappings) > 255 else np.uint8)
        rgb = array[..., :3]
        for part in mappings:
            ids[np.all(rgb == part.color, axis=-1)] = part.stencil_value
        return ids
    if array.ndim == 2:
        return array
    channel = inspect_mask(mask)["selected_channel_index"]
    if channel is None:
        raise ValueError("Mask channels are ambiguous; refusing to guess part IDs.")
    return array[..., int(channel)]


def _channel_names(mode: str | None, count: int) -> list[str]:
    if mode and len(mode) == count and mode in {"RGB", "RGBA", "CMYK", "LAB", "HSV"}:
        return list(mode)
    if count == 1:
        return ["gray"]
    return [f"channel_{index}" for index in range(count)]


def _value_counts(values: np.ndarray) -> dict[int | float, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        value.item(): int(count)
        for value, count in zip(unique, counts, strict=True)
    }


def _pixel_vector_counts(mask: np.ndarray) -> dict[str, int]:
    pixels = mask.reshape(-1, mask.shape[2])
    if np.issubdtype(mask.dtype, np.integer) and mask.min() >= 0 and mask.max() <= 255:
        packed = np.zeros(pixels.shape[0], dtype=np.uint64)
        for index in range(pixels.shape[1]):
            packed |= pixels[:, index].astype(np.uint64) << (8 * index)
        unique, counts = np.unique(packed, return_counts=True)
        result: dict[str, int] = {}
        for packed_value, count in zip(unique, counts, strict=True):
            number = int(packed_value)
            vector = tuple((number >> (8 * index)) & 255 for index in range(mask.shape[2]))
            result[",".join(map(str, vector))] = int(count)
        return result

    unique_rows, counts = np.unique(pixels, axis=0, return_counts=True)
    return {
        ",".join(map(str, row.tolist())): int(count)
        for row, count in zip(unique_rows, counts, strict=True)
    }


def inspect_mask(mask: ImageInfo | np.ndarray, mode: str | None = None) -> dict[str, Any]:
    """Inspect scalar and per-channel mask encodings without assuming a channel."""

    if isinstance(mask, ImageInfo):
        array = mask.array
        mode = mask.mode
    else:
        array = np.asarray(mask)
    if array.ndim not in {2, 3}:
        raise ValueError(f"A mask must be 2D or 3D, received shape {array.shape}")
    if array.size == 0:
        raise ValueError("A mask cannot be empty")

    height, width = array.shape[:2]
    channel_count = 1 if array.ndim == 2 else int(array.shape[2])
    names = _channel_names(mode, channel_count)
    channel_arrays = [array] if array.ndim == 2 else [array[..., i] for i in range(channel_count)]
    channel_stats = [
        {"index": index, "name": names[index], "value_counts": _value_counts(values)}
        for index, values in enumerate(channel_arrays)
    ]

    if array.ndim == 2:
        selected_index: int | None = 0
        selection_reason = "single-channel mask"
    else:
        informative = [i for i, values in enumerate(channel_arrays) if np.unique(values).size > 1]
        if len(informative) == 1:
            selected_index = informative[0]
            selection_reason = "only channel containing more than one value"
        elif all(np.array_equal(channel_arrays[0], values) for values in channel_arrays[1:]):
            selected_index = 0
            selection_reason = "all channels contain identical values"
        else:
            selected_index = None
            selection_reason = "multiple channels vary; no scalar ID channel can be inferred safely"

    selected_counts = (
        channel_stats[selected_index]["value_counts"] if selected_index is not None else None
    )
    return {
        "width": width,
        "height": height,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "channels": channel_count,
        "channel_stats": channel_stats,
        "pixel_vector_counts": (
            _pixel_vector_counts(array) if array.ndim == 3 else None
        ),
        "selected_channel_index": selected_index,
        "selected_channel_name": names[selected_index] if selected_index is not None else None,
        "selection_reason": selection_reason,
        "selected_value_counts": selected_counts,
    }


def discover_case_directories(dataset_root: str | Path) -> list[Path]:
    """Return directories containing mapping JSON files."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")
    return sorted({path.parent for path in root.rglob("*.json")})
