"""Dataset discovery and validation utilities for PartSight Phase 1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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


def load_part_mapping(path: str | Path) -> dict[str, list[PartMapping]]:
    """Load and validate the supplied mask-to-parts JSON mapping."""

    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON mapping file does not exist: {json_path}")
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{json_path}': {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise ValueError("Mapping JSON must be a non-empty object keyed by mask filename")

    result: dict[str, list[PartMapping]] = {}
    for mask_name, entries in raw.items():
        if not isinstance(mask_name, str) or not mask_name.strip():
            raise ValueError("Every mapping key must be a non-empty mask filename")
        if not isinstance(entries, list):
            raise ValueError(f"Mapping for '{mask_name}' must be a list")
        parsed: list[PartMapping] = []
        seen_values: set[int] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {index} for '{mask_name}' must be an object")
            name = entry.get("Name")
            value = entry.get("StencilValue")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Entry {index} for '{mask_name}' has an invalid Name")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Entry {index} for '{mask_name}' has an invalid StencilValue"
                )
            if value in seen_values:
                raise ValueError(f"Duplicate StencilValue {value} for '{mask_name}'")
            seen_values.add(value)
            parsed.append(PartMapping(name.strip(), value, entry.get("NodeId")))
        result[mask_name] = parsed
    return result


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


def compare_mapping_to_mask(
    mappings: Iterable[PartMapping], mask_inspection: dict[str, Any]
) -> dict[str, Any]:
    """Compare mapped scalar IDs with the safely inferred mask channel."""

    counts = mask_inspection.get("selected_value_counts")
    if counts is None:
        raise ValueError("Mask inspection did not identify an unambiguous scalar ID channel")
    mask_values = {int(value) for value in counts}
    entries = list(mappings)
    mapped_values = {entry.stencil_value for entry in entries}
    absent = [entry for entry in entries if entry.stencil_value not in mask_values]
    return {
        "mapped_values": sorted(mapped_values),
        "mask_values": sorted(mask_values),
        "absent_mapped_parts": [asdict(entry) for entry in absent],
        "unmapped_mask_values": sorted(mask_values - mapped_values - {0}),
        "background_value": 0 if 0 in mask_values else None,
    }


def _classify_image(path: Path, resolved_masks: set[Path]) -> str:
    if path.resolve() in resolved_masks:
        return "part mask"
    name = path.stem.lower()
    if name.startswith("fg_"):
        return "CAD reference image"
    if name.startswith("bg_") and "testimage" in name:
        return "additional real-world test photograph"
    if name.startswith("bg_"):
        return "real-world photograph"
    return "unclassified image"


def _resolve_mask(case_dir: Path, declared_name: str) -> tuple[Path, list[str]]:
    declared = case_dir / declared_name
    if declared.is_file():
        return declared, []
    candidates = sorted(
        path
        for path in case_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and ("mask" in path.stem.lower() or path.suffix.lower() == ".bmp")
    )
    if len(candidates) == 1:
        warning = (
            f"JSON declares mask '{declared_name}', but it is absent; resolved to the only "
            f"mask candidate '{candidates[0].name}'."
        )
        return candidates[0], [warning]
    names = ", ".join(path.name for path in candidates) or "none"
    raise FileNotFoundError(
        f"Declared mask '{declared_name}' is absent in {case_dir}; "
        f"unambiguous fallback is impossible (candidates: {names})"
    )


def discover_case_directories(dataset_root: str | Path) -> list[Path]:
    """Return directories containing mapping JSON files."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")
    return sorted({path.parent for path in root.rglob("*.json")})


def inspect_dataset(dataset_root: str | Path) -> dict[str, Any]:
    """Discover and inspect all JSON-defined dataset cases under a root."""

    root = Path(dataset_root).resolve()
    case_dirs = discover_case_directories(root)
    if not case_dirs:
        raise ValueError(f"No JSON mapping files found under dataset root: {root}")

    cases: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        json_files = sorted(case_dir.glob("*.json"))
        case_result: dict[str, Any] = {
            "name": "." if case_dir == root else case_dir.relative_to(root).as_posix(),
            "directory": str(case_dir),
            "json_files": [],
            "images": [],
            "warnings": [],
        }
        resolved_masks: set[Path] = set()
        mask_records: list[dict[str, Any]] = []
        for json_path in json_files:
            mapping_groups = load_part_mapping(json_path)
            json_record: dict[str, Any] = {"path": str(json_path), "masks": []}
            for declared_name, mappings in mapping_groups.items():
                mask_path, warnings = _resolve_mask(case_dir, declared_name)
                case_result["warnings"].extend(warnings)
                resolved_masks.add(mask_path.resolve())
                loaded_mask = load_image(mask_path)
                mask_stats = inspect_mask(loaded_mask)
                comparison = compare_mapping_to_mask(mappings, mask_stats)
                record = {
                    "declared_mask_name": declared_name,
                    "resolved_mask_path": str(mask_path),
                    "parts": [asdict(entry) for entry in mappings],
                    "inspection": mask_stats,
                    "comparison": comparison,
                }
                json_record["masks"].append(record)
                mask_records.append(record)
            case_result["json_files"].append(json_record)

        image_paths = sorted(
            path
            for path in case_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        for path in image_paths:
            loaded = load_image(path)
            compatible_masks = [
                Path(record["resolved_mask_path"]).name
                for record in mask_records
                if loaded.width == record["inspection"]["width"]
                and loaded.height == record["inspection"]["height"]
            ]
            metadata = loaded.metadata()
            metadata["role"] = _classify_image(path, resolved_masks)
            metadata["dimension_compatible_masks"] = compatible_masks
            case_result["images"].append(metadata)
        cases.append(case_result)

    return {"dataset_root": str(root), "case_count": len(cases), "cases": cases}
