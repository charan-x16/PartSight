from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.data_inspection import (
    PartMapping,
    compare_mapping_to_mask,
    inspect_mask,
    load_image,
    load_part_mapping,
)


def test_load_valid_mapping(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps(
            {"mask.png": [{"NodeId": 7, "Name": "washer", "StencilValue": 3}]}
        ),
        encoding="utf-8",
    )

    mapping = load_part_mapping(path)

    assert mapping["mask.png"] == [PartMapping("washer", 3, 7)]


def test_missing_mapping_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_part_mapping(tmp_path / "missing.json")


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_part_mapping(path)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"mask.png": "not-a-list"},
        {"mask.png": [{"Name": "part", "StencilValue": "1"}]},
        {"mask.png": [{"Name": "", "StencilValue": 1}]},
    ],
)
def test_invalid_mapping_structure_raises(tmp_path, payload):
    path = tmp_path / "bad-structure.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_part_mapping(path)


def test_load_image_preserves_mode_shape_and_dtype(tmp_path):
    path = tmp_path / "image.png"
    source = np.zeros((4, 6, 4), dtype=np.uint8)
    Image.fromarray(source).save(path)

    loaded = load_image(path)

    assert (loaded.width, loaded.height) == (6, 4)
    assert loaded.mode == "RGBA"
    assert loaded.channels == 4
    assert loaded.array.shape == (4, 6, 4)
    assert loaded.dtype == "uint8"


def test_single_channel_mask_counts_values():
    mask = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.uint8)

    result = inspect_mask(mask)

    assert result["selected_channel_name"] == "gray"
    assert result["selected_value_counts"] == {0: 1, 1: 2, 2: 3}


def test_multichannel_mask_finds_only_informative_channel():
    mask = np.zeros((2, 3, 3), dtype=np.uint8)
    mask[..., 1] = np.array([[0, 4, 4], [7, 7, 7]])

    result = inspect_mask(mask, mode="RGB")

    assert result["selected_channel_index"] == 1
    assert result["selected_channel_name"] == "G"
    assert result["selected_value_counts"] == {0: 1, 4: 2, 7: 3}
    assert result["pixel_vector_counts"] == {
        "0,0,0": 1,
        "0,4,0": 2,
        "0,7,0": 3,
    }


def test_ambiguous_multichannel_mask_is_not_guessed():
    mask = np.zeros((2, 2, 3), dtype=np.uint8)
    mask[..., 0] = [[0, 1], [2, 3]]
    mask[..., 1] = [[3, 2], [1, 0]]

    result = inspect_mask(mask, mode="RGB")

    assert result["selected_channel_index"] is None
    assert result["selected_value_counts"] is None


def test_compare_mapping_reports_absent_and_unmapped_values():
    mask_stats = inspect_mask(np.array([[0, 1], [1, 5]], dtype=np.uint8))
    mappings = [PartMapping("present", 1), PartMapping("absent", 2)]

    result = compare_mapping_to_mask(mappings, mask_stats)

    assert result["absent_mapped_parts"] == [
        {"name": "absent", "stencil_value": 2, "node_id": None}
    ]
    assert result["unmapped_mask_values"] == [5]
