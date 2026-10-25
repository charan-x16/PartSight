from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.data_inspection import (
    PartMapping,
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


@pytest.mark.parametrize(
    "payload",
    [
        {"washer_1": 3, "nut_1": 7},
        [{"name": "washer_1", "pixel_value": 3}, {"name": "nut_1", "pixel_value": 7}],
        {"parts": [{"Name": "washer_1", "StencilValue": 3}, {"Name": "nut_1", "StencilValue": 7}]},
    ],
)
def test_flat_and_list_part_lists_are_accepted(tmp_path, payload):
    path = tmp_path / "Sample_PartNamesToPixelMap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    mapping = load_part_mapping(path)

    assert [(part.name, part.stencil_value) for part in mapping[""]] == [("washer_1", 3), ("nut_1", 7)]


def test_rgb_part_colours_become_ids_in_the_mask(tmp_path):
    from src.data_inspection import load_image, mask_part_ids

    path = tmp_path / "Sample_PartNamesToPixelMap.json"
    path.write_text(json.dumps({"washer_1": [255, 0, 0], "nut_1": [0, 0, 255]}), encoding="utf-8")
    parts = load_part_mapping(path)[""]
    mask = np.zeros((4, 4, 3), np.uint8)
    mask[0, 0] = (255, 0, 0)
    mask[3, 3] = (0, 0, 255)
    Image.fromarray(mask).save(tmp_path / "Sample_Mask.png")

    ids = mask_part_ids(load_image(tmp_path / "Sample_Mask.png"), parts)

    assert parts[0].color == (255, 0, 0) and ids[0, 0] == parts[0].stencil_value
    assert ids[3, 3] == parts[1].stencil_value and ids[1, 1] == 0


def test_assignment_file_naming_is_resolved_per_photo(tmp_path):
    from src.data_inspection import discover_photos, resolve_case_files

    for name in ("A_Real.jpeg", "A_CAD.png", "A_Mask.png", "B_Real.jpeg", "B_CAD.png", "B_Mask.png"):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / name)
    for prefix in ("A", "B"):
        (tmp_path / f"{prefix}_PartNamesToPixelMap.json").write_text(json.dumps({"part_1": 1}), encoding="utf-8")

    files = resolve_case_files(tmp_path, "B_Real.jpeg")

    assert [photo.name for photo in discover_photos(tmp_path)] == ["A_Real.jpeg", "B_Real.jpeg"]
    assert (files.cad.name, files.mask.name, files.mapping.name) == ("B_CAD.png", "B_Mask.png", "B_PartNamesToPixelMap.json")
