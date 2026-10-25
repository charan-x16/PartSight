"""End-to-end geometry inspection and UI metric tests on a small synthetic case."""

import json

import cv2
import numpy as np
import pytest
from PIL import Image

from src.data_inspection import PartMapping
from src.detector import AlignmentResult, _annotate, extract_part_regions

HEIGHT, WIDTH = 480, 640
FASTENERS = [(170, 140), (320, 140), (470, 140), (170, 330), (320, 330), (470, 330)]


def _hexagon(center, radius=42, angle=15.0):
    angles = np.deg2rad(np.arange(0, 360, 60) + angle)
    return np.array([[center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)] for a in angles], np.int32)


def _scene(poses):
    """Textured plate with hex fasteners; ``poses`` maps index -> (dx, dy) or None (removed)."""
    rng = np.random.default_rng(7)
    image = np.full((HEIGHT, WIDTH, 3), 40, np.uint8)
    plate = np.clip(120 + cv2.GaussianBlur(rng.normal(0, 60, (HEIGHT, WIDTH)), (0, 0), 8), 0, 255).astype(np.uint8)
    image[40:440, 60:580] = plate[40:440, 60:580, None]
    texture = np.clip(190 + cv2.GaussianBlur(rng.normal(0, 60, (120, 120)), (0, 0), 6), 0, 255).astype(np.uint8)
    for index, center in enumerate(FASTENERS):
        pose = poses.get(index, (0, 0))
        if pose is None:
            continue
        moved = (center[0] + pose[0], center[1] + pose[1])
        hexagon = _hexagon(moved)
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.fillPoly(mask, [hexagon], 1)
        y0, x0 = moved[1] - 60, moved[0] - 60
        patch = np.zeros((HEIGHT, WIDTH), np.uint8)
        patch[y0:y0 + 120, x0:x0 + 120] = texture
        image[mask > 0] = patch[mask > 0, None]
        cv2.polylines(image, [hexagon], True, (20, 20, 20), 3)
        cv2.line(image, moved, (moved[0] + 25, moved[1] - 10), (30, 30, 30), 5)  # asymmetric marking
    return image


def _mask():
    ids = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    ids[40:440, 60:580, 0] = 1
    for index, center in enumerate(FASTENERS):
        channel = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.fillPoly(channel, [_hexagon(center)], 1)
        ids[channel > 0, 0] = index + 2
    return ids


@pytest.fixture(scope="module")
def synthetic_case(tmp_path_factory):
    case = tmp_path_factory.mktemp("geometry_case")
    reference = _scene({})
    Image.fromarray(reference).save(case / "FG_00.png")
    Image.fromarray(reference).save(case / "BG_00.png")
    Image.fromarray(_scene({1: (16, 0), 4: None})).save(case / "BG_01.png")
    Image.fromarray(_mask()).save(case / "Mask_00_00.bmp")
    mapping = {"Mask_00_00.bmp": [{"NodeId": 0, "Name": "plate", "StencilValue": 1}] + [
        {"NodeId": 0, "Name": f"nut_{index}", "StencilValue": index + 2} for index in range(len(FASTENERS))
    ]}
    (case / "0000.json").write_text(json.dumps(mapping), encoding="utf-8")
    return case


@pytest.fixture(scope="module")
def confirmed_result(synthetic_case):
    from src.geometric import inspect_geometry_case

    return inspect_geometry_case(synthetic_case, "BG_01.png", reference_photo="BG_00.png", reference_confirmed=True)


def _part(result, value):
    return next(row for row in result["parts"] if row["pixel_value"] == value)


def test_end_to_end_moved_part_is_out_of_tolerance_in_position(confirmed_result):
    assert confirmed_result["geometry_basis"] == "reference-relative"
    moved = _part(confirmed_result, 3)
    assert moved["status"] == "OUT_OF_TOLERANCE", moved["diagnostic"]
    assert moved["position_result"] == "OUT"
    assert abs(moved["position_offset_dx_px"] - 16) <= 1.5 and abs(moved["position_offset_dy_px"]) <= 1.5


def test_end_to_end_removed_part_is_not_applicable(confirmed_result):
    removed = _part(confirmed_result, 6)
    assert removed["presence_status"] == "MISSING"
    assert removed["status"] == "NOT_APPLICABLE"
    assert removed["position_offset_magnitude_px"] is None


def test_end_to_end_unchanged_parts_are_not_flagged(confirmed_result):
    for value in (2, 4, 5, 7):
        row = _part(confirmed_result, value)
        assert row["status"] == "WITHIN_TOLERANCE", (value, row["diagnostic"])
        assert row["position_offset_magnitude_px"] <= 2.0


def test_end_to_end_axis_counts_use_geometry_fields(confirmed_result):
    assert confirmed_result["axis_counts"]["position_out_of_tolerance"] == 1
    assert confirmed_result["counts"]["NOT_APPLICABLE"] == 1


def test_without_confirmation_geometry_uses_cad_baseline_and_cad_relative_basis(synthetic_case):
    from src.geometric import inspect_geometry_case

    result = inspect_geometry_case(synthetic_case, "BG_01.png", reference_photo="BG_00.png", reference_confirmed=False)

    assert result["geometry_basis"] == "CAD-relative"
    assert result["presence_source"] == "CAD baseline"
    assert all(row["status"] != "WITHIN_TOLERANCE" or row["presence_status"] == "PRESENT" for row in result["parts"])


def test_selected_alignment_method_reaches_geometry_mode(monkeypatch):
    import src.geometric as geometric

    calls = []

    def fake_inspect_case(case_dir, photo_name, *, alignment_method="ecc"):
        calls.append(alignment_method)
        raise RuntimeError("stop after recording the call")

    monkeypatch.setattr(geometric, "inspect_case", fake_inspect_case)
    with pytest.raises(RuntimeError):
        geometric.inspect_geometry_case("unused", "unused.png", alignment_method="loftr")
    assert calls == ["loftr"]


def test_annotations_render_geometry_outcomes_and_violated_axes():
    # Parts sit away from the legend drawn in the top-left corner.
    ids = np.zeros((400, 600), np.uint8)
    ids[220:260, 300:340], ids[220:260, 400:440], ids[320:360, 300:340], ids[320:360, 400:440] = 1, 2, 3, 4
    regions = extract_part_regions(ids, [PartMapping(f"p{v}", v) for v in (1, 2, 3, 4)])
    records = [
        {"status": status, "deviation_axes": axes, "pixel_value": value}
        for value, status, axes in (
            (1, "WITHIN_TOLERANCE", ""), (2, "OUT_OF_TOLERANCE", "orientation,size"),
            (3, "NOT_MEASURABLE", ""), (4, "NOT_APPLICABLE", ""),
        )
    ]
    image = np.zeros((400, 600, 3), np.uint8)

    cad, photo = _annotate(image, image, regions, records, AlignmentResult(True, np.eye(3), "ok"))

    def has(color, box):
        y0, y1, x0, x1 = box
        return (cad[y0:y1, x0:x1] == color).all(axis=-1).any()

    assert cad.shape == image.shape and photo.shape == image.shape
    assert has((20, 185, 70), (215, 265, 295, 345))  # within tolerance: green
    assert has((230, 45, 45), (215, 265, 395, 445))  # out of tolerance: red
    assert has((245, 170, 20), (315, 365, 295, 345))  # not measurable: amber
    assert has((150, 150, 150), (315, 365, 395, 445))  # not applicable: grey
