"""Semi-synthetic geometry regression tests built from the confirmed known-good photo.

Cases are generated on the fly from automatically found fastener groups; no
part IDs or coordinates are written here. Abstaining (NOT_MEASURABLE) is
allowed, but a definitive decision must never contradict the ground truth.
Skipped when the (git-ignored) sample dataset is not present.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.detector import inspect_case
from src.geometric import inspect_geometry_case
from src.geometry_validation import TRANSFORMS, fastener_groups, generate_case, ground_truth_rows, score_part

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "AI_sample_CAD"
REFERENCE = "BG_00.png"
TOLERANCES = {"position_tolerance_px": 12.0, "orientation_tolerance_deg": 5.0, "size_tolerance_ratio": 0.10}
SELECTED = ("unchanged", "shift_outside", "rotate_outside_cw", "scale_outside_up", "removed")

pytestmark = pytest.mark.skipif(not (CASE_DIR / REFERENCE).is_file(), reason="Sample dataset is not available.")


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    folder = tmp_path_factory.mktemp("geometry_validation")
    reference = inspect_case(CASE_DIR, REFERENCE)
    images = reference["images"]
    cad_to_photo = np.linalg.inv(images["matrix_photo_to_cad"])
    names = {row["pixel_value"]: row["part_name"] for row in reference["parts"]}
    groups = fastener_groups(images["mask_ids"])
    rows = []
    for index, transform in enumerate(t for t in TRANSFORMS if t["label"] in SELECTED):
        group = groups[index % len(groups)]
        edited, region, applied = generate_case(images["photo"], group, transform, cad_to_photo)
        path = folder / f"{transform['label']}.png"
        Image.fromarray(edited).save(path)
        truth = ground_truth_rows(path.name, transform, index, group, applied, region, images["mask_ids"], names,
                                  cad_to_photo, TOLERANCES)
        result = inspect_geometry_case(CASE_DIR, str(path), reference_photo=REFERENCE, reference_confirmed=True,
                                       reference_result=reference, **TOLERANCES)
        assert result["geometry_basis"] == "reference-relative"
        by_value = {row["pixel_value"]: row for row in result["parts"]}
        rows.extend(score_part(row, by_value[row["pixel_value"]]) for row in truth)
    return rows


def test_unchanged_parts_raise_no_false_alarms(scored):
    alarms = [r for r in scored if r["role"] in {"control", "control_adjacent"} and r["evaluation"] == "FALSE_ALARM"]
    assert alarms == []


def test_unchanged_parts_measure_near_zero_deviation(scored):
    measured = [r for r in scored if r["role"] == "control" and r["position_offset_magnitude_px"] is not None]
    assert measured, "no unchanged part was measurable"
    assert max(r["position_offset_magnitude_px"] for r in measured) <= 2.0


def test_out_of_tolerance_parts_are_never_reported_within_tolerance(scored):
    missed = [r for r in scored if r["role"] == "modified" and r["evaluation"] == "MISSED_DEVIATION"]
    assert missed == []


def test_at_least_one_deviation_per_axis_is_detected(scored):
    detected = {
        axis for r in scored if r["role"] == "modified"
        for axis in ("position", "orientation", "size")
        if r.get(f"expected_{axis}") == "OUT" and r.get(f"{axis}_result") == "OUT"
    }
    assert detected == {"position", "orientation", "size"}


def test_measured_values_agree_with_ground_truth_within_uncertainty(scored):
    for r in scored:
        if r["role"] != "modified" or r.get("error_dx_px") is None:
            continue
        position_error = float(np.hypot(r["error_dx_px"], r["error_dy_px"]))
        assert position_error <= r["position_uncertainty_px"] + 1.0, r
        if r.get("error_rotation_deg") is not None and r["intended_rotation_deg"]:
            assert np.sign(r["orientation_difference_deg"]) == np.sign(float(r["intended_rotation_deg"])), r
            assert abs(r["error_rotation_deg"]) <= r["orientation_uncertainty_deg"] + 0.5, r
        if r.get("error_scale") is not None:
            assert abs(r["error_scale"]) <= r["size_uncertainty"] + 0.01, r


def test_removed_parts_are_not_applicable(scored):
    removed = [r for r in scored if r["role"] == "removed"]
    assert removed and all(r["measured_outcome"] == "NOT_APPLICABLE" for r in removed)
