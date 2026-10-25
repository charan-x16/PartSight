"""Problem 1 regression tests for the defects introduced into the sample test images.

The defect locations were verified by pixel-differencing each test image against
``BG_00.png``; they live only in this test module, never in detection logic.
Skipped when the (git-ignored) sample dataset is not present.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.detector import inspect_case, largest_component_mask
from src.reference_comparison import inspect_with_real_reference

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "AI_sample_CAD"
REFERENCE = "BG_00.png"
# photo -> {mask value: (intended status, defect type)}
INTENDED_DEFECTS = {
    "BG_00_TestImage.png": {10: ("MISSING", "missing_fastener"), 24: ("MISSING", "missing_fastener")},
    "BG_00_TestImage2.png": {10: ("PRESENT", "rotated_fastener"), 24: ("PRESENT", "rotated_fastener")},
    "BG_00_TestImage3.png": {25: ("MISSING", "missing_bolt_nut")},
}

pytestmark = pytest.mark.skipif(
    not all((CASE_DIR / name).is_file() for name in [REFERENCE, *INTENDED_DEFECTS]),
    reason="Sample dataset with test images is not available.",
)


@pytest.fixture(scope="module")
def reference_result():
    return inspect_case(CASE_DIR, REFERENCE)


@pytest.fixture(scope="module")
def results(reference_result):
    return {
        photo: inspect_with_real_reference(
            CASE_DIR,
            photo,
            reference_photo=REFERENCE,
            reference_confirmed=True,
            reference_result=reference_result,
        )
        for photo in INTENDED_DEFECTS
    }


@pytest.mark.parametrize("photo", sorted(INTENDED_DEFECTS))
def test_verified_defect_location_matches_labelled_parts(photo, reference_result):
    """The edited pixels cover exactly the labelled parts (>= half of each core)."""
    baseline = np.asarray(Image.open(CASE_DIR / REFERENCE).convert("RGB")).astype(int)
    edited = np.asarray(Image.open(CASE_DIR / photo).convert("RGB")).astype(int)
    changed = cv2.morphologyEx(
        (np.abs(baseline - edited).max(axis=2) > 20).astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((15, 15), np.uint8),
    )
    images = reference_result["images"]
    ids = images["mask_ids"]
    changed_cad = cv2.warpPerspective(
        changed, images["matrix_photo_to_cad"], (ids.shape[1], ids.shape[0]), flags=cv2.INTER_NEAREST
    ) > 0
    covered = set()
    for value in np.unique(ids):
        core = largest_component_mask(ids == value)
        if value and core.sum() >= 400 and (core & changed_cad).sum() >= 0.5 * core.sum():
            covered.add(int(value))
    assert covered == set(INTENDED_DEFECTS[photo])


@pytest.mark.parametrize(
    ("photo", "value"),
    [(photo, value) for photo, defects in INTENDED_DEFECTS.items() for value in defects],
)
def test_intended_defect_outcome(photo, value, results):
    row = next(part for part in results[photo]["parts"] if part["pixel_value"] == value)
    intended, defect = INTENDED_DEFECTS[photo][value]
    expected = "ROTATED" if defect == "rotated_fastener" else intended
    assert row["inspection_outcome"] == expected, row["diagnostic"]
    if expected == "ROTATED":
        # Present parts keep a PRESENT presence status; the rotation found must
        # be the one verified by pixel analysis (70 deg clockwise = 290 deg CCW).
        assert row["status"] == "PRESENT"
        assert abs(row["reference_rotation_deg"] - 290.0) <= 5.0


@pytest.mark.parametrize("photo", sorted(INTENDED_DEFECTS))
def test_unchanged_parts_are_not_flagged(photo, results):
    flagged = [
        (part["pixel_value"], part["status"])
        for part in results[photo]["parts"]
        if part["pixel_value"] not in INTENDED_DEFECTS[photo]
        and part["inspection_outcome"] in {"MISSING", "ROTATED"}
    ]
    assert flagged == []


def test_reference_is_verified_as_aligned_for_every_test_image(results):
    for photo, result in results.items():
        assert result["reference"]["alignment_verified"], photo
