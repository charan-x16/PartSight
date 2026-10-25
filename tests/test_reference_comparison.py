import cv2
import numpy as np

from src.reference_comparison import (
    classify_reference_similarity,
    combine_reference_and_baseline,
    compare_reference_region,
)


def _texture(seed=21, size=96):
    rng = np.random.default_rng(seed)
    gray = rng.integers(35, 220, (size, size), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return np.repeat(gray[..., None], 3, axis=2)


def test_region_comparison_handles_small_shift_and_illumination_change():
    reference = _texture()
    gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    shifted = cv2.warpAffine(
        gray, np.float32([[1, 0, 4], [0, 1, -2]]), (gray.shape[1], gray.shape[0])
    )
    target = np.repeat(np.clip(shifted.astype(np.float32) * 0.72 + 35, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    mask = np.zeros(gray.shape, dtype=bool)
    mask[24:72, 24:72] = True

    result = compare_reference_region(reference, target, mask, max_shift_px=6, shift_step_px=2)

    assert result["measurable"]
    assert result["similarity"] > 0.80
    assert abs(result["shift_dx_px"]) >= 2
    assert abs(result["shift_dy_px"]) >= 2


def test_empty_recess_like_flat_region_is_dissimilar_to_textured_reference():
    reference = _texture()
    target = np.full_like(reference, 120)
    mask = np.zeros(reference.shape[:2], dtype=bool)
    mask[24:72, 24:72] = True

    result = compare_reference_region(reference, target, mask)
    decision = classify_reference_similarity(
        result["similarity"], alignment_valid=True, reference_confirmed=True
    )

    assert result["similarity"] < 0.20
    assert decision["candidate_status"] == "MISSING"
    assert decision["status"] == "MISSING"


def test_ambiguous_similarity_abstains():
    decision = classify_reference_similarity(
        0.38, alignment_valid=True, reference_confirmed=True
    )

    assert decision["candidate_status"] == "UNCERTAIN"
    assert decision["status"] == "UNCERTAIN"


def test_unconfirmed_reference_keeps_candidate_provisional_and_final_uncertain():
    decision = classify_reference_similarity(
        0.92, alignment_valid=True, reference_confirmed=False
    )

    assert decision["candidate_status"] == "PRESENT"
    assert decision["status"] == "UNCERTAIN"
    assert "completeness has not been confirmed" in decision["reason"]


def test_alignment_failure_always_abstains_even_with_high_similarity():
    decision = classify_reference_similarity(
        0.99, alignment_valid=False, reference_confirmed=True
    )

    assert decision["candidate_status"] == "UNCERTAIN"
    assert decision["status"] == "UNCERTAIN"
    assert "alignment failed" in decision["reason"]


def test_reference_can_resolve_uncertain_baseline_presence():
    status, _ = combine_reference_and_baseline("UNCERTAIN", "PRESENT")
    assert status == "PRESENT"


def test_reference_absence_alone_cannot_override_uncertain_or_present_baseline():
    uncertain_status, _ = combine_reference_and_baseline("UNCERTAIN", "MISSING")
    present_status, _ = combine_reference_and_baseline("PRESENT", "MISSING")
    assert uncertain_status == "UNCERTAIN"
    assert present_status == "UNCERTAIN"


def test_missing_requires_agreement_between_reference_and_baseline():
    status, _ = combine_reference_and_baseline("MISSING", "MISSING")
    assert status == "MISSING"


def test_no_reference_uses_conservative_fallback_without_cross_case_substitution(tmp_path):
    # The main-case reference is never imported here. With no same-case choice,
    # this function returns the supplied case's baseline records unchanged.
    from src.reference_comparison import inspect_with_real_reference

    baseline = {
        "case": "TestCase2",
        "photo": "BG_00.png",
        "cad": "FG_00.png",
        "mask": "Mask_00.bmp",
        "alignment": {"success": True},
        "parts": [{"part_name": "bolt", "pixel_value": 3, "status": "UNCERTAIN", "diagnostic": "baseline abstained"}],
        "counts": {"PRESENT": 0, "MISSING": 0, "UNCERTAIN": 1},
        "images": {
            "photo": np.zeros((8, 8, 3), dtype=np.uint8),
            "cad": np.zeros((8, 8, 3), dtype=np.uint8),
            "mask": np.zeros((8, 8), dtype=np.uint8),
            "aligned": np.zeros((8, 8, 3), dtype=np.uint8),
            "photo_annotated": np.zeros((8, 8, 3), dtype=np.uint8),
            "cad_annotated": np.zeros((8, 8, 3), dtype=np.uint8),
        },
    }

    result = inspect_with_real_reference(
        tmp_path, "BG_00.png", reference_photo=None, base_result=baseline
    )

    assert result["reference"]["available"] is False
    assert result["parts"][0]["status"] == "UNCERTAIN"
    assert "No separate same-case" in result["reference"]["warning"]


def _fastener_like(size=161):
    """Asymmetric disc pattern: rotation changes its appearance."""
    image = np.full((size, size, 3), 90, dtype=np.uint8)
    center = size // 2
    cv2.circle(image, (center, center), 50, (200, 200, 200), -1)
    cv2.rectangle(image, (center - 8, center - 45), (center + 8, center), (30, 30, 30), -1)
    cv2.circle(image, (center + 25, center + 20), 9, (250, 250, 250), -1)
    mask = np.zeros((size, size), dtype=bool)
    cv2.circle(mask.view(np.uint8), (center, center), 52, 1, -1)
    return image, mask


def test_rotation_search_recognises_present_but_rotated_part():
    from src.reference_comparison import best_rotated_reference_similarity

    reference, mask = _fastener_like()
    center = (reference.shape[1] // 2, reference.shape[0] // 2)
    rotated = cv2.warpAffine(
        reference, cv2.getRotationMatrix2D(center, 70.0, 1.0), reference.shape[1::-1],
        borderMode=cv2.BORDER_REFLECT,
    )

    unrotated = compare_reference_region(reference, rotated, mask)
    result = best_rotated_reference_similarity(reference, rotated, mask)

    assert unrotated["similarity"] < 0.55
    assert result["similarity"] >= 0.9
    assert abs(result["rotation_deg"] - 70.0) <= 5.0


def test_rotation_search_does_not_explain_an_obliterated_part():
    from src.reference_comparison import best_rotated_reference_similarity

    reference, mask = _fastener_like()
    covered = reference.copy()
    rng = np.random.default_rng(3)
    for _ in range(40):
        p0 = tuple(int(v) for v in rng.integers(30, 130, 2))
        p1 = tuple(int(v) for v in rng.integers(30, 130, 2))
        cv2.line(covered, p0, p1, (70, 55, 40), 7)

    result = best_rotated_reference_similarity(reference, covered, mask)

    assert result["similarity"] <= 0.20


def test_reference_only_absence_overrides_baseline_only_when_enabled():
    assert combine_reference_and_baseline("PRESENT", "MISSING")[0] == "UNCERTAIN"
    status, reason = combine_reference_and_baseline("PRESENT", "MISSING", reference_only_absence=True)
    assert status == "MISSING"
    assert "any tested shift or rotation" in reason
    # A reference-supported presence still never overrides baseline absence.
    assert combine_reference_and_baseline("MISSING", "PRESENT", reference_only_absence=True)[0] == "UNCERTAIN"


def test_rotation_search_uses_axis_of_partly_occluded_ring():
    """A washer ring partly hidden by a nut has its centroid off the rotation axis."""
    from src.reference_comparison import best_rotated_reference_similarity

    size, center = 241, (120, 120)
    reference = np.full((size, size, 3), 80, dtype=np.uint8)
    cv2.circle(reference, center, 90, (190, 190, 190), -1)
    cv2.circle(reference, center, 40, (80, 80, 80), -1)
    cv2.ellipse(reference, center, (70, 70), 0, 200, 260, (250, 250, 250), 9)
    cv2.circle(reference, (150, 110), 8, (30, 30, 30), -1)
    ring = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(ring, center, 92, 1, -1)
    cv2.circle(ring, center, 38, 0, -1)
    ring[:, 150:] = 0  # occluded side: the remaining region is a crescent
    mask = ring.astype(bool)
    ys, xs = np.nonzero(mask)
    assert np.hypot(xs.mean() - center[0], ys.mean() - center[1]) > 20
    rotated = cv2.warpAffine(
        reference, cv2.getRotationMatrix2D(center, -70.0, 1.0), (size, size), borderMode=cv2.BORDER_REFLECT
    )

    result = best_rotated_reference_similarity(reference, rotated, mask)

    assert result["similarity"] >= 0.8
    assert abs(result["rotation_deg"] - 290.0) <= 5.0
    assert np.hypot(result["rotation_center_xy"][0] - center[0], result["rotation_center_xy"][1] - center[1]) <= 3
