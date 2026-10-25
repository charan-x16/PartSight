import cv2
import numpy as np
import pytest

from src.detector import result_csv, result_json
from src.geometric import geometry_outcome, measure_local_component_geometry

SIZE = 400
CENTER = (200.0, 200.0)


def _hex_mask(radius=70):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    angles = np.deg2rad(np.arange(0, 360, 60) + 15)
    points = np.array([[CENTER[0] + radius * np.cos(a), CENTER[1] + radius * np.sin(a)] for a in angles], np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


def _washer_mask(radius=75):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(mask, (int(CENTER[0]), int(CENTER[1])), radius, 1, -1)
    return mask


def _photo(mask, dx=0.0, dy=0.0, angle=0.0, scale=1.0, clutter_seed=None, clutter_lines=60):
    """Render the part after a known deviation; angle follows cv2.getRotationMatrix2D (CCW positive)."""
    matrix = cv2.getRotationMatrix2D(CENTER, angle, scale)
    matrix[:, 2] += (dx, dy)
    photo = np.full((SIZE, SIZE), 90, np.uint8)
    if clutter_seed is not None:
        rng = np.random.default_rng(clutter_seed)
        for _ in range(clutter_lines):
            p = rng.integers(0, SIZE, 4)
            cv2.line(photo, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 140, 2)
    warped = cv2.warpAffine(mask * 255, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    photo[warped > 0] = 200
    return np.dstack([photo] * 3)


def _measure(mask, photo, **kwargs):
    return measure_local_component_geometry(mask, photo, np.eye(3), **kwargs)


def test_known_position_rotation_and_scale_are_recovered_at_fine_resolution():
    mask = _hex_mask()
    result = _measure(mask, _photo(mask, dx=7, dy=-5, angle=6.0, scale=1.06))

    assert result["measurable"], result
    assert abs(result["position_offset_dx_px"] - 7) <= 1
    assert abs(result["position_offset_dy_px"] + 5) <= 1
    assert abs(result["orientation_difference_deg"] - 6.0) <= 1.0
    # Mask contours lie on pixel centres while photo edges lie between pixels,
    # so a bias of about one scale step (0.01) is expected.
    assert abs(result["size_ratio"] - 1.06) <= 0.015
    assert result["position_uncertainty_px"] <= 2.0


@pytest.mark.parametrize("angle", [8.0, -8.0])
def test_rotation_sign_convention_is_counter_clockwise_positive(angle):
    mask = _hex_mask()
    result = _measure(mask, _photo(mask, angle=angle))

    assert result["measurable"], result
    assert np.sign(result["orientation_difference_deg"]) == np.sign(angle)
    assert abs(result["orientation_difference_deg"] - angle) <= 1.0


@pytest.mark.parametrize(
    ("deviation", "axis", "expected"),
    [
        ({"dx": 9}, "position_result", "WITHIN"),
        ({"dx": 15}, "position_result", "OUT"),
        ({"dx": 10, "dy": 6}, "position_result", "INCONCLUSIVE"),  # 11.7 px vs 12 px +/- uncertainty
        ({"angle": 3.0}, "orientation_result", "WITHIN"),
        ({"angle": -8.0}, "orientation_result", "OUT"),
        ({"scale": 1.05}, "size_result", "WITHIN"),
        ({"scale": 1.14}, "size_result", "OUT"),
    ],
)
def test_values_inside_outside_and_at_each_tolerance(deviation, axis, expected):
    mask = _hex_mask()
    result = _measure(mask, _photo(mask, **deviation))

    assert result["measurable"], result
    assert result[axis] == expected, result


def test_tolerances_change_decisions_but_not_measurements():
    mask = _hex_mask()
    photo = _photo(mask, dx=9, dy=0)
    strict = _measure(mask, photo, position_tolerance_px=5)
    loose = _measure(mask, photo, position_tolerance_px=30)

    for key in ("position_offset_dx_px", "position_offset_dy_px", "orientation_difference_deg", "size_ratio"):
        assert strict[key] == loose[key]
    assert strict["position_result"] == "OUT" and loose["position_result"] == "WITHIN"


def test_symmetric_washer_rotation_is_not_reported_even_with_clutter():
    mask = _washer_mask()
    for seed in (None, 4):
        result = _measure(mask, _photo(mask, dx=5, dy=-3, angle=8.0, clutter_seed=seed))
        assert result["measurable"], result
        assert result["orientation_result"] == "NOT_OBSERVABLE"
        assert result["orientation_difference_deg"] is None
        assert abs(result["position_offset_dx_px"] - 5) <= 1 and abs(result["position_offset_dy_px"] + 3) <= 1


def test_clutter_never_produces_a_contradicting_definitive_decision():
    truth = {"dx": 14, "dy": 0, "angle": 0.0, "scale": 1.0}
    expected = {"position_result": "OUT", "orientation_result": "WITHIN", "size_result": "WITHIN"}
    decided = 0
    for mask in (_hex_mask(), _washer_mask()):
        for seed in range(6):
            result = _measure(mask, _photo(mask, clutter_seed=seed, **truth))
            if not result["measurable"]:
                continue  # an ambiguous or weak fit must abstain rather than guess
            for axis, wanted in expected.items():
                if result[axis] in {"WITHIN", "OUT"}:
                    decided += 1
                    assert result[axis] == wanted, (seed, axis, result)
    assert decided > 0


def test_mild_clutter_is_measured_accurately():
    mask = _hex_mask()
    result = _measure(mask, _photo(mask, dx=5, dy=-3, clutter_seed=1, clutter_lines=20))

    assert result["measurable"], result
    assert abs(result["position_offset_dx_px"] - 5) <= 1 and abs(result["position_offset_dy_px"] + 3) <= 1


def test_fit_beyond_the_search_range_is_not_measurable():
    mask = _hex_mask()
    result = _measure(mask, _photo(mask, dx=40))

    assert not result["measurable"]


def test_two_equally_good_fits_are_reported_as_ambiguous():
    # Two complete copies of the outline 24 px apart: both poses explain the edges.
    mask = _hex_mask(radius=50)
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    gray = np.full((SIZE, SIZE), 90, np.uint8)
    for shift in (-12, 12):
        cv2.polylines(gray, [contour + np.array([shift, 0])], True, 220, 2)
    photo = np.dstack([gray] * 3)

    result = _measure(mask, photo)

    assert not result["measurable"]
    assert "Ambiguous" in result["reason"]


def test_local_geometry_returns_uncertain_measurement_for_blank_photo():
    mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (30, 30), 1, thickness=-1)
    result = measure_local_component_geometry(mask, np.zeros((40, 40, 3), dtype=np.uint8), np.eye(3))

    assert not result["measurable"]
    assert "few local edges" in result["reason"]


def test_local_geometry_rejects_singular_global_transform():
    mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (30, 30), 1, thickness=-1)
    photo = np.repeat((mask * 255)[..., None], 3, axis=2)

    result = measure_local_component_geometry(mask, photo, np.zeros((3, 3), dtype=np.float32))

    assert not result["measurable"]
    assert "transform is invalid" in result["reason"]


def _measurement(position=("WITHIN", 3.0), orientation=("WITHIN", 1.0), size=("WITHIN", 1.02)):
    return {
        "measurable": True,
        "position_offset_magnitude_px": position[1], "position_uncertainty_px": 1.5, "position_result": position[0],
        "orientation_difference_deg": orientation[1], "orientation_uncertainty_deg": 0.5,
        "orientation_result": orientation[0],
        "size_ratio": size[1], "size_uncertainty": 0.01, "size_result": size[0],
    }


def test_missing_part_geometry_is_not_applicable_never_correct():
    outcome, reason, _ = geometry_outcome("MISSING", _measurement())
    assert outcome == "NOT_APPLICABLE"
    assert "missing" in reason


def test_contour_fit_alone_does_not_establish_presence():
    outcome, reason, _ = geometry_outcome("UNCERTAIN", _measurement())
    assert outcome == "NOT_MEASURABLE"
    assert "cannot establish" in reason


def test_orientation_violation_alone_makes_the_part_out_of_tolerance():
    outcome, reason, axes = geometry_outcome("PRESENT", _measurement(orientation=("OUT", 9.0)))
    assert outcome == "OUT_OF_TOLERANCE"
    assert axes == ["orientation"]
    assert "orientation" in reason


def test_size_and_position_violations_are_listed_independently():
    outcome, _, axes = geometry_outcome("PRESENT", _measurement(position=("OUT", 20.0), size=("OUT", 1.2)))
    assert outcome == "OUT_OF_TOLERANCE"
    assert axes == ["position", "size"]


def test_unobservable_orientation_still_allows_within_tolerance_with_a_note():
    measurement = _measurement(orientation=("NOT_OBSERVABLE", None))
    measurement["orientation_uncertainty_deg"] = None
    outcome, reason, _ = geometry_outcome("PRESENT", measurement)
    assert outcome == "WITHIN_TOLERANCE"
    assert "not observable" in reason


def test_uncertainty_overlapping_a_tolerance_is_not_a_definitive_decision():
    outcome, reason, _ = geometry_outcome("PRESENT", _measurement(position=("INCONCLUSIVE", 11.5)))
    assert outcome == "NOT_MEASURABLE"
    assert "overlaps" in reason


def _textured_photo(mask, dx=0.0, dy=0.0, angle=0.0, scale=1.0):
    """Smooth plate with a textured part carrying an asymmetric mark, moved by a known deviation."""
    rng = np.random.default_rng(11)
    plate = np.clip(110 + cv2.GaussianBlur(rng.normal(0, 50, (SIZE, SIZE)), (0, 0), 8), 0, 255).astype(np.uint8)
    part = np.clip(185 + cv2.GaussianBlur(rng.normal(0, 70, (SIZE, SIZE)), (0, 0), 4), 0, 255).astype(np.uint8)
    cv2.line(part, (200, 200), (240, 170), 40, 7)
    matrix = cv2.getRotationMatrix2D(CENTER, angle, scale)
    matrix[:, 2] += (dx, dy)
    moved_part = cv2.warpAffine(part, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    moved_mask = cv2.warpAffine(mask, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST) > 0
    photo = plate.copy()
    photo[moved_mask] = moved_part[moved_mask]
    return np.dstack([photo] * 3)


def _reference_deviation(mask, reference, target, cad_to_reference=np.eye(3)):
    from src.geometric import measure_reference_deviation

    return measure_reference_deviation(mask, reference, target, cad_to_reference, np.eye(3), registration_uncertainty_px=0.5)


def test_reference_relative_measurement_cancels_the_reference_offset():
    mask = _hex_mask()
    # The known-good part sits 20 px from the CAD outline (systematic offset);
    # the inspected part has moved a further 17 px and turned 6 deg.
    reference = _textured_photo(mask, dx=20, dy=2)
    target = _textured_photo(mask, dx=37, dy=2, angle=6.0)
    cad_to_reference = np.array([[1.0, 0, 20], [0, 1.0, 2], [0, 0, 1.0]])

    result = _reference_deviation(mask, reference, target, cad_to_reference)

    assert result["measurable"], result
    assert abs(result["position_offset_dx_px"] - 17) <= 1.5 and abs(result["position_offset_dy_px"]) <= 1.5
    assert abs(result["orientation_difference_deg"] - 6.0) <= 1.0
    assert abs(result["size_ratio"] - 1.0) <= 0.02
    assert result["position_result"] == "OUT" and result["orientation_result"] == "OUT"


def test_reference_relative_measurement_of_an_unchanged_part_is_near_zero():
    mask = _hex_mask()
    photo = _textured_photo(mask)

    result = _reference_deviation(mask, photo, photo.copy())

    assert result["measurable"], result
    assert result["position_offset_magnitude_px"] <= 1.0
    assert abs(result["orientation_difference_deg"]) <= 0.5 and abs(result["size_ratio"] - 1.0) <= 0.01
    assert (result["position_result"], result["orientation_result"], result["size_result"]) == ("WITHIN",) * 3


def test_reference_relative_measurement_recovers_scale():
    mask = _hex_mask()
    result = _reference_deviation(mask, _textured_photo(mask), _textured_photo(mask, scale=1.14))

    assert result["measurable"], result
    assert abs(result["size_ratio"] - 1.14) <= 0.02
    assert result["size_result"] == "OUT"


def test_reference_relative_keeps_symmetric_orientation_unobservable():
    mask = _washer_mask()
    result = _reference_deviation(mask, _textured_photo(mask), _textured_photo(mask, angle=8))

    assert result["measurable"], result
    assert result["orientation_result"] == "NOT_OBSERVABLE"
    assert result["orientation_difference_deg"] is None


def test_reference_relative_measurement_of_a_removed_part_is_not_measurable():
    mask = _hex_mask()
    reference = _textured_photo(mask)
    removed = _textured_photo(np.zeros_like(mask))

    result = _reference_deviation(mask, reference, removed)

    assert not result["measurable"]


def test_geometric_reports_serialize_geometry_fields():
    result = {
        "mode": "Geometric Validation",
        "parts": [{
            "part_name": "bracket",
            "pixel_value": 7,
            "status": "WITHIN_TOLERANCE",
            "evidence_score": 0.8,
            "cad_bbox_xyxy": [10, 20, 40, 50],
            "position_offset_dx_px": 3.0,
            "position_offset_dy_px": -2.0,
            "size_ratio": 1.05,
            "diagnostic": "local edge match",
        }],
        "images": {"photo": np.zeros((2, 2, 3), dtype=np.uint8)},
    }
    csv_text = result_csv(result)
    json_text = result_json(result)

    assert "position_offset_dx_px" in csv_text
    assert '"size_ratio": 1.05' in json_text
    assert '"images"' not in json_text


@pytest.mark.parametrize("center", [(150.0, 100.0), (8.0, 6.0), (290.0, 195.0), (-20.0, 110.0)])
def test_search_window_lookups_are_bit_identical_to_full_map_lookups(center):
    from src.geometric import (
        ROTATION_SEARCH_DEG, SCALE_SEARCH, TRANSLATION_SEARCH_PX, _bilinear, _rotate, _search_window,
    )

    rng = np.random.default_rng(5)
    full = rng.random((200, 300)).astype(np.float32)
    center = np.array(center)
    centered = rng.normal(0, 25, (300, 2))
    r0, r1, c0, c1 = _search_window(full.shape, center, np.linalg.norm(centered, axis=1).max())
    window = np.ascontiguousarray(full[r0:r1, c0:c1])
    for _ in range(40):
        angle = rng.uniform(-ROTATION_SEARCH_DEG, ROTATION_SEARCH_DEG)
        scale = rng.uniform(*SCALE_SEARCH)
        shift = rng.uniform(-TRANSLATION_SEARCH_PX, TRANSLATION_SEARCH_PX, 2)
        shape = _rotate(centered, angle, scale) + center + shift
        expected, expected_valid = _bilinear(full, shape[:, 0], shape[:, 1])
        values, valid = _bilinear(window, shape[:, 0], shape[:, 1], (r0, c0), full.shape)
        assert np.array_equal(valid, expected_valid)
        assert np.array_equal(values, expected)


def test_windowed_cost_functions_match_full_map_costs():
    import src.geometric as geometric

    rng = np.random.default_rng(9)
    distance = rng.random((240, 320)).astype(np.float32) * 10
    gray, gradient = (rng.random((240, 320)).astype(np.float32) for _ in range(2))
    center = np.array([30.0, 200.0])  # near the image corner so clipping is exercised
    centered = rng.normal(0, 20, (300, 2))
    reference = geometric._normalised(rng.random(300)), geometric._normalised(rng.random(300))
    dxs = np.arange(-28, 29, 4.0)
    dys = np.full_like(dxs, 3.0)
    windowed_outline = geometric._outline_cost(centered, center, distance)
    windowed_appearance = geometric._appearance_cost(*reference, (gray, gradient), centered + center, center)

    original = geometric._search_window
    geometric._search_window = lambda shape, *_: (0, shape[0], 0, shape[1])  # no cropping
    try:
        full_outline = geometric._outline_cost(centered, center, distance)
        full_appearance = geometric._appearance_cost(*reference, (gray, gradient), centered + center, center)
    finally:
        geometric._search_window = original
    for angle, scale in ((0.0, 1.0), (-15.0, 0.8), (12.5, 1.25)):
        for windowed, full in ((windowed_outline, full_outline), (windowed_appearance, full_appearance)):
            cost_w, support_w = windowed(dxs, dys, angle, scale)
            cost_f, support_f = full(dxs, dys, angle, scale)
            assert np.array_equal(cost_w, cost_f) and np.array_equal(support_w, support_f)
