import numpy as np
import cv2
import json
import csv
import io

from src.data_inspection import PartMapping
from src.detector import (
    AlignmentResult,
    align_masked_edges_ecc,
    align_sift_ransac,
    classify_regions,
    classify_part_measurements,
    extract_part_regions,
    result_csv,
    result_json,
)


def test_extract_part_regions_geometry_and_empty_values():
    mask = np.array(
        [[0, 3, 3, 0], [0, 3, 0, 0], [5, 0, 0, 0]], dtype=np.uint8
    )
    regions = extract_part_regions(
        mask, [PartMapping("washer", 3), PartMapping("absent", 7)]
    )

    assert regions[0].area == 3
    assert regions[0].bbox == (1, 0, 3, 2)
    assert regions[0].centroid == (4 / 3, 1 / 3)
    assert regions[1].area == 0
    assert regions[1].bbox is None
    assert regions[1].centroid is None
    assert not regions[1].binary_mask.any()
    assert regions[0].component_count == 1
    assert regions[0].largest_component_area == 3
    assert regions[1].component_count == 0
    assert regions[1].largest_component_area == 0
    assert regions[0].largest_component_bbox == (1, 0, 3, 2)
    assert regions[0].largest_component_centroid == (4 / 3, 1 / 3)
    assert regions[1].largest_component_bbox is None
    assert regions[1].largest_component_centroid is None


def test_extract_part_region_reports_disconnected_components_and_union_geometry():
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[1:3, 1:3] = 4
    mask[6, 10] = 4

    region = extract_part_regions(mask, [PartMapping("fragmented", 4)])[0]

    assert region.area == 5
    assert region.bbox == (1, 1, 11, 7)
    assert region.centroid == (3.2, 2.4)
    assert region.component_count == 2
    assert region.largest_component_area == 4
    assert region.largest_component_bbox == (1, 1, 3, 3)
    assert region.largest_component_centroid == (1.5, 1.5)


def test_highly_fragmented_geometry_remains_uncertain():
    mask = np.zeros((40, 40), dtype=np.uint8)
    for y in range(2, 38, 4):
        for x in range(2, 38, 4):
            mask[y, x] = 8
    region = extract_part_regions(mask, [PartMapping("fragmented", 8)])
    alignment = AlignmentResult(True, np.eye(3), "ok")
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    result = classify_regions(region, image, image, alignment)[0]

    assert region[0].component_count > 20
    assert result["status"] == "UNCERTAIN"
    assert "highly fragmented" in result["diagnostic"]


def test_extract_red_channel_from_multichannel_mask():
    mask = np.zeros((2, 2, 3), dtype=np.uint8)
    mask[0, 1, 0] = 9
    mask[1, 0, 1] = 9
    region = extract_part_regions(mask, [PartMapping("red id", 9)], 0)[0]
    assert region.area == 1
    assert region.bbox == (1, 0, 2, 1)


def test_alignment_failure_makes_every_part_uncertain():
    regions = extract_part_regions(
        np.array([[0, 1]], dtype=np.uint8), [PartMapping("part", 1)]
    )
    alignment = AlignmentResult(False, None, "too few matches")

    results = classify_regions(regions, None, np.zeros((1, 2, 3), np.uint8), alignment)

    assert results[0]["status"] == "UNCERTAIN"
    assert "too few matches" in results[0]["diagnostic"]


def test_global_matching_inliers_alone_do_not_prove_local_part_presence():
    regions = extract_part_regions(
        np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8),
        [PartMapping("part", 1)],
    )
    alignment = AlignmentResult(
        True,
        np.eye(3),
        "ok",
        good_matches=12,
        inliers=10,
        inlier_ratio=10 / 12,
        cad_inlier_points=((1.0, 0.0), (1.0, 1.0)),
    )

    result = classify_regions(
        regions, np.zeros((2, 3, 3), np.uint8), np.zeros((2, 3, 3), np.uint8), alignment
    )[0]

    assert result["status"] == "UNCERTAIN"
    assert "Too little per-part" in result["diagnostic"]


def test_global_feature_matches_are_not_positive_classification_evidence():
    result = classify_part_measurements(
        area=1200,
        matching_inliers=6,
        cad_std=20.0,
        real_std=12.0,
        real_to_ring_mean_delta=10.0,
        cad_edge_density=0.03,
        real_edge_density=0.02,
        edge_support=0.15,
    )
    assert result["status"] == "UNCERTAIN"


def test_localized_part_is_flagged_when_silhouette_is_displaced(monkeypatch):
    import src.geometric as geometric

    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 1
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[20:60, 20:60] = 180
    region = extract_part_regions(mask, [PartMapping("nut", 1)])[0]
    alignment = AlignmentResult(
        True, np.eye(3), "ok", cad_inlier_points=((25.0, 25.0), (50.0, 50.0))
    )
    monkeypatch.setattr(
        geometric,
        "measure_local_component_geometry",
        lambda *_args, **_kwargs: {
            "measurable": True,
            "edge_support": 0.80,
            "position_offset_dx_px": 14.0,
            "position_offset_dy_px": -5.0,
            "position_offset_magnitude_px": float(np.hypot(14, 5)),
            "position_out_of_tolerance": True,
        },
    )

    result = classify_regions([region], image, image, alignment)[0]

    assert result["status"] == "PRESENT"
    assert result["position_deviation"] is True
    assert result["position_offset_dx_px"] == 14.0
    assert "outside the 12 px tolerance" in result["diagnostic"]


def test_loftr_alignment_reports_missing_optional_dependencies(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_loftr(name, *args, **kwargs):
        if name == "torch" or name.startswith("kornia"):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_loftr)
    from src.loftr_alignment import align_loftr_ransac

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    alignment, aligned = align_loftr_ransac(image, image)

    assert not alignment.success
    assert aligned is None
    assert "optional dependencies" in alignment.reason


def test_flat_region_heuristic_can_report_missing_only_with_cad_structure():
    result = classify_part_measurements(
        area=900,
        matching_inliers=0,
        cad_std=25.0,
        real_std=4.0,
        real_to_ring_mean_delta=1.5,
        cad_edge_density=0.04,
        real_edge_density=0.001,
    )

    assert result["status"] == "MISSING"
    assert "heuristic" in result["evidence"]


def test_local_edge_correspondence_is_positive_presence_evidence():
    result = classify_part_measurements(
        area=1600,
        matching_inliers=0,
        cad_std=25.0,
        real_std=22.0,
        real_to_ring_mean_delta=18.0,
        cad_edge_density=0.04,
        real_edge_density=0.03,
        edge_support=0.72,
    )
    assert result["status"] == "PRESENT"
    assert "72%" in result["evidence"]


def test_weak_local_evidence_abstains():
    result = classify_part_measurements(
        area=1600,
        matching_inliers=0,
        cad_std=24.0,
        real_std=14.0,
        real_to_ring_mean_delta=8.0,
        cad_edge_density=0.04,
        real_edge_density=0.025,
        edge_support=0.18,
    )
    assert result["status"] == "UNCERTAIN"


def test_clear_local_edge_absence_with_flat_blended_photo_is_missing_evidence():
    result = classify_part_measurements(
        area=1800,
        matching_inliers=0,
        cad_std=13.0,
        real_std=0.3,
        real_to_ring_mean_delta=0.2,
        cad_edge_density=0.025,
        real_edge_density=0.0,
        edge_support=0.0,
    )
    assert result["status"] == "MISSING"
    assert "Possible absence" in result["evidence"]


def test_csv_and_json_reports_include_part_evidence_only():
    result = {
        "case": "case-a",
        "photo": "BG_00.png",
        "parts": [{
            "part_name": "washer",
            "pixel_value": 4,
            "status": "UNCERTAIN",
            "evidence_score": 0.2,
            "mask_area_px": 60,
            "bbox_xyxy": [1, 2, 4, 5],
            "centroid_xy": [2.0, 3.0],
            "diagnostic": "small region",
        }],
        "images": {"large": np.zeros((500, 500, 3), dtype=np.uint8)},
    }
    payload = json.loads(result_json(result))
    rows = list(csv.DictReader(io.StringIO(result_csv(result))))
    assert "images" not in payload
    assert payload["parts"][0]["diagnostic"] == "small region"
    assert rows[0]["part_name"] == "washer"
    assert rows[0]["status"] == "UNCERTAIN"


def test_empty_mask_region_is_not_reported_missing():
    regions = extract_part_regions(
        np.zeros((8, 8), dtype=np.uint8), [PartMapping("no pixels", 4)]
    )
    alignment = AlignmentResult(True, np.eye(3), "ok")

    result = classify_regions(regions, np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), alignment)[0]

    assert result["status"] == "UNCERTAIN"
    assert result["bbox_xyxy"] is None
    assert "no pixels" in result["diagnostic"].lower()
    assert result["component_count"] == 0


def test_sift_reports_unavailable_descriptors_for_blank_inputs():
    blank = np.zeros((240, 320, 4), dtype=np.uint8)

    result, aligned = align_sift_ransac(blank, blank)

    assert not result.success
    assert not result.descriptors_photo_available
    assert not result.descriptors_cad_available
    assert aligned is None


def test_sift_rejects_insufficient_ratio_filtered_matches(monkeypatch):
    keypoints = [cv2.KeyPoint(float(x * 10 + 5), float(y * 10 + 5), 3) for y in range(3) for x in range(4)]
    descriptors = np.zeros((len(keypoints), 128), dtype=np.float32)

    class FlatSift:
        def detectAndCompute(self, _image, _mask):
            return keypoints, descriptors

    monkeypatch.setattr(cv2, "SIFT_create", lambda **_kwargs: FlatSift())
    image = np.zeros((80, 80, 3), dtype=np.uint8)

    result, _ = align_sift_ransac(image, image)

    assert result.raw_matches > 0
    assert result.filtered_matches == 0
    assert not result.success
    assert "ratio-filtered" in result.reason


def test_sift_rejects_nonfinite_homography(monkeypatch):
    keypoints = [cv2.KeyPoint(float(x * 20 + 5), float(y * 20 + 5), 3) for y in range(3) for x in range(4)]
    descriptors = np.zeros((len(keypoints), 128), dtype=np.float32)
    for index in range(len(keypoints)):
        descriptors[index, index] = 100.0

    class UniqueSift:
        def detectAndCompute(self, _image, _mask):
            return keypoints, descriptors

    monkeypatch.setattr(cv2, "SIFT_create", lambda **_kwargs: UniqueSift())
    monkeypatch.setattr(
        cv2,
        "findHomography",
        lambda *_args, **_kwargs: (np.full((3, 3), np.nan), np.ones((len(keypoints), 1), np.uint8)),
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result, _ = align_sift_ransac(image, image)

    assert result.filtered_matches == len(keypoints)
    assert result.inliers == len(keypoints)
    assert not result.homography_valid
    assert "non-finite homography" in result.reason


def test_sift_aligns_synthetic_textured_rgba_pair_in_photo_to_cad_direction():
    rng = np.random.default_rng(42)
    cad = np.full((480, 640, 3), 245, dtype=np.uint8)
    for _ in range(150):
        center = (int(rng.integers(10, 630)), int(rng.integers(10, 470)))
        color = tuple(int(value) for value in rng.integers(0, 220, size=3))
        radius = int(rng.integers(3, 16))
        cv2.circle(cad, center, radius, color, thickness=-1)
        if rng.random() > 0.5:
            cv2.line(cad, center, (min(639, center[0] + radius * 2), min(479, center[1] + radius)), (20, 20, 20), 2)
    known_photo_to_cad = np.array(
        [[1.0, 0.012, 7.0], [-0.008, 1.0, 6.0], [0.00001, -0.00002, 1.0]],
        dtype=np.float64,
    )
    photo = cv2.warpPerspective(cad, np.linalg.inv(known_photo_to_cad), (640, 480))
    photo_rgba = np.dstack((photo, np.full(photo.shape[:2], 255, dtype=np.uint8)))
    cad_rgba = np.dstack((cad, np.full(cad.shape[:2], 255, dtype=np.uint8)))

    result, aligned = align_sift_ransac(photo_rgba, cad_rgba)

    assert result.success, result.reason
    assert result.homography_valid
    assert result.inlier_ratio > 0.2
    assert result.median_reprojection_error is not None
    assert aligned is not None and aligned.shape == cad.shape
    point = np.array([[[200.0, 160.0]]], dtype=np.float32)
    expected = cv2.perspectiveTransform(point, known_photo_to_cad)[0, 0]
    actual = cv2.perspectiveTransform(point, result.matrix_photo_to_cad)[0, 0]
    assert np.linalg.norm(expected - actual) < 3.0


def test_masked_edge_ecc_aligns_synthetic_assembly_in_photo_to_cad_direction():
    cad = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(cad, (110, 55), (535, 435), (75, 100, 125), thickness=-1)
    cv2.rectangle(mask, (110, 55), (535, 435), 1, thickness=-1)
    cv2.rectangle(cad, (160, 105), (485, 370), (120, 145, 170), thickness=5)
    for center in ((180, 140), (460, 140), (320, 260), (180, 340), (460, 340)):
        cv2.circle(cad, center, 18, (20, 25, 30), thickness=-1)
        cv2.circle(cad, center, 7, (230, 230, 230), thickness=3)
    for x in (250, 320, 390):
        cv2.line(cad, (x, 75), (x, 205), (15, 20, 25), 10)
    known_photo_to_cad = np.array(
        [[1.0, 0.01, 8.0], [-0.01, 1.0, 5.0], [0.00001, 0.00001, 1.0]],
        dtype=np.float64,
    )
    photo = cv2.warpPerspective(cad, np.linalg.inv(known_photo_to_cad), (640, 480))

    result, aligned = align_masked_edges_ecc(photo, cad, mask)

    assert result.success, result.reason
    assert result.correlation_score is not None and result.correlation_score >= 0.30
    assert aligned is not None and aligned.shape == cad.shape
    point = np.array([[[300.0, 250.0]]], dtype=np.float32)
    expected = cv2.perspectiveTransform(point, known_photo_to_cad)[0, 0]
    actual = cv2.perspectiveTransform(point, result.matrix_photo_to_cad)[0, 0]
    assert np.linalg.norm(expected - actual) < 10.0


def test_stray_mask_fragments_do_not_contribute_local_presence_evidence():
    cad = np.zeros((200, 200, 3), dtype=np.uint8)
    cad[40:100, 40:100] = 200
    photo = np.full((200, 200, 3), 90, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[40:100, 40:100] = 1
    with_strays = mask.copy()
    # Boundary-artifact IDs near unrelated structure that the photo matches.
    for x in range(120, 190, 10):
        for y in range(20, 190, 12):
            cv2.rectangle(cad, (x - 3, y - 3), (x + 3, y + 3), (255, 255, 255), -1)
            cv2.rectangle(photo, (x - 3, y - 3), (x + 3, y + 3), (255, 255, 255), -1)
            with_strays[y, x] = 1
    alignment = AlignmentResult(True, np.eye(3), "ok")
    mapping = [PartMapping("washer", 1)]

    clean = classify_regions(extract_part_regions(mask, mapping), photo, cad, alignment)[0]
    noisy = classify_regions(extract_part_regions(with_strays, mapping), photo, cad, alignment)[0]

    assert noisy["component_count"] > 20
    assert noisy["status"] == clean["status"] != "PRESENT"
    assert noisy["diagnostic"] == clean["diagnostic"]


def test_foreground_edge_distance_increases_with_misregistration():
    from src.detector import foreground_edge_distance

    cad = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2.rectangle(cad, (40, 40), (120, 120), (220, 220, 220), -1)
    mask = np.zeros((160, 160), dtype=np.uint8)
    mask[40:121, 40:121] = 1
    shifted = np.roll(cad, (9, 9), axis=(0, 1))

    assert foreground_edge_distance(cad, cad, mask) == 0.0
    assert foreground_edge_distance(shifted, cad, mask) >= 5.0


def _run_loftr_with_fake_matches(monkeypatch, points):
    import pytest

    torch = pytest.importorskip("torch")
    KF = pytest.importorskip("kornia.feature")

    class FakeLoFTR:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, batch):
            tensor = torch.from_numpy(points)
            return {"keypoints0": tensor, "keypoints1": tensor.clone(), "confidence": torch.ones(len(points))}

    monkeypatch.setattr(KF, "LoFTR", FakeLoFTR)
    from src.loftr_alignment import align_loftr_ransac

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    return align_loftr_ransac(image, image)


def test_loftr_rejects_nearly_collinear_inliers(monkeypatch):
    xs = np.linspace(10, 300, 40)
    points = np.column_stack((xs, 100 + 3 * np.sin(xs))).astype(np.float32)

    alignment, aligned = _run_loftr_with_fake_matches(monkeypatch, points)

    assert not alignment.success and aligned is None
    assert "nearly collinear" in alignment.reason


def test_loftr_accepts_well_spread_inliers(monkeypatch):
    grid = np.stack(np.meshgrid(np.linspace(20, 300, 8), np.linspace(20, 220, 6)), axis=-1)
    points = grid.reshape(-1, 2).astype(np.float32)

    alignment, aligned = _run_loftr_with_fake_matches(monkeypatch, points)

    assert alignment.success, alignment.reason
    assert np.allclose(alignment.matrix_photo_to_cad, np.eye(3), atol=1e-3)


def test_fixed_mask_regions_are_cached_read_only_and_keyed_by_content():
    from src.detector import region_core

    mask = np.zeros((40, 50), np.uint8)
    mask[5:20, 5:20] = 3
    mapping = [PartMapping("washer", 3)]

    first = extract_part_regions(mask, mapping)
    second = extract_part_regions(mask.copy(), mapping)
    assert first[0] is second[0]
    assert not first[0].binary_mask.flags.writeable
    assert region_core(first[0]) is region_core(second[0])

    changed = mask.copy()
    changed[25:35, 25:45] = 3
    third = extract_part_regions(changed, mapping)
    assert third[0] is not first[0] and third[0].area == first[0].area + 200
