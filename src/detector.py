"""Conservative SIFT baseline for PartSight part-presence inspection.

This is an explainable heuristic, not a validated production detector. A part
is only reported missing when alignment is reliable and its expected region
looks unusually empty compared with the CAD part and its local surroundings.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data_inspection import (
    PartMapping,
    discover_case_directories,
    discover_photos,
    load_image,
    load_part_mapping,
    mask_part_ids,
    resolve_case_files,
)

LOGGER = logging.getLogger(__name__)
PRESENCE_STATUSES = ("PRESENT", "MISSING", "UNCERTAIN", "NOT_VISIBLE")

# Fixed per-case data (CAD image, mask, mask regions) is identical for every
# photograph of a case, so it is computed once and shared read-only. Each
# cache keeps at most two cases; a region list holds full-size masks.
_FIXED_INPUT_CACHE: dict[tuple, tuple] = {}
_REGION_CACHE: dict[tuple, list] = {}
_CORE_CACHE: dict[int, np.ndarray] = {}
_CACHE_CASES = 2


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _remember(cache: dict, key: tuple, value: Any, on_evict=None) -> Any:
    if len(cache) >= _CACHE_CASES:
        old_key = next(iter(cache))
        evicted = cache.pop(old_key)
        if on_evict is not None:
            on_evict(evicted)
    cache[key] = value
    return value


@dataclass(frozen=True)
class PartRegion:
    name: str
    stencil_value: int
    area: int
    bbox: tuple[int, int, int, int] | None
    centroid: tuple[float, float] | None
    binary_mask: np.ndarray
    component_count: int = 0
    largest_component_area: int = 0
    largest_component_bbox: tuple[int, int, int, int] | None = None
    largest_component_centroid: tuple[float, float] | None = None


@dataclass(frozen=True)
class AlignmentResult:
    success: bool
    matrix_photo_to_cad: np.ndarray | None
    reason: str
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    median_reprojection_error: float | None = None
    cad_inlier_points: tuple[tuple[float, float], ...] = ()
    method: str = "SIFT"
    keypoints_photo: int = 0
    keypoints_cad: int = 0
    correspondences: int = 0
    descriptors_photo_available: bool = False
    descriptors_cad_available: bool = False
    raw_matches: int = 0
    knn_candidates: int = 0
    filtered_matches: int = 0
    homography_valid: bool = False
    inlier_spread_ratio: float | None = None
    projected_area_ratio: float | None = None
    overlap_ratio: float | None = None
    match_points: tuple[tuple[float, float, float, float, bool], ...] = ()
    correlation_score: float | None = None


def extract_part_regions(
    mask: np.ndarray,
    mappings: list[PartMapping],
    channel_index: int = 0,
) -> list[PartRegion]:
    """Extract mapped regions and geometry from a scalar or multichannel mask."""

    array = np.asarray(mask)
    if array.ndim == 2:
        ids = array
    elif array.ndim == 3 and 0 <= channel_index < array.shape[2]:
        ids = array[..., channel_index]
    else:
        raise ValueError(f"Unsupported mask shape or channel: {array.shape}")
    key = (
        hashlib.blake2b(np.ascontiguousarray(ids).view(np.uint8), digest_size=16).hexdigest(),
        ids.shape, str(ids.dtype), tuple((part.name, part.stencil_value) for part in mappings),
    )
    cached = _REGION_CACHE.get(key)
    if cached is not None:
        return list(cached)
    regions: list[PartRegion] = []
    for part in mappings:
        binary = ids == part.stencil_value
        ys, xs = np.nonzero(binary)
        area = int(xs.size)
        if area:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            centroid = (float(xs.mean()), float(ys.mean()))
        else:
            bbox, centroid = None, None
        import cv2

        component_count_with_background, _, component_stats, component_centroids = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        component_areas = component_stats[1:, cv2.CC_STAT_AREA]
        component_count = int(component_count_with_background - 1)
        if component_count:
            largest_label = int(np.argmax(component_areas)) + 1
            largest_component_area = int(component_stats[largest_label, cv2.CC_STAT_AREA])
            component_x = int(component_stats[largest_label, cv2.CC_STAT_LEFT])
            component_y = int(component_stats[largest_label, cv2.CC_STAT_TOP])
            component_w = int(component_stats[largest_label, cv2.CC_STAT_WIDTH])
            component_h = int(component_stats[largest_label, cv2.CC_STAT_HEIGHT])
            largest_component_bbox = (
                component_x,
                component_y,
                component_x + component_w,
                component_y + component_h,
            )
            largest_component_centroid = tuple(
                float(value) for value in component_centroids[largest_label]
            )
        else:
            largest_component_area = 0
            largest_component_bbox = None
            largest_component_centroid = None
        regions.append(
            PartRegion(
                part.name,
                part.stencil_value,
                area,
                bbox,
                centroid,
                binary,
                component_count,
                largest_component_area,
                largest_component_bbox,
                largest_component_centroid,
            )
        )
    for region in regions:
        _read_only(region.binary_mask)

    def forget_cores(evicted: list) -> None:
        for region in evicted:
            _CORE_CACHE.pop(id(region.binary_mask), None)

    _remember(_REGION_CACHE, key, regions, forget_cores)
    return list(regions)


def region_core(region: PartRegion) -> np.ndarray:
    """Largest connected component of a region's mask, cached for cached regions."""
    key = id(region.binary_mask)
    core = _CORE_CACHE.get(key)
    if core is not None:
        return core
    core = largest_component_mask(region.binary_mask)
    if any(region is other for cached in _REGION_CACHE.values() for other in cached):
        _CORE_CACHE[key] = _read_only(core)
    return core


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Return the largest 8-connected component of a binary mask."""

    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(binary, dtype=bool)
    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return labels == largest_label


def _align_ransac(
    photo_image: np.ndarray, cad_image: np.ndarray, method: str
) -> tuple[AlignmentResult, np.ndarray | None]:
    """Shared SIFT/ORB implementation with detailed quality diagnostics."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError("OpenCV is required. Install the project with `uv sync`.") from exc

    diag: dict[str, Any] = {
        "keypoints_photo": 0,
        "keypoints_cad": 0,
        "descriptors_photo_available": False,
        "descriptors_cad_available": False,
        "raw_matches": 0,
        "knn_candidates": 0,
        "filtered_matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error": None,
        "homography_valid": False,
        "inlier_spread_ratio": None,
        "projected_area_ratio": None,
        "overlap_ratio": None,
        "match_points": (),
        "correlation_score": None,
        "method": method,
    }

    def finish(reason: str, success: bool = False, matrix: np.ndarray | None = None,
               aligned: np.ndarray | None = None,
               cad_points: tuple[tuple[float, float], ...] = ()):
        result = AlignmentResult(
            success=success,
            matrix_photo_to_cad=matrix if success else None,
            reason=reason,
            cad_inlier_points=cad_points if success else (),
            **diag,
        )
        LOGGER.info(
            "%s alignment: kp(photo/CAD)=%d/%d descriptors=%s/%s raw=%d knn-candidates=%d filtered=%d "
            "inliers=%d ratio=%.3f H-valid=%s median-reproj=%s spread=%s area=%s overlap=%s result=%s (%s)",
            method,
            result.keypoints_photo,
            result.keypoints_cad,
            result.descriptors_photo_available,
            result.descriptors_cad_available,
            result.raw_matches,
            result.knn_candidates,
            result.filtered_matches,
            result.inliers,
            result.inlier_ratio,
            result.homography_valid,
            result.median_reprojection_error,
            result.inlier_spread_ratio,
            result.projected_area_ratio,
            result.overlap_ratio,
            "accepted" if success else "rejected",
            reason,
        )
        return result, aligned if success else None

    photo = np.asarray(photo_image)
    cad = np.asarray(cad_image)
    if photo.ndim == 2:
        photo_gray = photo
        photo_rgb = np.repeat(photo[..., None], 3, axis=2)
    elif photo.ndim == 3 and photo.shape[2] == 4:
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_RGBA2GRAY)
        photo_rgb = cv2.cvtColor(photo, cv2.COLOR_RGBA2RGB)
    elif photo.ndim == 3 and photo.shape[2] == 3:
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        photo_rgb = photo
    else:
        return finish(f"Unsupported photograph shape {photo.shape}.")
    if cad.ndim == 2:
        cad_gray = cad
    elif cad.ndim == 3 and cad.shape[2] == 4:
        cad_gray = cv2.cvtColor(cad, cv2.COLOR_RGBA2GRAY)
    elif cad.ndim == 3 and cad.shape[2] == 3:
        cad_gray = cv2.cvtColor(cad, cv2.COLOR_RGB2GRAY)
    else:
        return finish(f"Unsupported CAD image shape {cad.shape}.")

    photo_h, photo_w = photo_gray.shape
    cad_h, cad_w = cad_gray.shape
    max_side = 1600

    # Use actual resized dimensions for coordinate conversion; independent
    # rounded resize factors are needed when source aspect ratios differ.
    photo_orig_h, photo_orig_w = photo_gray.shape
    cad_orig_h, cad_orig_w = cad_gray.shape

    def resize_exact(gray: np.ndarray, orig_w: int, orig_h: int) -> tuple[np.ndarray, float, float]:
        scale = min(1.0, max_side / max(orig_w, orig_h))
        width, height = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
        small = gray if (width, height) == (orig_w, orig_h) else cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        return small, width / orig_w, height / orig_h

    photo_gray, photo_sx, photo_sy = resize_exact(photo_gray, photo_orig_w, photo_orig_h)
    cad_gray, cad_sx, cad_sy = resize_exact(cad_gray, cad_orig_w, cad_orig_h)
    detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.025)
    norm, ratio_threshold = cv2.NORM_L2, 0.72
    keypoints_photo, descriptors_photo = detector.detectAndCompute(photo_gray, None)
    keypoints_cad, descriptors_cad = detector.detectAndCompute(cad_gray, None)
    diag.update(
        keypoints_photo=len(keypoints_photo),
        keypoints_cad=len(keypoints_cad),
        descriptors_photo_available=descriptors_photo is not None,
        descriptors_cad_available=descriptors_cad is not None,
    )
    if descriptors_photo is None or descriptors_cad is None:
        return finish(f"{method} descriptors unavailable for one or both images.")
    if len(descriptors_cad) < 2:
        return finish(f"CAD image has too few {method} descriptors for ratio matching.")
    pairs = cv2.BFMatcher(norm).knnMatch(descriptors_photo, descriptors_cad, k=2)
    diag["raw_matches"] = len(pairs)
    diag["knn_candidates"] = sum(len(pair) for pair in pairs)
    matches = [pair[0] for pair in pairs if len(pair) >= 2 and pair[0].distance < ratio_threshold * pair[1].distance]
    diag["filtered_matches"] = diag["good_matches"] = len(matches)
    if len(matches) < 12:
        return finish(f"Only {len(matches)} ratio-filtered {method} matches; need at least 12.")

    photo_points = np.float32([keypoints_photo[m.queryIdx].pt for m in matches])
    cad_points = np.float32([keypoints_cad[m.trainIdx].pt for m in matches])

    def diagnostic_points(inlier_flags: np.ndarray | None = None) -> tuple[tuple[float, float, float, float, bool], ...]:
        total = len(matches)
        inlier_flags = np.zeros(total, dtype=bool) if inlier_flags is None else inlier_flags
        inlier_indices = np.flatnonzero(inlier_flags).tolist()
        outlier_indices = np.flatnonzero(~inlier_flags).tolist()
        remaining = max(0, 160 - len(inlier_indices))
        if len(outlier_indices) > remaining:
            positions = np.linspace(0, len(outlier_indices) - 1, remaining, dtype=int) if remaining else []
            outlier_indices = [outlier_indices[int(index)] for index in positions]
        keep = set(inlier_indices + outlier_indices)
        return tuple(
            (
                float(photo_points[index, 0] / photo_sx),
                float(photo_points[index, 1] / photo_sy),
                float(cad_points[index, 0] / cad_sx),
                float(cad_points[index, 1] / cad_sy),
                bool(inlier_flags[index]),
            )
            for index in sorted(keep)
        )

    diag["match_points"] = diagnostic_points()
    try:
        h_small, inlier_mask = cv2.findHomography(
            photo_points, cad_points, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.995
        )
    except cv2.error as exc:
        return finish(f"RANSAC failed: {exc}")
    if inlier_mask is not None:
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        ratio = inlier_count / len(matches)
        diag.update(inliers=inlier_count, inlier_ratio=ratio)
    diag["match_points"] = diagnostic_points(inliers)
    if h_small is None or inlier_mask is None:
        return finish("RANSAC could not estimate a homography.")
    if h_small.shape != (3, 3) or not np.isfinite(h_small).all():
        return finish("RANSAC returned a malformed or non-finite homography.")

    if inlier_count < 8 or ratio < 0.20:
        return finish(f"Weak RANSAC fit: {inlier_count}/{len(matches)} inliers ({ratio:.0%}).")
    try:
        projected_points = cv2.perspectiveTransform(photo_points[inliers, None, :], h_small)
    except cv2.error:
        projected_points = None
    if projected_points is None or projected_points.shape[0] != inlier_count:
        return finish("RANSAC homography could not project its inlier points.")
    projected = projected_points[:, 0, :]
    if not np.isfinite(projected).all():
        return finish("RANSAC homography projected non-finite inlier points.")
    errors = np.linalg.norm(projected - cad_points[inliers], axis=1)
    median_error = float(np.median(errors))
    hull = cv2.convexHull(cad_points[inliers])
    coverage = float(cv2.contourArea(hull)) / max(1.0, cad_gray.shape[0] * cad_gray.shape[1])
    diag.update(median_reprojection_error=median_error, homography_valid=True, inlier_spread_ratio=coverage)
    if median_error > 5.0 or coverage < 0.001:
        return finish(f"RANSAC inliers lack spatial support or have high reprojection error (median={median_error:.2f}px, spread={coverage:.4%}).")

    photo_scale = np.diag([photo_sx, photo_sy, 1.0])
    cad_scale = np.diag([cad_sx, cad_sy, 1.0])
    matrix = np.linalg.inv(cad_scale) @ h_small @ photo_scale
    photo_corners = np.float32([[[0, 0], [photo_w - 1, 0], [photo_w - 1, photo_h - 1], [0, photo_h - 1]]])
    cad_corners = cv2.perspectiveTransform(photo_corners, matrix)
    if cad_corners is None or not np.isfinite(cad_corners).all():
        return finish("Photo-to-CAD homography could not project the image bounds.")
    cad_corners = cad_corners[0]
    quad_area = abs(float(cv2.contourArea(cad_corners)))
    cad_area = float(cad_w * cad_h)
    area_ratio = quad_area / cad_area
    diag.update(projected_area_ratio=area_ratio)
    if not (0.08 <= area_ratio <= 5.0):
        diag["homography_valid"] = False
        return finish(f"Projected photograph area is implausible relative to CAD ({area_ratio:.3f}x).")
    cad_rectangle = np.float32([[[0, 0], [cad_w - 1, 0], [cad_w - 1, cad_h - 1], [0, cad_h - 1]]])
    try:
        overlap_area, _ = cv2.intersectConvexConvex(cad_corners.astype(np.float32), cad_rectangle)
    except cv2.error:
        overlap_area = 0.0
    overlap_ratio = float(overlap_area) / cad_area
    diag["overlap_ratio"] = overlap_ratio
    if overlap_ratio < 0.20:
        diag["homography_valid"] = False
        return finish(f"Projected photograph overlaps only {overlap_ratio:.1%} of the CAD canvas.")

    cad_inlier_points = tuple((float(x / cad_sx), float(y / cad_sy)) for x, y in cad_points[inliers])
    aligned = cv2.warpPerspective(photo_rgb, matrix, (cad_w, cad_h))
    return finish(f"{method}/RANSAC alignment passed quality checks.", True, matrix, aligned, cad_inlier_points)


def align_sift_ransac(photo_rgb: np.ndarray, cad_rgb: np.ndarray) -> tuple[AlignmentResult, np.ndarray | None]:
    """Estimate photo-to-CAD homography using SIFT ratio matching and RANSAC."""

    return _align_ransac(photo_rgb, cad_rgb, "SIFT")


def align_masked_edges_ecc(
    photo_rgb: np.ndarray, cad_rgb: np.ndarray, part_mask: np.ndarray
) -> tuple[AlignmentResult, np.ndarray | None]:
    """Register shared assembly edges with ECC inside the CAD foreground ROI.

    OpenCV returns the template-CAD to input-photo sampling transform for this
    call order. The crop/resize offsets are composed around its inverse to
    produce the conventional photo-to-CAD transform.
    """

    import cv2

    photo = np.asarray(photo_rgb)
    cad = np.asarray(cad_rgb)
    if photo.ndim == 3 and photo.shape[2] == 4:
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_RGBA2GRAY)
        photo_color = cv2.cvtColor(photo, cv2.COLOR_RGBA2RGB)
    elif photo.ndim == 3 and photo.shape[2] == 3:
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        photo_color = photo
    elif photo.ndim == 2:
        photo_gray = photo
        photo_color = np.repeat(photo[..., None], 3, axis=2)
    else:
        return AlignmentResult(False, None, f"Unsupported photograph shape {photo.shape}.", method="ECC-edge"), None
    if cad.ndim == 3 and cad.shape[2] == 4:
        cad_gray = cv2.cvtColor(cad, cv2.COLOR_RGBA2GRAY)
    elif cad.ndim == 3 and cad.shape[2] == 3:
        cad_gray = cv2.cvtColor(cad, cv2.COLOR_RGB2GRAY)
    elif cad.ndim == 2:
        cad_gray = cad
    else:
        return AlignmentResult(False, None, f"Unsupported CAD image shape {cad.shape}.", method="ECC-edge"), None

    cad_h, cad_w = cad_gray.shape
    photo_h, photo_w = photo_gray.shape
    mask = np.asarray(part_mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2 or mask.shape != (cad_h, cad_w):
        return AlignmentResult(False, None, "ECC requires a single-channel CAD mask with CAD dimensions.", method="ECC-edge"), None
    foreground = np.where(mask > 0, 255, 0).astype(np.uint8)
    if cv2.countNonZero(foreground) < 1000:
        return AlignmentResult(False, None, "CAD mask has too little foreground for geometry registration.", method="ECC-edge"), None

    max_side = 1600
    scale = min(1.0, max_side / max(cad_h, cad_w))
    small_w, small_h = max(1, round(cad_w * scale)), max(1, round(cad_h * scale))
    photo_small = cv2.resize(photo_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    cad_small = cv2.resize(cad_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(foreground, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    dilation = max(3, int(round(min(small_w, small_h) * 0.007)) | 1)
    mask_small = cv2.dilate(mask_small, np.ones((dilation, dilation), np.uint8))
    x, y, width, height = cv2.boundingRect(mask_small)
    padding = max(8, int(round(max(width, height) * 0.10)))
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(small_w, x + width + padding), min(small_h, y + height + padding)
    roi_mask = mask_small[y0:y1, x0:x1]
    template_edges = cv2.Canny(cad_small[y0:y1, x0:x1], 40, 120)
    input_edges = cv2.Canny(photo_small[y0:y1, x0:x1], 40, 120)
    template_edges = np.where(roi_mask > 0, template_edges, 0).astype(np.uint8)
    input_edges = np.where(roi_mask > 0, input_edges, 0).astype(np.uint8)
    if cv2.countNonZero(template_edges) < 64 or cv2.countNonZero(input_edges) < 64:
        return AlignmentResult(False, None, "Too few assembly edges inside the foreground ROI.", method="ECC-edge"), None

    warp_template_to_photo = np.eye(3, dtype=np.float32)
    try:
        correlation, warp_template_to_photo = cv2.findTransformECC(
            template_edges,
            input_edges,
            warp_template_to_photo,
            cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-5),
            roi_mask,
            5,
        )
    except cv2.error as exc:
        return AlignmentResult(False, None, f"Masked-edge ECC failed: {exc}", method="ECC-edge"), None
    correlation = float(correlation)
    if not np.isfinite(warp_template_to_photo).all() or not np.isfinite(correlation):
        return AlignmentResult(False, None, "ECC returned a non-finite transform or score.", method="ECC-edge", correlation_score=correlation), None

    crop_to_small = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])
    cad_to_photo_small = crop_to_small @ warp_template_to_photo @ np.linalg.inv(crop_to_small)
    scale_photo = np.diag([small_w / photo_w, small_h / photo_h, 1.0])
    scale_cad = np.diag([small_w / cad_w, small_h / cad_h, 1.0])
    try:
        matrix = np.linalg.inv(scale_cad) @ np.linalg.inv(cad_to_photo_small) @ scale_photo
    except np.linalg.LinAlgError:
        return AlignmentResult(False, None, "ECC transform is singular.", method="ECC-edge", correlation_score=correlation), None
    if abs(float(np.linalg.det(matrix))) < 1e-10:
        return AlignmentResult(False, None, "ECC photo-to-CAD transform is singular.", method="ECC-edge", correlation_score=correlation), None

    corners = np.float32([[[0, 0], [photo_w - 1, 0], [photo_w - 1, photo_h - 1], [0, photo_h - 1]]])
    projected_corners = cv2.perspectiveTransform(corners, matrix)
    if projected_corners is None or not np.isfinite(projected_corners).all():
        return AlignmentResult(False, None, "ECC transform cannot project photograph bounds.", method="ECC-edge", correlation_score=correlation), None
    projected_corners = projected_corners[0]
    cad_area = float(cad_w * cad_h)
    area_ratio = abs(float(cv2.contourArea(projected_corners))) / cad_area
    cad_rect = np.float32([[[0, 0], [cad_w - 1, 0], [cad_w - 1, cad_h - 1], [0, cad_h - 1]]])
    try:
        overlap, _ = cv2.intersectConvexConvex(projected_corners.astype(np.float32), cad_rect)
    except cv2.error:
        overlap = 0.0
    overlap_ratio = float(overlap) / cad_area
    diag = {
        "method": "ECC-edge",
        "correlation_score": correlation,
        "homography_valid": True,
        "projected_area_ratio": area_ratio,
        "overlap_ratio": overlap_ratio,
    }
    if correlation < 0.30:
        return AlignmentResult(False, None, f"Masked-edge correlation is weak ({correlation:.3f}).", **diag), None
    if not (0.08 <= area_ratio <= 5.0) or overlap_ratio < 0.20:
        return AlignmentResult(False, None, f"ECC transform has implausible projection (area={area_ratio:.3f}x, overlap={overlap_ratio:.1%}).", **diag), None
    aligned = cv2.warpPerspective(photo_color, matrix, (cad_w, cad_h))
    LOGGER.info("ECC-edge alignment: correlation=%.4f projected-area=%.4f overlap=%.1f%% result=accepted", correlation, area_ratio, overlap_ratio * 100)
    return AlignmentResult(True, matrix, "Masked-edge ECC passed correlation and projection checks.", **diag), aligned


def _alignment_record(result: AlignmentResult) -> dict[str, Any]:
    """JSON-safe summary of alignment diagnostics, excluding matrices/points."""

    return {
        "method": result.method,
        "success": result.success,
        "reason": result.reason,
        "keypoints_photo": result.keypoints_photo,
        "keypoints_cad": result.keypoints_cad,
        "correspondences": result.correspondences,
        "descriptors_photo_available": result.descriptors_photo_available,
        "descriptors_cad_available": result.descriptors_cad_available,
        "raw_matches": result.raw_matches,
        "knn_candidates": result.knn_candidates,
        "filtered_matches": result.filtered_matches,
        "ransac_inliers": result.inliers,
        "inlier_ratio": result.inlier_ratio,
        "homography_valid": result.homography_valid,
        "median_reprojection_error_px": result.median_reprojection_error,
        "inlier_spread_ratio": result.inlier_spread_ratio,
        "projected_area_ratio": result.projected_area_ratio,
        "overlap_ratio": result.overlap_ratio,
        "correlation_score": result.correlation_score,
    }


def classify_regions(
    regions: list[PartRegion], aligned_rgb: np.ndarray | None, cad_rgb: np.ndarray, alignment: AlignmentResult
) -> list[dict[str, Any]]:
    """Return one present/absent record for every mapped part (see ``presence``)."""

    from .presence import classify_presence

    return classify_presence(regions, aligned_rgb, cad_rgb, alignment)


def _annotate(cad_rgb: np.ndarray, photo_rgb: np.ndarray, regions: list[PartRegion], records: list[dict[str, Any]], alignment: AlignmentResult) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    cad = Image.fromarray(cad_rgb.copy())
    photo = Image.fromarray(photo_rgb.copy())
    draw_cad, draw_photo = ImageDraw.Draw(cad), ImageDraw.Draw(photo)
    font_size = max(18, min(34, min(cad.width, cad.height) // 90))
    font = ImageFont.load_default(size=font_size)
    colors = {
        "PRESENT": (20, 185, 70), "ROTATED": (60, 140, 255),
        "MISSING": (230, 45, 45), "UNCERTAIN": (245, 170, 20),
    }
    # Absent parts are labelled "A"; the others by their first letter.
    short_labels = {"MISSING": "A"}
    legend_names = {"MISSING": "ABSENT"}
    occupied_cad: list[tuple[int, int, int, int]] = []
    occupied_photo: list[tuple[int, int, int, int]] = []

    def draw_label(draw, text, anchor, color, occupied, canvas_size):
        x, y = anchor
        width, height = canvas_size
        text_width = draw.textbbox((0, 0), text, font=font, stroke_width=2)[2]
        text_height = font_size + 4
        candidates = [
            (x, y - text_height), (x, y + 4), (x - text_width, y - text_height),
            (x + 4, y - text_height * 2), (x - text_width, y + 4),
            (x + 4, y + text_height), (x - text_width // 2, y - text_height * 2),
        ]
        for label_x, label_y in candidates:
            label_x = min(max(0, label_x), max(0, width - text_width))
            label_y = min(max(0, label_y), max(0, height - text_height))
            rect = (label_x, label_y, label_x + text_width, label_y + text_height)
            if any(not (rect[2] <= other[0] or rect[0] >= other[2] or rect[3] <= other[1] or rect[1] >= other[3]) for other in occupied):
                continue
            draw.text((label_x, label_y), text, font=font, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
            occupied.append(rect)
            return
    for region, result in zip(regions, records, strict=True):
        outcome = result.get("inspection_outcome", result["status"])
        # Parts hidden in the CAD view cannot be checked and are listed only in the results table.
        if region.bbox is None or outcome == "NOT_VISIBLE":
            continue
        color = colors[outcome]
        # Outline the part's main region; stray boundary IDs in the mask would only add clutter.
        contours, _ = cv2.findContours(
            region_core(region).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            cad_points = [(int(point[0][0]), int(point[0][1])) for point in contour]
            if len(cad_points) >= 2:
                draw_cad.line(cad_points + [cad_points[0]], fill=color, width=3)
        fragmented = (
            region.component_count > 20
            and region.largest_component_area / max(region.area, 1) < 0.10
        )
        label = f"{region.stencil_value} {short_labels.get(outcome, outcome[0])}"
        if not fragmented:
            label_centroid = region.largest_component_centroid or region.centroid
            draw_label(
                draw_cad,
                label,
                (int(label_centroid[0]), int(label_centroid[1])),
                color,
                occupied_cad,
                (cad.width, cad.height),
            )
        if alignment.success and alignment.matrix_photo_to_cad is not None:
            inverse = np.linalg.inv(alignment.matrix_photo_to_cad)
            for contour in contours:
                polygon = cv2.perspectiveTransform(contour.astype(np.float32), inverse)[:, 0, :]
                if len(polygon) >= 2 and np.isfinite(polygon).all():
                    coords = [(int(x), int(y)) for x, y in polygon]
                    draw_photo.line(coords + [coords[0]], fill=color, width=3)
            label_centroid = region.largest_component_centroid or region.centroid
            center = np.float32([[[label_centroid[0], label_centroid[1]]]])
            photo_center = cv2.perspectiveTransform(center, inverse)[0, 0]
            if not fragmented and np.isfinite(photo_center).all():
                px, py = (int(value) for value in photo_center)
                draw_label(
                    draw_photo,
                    label,
                    (px, py),
                    color,
                    occupied_photo,
                    (photo.width, photo.height),
                )
    legend_font = ImageFont.load_default(size=max(16, font_size - 2))
    shown = ["PRESENT", "MISSING", "UNCERTAIN"]
    if any(record.get("inspection_outcome") == "ROTATED" for record in records):
        shown.insert(1, "ROTATED")
    legend_rows = tuple((status, colors[status]) for status in shown)
    for canvas, draw in ((cad, draw_cad), (photo, draw_photo)):
        legend_width, row_height, margin = 230, 27, 10
        legend_height = row_height * len(legend_rows) + 2 * margin
        draw.rectangle((8, 8, 8 + legend_width, 8 + legend_height), fill=(20, 24, 30))
        for row, (status, color) in enumerate(legend_rows):
            top = 8 + margin + row * row_height
            draw.rectangle((18, top + 4, 36, top + 20), fill=color)
            draw.text((44, top), legend_names.get(status, status), font=legend_font, fill="white")
    if not alignment.success:
        draw_photo.rectangle((8, 8, min(photo.width - 8, 1250), font_size * 2 + 18), fill=(130, 15, 15))
        draw_photo.text((16, 14), "ALIGNMENT FAILED — all part statuses are UNCERTAIN", font=font, fill="white")
    return np.asarray(cad), np.asarray(photo)


def inspect_case(case_dir: str | Path, photo_name: str) -> dict[str, Any]:
    """Align a real photo to its CAD image and decide presence for every listed part."""

    directory = Path(case_dir).resolve()
    files = resolve_case_files(directory, photo_name)
    photo_path, cad_path, mask_path = files.photo, files.cad, files.mask
    photo_info = load_image(photo_path)
    photo_rgb = np.asarray(Image.fromarray(photo_info.array).convert("RGB"))
    fixed_key = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in (cad_path, mask_path, files.mapping)
    )
    fixed = _FIXED_INPUT_CACHE.get(fixed_key)
    if fixed is None:
        mappings = load_part_mapping(files.mapping)[files.mapping_group]
        if not mappings:
            raise ValueError(f"No parts in {files.mapping}")
        cad_info, mask_info = load_image(cad_path), load_image(mask_path)
        cad_rgb = _read_only(np.array(Image.fromarray(cad_info.array).convert("RGB")))
        _read_only(mask_info.array)
        mask_ids = mask_part_ids(mask_info, mappings)
        if mask_ids.flags.writeable:
            _read_only(mask_ids)
        fixed = _remember(_FIXED_INPUT_CACHE, fixed_key, (cad_rgb, mask_info, mask_ids, mappings))
    cad_rgb, mask_info, mask_ids, mappings = fixed
    if mask_info.array.shape[:2] != cad_rgb.shape[:2]:
        raise ValueError(
            f"CAD image and part mask dimensions differ: {cad_rgb.shape[1]}x{cad_rgb.shape[0]} vs "
            f"{mask_info.width}x{mask_info.height}"
        )
    regions = extract_part_regions(mask_ids, mappings)
    # Masked-edge ECC registers photo and CAD edges inside the assembly; keypoint
    # matching (SIFT) between a real photo and a rendered CAD image fails on the
    # sample data, so it is only tried when ECC fails.
    attempted_alignments: list[AlignmentResult] = []
    alignment, aligned = align_masked_edges_ecc(photo_rgb, cad_rgb, mask_ids)
    attempted_alignments.append(alignment)
    if not alignment.success:
        LOGGER.warning("ECC failed for %s: %s; trying SIFT.", photo_name, alignment.reason)
        alignment, aligned = align_sift_ransac(photo_rgb, cad_rgb)
        attempted_alignments.append(alignment)
    if not alignment.success:
        # Retain whichever failure provides the stronger geometric diagnostic.
        alignment = max(
            attempted_alignments,
            key=lambda result: (
                result.homography_valid,
                result.inliers,
                result.inlier_spread_ratio or 0.0,
                result.inlier_ratio,
                result.correlation_score or 0.0,
            ),
        )
        aligned = None
    records = classify_regions(regions, aligned, cad_rgb, alignment)
    cad_annotated, photo_annotated = _annotate(cad_rgb, photo_rgb, regions, records, alignment)
    return {
        "case": directory.name,
        "photo": photo_path.name,
        "cad": cad_path.name,
        "mask": mask_path.name,
        "alignment": {
            **_alignment_record(alignment),
            "attempts": [_alignment_record(attempt) for attempt in attempted_alignments],
        },
        "parts": records,
        "counts": {status: sum(part["status"] == status for part in records) for status in PRESENCE_STATUSES},
        "images": {
            "photo": photo_rgb,
            "cad": cad_rgb,
            "mask": mask_info.array,
            "aligned": aligned,
            "photo_annotated": photo_annotated,
            "cad_annotated": cad_annotated,
            # Internal context reused by the orientation check; image arrays
            # and matrices are excluded by result_json().
            "mask_ids": mask_ids,
            "regions": regions,
            "matrix_photo_to_cad": alignment.matrix_photo_to_cad,
            "alignment_result": alignment,
        },
    }


def result_json(result: dict[str, Any]) -> str:
    """Serialize inspection records without large image arrays."""

    return json.dumps({key: value for key, value in result.items() if key != "images"}, indent=2)


def result_csv(result: dict[str, Any]) -> str:
    buffer = io.StringIO()
    preferred = ["part_name", "pixel_value", "status", "evidence_score", "mask_area_px", "component_count", "largest_component_area_px", "largest_component_fraction", "bbox_xyxy", "centroid_xy", "largest_component_bbox_xyxy", "largest_component_centroid_xy", "diagnostic"]
    extra = sorted({key for row in result["parts"] for key in row} - set(preferred))
    fields = [*preferred, *extra]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in result["parts"]:
        writer.writerow(row)
    return buffer.getvalue()


def discover_inspection_photos(dataset_root: str | Path) -> list[tuple[Path, str]]:
    """List (case directory, photo filename) pairs supported by the dataset."""

    root = Path(dataset_root).resolve()
    return [(case_dir, photo.name) for case_dir in discover_case_directories(root) for photo in discover_photos(case_dir)]
