"""Local geometric validation of part position, orientation, and size.

Measurements are image-space estimates in photo pixels, not millimetres. Two
bases are supported:

* CAD-relative: the global registration projects each part outline into the
  photo and a local search fits translation, in-plane rotation, and uniform
  scale of that outline to photo edges (truncated chamfer cost). On real photos
  this is often ambiguous for rotation and scale, which is reported as such.
* Reference-relative (confirmed known-good photo available): the part's region
  in the reference photo is matched in the inspected photo by masked appearance
  correlation after photo-to-photo registration, so offsets shared with the
  reference (CAD placement, height above the aligned surface) cancel.

Conventions
-----------
* Position offset ``(dx, dy)``: displacement of the part's projected centroid,
  +x to the right and +y down in the photo.
* Orientation: in-plane rotation of the outline about its projected centroid in
  degrees, positive = counter-clockwise as the image is displayed (the same
  sign as ``cv2.getRotationMatrix2D``). It approximates rotation about the
  viewing axis and is only meaningful for near-frontal views.
* Size: uniform scale of the outline, 1.0 = as expected.

Search ranges and resolution are fixed module constants, independent of the
tolerances, so changing a tolerance only changes the decision. No pose prior
is added to the cost; equal-cost candidates are tie-broken toward the smaller
deviation. Each axis gets an uncertainty from its cost profile (the interval in
which the cost rises by less than ``PROFILE_COST_DELTA_PX``), never smaller than
the search resolution, plus the registration uncertainty for position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detector import (
    AlignmentResult,
    PartRegion,
    _annotate,
    align_sift_ransac,
    foreground_edge_distance,
    inspect_case,
    region_core,
    result_csv,
    result_json,
)

TRANSLATION_SEARCH_PX = 28
COARSE_TRANSLATION_STEP_PX = 4
FINE_TRANSLATION_STEP_PX = 1.0
ROTATION_SEARCH_DEG = 15.0
COARSE_ROTATION_STEP_DEG = 2.5
FINE_ROTATION_STEP_DEG = 0.5
SCALE_SEARCH = (0.80, 1.25)
COARSE_SCALE_STEP = 0.05
FINE_SCALE_STEP = 0.01
EDGE_TOLERANCE_PX = 4.0
MIN_EDGE_SUPPORT = 0.42
# A quarter pixel of mean contour-to-edge distance is below what Canny edge
# localisation (about +/-0.5 px) can distinguish.
PROFILE_COST_DELTA_PX = 0.25
# An outline that moves less than one pixel on average when rotated across the
# whole search range carries no rotation information (e.g. a round washer).
SYMMETRY_DISPLACEMENT_PX = 1.0
# Photo-to-photo matching: a part must reach the reference-comparison presence
# similarity, and profile/ambiguity intervals use a 0.02 similarity margin.
MIN_APPEARANCE_SIMILARITY = 0.55
APPEARANCE_SCORE_DELTA = 0.02

GEOMETRY_OUTCOMES = ("WITHIN_TOLERANCE", "OUT_OF_TOLERANCE", "NOT_MEASURABLE", "NOT_APPLICABLE")
WITHIN, OUT, INCONCLUSIVE = "WITHIN", "OUT", "INCONCLUSIVE"
NOT_OBSERVABLE, AXIS_NOT_MEASURABLE = "NOT_OBSERVABLE", "NOT_MEASURABLE"


def _bbox_from_points(points: np.ndarray) -> list[int] | None:
    if points.size == 0 or not np.isfinite(points).all():
        return None
    low = np.floor(points.min(axis=0)).astype(int)
    high = np.ceil(points.max(axis=0)).astype(int) + 1
    return [int(low[0]), int(low[1]), int(high[0]), int(high[1])]


def edge_distance_map(photo_rgb: np.ndarray) -> np.ndarray | None:
    """Distance (px) to the nearest contrast-normalised Canny edge of a photo."""
    photo = np.asarray(photo_rgb)
    code = cv2.COLOR_RGBA2GRAY if photo.shape[2] == 4 else cv2.COLOR_RGB2GRAY
    gray = cv2.cvtColor(photo, code)
    edges = cv2.Canny(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray), 45, 130)
    if cv2.countNonZero(edges) < 20:
        return None
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)


def _bilinear(
    image: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    origin: tuple[int, int] = (0, 0),
    full_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear lookup; ``image`` may be a window at ``origin`` of a map of ``full_shape``.

    Validity and clipping always use the full map bounds, so a window that
    contains every reachable pixel gives exactly the full-map result.
    """
    height, width = full_shape or image.shape
    valid = (xs >= 0) & (xs <= width - 1) & (ys >= 0) & (ys <= height - 1)
    x = np.clip(xs, 0, width - 1.001)
    y = np.clip(ys, 0, height - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    xi, yi = x0 - origin[1], y0 - origin[0]
    values = (
        image[yi, xi] * (1 - fx) * (1 - fy) + image[yi, xi + 1] * fx * (1 - fy)
        + image[yi + 1, xi] * (1 - fx) * fy + image[yi + 1, xi + 1] * fx * fy
    )
    return values, valid


def _search_window(shape: tuple[int, int], center: np.ndarray, radius: float) -> tuple[int, int, int, int]:
    """Rows/columns ``(r0, r1, c0, c1)`` containing every pixel the pose search can read.

    A posed sample lies within ``radius * max_scale + max_translation`` of the
    centre on each axis (rotation preserves the radius); one extra pixel on
    each side covers truncation and the bilinear neighbour.
    """
    height, width = shape
    reach = float(radius) * SCALE_SEARCH[1] + TRANSLATION_SEARCH_PX + 2.0
    c0 = max(0, int(np.floor(center[0] - reach)) - 1)
    c1 = min(width, int(np.ceil(center[0] + reach)) + 2)
    r0 = max(0, int(np.floor(center[1] - reach)) - 1)
    r1 = min(height, int(np.ceil(center[1] + reach)) + 2)
    return r0, r1, c0, c1


def _rotate(points: np.ndarray, angle_deg: float, scale: float = 1.0) -> np.ndarray:
    """Rotate centred points; positive angle is counter-clockwise on screen."""
    radians = np.deg2rad(angle_deg)
    c, s = np.cos(radians), np.sin(radians)
    return points @ np.array([[c, s], [-s, c]], dtype=np.float64).T * scale


def _outline_cost(centered, center, distance_map):
    """Truncated chamfer cost of the posed outline against photo edges (lower is better)."""
    r0, r1, c0, c1 = _search_window(distance_map.shape, center, np.linalg.norm(centered, axis=1).max())
    window, origin, full_shape = np.ascontiguousarray(distance_map[r0:r1, c0:c1]), (r0, c0), distance_map.shape

    def evaluate(dxs, dys, angle, scale):
        shape = _rotate(centered, angle, scale) + center
        xs = shape[None, :, 0] + dxs[:, None]
        ys = shape[None, :, 1] + dys[:, None]
        distances, valid = _bilinear(window, xs, ys, origin, full_shape)
        count = valid.sum(axis=1)
        in_frame = count / centered.shape[0]
        truncated = np.where(valid, np.minimum(distances, EDGE_TOLERANCE_PX * 3), 0.0)
        # The out-of-frame term is not a pose prior.
        cost = truncated.sum(axis=1) / np.maximum(count, 1) + (1.0 - in_frame) * 10.0
        cost[in_frame < 0.75] = np.inf
        support = (valid & (distances <= EDGE_TOLERANCE_PX)).sum(axis=1) / centered.shape[0]
        return cost, support
    return evaluate


def appearance_maps(photo_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grey level and gradient magnitude used for photo-to-photo appearance matching."""
    photo = np.asarray(photo_rgb)
    code = cv2.COLOR_RGBA2GRAY if photo.shape[2] == 4 else cv2.COLOR_RGB2GRAY
    gray = cv2.cvtColor(photo, code).astype(np.float32)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return gray, gradient


def _normalised(values: np.ndarray) -> np.ndarray:
    values = values - values.mean()
    return values / max(float(np.linalg.norm(values)), 1e-6)


def _appearance_cost(reference_intensity, reference_gradient, target_maps, points, center):
    """1 - mean(intensity NCC, gradient-magnitude NCC) of reference samples vs the posed target."""
    centered = points - center
    r0, r1, c0, c1 = _search_window(target_maps[0].shape, center, np.linalg.norm(centered, axis=1).max())
    target_gray, target_gradient = (np.ascontiguousarray(target_map[r0:r1, c0:c1]) for target_map in target_maps)
    origin, full_shape = (r0, c0), target_maps[0].shape

    def ncc(values, reference):
        values = values - values.mean(axis=1, keepdims=True)
        return (values @ reference) / np.maximum(np.linalg.norm(values, axis=1), 1e-6)

    def evaluate(dxs, dys, angle, scale):
        shape = _rotate(centered, angle, scale) + center
        xs = shape[None, :, 0] + dxs[:, None]
        ys = shape[None, :, 1] + dys[:, None]
        intensity, valid = _bilinear(target_gray, xs, ys, origin, full_shape)
        gradient, _ = _bilinear(target_gradient, xs, ys, origin, full_shape)
        score = 0.5 * ncc(intensity, reference_intensity) + 0.5 * ncc(gradient, reference_gradient)
        cost = 1.0 - score
        cost[valid.mean(axis=1) < 0.95] = np.inf
        return cost, score
    return evaluate


def _grid_search(evaluate, txs, tys, angles, scales, record=None) -> dict[str, Any] | None:
    grid_x, grid_y = np.meshgrid(np.asarray(txs, float), np.asarray(tys, float))
    dxs, dys = grid_x.ravel(), grid_y.ravel()
    best: dict[str, Any] | None = None
    for scale in scales:
        for angle in angles:
            cost, support = evaluate(dxs, dys, float(angle), float(scale))
            if record is not None:
                record.append(np.column_stack((cost, dxs, dys, np.full_like(dxs, angle), np.full_like(dxs, scale))))
            for index in np.flatnonzero(cost <= cost.min() + 1e-9):
                candidate = {
                    "cost": float(cost[index]), "support": float(support[index]),
                    "dx": float(dxs[index]), "dy": float(dys[index]),
                    "angle": float(angle), "scale": float(scale),
                }
                # Equal-cost tie-break toward the smaller deviation (not a prior).
                key = (round(candidate["cost"], 9), abs(candidate["angle"]), abs(candidate["scale"] - 1.0),
                       np.hypot(candidate["dx"], candidate["dy"]))
                if best is None or key < best["key"]:
                    best = {**candidate, "key": key}
    return best


def _profile(values: np.ndarray, costs: np.ndarray, optimum: float, delta: float) -> tuple[float, bool]:
    """Half-width of the contiguous low-cost interval around the optimum and whether it is unbounded."""
    index = int(np.argmin(np.abs(values - optimum)))
    limit = costs[index] + delta
    low = high = index
    while low > 0 and costs[low - 1] <= limit:
        low -= 1
    while high < len(values) - 1 and costs[high + 1] <= limit:
        high += 1
    half_width = float(max(values[index] - values[low], values[high] - values[index]))
    return half_width, low == 0 or high == len(values) - 1


def _fit_pose(evaluate, symmetric: bool, delta: float) -> dict[str, Any] | None:
    """Coarse-to-fine pose search with per-axis profiles, bound checks, and ambiguity checks."""
    coarse_t = np.arange(-TRANSLATION_SEARCH_PX, TRANSLATION_SEARCH_PX + 1, COARSE_TRANSLATION_STEP_PX)
    coarse_a = [0.0] if symmetric else np.arange(-ROTATION_SEARCH_DEG, ROTATION_SEARCH_DEG + 1e-9, COARSE_ROTATION_STEP_DEG)
    coarse_s = np.round(np.arange(SCALE_SEARCH[0], SCALE_SEARCH[1] + 1e-9, COARSE_SCALE_STEP), 4)
    evaluated: list[np.ndarray] = []
    best = _grid_search(evaluate, coarse_t, coarse_t, coarse_a, coarse_s, evaluated)
    if best is None or not np.isfinite(best["cost"]):
        return None
    # Per-axis ambiguity: a pose more than two coarse steps away on that axis
    # explaining the image almost as well means the axis is not identifiable.
    candidates = np.vstack(evaluated)

    def competing(distinct: np.ndarray) -> bool:
        return bool(distinct.any() and candidates[distinct, 0].min() <= best["cost"] + delta)

    ambiguous_t = competing(
        (np.abs(candidates[:, 1] - best["dx"]) > 2 * COARSE_TRANSLATION_STEP_PX)
        | (np.abs(candidates[:, 2] - best["dy"]) > 2 * COARSE_TRANSLATION_STEP_PX)
    )
    ambiguous_s = competing(np.abs(candidates[:, 4] - best["scale"]) > 2 * COARSE_SCALE_STEP + 1e-9)
    ambiguous_a = not symmetric and competing(np.abs(candidates[:, 3] - best["angle"]) > 2 * COARSE_ROTATION_STEP_DEG)

    def fine(value, step, coarse_step, low, high):
        return np.unique(np.round(np.clip(np.arange(value - coarse_step, value + coarse_step + 1e-9, step), low, high), 4))

    best = _grid_search(
        evaluate,
        fine(best["dx"], FINE_TRANSLATION_STEP_PX, COARSE_TRANSLATION_STEP_PX, -TRANSLATION_SEARCH_PX, TRANSLATION_SEARCH_PX),
        fine(best["dy"], FINE_TRANSLATION_STEP_PX, COARSE_TRANSLATION_STEP_PX, -TRANSLATION_SEARCH_PX, TRANSLATION_SEARCH_PX),
        [0.0] if symmetric else fine(best["angle"], FINE_ROTATION_STEP_DEG, COARSE_ROTATION_STEP_DEG, -ROTATION_SEARCH_DEG, ROTATION_SEARCH_DEG),
        fine(best["scale"], FINE_SCALE_STEP, COARSE_SCALE_STEP, *SCALE_SEARCH),
    )
    full_t = np.arange(-TRANSLATION_SEARCH_PX, TRANSLATION_SEARCH_PX + 1, FINE_TRANSLATION_STEP_PX).astype(float)
    one_x, one_y = np.array([best["dx"]]), np.array([best["dy"]])
    cost_x, _ = evaluate(full_t, np.full_like(full_t, best["dy"]), best["angle"], best["scale"])
    cost_y, _ = evaluate(np.full_like(full_t, best["dx"]), full_t, best["angle"], best["scale"])
    full_s = np.round(np.arange(SCALE_SEARCH[0], SCALE_SEARCH[1] + 1e-9, FINE_SCALE_STEP), 4)
    cost_s = np.array([evaluate(one_x, one_y, best["angle"], s)[0][0] for s in full_s])
    fit = {
        "best": best,
        "ambiguous_translation": ambiguous_t, "ambiguous_scale": ambiguous_s, "ambiguous_rotation": ambiguous_a,
        "translation_at_bound": max(abs(best["dx"]), abs(best["dy"])) >= TRANSLATION_SEARCH_PX,
        "scale_at_bound": best["scale"] <= SCALE_SEARCH[0] or best["scale"] >= SCALE_SEARCH[1],
        "rotation_at_bound": not symmetric and abs(best["angle"]) >= ROTATION_SEARCH_DEG,
    }
    fit["half_x"], fit["open_x"] = _profile(full_t, cost_x, best["dx"], delta)
    fit["half_y"], fit["open_y"] = _profile(full_t, cost_y, best["dy"], delta)
    fit["half_s"], fit["open_s"] = _profile(full_s, cost_s, best["scale"], delta)
    if not symmetric:
        full_a = np.round(np.arange(-ROTATION_SEARCH_DEG, ROTATION_SEARCH_DEG + 1e-9, FINE_ROTATION_STEP_DEG), 4)
        cost_a = np.array([evaluate(one_x, one_y, a, best["scale"])[0][0] for a in full_a])
        fit["half_a"], fit["open_a"] = _profile(full_a, cost_a, best["angle"], delta)
    return fit


def _axis_results(
    fit: dict[str, Any],
    symmetric: bool,
    *,
    position_tolerance_px: float,
    size_tolerance_ratio: float,
    orientation_tolerance_deg: float,
    registration_uncertainty_px: float,
) -> dict[str, Any]:
    """Turn a pose fit into per-axis values, uncertainties, and tolerance results."""
    best = fit["best"]
    if fit["translation_at_bound"] or fit["scale_at_bound"]:
        limits = []
        if fit["translation_at_bound"]:
            limits.append(f"translation ({best['dx']:.0f}, {best['dy']:.0f}) px at the ±{TRANSLATION_SEARCH_PX} px limit")
        if fit["scale_at_bound"]:
            limits.append(f"scale {best['scale']:.2f} at the search limit")
        return _not_measurable("Fit hit the search boundary (" + "; ".join(limits) + ").")
    if fit["open_x"] or fit["open_y"]:
        return _not_measurable("Position is not resolved: the fit cost stays flat across the translation search range.")
    if fit["ambiguous_translation"]:
        return _not_measurable("Ambiguous fit: a distinctly different position matches almost as well.")
    notes = []
    position_fit = max(FINE_TRANSLATION_STEP_PX, float(np.hypot(fit["half_x"], fit["half_y"])))
    position_uncertainty = float(np.hypot(position_fit, registration_uncertainty_px))
    magnitude = float(np.hypot(best["dx"], best["dy"]))
    if symmetric:
        angle, angle_uncertainty, orientation = None, None, NOT_OBSERVABLE
    elif fit["rotation_at_bound"] or fit["open_a"] or fit["ambiguous_rotation"]:
        angle, angle_uncertainty, orientation = None, None, AXIS_NOT_MEASURABLE
        notes.append("rotation " + ("at the search limit" if fit["rotation_at_bound"] else "not uniquely determined"))
    else:
        angle = best["angle"]
        angle_uncertainty = max(FINE_ROTATION_STEP_DEG, fit["half_a"])
        orientation = _tolerance_result(abs(angle), angle_uncertainty, orientation_tolerance_deg)
    if fit["open_s"] or fit["ambiguous_scale"]:
        size, size_uncertainty, size_result = None, None, AXIS_NOT_MEASURABLE
        notes.append("scale not uniquely determined")
    else:
        size = best["scale"]
        size_uncertainty = max(FINE_SCALE_STEP, fit["half_s"])
        size_result = _tolerance_result(abs(size - 1.0), size_uncertainty, size_tolerance_ratio)
    position_result = _tolerance_result(magnitude, position_uncertainty, position_tolerance_px)
    return {
        "measurable": True,
        "position_offset_dx_px": best["dx"],
        "position_offset_dy_px": best["dy"],
        "position_offset_magnitude_px": magnitude,
        "position_uncertainty_fit_px": position_fit,
        "position_uncertainty_px": position_uncertainty,
        "position_result": position_result,
        "orientation_observable": not symmetric,
        "orientation_difference_deg": angle,
        "orientation_uncertainty_deg": angle_uncertainty,
        "orientation_result": orientation,
        "size_ratio": size,
        "size_uncertainty": size_uncertainty,
        "size_result": size_result,
        "position_out_of_tolerance": position_result == OUT,
        "orientation_out_of_tolerance": orientation == OUT,
        "size_out_of_tolerance": size_result == OUT,
        "axis_notes": notes,
    }


def _rotationally_symmetric(centered: np.ndarray) -> bool:
    """True when rotating the outline across the search range barely moves it."""
    offset = -centered.min(axis=0) + 5
    size = np.ceil(centered.max(axis=0) + offset + 6).astype(int)
    canvas = np.full((size[1], size[0]), 255, np.uint8)
    cv2.polylines(canvas, [np.rint(centered + offset).astype(np.int32)], True, 0, 1)
    distance = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)
    worst = 0.0
    for angle in (-ROTATION_SEARCH_DEG, -ROTATION_SEARCH_DEG / 2, ROTATION_SEARCH_DEG / 2, ROTATION_SEARCH_DEG):
        rotated = _rotate(centered, angle) + offset
        values, valid = _bilinear(distance, rotated[:, 0], rotated[:, 1])
        worst = max(worst, float(values[valid].mean()) if valid.any() else np.inf)
    return worst < SYMMETRY_DISPLACEMENT_PX


def _tolerance_result(magnitude: float, uncertainty: float, tolerance: float) -> str:
    if magnitude - uncertainty > tolerance:
        return OUT
    if magnitude + uncertainty <= tolerance:
        return WITHIN
    return INCONCLUSIVE


def _not_measurable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"measurable": False, "reason": reason, **extra}


def _sampled_outline(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(contour) < 20:
        return None
    if len(contour) > 300:
        contour = contour[np.linspace(0, len(contour) - 1, 300, dtype=int)]
    return contour


def measure_local_component_geometry(
    component_mask: np.ndarray,
    photo_rgb: np.ndarray,
    cad_to_photo: np.ndarray,
    *,
    position_tolerance_px: float = 12.0,
    size_tolerance_ratio: float = 0.10,
    orientation_tolerance_deg: float = 5.0,
    registration_uncertainty_px: float = 0.0,
    distance_map: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure a part's position, orientation, and size residual against its CAD outline.

    Tolerances only drive the per-axis decisions. A definitive decision is
    made only when value +/- uncertainty lies entirely on one side of the
    tolerance; otherwise the axis is ``INCONCLUSIVE``. Rotation of a
    rotationally symmetric outline is ``NOT_OBSERVABLE`` and not reported;
    an axis with a competing, distinctly different fit is ``NOT_MEASURABLE``.
    """
    mask = (np.asarray(component_mask) > 0).astype(np.uint8)
    photo = np.asarray(photo_rgb)
    if mask.ndim != 2 or photo.ndim != 3 or photo.shape[2] not in {3, 4}:
        return _not_measurable("Unsupported mask or photograph format.")
    if not np.isfinite(cad_to_photo).all() or abs(float(np.linalg.det(cad_to_photo))) < 1e-12:
        return _not_measurable("CAD-to-photo transform is invalid.")
    contour = _sampled_outline(mask)
    if contour is None:
        return _not_measurable("Too few contour pixels for a stable measurement.")
    transform = np.asarray(cad_to_photo, np.float64)
    projected = cv2.perspectiveTransform(contour[None, ...].astype(np.float64), transform)[0]
    if not np.isfinite(projected).all():
        return _not_measurable("Global transform projects non-finite component coordinates.")
    moments = cv2.moments(mask)
    if abs(moments["m00"]) < 1e-8:
        return _not_measurable("Part mask has no measurable area.")
    center = cv2.perspectiveTransform(
        np.array([[[moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]]]), transform
    )[0, 0]
    if distance_map is None:
        distance_map = edge_distance_map(photo)
    if distance_map is None:
        return _not_measurable("Photograph has too few local edges.")
    centered = projected - center
    symmetric = _rotationally_symmetric(centered)
    fit = _fit_pose(_outline_cost(centered, center, distance_map), symmetric, PROFILE_COST_DELTA_PX)
    if fit is None:
        return _not_measurable("Expected part contour falls outside the photograph.")
    best = fit["best"]
    if best["support"] < MIN_EDGE_SUPPORT:
        return _not_measurable(f"Local silhouette support is weak ({best['support']:.0%}).", edge_support=best["support"])
    axes = _axis_results(
        fit, symmetric, position_tolerance_px=position_tolerance_px, size_tolerance_ratio=size_tolerance_ratio,
        orientation_tolerance_deg=orientation_tolerance_deg, registration_uncertainty_px=registration_uncertainty_px,
    )
    if not axes["measurable"]:
        return {**axes, "edge_support": best["support"]}
    adjusted = _rotate(centered, best["angle"], best["scale"]) + center + np.array([best["dx"], best["dy"]])
    x, y, width, height = cv2.boundingRect(mask)
    corners_cad = np.float32([[[x, y], [x + width - 1, y], [x + width - 1, y + height - 1], [x, y + height - 1]]])
    expected_bbox = _bbox_from_points(cv2.perspectiveTransform(corners_cad.astype(np.float64), transform)[0])
    observed_bbox = _bbox_from_points(adjusted)
    outline_bbox = _bbox_from_points(projected)
    width_ratio = height_ratio = None
    if outline_bbox and observed_bbox:
        width_ratio = (observed_bbox[2] - observed_bbox[0]) / max(1, outline_bbox[2] - outline_bbox[0])
        height_ratio = (observed_bbox[3] - observed_bbox[1]) / max(1, outline_bbox[3] - outline_bbox[1])
    note = f" ({'; '.join(axes['axis_notes'])})" if axes["axis_notes"] else ""
    return {
        **axes,
        "edge_support": best["support"],
        "expected_photo_bbox_xyxy": expected_bbox,
        "observed_photo_bbox_xyxy": observed_bbox,
        "expected_center_xy": [float(center[0]), float(center[1])],
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "reason": f"Local silhouette fit passed the edge-support threshold{note}.",
    }


def measure_reference_deviation(
    component_mask: np.ndarray,
    reference_rgb: np.ndarray,
    photo_rgb: np.ndarray,
    cad_to_reference: np.ndarray,
    reference_to_photo: np.ndarray,
    *,
    position_tolerance_px: float = 12.0,
    size_tolerance_ratio: float = 0.10,
    orientation_tolerance_deg: float = 5.0,
    registration_uncertainty_px: float = 0.0,
    reference_maps: tuple[np.ndarray, np.ndarray] | None = None,
    photo_maps: tuple[np.ndarray, np.ndarray] | None = None,
    max_samples: int = 3000,
) -> dict[str, Any]:
    """Deviation of a part from the same part in a confirmed known-good photo.

    The part's region in the reference photo (CAD mask mapped by the
    reference alignment, plus a 3 px rim for its outline) is matched in the
    target photo by masked intensity and gradient-magnitude correlation over
    translation, rotation, and scale about its centroid, after the
    photo-to-photo registration. No CAD outline enters the measurement, so CAD
    placement and height offsets shared with the reference cancel; viewpoint
    differences between the photos are only modelled by the registration.
    """
    mask = (np.asarray(component_mask) > 0).astype(np.uint8)
    reference = np.asarray(reference_rgb)
    height, width = reference.shape[:2]
    region = cv2.warpPerspective(mask, np.asarray(cad_to_reference, np.float64), (width, height), flags=cv2.INTER_NEAREST)
    region = cv2.dilate(region, np.ones((7, 7), np.uint8))
    outline = _sampled_outline(region)
    ys, xs = np.nonzero(region)
    if outline is None or xs.size < 200:
        return _not_measurable("Part region in the reference photo is too small to match.")
    if xs.size > max_samples:
        keep = np.linspace(0, xs.size - 1, max_samples, dtype=int)
        xs, ys = xs[keep], ys[keep]
    reference_maps = reference_maps or appearance_maps(reference)
    photo_maps = photo_maps or appearance_maps(np.asarray(photo_rgb))
    reference_intensity = _normalised(reference_maps[0][ys, xs].astype(np.float64))
    reference_gradient = _normalised(reference_maps[1][ys, xs].astype(np.float64))
    if np.allclose(reference_intensity, 0) and np.allclose(reference_gradient, 0):
        return _not_measurable("Reference part region has no appearance structure to match.")
    moments = cv2.moments(region)
    reference_center = np.array([[[moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]]])
    registration = np.asarray(reference_to_photo, np.float64)
    points = cv2.perspectiveTransform(np.column_stack((xs, ys)).astype(np.float64)[None], registration)[0]
    center = cv2.perspectiveTransform(reference_center, registration)[0, 0]
    outline_photo = cv2.perspectiveTransform(outline[None].astype(np.float64), registration)[0]
    symmetric = _rotationally_symmetric(outline_photo - center)
    evaluate = _appearance_cost(reference_intensity, reference_gradient, photo_maps, points, center)
    fit = _fit_pose(evaluate, symmetric, APPEARANCE_SCORE_DELTA)
    if fit is None:
        return _not_measurable("Part region falls outside the photograph.")
    best = fit["best"]
    if best["support"] < MIN_APPEARANCE_SIMILARITY:
        return _not_measurable(
            f"The part does not match its reference appearance under any tested pose "
            f"(best similarity {best['support']:.2f} < {MIN_APPEARANCE_SIMILARITY:.2f}).",
            edge_support=best["support"],
        )
    axes = _axis_results(
        fit, symmetric, position_tolerance_px=position_tolerance_px, size_tolerance_ratio=size_tolerance_ratio,
        orientation_tolerance_deg=orientation_tolerance_deg, registration_uncertainty_px=registration_uncertainty_px,
    )
    if not axes["measurable"]:
        return {**axes, "edge_support": best["support"]}
    posed = _rotate(outline_photo - center, best["angle"], best["scale"]) + center + np.array([best["dx"], best["dy"]])
    note = f" ({'; '.join(axes['axis_notes'])})" if axes["axis_notes"] else ""
    return {
        **axes,
        "edge_support": best["support"],
        "expected_photo_bbox_xyxy": _bbox_from_points(outline_photo),
        "observed_photo_bbox_xyxy": _bbox_from_points(posed),
        "expected_center_xy": [float(center[0]), float(center[1])],
        "width_ratio": None,
        "height_ratio": None,
        "reason": f"Matched to the confirmed reference part (similarity {best['support']:.2f}){note}.",
    }


def geometry_outcome(presence_status: str, measurement: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Combine presence and per-axis results into one geometry outcome.

    Returns the outcome, a message, and the list of axes out of tolerance.
    """
    if presence_status == "MISSING":
        return "NOT_APPLICABLE", "Part is reported missing; its geometry is not evaluated.", []
    if presence_status != "PRESENT":
        return (
            "NOT_MEASURABLE",
            f"Presence is {presence_status}; a contour fit alone cannot establish that the part is there.",
            [],
        )
    if not measurement.get("measurable"):
        return "NOT_MEASURABLE", measurement.get("reason", "Local geometry could not be measured."), []
    axes = {
        "position": measurement["position_result"],
        "orientation": measurement["orientation_result"],
        "size": measurement["size_result"],
    }
    out = [axis for axis, result in axes.items() if result == OUT]

    def describe(label, value, uncertainty, fmt, unit, result):
        if value is None:
            return f"{label} {result}"
        return f"{label} {value:{fmt}}±{uncertainty:{fmt.lstrip('+')}}{unit} {result}"

    summary = "; ".join((
        describe("position", measurement.get("position_offset_magnitude_px"), measurement.get("position_uncertainty_px"), ".1f", "px", axes["position"]),
        describe("orientation", measurement.get("orientation_difference_deg"), measurement.get("orientation_uncertainty_deg"), "+.1f", "°", axes["orientation"]),
        describe("size", measurement.get("size_ratio"), measurement.get("size_uncertainty"), ".3f", "", axes["size"]),
    ))
    if out:
        return "OUT_OF_TOLERANCE", f"Out of tolerance in {', '.join(out)}: {summary}.", out
    required_ok = axes["position"] == WITHIN and axes["size"] == WITHIN
    orientation_ok = axes["orientation"] in {WITHIN, NOT_OBSERVABLE}
    if required_ok and orientation_ok:
        note = " Orientation is not observable from this outline." if axes["orientation"] == NOT_OBSERVABLE else ""
        return "WITHIN_TOLERANCE", f"Within tolerance: {summary}.{note}", []
    return (
        "NOT_MEASURABLE",
        f"No definitive decision: uncertainty overlaps a tolerance or an axis is unresolved ({summary}).",
        [],
    )


def orientation_from_reference_rotation(
    rotation_deg: float,
    component_mask: np.ndarray,
    cad_to_photo: np.ndarray,
    *,
    orientation_tolerance_deg: float = 5.0,
    step_deg: float = 5.0,
) -> dict[str, Any]:
    """Orientation evidence from the confirmed-reference rotation search of presence inspection.

    That search matches the part only after rotating the reference by
    ``rotation_deg`` (counter-clockwise, 5 deg steps), so the orientation
    deviation is that angle wrapped to (-180, 180] with half a step of
    uncertainty. Position and size are not measured by it. A rotationally
    symmetric outline keeps orientation ``NOT_OBSERVABLE``.
    """
    contour = _sampled_outline((np.asarray(component_mask) > 0).astype(np.uint8))
    symmetric = True
    if contour is not None:
        projected = cv2.perspectiveTransform(contour[None].astype(np.float64), np.asarray(cad_to_photo, np.float64))[0]
        symmetric = _rotationally_symmetric(projected - projected.mean(axis=0))
    angle = float((rotation_deg + 180.0) % 360.0 - 180.0)
    if angle == -180.0:
        angle = 180.0
    uncertainty = step_deg / 2.0
    orientation = NOT_OBSERVABLE if symmetric else _tolerance_result(abs(angle), uncertainty, orientation_tolerance_deg)
    return {
        "measurable": True,
        "edge_support": None,
        "position_offset_dx_px": None, "position_offset_dy_px": None, "position_offset_magnitude_px": None,
        "position_uncertainty_fit_px": None, "position_uncertainty_px": None, "position_result": AXIS_NOT_MEASURABLE,
        "orientation_observable": not symmetric,
        "orientation_difference_deg": None if symmetric else angle,
        "orientation_uncertainty_deg": None if symmetric else uncertainty,
        "orientation_result": orientation,
        "size_ratio": None, "size_uncertainty": None, "size_result": AXIS_NOT_MEASURABLE,
        "position_out_of_tolerance": False,
        "orientation_out_of_tolerance": orientation == OUT,
        "size_out_of_tolerance": False,
        "reason": (
            f"Orientation from the confirmed-reference rotation search ({rotation_deg:.0f} deg CCW); "
            "the rotation exceeds the local geometry search range, so position and size are not measured."
        ),
    }


def register_reference_to_photo(reference_rgb: np.ndarray, photo_rgb: np.ndarray) -> tuple[np.ndarray | None, float | None, str]:
    """Reference-photo -> target-photo homography and its residual in target pixels."""
    result, _ = align_sift_ransac(reference_rgb, photo_rgb)
    if not result.success or result.matrix_photo_to_cad is None:
        return None, None, f"Photo-to-photo registration failed: {result.reason}"
    height, width = np.asarray(photo_rgb).shape[:2]
    # SIFT runs on images resized to at most 1600 px; report the residual at full resolution.
    residual = float(result.median_reprojection_error or 0.0) * max(1.0, max(height, width) / 1600.0)
    return result.matrix_photo_to_cad, residual, f"SIFT/RANSAC photo-to-photo registration ({result.inliers} inliers)"


_EMPTY_GEOMETRY = {
    "expected_photo_bbox_xyxy": None,
    "observed_photo_bbox_xyxy": None,
    "position_offset_dx_px": None,
    "position_offset_dy_px": None,
    "position_offset_magnitude_px": None,
    "position_uncertainty_px": None,
    "position_result": None,
    "orientation_difference_deg": None,
    "orientation_uncertainty_deg": None,
    "orientation_result": None,
    "size_ratio": None,
    "size_uncertainty": None,
    "size_result": None,
    "width_ratio": None,
    "height_ratio": None,
    "position_out_of_tolerance": None,
    "orientation_out_of_tolerance": None,
    "size_out_of_tolerance": None,
    "edge_support": None,
    "cad_relative_dx_px": None,
    "cad_relative_dy_px": None,
    "cad_relative_orientation_deg": None,
    "cad_relative_size_ratio": None,
    "reference_cad_relative_dx_px": None,
    "reference_cad_relative_dy_px": None,
}


def inspect_geometry_case(
    case_dir: str | Path,
    photo_name: str,
    *,
    position_tolerance_px: float = 12.0,
    size_tolerance_ratio: float = 0.10,
    orientation_tolerance_deg: float = 5.0,
    base_result: dict[str, Any] | None = None,
    alignment_method: str = "ecc",
    reference_photo: str | Path | None = None,
    reference_confirmed: bool = False,
    reference_result: dict[str, Any] | None = None,
    presence_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run geometric validation for one photograph.

    Presence comes from the confirmed-reference comparison when one is
    available and verified, otherwise from the CAD baseline. With a usable
    reference, deviations are measured relative to the same part in the
    reference photo; otherwise relative to the CAD outline, with the global
    registration error added to the position uncertainty.
    """
    base = base_result if base_result is not None else inspect_case(case_dir, photo_name, alignment_method=alignment_method)
    images = base["images"]
    alignment: AlignmentResult = images["alignment_result"]
    matrix = images["matrix_photo_to_cad"]
    photo_rgb = images["photo"]
    cad_to_photo = np.linalg.inv(matrix) if alignment.success and matrix is not None else None
    case_path = Path(case_dir).resolve()

    presence = base
    presence_source = "CAD baseline"
    reference_usable = False
    reference_note = None
    if reference_photo is not None and reference_confirmed:
        from .reference_comparison import inspect_with_real_reference

        if reference_result is None and (case_path / Path(reference_photo).name).is_file():
            reference_result = inspect_case(case_path, Path(reference_photo).name, alignment_method=alignment_method)
        presence = presence_result if presence_result is not None else inspect_with_real_reference(
            case_path, photo_name, reference_photo=reference_photo, reference_confirmed=True,
            base_result=base, reference_result=reference_result, alignment_method=alignment_method,
        )
        reference_usable = bool(presence.get("reference", {}).get("alignment_verified"))
        if reference_usable:
            presence_source = f"confirmed reference {Path(reference_photo).name}"
        else:
            presence = base
            reference_note = (
                "Reference not usable for this photo (not verified as aligned); "
                "CAD baseline presence and CAD-relative geometry are used."
            )

    # Registration between photos for reference-relative geometry.
    reference_to_photo = cad_to_reference = None
    photo_registration_residual = None
    registration_note = None
    if reference_usable and reference_result is not None and cad_to_photo is not None:
        reference_matrix = reference_result["images"]["matrix_photo_to_cad"]
        cad_to_reference = np.linalg.inv(reference_matrix)
        reference_to_photo, photo_registration_residual, registration_note = register_reference_to_photo(
            reference_result["images"]["photo"], photo_rgb
        )
        if reference_to_photo is None:
            reference_usable = False
            reference_note = f"{registration_note}; geometry falls back to CAD-relative measurement."

    cad_registration_uncertainty = alignment.foreground_edge_distance_px
    if cad_registration_uncertainty is None and images.get("aligned") is not None:
        cad_registration_uncertainty = foreground_edge_distance(images["aligned"], images["cad"], images["mask_ids"])
    cad_registration_uncertainty = float(cad_registration_uncertainty or 0.0)

    regions: list[PartRegion] = images["regions"]
    presence_by_value = {row["pixel_value"]: row for row in presence["parts"]}
    tolerances = {
        "position_tolerance_px": position_tolerance_px,
        "size_tolerance_ratio": size_tolerance_ratio,
        "orientation_tolerance_deg": orientation_tolerance_deg,
    }
    photo_distance = edge_distance_map(photo_rgb) if cad_to_photo is not None else None
    reference_distance = reference_maps = photo_maps = None
    reference_cache: dict[Any, Any] = {}
    if reference_usable and reference_result is not None:
        # The confirmed reference is reused for every inspected photo; its
        # derived maps and per-part measurements are computed once.
        reference_cache = reference_result["images"].setdefault("_derived_cache", {})
        if "edge_distance_map" not in reference_cache:
            reference_cache["edge_distance_map"] = edge_distance_map(reference_result["images"]["photo"])
            reference_cache["appearance_maps"] = appearance_maps(reference_result["images"]["photo"])
        reference_distance = reference_cache["edge_distance_map"]
        reference_maps = reference_cache["appearance_maps"]
        photo_maps = appearance_maps(photo_rgb)
    records: list[dict[str, Any]] = []
    for region in regions:
        presence_row = presence_by_value[region.stencil_value]
        presence_status = str(presence_row["status"])
        cad_measurement: dict[str, Any] = {}
        if region.area == 0:
            measurement = _not_measurable("Mapped value has no pixels in the CAD mask.")
        elif cad_to_photo is None:
            measurement = _not_measurable(f"Global alignment failed: {alignment.reason}")
        elif region.component_count > 20 and region.largest_component_area / region.area < 0.10:
            measurement = _not_measurable(
                f"Mask geometry is fragmented ({region.component_count} components; "
                f"largest is {region.largest_component_area / region.area:.1%}); local dimensions are unreliable."
            )
        else:
            core = region_core(region)
            cad_measurement = measure_local_component_geometry(
                core, photo_rgb, cad_to_photo, registration_uncertainty_px=cad_registration_uncertainty,
                distance_map=photo_distance, **tolerances,
            )
            measurement = cad_measurement
            if reference_usable and reference_to_photo is not None:
                measurement = measure_reference_deviation(
                    core, reference_result["images"]["photo"], photo_rgb, cad_to_reference, reference_to_photo,
                    registration_uncertainty_px=photo_registration_residual or 0.0,
                    reference_maps=reference_maps, photo_maps=photo_maps, **tolerances,
                )
                rotation_deg = presence_row.get("reference_rotation_deg")
                if (
                    not measurement.get("measurable")
                    and presence_row.get("inspection_outcome") == "ROTATED"
                    and rotation_deg is not None
                ):
                    measurement = orientation_from_reference_rotation(
                        float(rotation_deg), core, cad_to_photo, orientation_tolerance_deg=orientation_tolerance_deg
                    )
                # Informational: the reference part's own offset from the CAD outline.
                part_key = ("reference_part", region.stencil_value, tuple(sorted(tolerances.items())))
                if part_key not in reference_cache:
                    reference_cache[part_key] = measure_local_component_geometry(
                        core, reference_result["images"]["photo"], cad_to_reference,
                        distance_map=reference_distance, **tolerances,
                    )
                measurement["_reference_part"] = reference_cache[part_key]
        outcome, reason, deviation_axes = geometry_outcome(presence_status, measurement)
        cad_outcome = geometry_outcome(presence_status, cad_measurement)[0] if cad_measurement else outcome
        record = {
            "part_name": region.name,
            "pixel_value": region.stencil_value,
            "status": outcome,
            "presence_status": presence_status,
            "presence_outcome": presence_row.get("inspection_outcome", presence_status),
            "presence_source": presence_source,
            "geometry_basis": "reference-relative" if reference_usable else "CAD-relative",
            "cad_relative_status": cad_outcome,
            "deviation_axes": ",".join(deviation_axes),
            "evidence_score": float(measurement.get("edge_support") or 0.0),
            "diagnostic": reason,
            "mask_area_px": region.area,
            "component_count": region.component_count,
            "largest_component_area_px": region.largest_component_area,
            "largest_component_fraction": region.largest_component_area / region.area if region.area else 0.0,
            "cad_bbox_xyxy": list(region.bbox) if region.bbox else None,
            "cad_centroid_xy": list(region.centroid) if region.centroid else None,
            **_EMPTY_GEOMETRY,
        }
        # Geometry values are only published for parts established as present;
        # a recess or panel edge can produce a plausible contour fit.
        if presence_status == "PRESENT" and measurement.get("measurable"):
            record.update({
                key: value for key, value in measurement.items()
                if key in _EMPTY_GEOMETRY or key in {"position_uncertainty_fit_px", "orientation_observable"}
            })
            if cad_measurement.get("measurable"):
                record.update({
                    "cad_relative_dx_px": cad_measurement["position_offset_dx_px"],
                    "cad_relative_dy_px": cad_measurement["position_offset_dy_px"],
                    "cad_relative_orientation_deg": cad_measurement["orientation_difference_deg"],
                    "cad_relative_size_ratio": cad_measurement["size_ratio"],
                })
            reference_part = measurement.get("_reference_part") or {}
            if reference_part.get("measurable"):
                record.update({
                    "reference_cad_relative_dx_px": reference_part["position_offset_dx_px"],
                    "reference_cad_relative_dy_px": reference_part["position_offset_dy_px"],
                })
        records.append(record)

    cad_annotated, photo_annotated = _annotate(images["cad"], photo_rgb, regions, records, alignment)
    photo_view = Image.fromarray(photo_annotated)
    draw = ImageDraw.Draw(photo_view)
    colors = {"WITHIN_TOLERANCE": (20, 185, 70), "OUT_OF_TOLERANCE": (230, 45, 45)}
    font = ImageFont.load_default(size=22)
    for record in records:
        bbox = record["observed_photo_bbox_xyxy"]
        if bbox is None or record["status"] not in colors:
            continue
        color = colors[record["status"]]
        draw.rectangle(tuple(bbox), outline=color, width=3)
        if record["deviation_axes"]:
            draw.text((bbox[0], bbox[3] + 4), record["deviation_axes"], font=font, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
    photo_annotated = np.asarray(photo_view)

    axis_counts = {
        f"{axis}_out_of_tolerance": sum(bool(row[f"{axis}_out_of_tolerance"]) for row in records)
        for axis in ("position", "orientation", "size")
    }
    return {
        "case": base["case"],
        "photo": base["photo"],
        "cad": base["cad"],
        "mask": base["mask"],
        "mode": "Geometric Validation",
        "alignment": base["alignment"],
        "geometry_basis": "reference-relative" if reference_usable else "CAD-relative",
        "presence_source": presence_source,
        "registration": {
            "cad_registration_uncertainty_px": cad_registration_uncertainty,
            "photo_registration": registration_note,
            "photo_registration_residual_px": photo_registration_residual,
        },
        "reference": {"warning": reference_note} if reference_note else {},
        "tolerances": tolerances,
        "parts": records,
        "counts": {outcome: sum(row["status"] == outcome for row in records) for outcome in GEOMETRY_OUTCOMES},
        "axis_counts": axis_counts,
        "images": {
            "photo": photo_rgb,
            "cad": images["cad"],
            "mask": images["mask"],
            "photo_annotated": photo_annotated,
            "cad_annotated": cad_annotated,
        },
    }


__all__ = [
    "GEOMETRY_OUTCOMES",
    "geometry_outcome",
    "inspect_geometry_case",
    "measure_local_component_geometry",
    "measure_reference_deviation",
    "orientation_from_reference_rotation",
    "register_reference_to_photo",
    "result_csv",
    "result_json",
]
