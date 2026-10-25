"""CAD-only part presence decisions (Problem Statement 1).

Each real photo is aligned to the CAD render. Every part listed in the JSON is
then judged at its expected location, using four local cues measured on the
aligned photo:

* shape support: fraction of the part's CAD edges with a photo edge within
  3 px, allowing a local shift of up to 16 px for residual misalignment;
* edge density: photo edge pixels per part pixel;
* texture: grey-level standard deviation inside the part;
* contrast: difference between the mean grey level inside the part and in a
  ring around it.

Identical parts in one photo (names that differ only in a trailing number,
e.g. ``large_washer_1`` .. ``large_washer_5``) should give similar cues, so a
part that is clearly weaker than its same-type siblings on the cues together
is absent, whatever removed or covered it. A part whose region is essentially
blank is absent. A part with no assessable sibling falls back to its absolute
outline support. Parts too small in the CAD view to assess are reported
``NOT_VISIBLE`` and never guessed.

The decision limits below were chosen by inspecting the sample images; they
have no independent validation.
"""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

MIN_PART_PIXELS = 400
SHIFT_SEARCH_PX = 16
SHIFT_STEP_PX = 2
EDGE_MATCH_PX = 3.0
RING_WIDTH_PX = 20
RELATIVE_EVIDENCE_LIMIT = 0.6
ABSOLUTE_SHAPE_LIMIT = 0.5
BLANK_TEXTURE_STD = 3.0
BLANK_EDGE_DENSITY = 0.005
LOW_MARGIN = 0.15
RATIO_CAP = 2.0
CUES = ("shape_support", "edge_density", "texture_std", "ring_contrast")
# Floors keep ratios finite when a sibling cue is near zero.
CUE_FLOORS = {"shape_support": 0.01, "edge_density": 0.001, "texture_std": 1.0, "ring_contrast": 1.0}
CUE_LABELS = {
    "shape_support": "outline support", "edge_density": "edge density",
    "texture_std": "texture", "ring_contrast": "contrast with surroundings",
}


def part_type(name: str) -> str:
    """Type of a part: its name without a trailing instance number (``bolt_3`` -> ``bolt``)."""
    return re.sub(r"[\s_\-.]*\d+(\.\d+)?$", "", name.strip()) or name


def _photo_features(aligned_rgb: np.ndarray, cad_rgb: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray), 45, 130) > 0
    return {
        "gray": gray,
        "edges": edges,
        "distance": cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3),
        "cad_edges": cv2.Canny(cv2.cvtColor(cad_rgb, cv2.COLOR_RGB2GRAY), 50, 140) > 0,
    }


def measure_cues(core: np.ndarray, features: dict[str, np.ndarray]) -> dict[str, Any]:
    """Local presence cues of one part (``core`` is its mask in CAD coordinates)."""
    ys, xs = np.nonzero(core)
    pad = RING_WIDTH_PX + SHIFT_SEARCH_PX + 8
    height, width = core.shape
    y0, y1 = max(0, ys.min() - pad), min(height, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(width, xs.max() + pad + 1)
    part = core[y0:y1, x0:x1].astype(np.uint8)
    gray = features["gray"][y0:y1, x0:x1]
    edges = features["edges"][y0:y1, x0:x1]
    distance = features["distance"][y0:y1, x0:x1]
    near = cv2.dilate(part, np.ones((7, 7), np.uint8)) > 0
    inner = cv2.erode(part, np.ones((5, 5), np.uint8)) > 0
    if not inner.any():
        inner = part > 0
    ring = (cv2.dilate(part, np.ones((2 * RING_WIDTH_PX + 1,) * 2, np.uint8)) > 0) & ~near

    edge_y, edge_x = np.nonzero(features["cad_edges"][y0:y1, x0:x1] & near)
    shape_support, shift = None, None
    if edge_x.size:
        offsets = np.arange(-SHIFT_SEARCH_PX, SHIFT_SEARCH_PX + 1, SHIFT_STEP_PX)
        dy, dx = (grid.ravel() for grid in np.meshgrid(offsets, offsets, indexing="ij"))
        py = np.clip(edge_y[None, :] + dy[:, None], 0, distance.shape[0] - 1)
        px = np.clip(edge_x[None, :] + dx[:, None], 0, distance.shape[1] - 1)
        support = (distance[py, px] <= EDGE_MATCH_PX).mean(axis=1)
        best = int(np.argmax(support))
        shape_support, shift = float(support[best]), (int(dx[best]), int(dy[best]))
    return {
        "shape_support": shape_support,
        "shape_shift_px": shift,
        "edge_density": float(edges[inner].mean()),
        "texture_std": float(gray[inner].std()),
        "ring_contrast": float(abs(gray[inner].mean() - gray[ring].mean())) if ring.any() else 0.0,
    }


def _is_blank(cues: dict[str, Any]) -> bool:
    return cues["texture_std"] < BLANK_TEXTURE_STD and cues["edge_density"] < BLANK_EDGE_DENSITY


def relative_evidence(cues: dict[str, Any], siblings: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    """Geometric mean of the part's cues relative to the median of its siblings (1 = typical)."""
    ratios = {}
    for cue in CUES:
        values = [sibling[cue] for sibling in siblings if sibling[cue] is not None]
        if cues[cue] is None or not values:
            continue
        floor = CUE_FLOORS[cue]
        ratios[cue] = min(RATIO_CAP, (cues[cue] + floor) / (float(np.median(values)) + floor))
    score = float(np.exp(np.mean(np.log(list(ratios.values()))))) if ratios else 0.0
    return score, ratios


def _record(region, status: str, score: float, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "part_name": region.name,
        "pixel_value": region.stencil_value,
        "status": status,
        "evidence_score": score,
        "mask_area_px": region.area,
        "component_count": region.component_count,
        "largest_component_area_px": region.largest_component_area,
        "largest_component_fraction": region.largest_component_area / region.area if region.area else 0.0,
        "largest_component_bbox_xyxy": list(region.largest_component_bbox) if region.largest_component_bbox else None,
        "largest_component_centroid_xy": list(region.largest_component_centroid) if region.largest_component_centroid else None,
        "bbox_xyxy": list(region.bbox) if region.bbox else None,
        "centroid_xy": list(region.centroid) if region.centroid else None,
        "part_type": part_type(region.name),
        "diagnostic": reason,
        **extra,
    }


def classify_presence(regions: list, aligned_rgb: np.ndarray | None, cad_rgb: np.ndarray, alignment) -> list[dict[str, Any]]:
    """Present / absent decision for every mapped part from CAD-only evidence."""
    from .detector import region_core

    if not alignment.success or aligned_rgb is None:
        return [_record(region, "UNCERTAIN", 0.0, f"Alignment failed: {alignment.reason}") for region in regions]
    features = _photo_features(aligned_rgb, cad_rgb)
    cues_by_value: dict[int, dict[str, Any]] = {}
    hidden: dict[int, str] = {}
    for region in regions:
        if region.area == 0:
            hidden[region.stencil_value] = "No pixels in the CAD mask: the part is hidden in this CAD view."
            continue
        core = region_core(region)
        core_area = int(core.sum())
        if core_area < MIN_PART_PIXELS:
            hidden[region.stencil_value] = (
                f"Only {core_area} px of the part is visible in the CAD view (minimum {MIN_PART_PIXELS}); "
                "it cannot be checked reliably."
            )
            continue
        cues_by_value[region.stencil_value] = measure_cues(core, features)

    by_type: dict[str, list[int]] = {}
    for region in regions:
        if region.stencil_value in cues_by_value:
            by_type.setdefault(part_type(region.name), []).append(region.stencil_value)

    records = []
    for region in regions:
        value = region.stencil_value
        if value in hidden:
            records.append(_record(region, "NOT_VISIBLE", 0.0, hidden[value], decision_basis="not visible in CAD"))
            continue
        cues = cues_by_value[value]
        shape = cues["shape_support"]
        siblings = [cues_by_value[other] for other in by_type[part_type(region.name)] if other != value]
        reference = [sibling for sibling in siblings if not _is_blank(sibling)]
        extra = {**cues, "sibling_count": len(reference), "relative_evidence": None}
        if _is_blank(cues):
            records.append(_record(
                region, "MISSING", 0.0,
                f"The part's region is blank (texture {cues['texture_std']:.1f}, edge density "
                f"{cues['edge_density']:.3f}): nothing is there.",
                decision_basis="blank region", **extra,
            ))
            continue
        if reference:
            score, ratios = relative_evidence(cues, reference)
            weak = [CUE_LABELS[cue] for cue, ratio in ratios.items() if ratio < RELATIVE_EVIDENCE_LIMIT]
            margin = " Close to the decision limit." if abs(score - RELATIVE_EVIDENCE_LIMIT) < LOW_MARGIN else ""
            extra.update(relative_evidence=score, decision_basis=f"compared with {len(reference)} same-type part(s)")
            if score >= RELATIVE_EVIDENCE_LIMIT:
                status = "PRESENT"
                reason = f"Looks like its {len(reference)} same-type part(s) (relative evidence {score:.2f})."
            else:
                status = "MISSING"
                reason = (
                    f"Much weaker than its {len(reference)} same-type part(s) (relative evidence {score:.2f} "
                    f"< {RELATIVE_EVIDENCE_LIMIT}); weak cues: {', '.join(weak) or 'all slightly'}."
                )
            records.append(_record(region, status, score, reason + margin, **extra))
            continue
        extra["decision_basis"] = "outline support (no same-type part to compare)"
        if shape is not None and shape >= ABSOLUTE_SHAPE_LIMIT:
            records.append(_record(
                region, "PRESENT", shape,
                f"No same-type part to compare with; {shape:.0%} of its CAD outline is found in the photo.", **extra,
            ))
        else:
            records.append(_record(
                region, "UNCERTAIN", shape or 0.0,
                "No same-type part to compare with, and its CAD outline is only "
                f"{0 if shape is None else shape:.0%} supported by photo edges.", **extra,
            ))
    return records
