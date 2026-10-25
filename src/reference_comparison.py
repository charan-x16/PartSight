"""Experimental real-photo-to-real-photo part comparison.

CAD and its part mask supply expected regions. A same-case photograph supplies
the local appearance template. The template's completeness must be confirmed
by an operator before candidate statuses are used as inspection decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import (
    _annotate,
    inspect_case,
    region_core,
)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson correlation, returning zero when a region has no variation."""
    a = np.asarray(first, dtype=np.float32).reshape(-1)
    b = np.asarray(second, dtype=np.float32).reshape(-1)
    if a.size < 16 or a.size != b.size:
        return 0.0
    a -= float(a.mean())
    b -= float(b.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-6:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def _appearance_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image)
    code = cv2.COLOR_RGBA2GRAY if rgb.shape[2] == 4 else cv2.COLOR_RGB2GRAY
    gray = cv2.cvtColor(rgb, code).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gray, cv2.magnitude(gx, gy)


def compare_reference_region(
    reference_rgb: np.ndarray,
    target_rgb: np.ndarray,
    region_mask: np.ndarray,
    *,
    max_shift_px: int = 6,
    shift_step_px: int = 2,
    reference_features: tuple[np.ndarray, np.ndarray] | None = None,
    target_features: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare local normalized intensity and gradient structure.

    Both images must already be warped into the CAD coordinate frame. A small
    translation search tolerates residual registration error. The score is the
    mean of intensity NCC and gradient-magnitude NCC; it is a similarity score,
    not a calibrated probability.
    """
    reference = np.asarray(reference_rgb)
    target = np.asarray(target_rgb)
    mask = np.asarray(region_mask, dtype=bool)
    if reference.ndim != 3 or target.ndim != 3 or reference.shape[:2] != target.shape[:2]:
        raise ValueError("Reference and target must be aligned RGB images of equal dimensions.")
    if reference.shape[2] not in {3, 4} or target.shape[2] not in {3, 4}:
        raise ValueError("Reference and target images must have three or four channels.")
    if mask.shape != reference.shape[:2]:
        raise ValueError("Part mask and aligned image dimensions must match.")
    if not mask.any():
        return {
            "measurable": False,
            "similarity": None,
            "intensity_ncc": None,
            "gradient_ncc": None,
            "shift_dx_px": None,
            "shift_dy_px": None,
            "pixel_count": 0,
            "reason": "Mapped part mask is empty.",
        }
    if max_shift_px < 0 or shift_step_px < 1:
        raise ValueError("max_shift_px must be non-negative and shift_step_px must be positive.")

    ref_gray, ref_gradient = reference_features or _appearance_features(reference)
    tar_gray, tar_gradient = target_features or _appearance_features(target)
    # Include a narrow border around the semantic mask to capture the part edge
    # while keeping distant background texture out of the comparison.
    support = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    support_y, support_x = np.nonzero(support)
    y0 = max(0, int(support_y.min()) - max_shift_px - 2)
    y1 = min(support.shape[0], int(support_y.max()) + max_shift_px + 3)
    x0 = max(0, int(support_x.min()) - max_shift_px - 2)
    x1 = min(support.shape[1], int(support_x.max()) + max_shift_px + 3)
    ref_gray = ref_gray[y0:y1, x0:x1]
    ref_gradient = ref_gradient[y0:y1, x0:x1]
    tar_gray = tar_gray[y0:y1, x0:x1]
    tar_gradient = tar_gradient[y0:y1, x0:x1]
    local_support = support[y0:y1, x0:x1]
    local_y, local_x = np.nonzero(local_support)
    # Large semantic regions need not spend time comparing every pixel. A
    # deterministic evenly spaced sample keeps the score reproducible.
    max_samples = 50_000
    if len(local_x) > max_samples:
        keep = np.linspace(0, len(local_x) - 1, max_samples, dtype=np.int64)
        local_x, local_y = local_x[keep], local_y[keep]
    reference_intensity = ref_gray[local_y, local_x]
    reference_gradient_values = ref_gradient[local_y, local_x]
    best: dict[str, Any] | None = None
    offsets = list(range(-max_shift_px, max_shift_px + 1, shift_step_px))
    if 0 not in offsets:
        offsets.append(0)
    for dy in offsets:
        for dx in offsets:
            sample_x, sample_y = local_x - dx, local_y - dy
            intensity_score = _correlation(
                reference_intensity, tar_gray[sample_y, sample_x]
            )
            gradient_score = _correlation(
                reference_gradient_values, tar_gradient[sample_y, sample_x]
            )
            similarity = 0.5 * intensity_score + 0.5 * gradient_score
            candidate = {
                "similarity": similarity,
                "intensity_ncc": intensity_score,
                "gradient_ncc": gradient_score,
                "shift_dx_px": dx,
                "shift_dy_px": dy,
                "pixel_count": int(np.count_nonzero(support)),
            }
            if best is None or candidate["similarity"] > best["similarity"]:
                best = candidate
    assert best is not None
    return {"measurable": True, **best, "reason": "Normalized local intensity and gradient comparison."}


def best_rotated_reference_similarity(
    reference_rgb: np.ndarray,
    target_rgb: np.ndarray,
    region_mask: np.ndarray,
    *,
    angle_step_deg: float = 5.0,
    max_shift_px: int = 24,
    shift_step_px: int = 4,
) -> dict[str, Any]:
    """Best reference similarity after rotating the reference about the part's axis.

    A fastener can be present but turned, which also produces a low unrotated
    similarity. If some in-plane rotation reproduces the target appearance,
    the part is still there; if none does, its appearance has changed.

    A round fastener turns about the centre of its outline. The centroid of a
    partly occluded part (e.g. a washer ring under a nut) lies off that axis,
    and rotating about it displaces the content far beyond any local shift, so
    the centre of the minimum enclosing circle of the outer contour is used.
    The ±24 px shift window covers residual alignment error plus the offset
    between the circle centre and the true rotation axis.
    """
    mask = np.asarray(region_mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {"measurable": False, "similarity": None, "rotation_deg": None, "rotation_center_xy": None}
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), _ = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    cx, cy = float(cx), float(cy)
    radius = int(np.ceil(np.hypot(xs - cx, ys - cy).max())) + max_shift_px + 6
    x0, y0 = max(0, int(cx) - radius), max(0, int(cy) - radius)
    x1, y1 = min(width, int(cx) + radius + 1), min(height, int(cy) + radius + 1)
    reference_crop = np.ascontiguousarray(np.asarray(reference_rgb)[y0:y1, x0:x1, :3])
    target_crop = np.ascontiguousarray(np.asarray(target_rgb)[y0:y1, x0:x1, :3])
    mask_crop = mask[y0:y1, x0:x1]
    target_features = _appearance_features(target_crop)
    best: dict[str, Any] | None = None
    # 0 deg is included: with the wider shift window it recognises a part that
    # is displaced (beyond the +/-6 px comparison window) but not rotated.
    for angle in np.arange(0.0, 360.0, angle_step_deg):
        rotation = cv2.getRotationMatrix2D((cx - x0, cy - y0), float(angle), 1.0)
        rotated = cv2.warpAffine(
            reference_crop, rotation, (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        result = compare_reference_region(
            rotated, target_crop, mask_crop,
            max_shift_px=max_shift_px, shift_step_px=shift_step_px,
            target_features=target_features,
        )
        if best is None or result["similarity"] > best["similarity"]:
            best = {**result, "rotation_deg": float(angle), "rotation_center_xy": [cx, cy]}
    assert best is not None
    return best


def classify_reference_similarity(
    similarity: float | None,
    *,
    alignment_valid: bool,
    reference_confirmed: bool,
    missing_threshold: float = 0.20,
    present_threshold: float = 0.55,
) -> dict[str, str]:
    """Return a candidate decision and a trust-gated final status."""
    if not alignment_valid:
        return {
            "candidate_status": "UNCERTAIN",
            "status": "UNCERTAIN",
            "reason": "Reference or target alignment failed; local regions are not comparable.",
        }
    if similarity is None:
        return {
            "candidate_status": "UNCERTAIN",
            "status": "UNCERTAIN",
            "reason": "No measurable local similarity score is available.",
        }
    if similarity >= present_threshold:
        candidate, evidence = "PRESENT", "Local appearance matches the real-photo reference."
    elif similarity <= missing_threshold:
        candidate, evidence = "MISSING", "Local appearance differs strongly from the real-photo reference."
    else:
        candidate, evidence = "UNCERTAIN", "Local similarity falls between the conservative decision thresholds."
    if reference_confirmed:
        return {"candidate_status": candidate, "status": candidate, "reason": evidence}
    return {
        "candidate_status": candidate,
        "status": "UNCERTAIN",
        "reason": f"{evidence} Candidate only: reference completeness has not been confirmed.",
    }


def combine_reference_and_baseline(
    baseline_status: str, reference_status: str, *, reference_only_absence: bool = False
) -> tuple[str, str]:
    """Require agreement before a reference-only absence claim is decisive.

    ``reference_only_absence`` may be set only when the reference is confirmed
    complete, the rest of the image matches it (so lighting and viewpoint are
    comparable), and no shift or in-plane rotation of the reference explains
    the part. CAD edge evidence cannot rule out absence when the expected
    region is covered by unrelated texture, so it does not veto that case.
    """
    if reference_status == "PRESENT":
        if baseline_status == "MISSING":
            return "UNCERTAIN", "Reference and baseline disagree about presence; manual review is needed."
        return "PRESENT", "Reference similarity supports presence and does not conflict with baseline evidence."
    if reference_status == "MISSING":
        if baseline_status == "MISSING":
            return "MISSING", "Reference dissimilarity agrees with baseline absence evidence."
        if reference_only_absence:
            return "MISSING", (
                "The part no longer matches the confirmed reference under any tested shift or rotation, "
                f"while the rest of the image does; baseline CAD-edge evidence was {baseline_status}, "
                "which edge texture covering the part can produce."
            )
        return "UNCERTAIN", (
            "Reference appearance differs, but the baseline does not independently support absence; "
            "lighting, reflections, or viewpoint may explain the difference."
        )
    return "UNCERTAIN", "Reference comparison is ambiguous; retaining UNCERTAIN."


def inspect_with_real_reference(
    case_dir: str | Path,
    photo_name: str,
    *,
    reference_photo: str | Path | None,
    reference_confirmed: bool = False,
    base_result: dict[str, Any] | None = None,
    reference_result: dict[str, Any] | None = None,
    max_shift_px: int = 6,
    shift_step_px: int = 2,
    missing_threshold: float = 0.20,
    present_threshold: float = 0.55,
) -> dict[str, Any]:
    """Run reference comparison, falling back to baseline when unavailable.

    References are case-local: this function never imports a reference from a
    different dataset case. The caller must explicitly confirm completeness.
    """
    directory = Path(case_dir).resolve()
    base = base_result if base_result is not None else inspect_case(
        directory, photo_name
    )
    photo_path = directory / photo_name
    reference_path = None if reference_photo is None else Path(reference_photo)
    if reference_path is not None and not reference_path.is_absolute():
        reference_path = (directory / reference_path).resolve()
    same_image = reference_path is not None and reference_path.is_file() and reference_path.resolve() == photo_path.resolve()
    if reference_path is None or not reference_path.is_file() or same_image:
        note = (
            "No separate same-case known-good reference is available; using the existing baseline."
            if reference_path is None or not reference_path.is_file()
            else "Selected reference is the inspection photo itself; no independent comparison is possible."
        )
        parts = [
            {
                **row,
                "baseline_status": row["status"],
                "candidate_status": row["status"],
                "reference_candidate_status": "UNCERTAIN" if same_image else row["status"],
                "status": row["status"],
                "inspection_outcome": row["status"],
                "reference_similarity": None,
                "reference_intensity_ncc": None,
                "reference_gradient_ncc": None,
                "comparison_bbox_xyxy": row.get("bbox_xyxy"),
                "reference_shift_dx_px": None,
                "reference_shift_dy_px": None,
                "diagnostic": f"{row['diagnostic']} {note}",
            }
            for row in base["parts"]
        ]
        fallback_counts = {
            status: sum(row["status"] == status for row in parts)
            for status in ("PRESENT", "MISSING", "UNCERTAIN")
        }
        return {
            **base,
            "mode": "Real-reference comparison (baseline fallback)",
            "reference": {
                "path": None if reference_path is None else str(reference_path),
                "confirmed_complete": False,
                "available": False,
                "self_comparison": bool(same_image),
                "warning": note,
            },
            "parts": parts,
            "counts": fallback_counts,
            "candidate_counts": {
                status: sum(row["candidate_status"] == status for row in parts)
                for status in ("PRESENT", "MISSING", "UNCERTAIN")
            },
            "images": {
                **base["images"],
                **({"photo_annotated": base["images"]["photo"], "cad_annotated": base["images"]["cad"]} if same_image else {}),
            },
        }

    if reference_path.parent.resolve() != directory:
        raise ValueError("Reference image must belong to the selected dataset case; cross-case references are disabled.")

    reference = reference_result if reference_result is not None else inspect_case(
        directory, reference_path.name
    )
    if (reference["cad"], reference["mask"]) != (base["cad"], base["mask"]):
        raise ValueError(
            f"Reference photo {reference_path.name} belongs to a different assembly "
            f"({reference['cad']}, {reference['mask']}); choose a photo of the same assembly."
        )
    target_aligned = base["images"].get("aligned")
    reference_aligned = reference["images"].get("aligned")
    alignment_valid = bool(
        base["alignment"].get("success")
        and reference["alignment"].get("success")
        and target_aligned is not None
        and reference_aligned is not None
        and target_aligned.shape[:2] == reference_aligned.shape[:2]
    )
    regions = base["images"]["regions"]
    reference_features = None
    if alignment_valid:
        # The reference is reused for every inspected photo; derive it once.
        derived = reference["images"].setdefault("_derived_cache", {})
        if "appearance_features" not in derived:
            derived["appearance_features"] = _appearance_features(reference_aligned)
        reference_features = derived["appearance_features"]
    target_features = _appearance_features(target_aligned) if alignment_valid else None
    baseline_by_value = {row["pixel_value"]: row for row in base["parts"]}
    # Stray boundary IDs in the rendered mask would compare unrelated, usually
    # unchanged pixels; compare each part's dominant component.
    core_masks = {
        region.stencil_value: region_core(region) if region.area else None
        for region in regions
    }
    # Same minimum per-part evidence as the CAD baseline: a few boundary
    # pixels cannot establish presence or absence of a physical part.
    assessable = {
        value: mask is not None and int(np.count_nonzero(mask)) >= 400
        for value, mask in core_masks.items()
    }
    comparisons: dict[int, dict[str, Any]] = {}
    if alignment_valid:
        for region in regions:
            if assessable[region.stencil_value]:
                comparisons[region.stencil_value] = compare_reference_region(
                    reference_aligned,
                    target_aligned,
                    core_masks[region.stencil_value],
                    max_shift_px=max_shift_px,
                    shift_step_px=shift_step_px,
                    reference_features=reference_features,
                    target_features=target_features,
                )
    # A part-level difference only indicates a change in that part when the
    # image as a whole still matches the reference.
    image_similarity = (
        float(np.median([c["similarity"] for c in comparisons.values()])) if comparisons else None
    )
    image_comparable = image_similarity is not None and image_similarity >= present_threshold
    # A confirmed reference is only used when it is also verified to be
    # registered with this photo: its unchanged parts must match.
    reference_usable = bool(reference_confirmed and alignment_valid and image_comparable)
    alignment_note = (
        "Reference alignment to this photo is not verified: "
        + (
            "global alignment failed."
            if not alignment_valid
            else f"median part similarity {image_similarity if image_similarity is not None else float('nan'):.3f} "
            f"is below {present_threshold:.2f}."
        )
    )
    parts: list[dict[str, Any]] = []
    for region in regions:
        baseline = baseline_by_value[region.stencil_value]
        rotation: dict[str, Any] | None = None
        if not assessable[region.stencil_value] or not alignment_valid:
            comparison = {
                "similarity": None,
                "intensity_ncc": None,
                "gradient_ncc": None,
                "shift_dx_px": None,
                "shift_dy_px": None,
                "pixel_count": 0,
            }
            decision = classify_reference_similarity(
                None,
                alignment_valid=alignment_valid,
                reference_confirmed=reference_confirmed,
                missing_threshold=missing_threshold,
                present_threshold=present_threshold,
            )
            if region.area == 0:
                reason = "Mapped part has no pixels in the CAD mask."
            elif not assessable[region.stencil_value]:
                reason = "Largest connected mask component is under 400 px; too little per-part evidence."
            else:
                reason = decision["reason"]
            if not assessable[region.stencil_value]:
                decision = {"candidate_status": "UNCERTAIN", "status": "UNCERTAIN", "reason": reason}
        else:
            comparison = comparisons[region.stencil_value]
            decision = classify_reference_similarity(
                float(comparison["similarity"]),
                alignment_valid=True,
                reference_confirmed=reference_confirmed,
                missing_threshold=missing_threshold,
                present_threshold=present_threshold,
            )
            if decision["candidate_status"] != "PRESENT":
                rotation = best_rotated_reference_similarity(
                    reference_aligned, target_aligned, core_masks[region.stencil_value]
                )
                if rotation["similarity"] is not None and rotation["similarity"] >= present_threshold:
                    if rotation["rotation_deg"] == 0.0:
                        explanation = (
                            f"Local appearance matches the reference after a shift of ({rotation['shift_dx_px']}, "
                            f"{rotation['shift_dy_px']}) px without rotation (similarity {rotation['similarity']:.3f}): "
                            "the part is present but displaced."
                        )
                    else:
                        explanation = (
                            f"Local appearance matches the reference after rotating it {rotation['rotation_deg']:.0f} deg "
                            f"(similarity {rotation['similarity']:.3f}): the part is present but its orientation differs."
                        )
                    decision = {**decision, "candidate_status": "PRESENT", "reason": explanation}
            reason = decision["reason"]
        reference_candidate = decision["candidate_status"]
        proposed_status, fusion_reason = combine_reference_and_baseline(
            str(baseline["status"]),
            reference_candidate,
            # Overriding the baseline requires that no tested pose of the
            # reference resembles the part, not merely that none matches well.
            reference_only_absence=bool(
                reference_confirmed
                and image_comparable
                and rotation is not None
                and rotation["similarity"] is not None
                and rotation["similarity"] <= missing_threshold
            ),
        )
        orientation_changed = bool(
            rotation is not None and reference_candidate == "PRESENT" and rotation["rotation_deg"] != 0.0
        )
        if rotation is not None and reference_candidate == "PRESENT":
            fusion_reason = f"{fusion_reason} {decision['reason']}"
        final_status = proposed_status if reference_usable else "UNCERTAIN"
        if not reference_confirmed:
            reason = f"{fusion_reason} Reference completeness is unconfirmed; final status remains UNCERTAIN."
        elif not reference_usable:
            reason = f"{fusion_reason} {alignment_note} Final status remains UNCERTAIN."
        else:
            reason = fusion_reason
        # Rotated parts remain PRESENT for presence counts; the separate outcome
        # keeps present, rotated, missing, and uncertain distinguishable.
        outcome = "ROTATED" if final_status == "PRESENT" and orientation_changed else final_status
        parts.append({
            **baseline,
            "baseline_status": baseline["status"],
            "reference_candidate_status": reference_candidate,
            "candidate_status": proposed_status,
            "status": final_status,
            "inspection_outcome": outcome,
            "evidence_score": comparison["similarity"] if comparison["similarity"] is not None else 0.0,
            "reference_similarity": comparison["similarity"],
            "reference_intensity_ncc": comparison["intensity_ncc"],
            "reference_gradient_ncc": comparison["gradient_ncc"],
            "comparison_bbox_xyxy": list(region.bbox) if region.bbox else None,
            "reference_shift_dx_px": comparison["shift_dx_px"],
            "reference_shift_dy_px": comparison["shift_dy_px"],
            "comparison_pixel_count": comparison["pixel_count"],
            "reference_rotated_similarity": rotation["similarity"] if rotation else None,
            "reference_rotation_deg": rotation["rotation_deg"] if rotation else None,
            "reference_orientation_changed": orientation_changed,
            "reference_rotation_center_xy": rotation.get("rotation_center_xy") if rotation else None,
            "diagnostic": f"{reason} (reference score={comparison['similarity']:.3f}; {decision['reason']})" if comparison["similarity"] is not None else reason,
        })
    cad_annotated, photo_annotated = _annotate(
        base["images"]["cad"], base["images"]["photo"], regions, parts,
        base["images"]["alignment_result"],
    )
    if not reference_confirmed:
        warning = (
            "Reference image completeness is unconfirmed. Candidate statuses are experimental; "
            "final statuses remain UNCERTAIN until a complete reference is confirmed."
        )
    elif not reference_usable:
        warning = f"{alignment_note} Reference-based statuses remain UNCERTAIN."
    else:
        warning = None
    return {
        **base,
        "mode": "Real-reference comparison",
        "reference": {
            "path": str(reference_path),
            "photo": reference_path.name,
            "confirmed_complete": reference_confirmed,
            "alignment_verified": reference_usable,
            "available": True,
            "warning": warning,
            "alignment": reference["alignment"],
        },
        "thresholds": {
            "missing_similarity_max": missing_threshold,
            "present_similarity_min": present_threshold,
            "max_shift_px": max_shift_px,
            "shift_step_px": shift_step_px,
        },
        "image_reference_similarity_median": image_similarity,
        "image_comparable_to_reference": image_comparable,
        "parts": parts,
        "counts": {status: sum(row["status"] == status for row in parts) for status in ("PRESENT", "MISSING", "UNCERTAIN")},
        "candidate_counts": {status: sum(row["candidate_status"] == status for row in parts) for status in ("PRESENT", "MISSING", "UNCERTAIN")},
        "outcome_counts": {
            outcome: sum(row["inspection_outcome"] == outcome for row in parts)
            for outcome in ("PRESENT", "ROTATED", "MISSING", "UNCERTAIN")
        },
        "images": {
            **base["images"],
            "photo_annotated": photo_annotated,
            "cad_annotated": cad_annotated,
        },
    }


def add_orientation_check(base: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    """Mark CAD-present parts that a confirmed known-good photo shows rotated.

    Present/absent decisions stay those of the CAD comparison (``base``). The
    known-good photo comparison only adds an orientation note: a part it
    matches only after rotating the reference is shown as present but rotated.
    """
    reference = comparison.get("reference", {})
    if not reference.get("alignment_verified"):
        warning = reference.get("warning") or "The known-good photo could not be verified as aligned with this photo."
        return {**base, "reference": {"warning": f"Orientation check skipped: {warning}"}}
    by_value = {row["pixel_value"]: row for row in comparison["parts"]}
    parts = []
    for row in base["parts"]:
        checked = by_value.get(row["pixel_value"], {})
        part = {**row, "inspection_outcome": row["status"]}
        rotation = checked.get("reference_rotation_deg")
        if row["status"] == "PRESENT" and checked.get("inspection_outcome") == "ROTATED" and rotation is not None:
            angle = (float(rotation) + 180.0) % 360.0 - 180.0
            direction = "counter-clockwise" if angle > 0 else "clockwise"
            part.update(
                inspection_outcome="ROTATED",
                reference_rotation_deg=rotation,
                diagnostic=(
                    f"{row['diagnostic']} Rotated about {abs(angle):.0f} deg {direction} compared with "
                    f"the known-good photo {reference.get('photo')}."
                ),
            )
        parts.append(part)
    cad_annotated, photo_annotated = _annotate(
        base["images"]["cad"], base["images"]["photo"], base["images"]["regions"], parts,
        base["images"]["alignment_result"],
    )
    return {
        **base,
        "parts": parts,
        "orientation_check": {"reference_photo": reference.get("photo")},
        "outcome_counts": {
            outcome: sum(part["inspection_outcome"] == outcome for part in parts)
            for outcome in ("PRESENT", "ROTATED", "MISSING", "UNCERTAIN", "NOT_VISIBLE")
        },
        "images": {**base["images"], "photo_annotated": photo_annotated, "cad_annotated": cad_annotated},
    }


__all__ = [
    "add_orientation_check",
    "best_rotated_reference_similarity",
    "compare_reference_region",
    "classify_reference_similarity",
    "combine_reference_and_baseline",
    "inspect_with_real_reference",
]
