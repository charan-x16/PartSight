"""Semi-synthetic geometry validation data and scoring.

This is evaluation tooling, not detection logic. Fastener groups are found from
the mask: small parts whose dominant regions touch form one physical stack
(e.g. a washer under a nut). Each generated image applies one known similarity
transform to one group about the group centre, or removes the group; every
other part is an unchanged control. The intended transform is the ground truth
relative to the known-good reference photograph.

Limitation: a pasted patch rotates its shading and highlights with it, which a
physically rotated part under fixed lighting would not do, and resampling
softens the pasted pixels. Results on these images do not establish accuracy
on real displaced parts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import largest_component_mask

TRANSFORMS: tuple[dict[str, Any], ...] = (
    {"label": "unchanged"},
    {"label": "shift_inside", "dx": 7.0, "dy": -5.0},
    {"label": "shift_outside", "dx": 14.0, "dy": 6.0},
    {"label": "shift_far", "dx": -18.0, "dy": 10.0},
    {"label": "rotate_inside_ccw", "angle": 3.0},
    {"label": "rotate_outside_ccw", "angle": 8.0},
    {"label": "rotate_outside_cw", "angle": -8.0},
    {"label": "scale_inside", "scale": 1.05},
    {"label": "scale_outside_up", "scale": 1.14},
    {"label": "scale_outside_down", "scale": 0.88},
    {"label": "combined_inside", "dx": 5.0, "dy": 4.0, "angle": 2.0, "scale": 1.04},
    {"label": "removed", "remove": True},
)
GROUND_TRUTH_FIELDS = (
    "image", "transform", "group", "pixel_value", "part_name", "role",
    "intended_dx_px", "intended_dy_px", "intended_offset_px", "intended_rotation_deg", "intended_scale",
    "expected_position", "expected_orientation", "expected_size", "expected_outcome",
)


def fastener_groups(mask_ids: np.ndarray, *, min_area: int = 2000, max_fraction: float = 0.02) -> list[dict[str, Any]]:
    """Group small parts whose dominant mask regions touch into fastener stacks."""
    ids = np.asarray(mask_ids)
    cores: dict[int, np.ndarray] = {}
    for value in np.unique(ids):
        if value == 0:
            continue
        core = largest_component_mask(ids == value)
        area = int(core.sum())
        if min_area <= area <= max_fraction * ids.size:
            cores[int(value)] = core
    union = np.zeros(ids.shape, np.uint8)
    for core in cores.values():
        union |= cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8))
    _, labels = cv2.connectedComponents(union)
    grouped: dict[int, list[int]] = {}
    for value, core in cores.items():
        label = int(np.bincount(labels[core]).argmax())
        grouped.setdefault(label, []).append(value)
    groups = []
    for values in grouped.values():
        mask = np.zeros(ids.shape, bool)
        for value in values:
            mask |= cores[value]
        groups.append({"values": sorted(values), "cad_mask": mask, "area": int(mask.sum())})
    return sorted(groups, key=lambda group: -group["area"])


def _expected(value: float, tolerance: float) -> str:
    return "OUT" if value > tolerance else "WITHIN"


def generate_case(
    photo_rgb: np.ndarray,
    group: dict[str, Any],
    transform: dict[str, Any],
    cad_to_photo: np.ndarray,
    *,
    margin_px: int = 15,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (edited image, edited-region mask, the applied similarity) for one group."""
    photo = np.asarray(photo_rgb)
    height, width = photo.shape[:2]
    if transform["label"] == "unchanged":
        return photo.copy(), np.zeros((height, width), bool), {"matrix": np.eye(3)[:2], "center": None}
    group_mask = cv2.warpPerspective(
        group["cad_mask"].astype(np.uint8), cad_to_photo, (width, height), flags=cv2.INTER_NEAREST
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
    patch = cv2.dilate(group_mask, kernel)
    hole = cv2.dilate(patch, np.ones((11, 11), np.uint8))
    contour = max(cv2.findContours(group_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    matrix = cv2.getRotationMatrix2D((cx, cy), transform.get("angle", 0.0), transform.get("scale", 1.0))
    matrix[:, 2] += (transform.get("dx", 0.0), transform.get("dy", 0.0))
    # Work in a region of interest around the edit.
    pad = int(radius + 2 * margin_px + 60)
    x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
    x1, y1 = min(width, int(cx) + pad), min(height, int(cy) + pad)
    roi = np.ascontiguousarray(photo[y0:y1, x0:x1])
    background = cv2.inpaint(roi, hole[y0:y1, x0:x1], 9, cv2.INPAINT_TELEA)
    edited = photo.copy()
    edit_region = hole.astype(bool)
    if transform.get("remove"):
        edited[y0:y1, x0:x1] = background
    else:
        local = matrix.copy()
        local[:, 2] += local[:, :2] @ np.array([x0, y0]) - np.array([x0, y0])
        warped = cv2.warpAffine(roi, local, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        moved = cv2.warpAffine(patch[y0:y1, x0:x1].astype(np.float32), local, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        alpha = np.clip(cv2.GaussianBlur(moved, (0, 0), 3), 0, 1)[..., None]
        edited[y0:y1, x0:x1] = np.clip(alpha * warped + (1 - alpha) * background, 0, 255).astype(np.uint8)
        edit_region[y0:y1, x0:x1] |= moved > 0
    return edited, edit_region, {"matrix": matrix, "center": (cx, cy)}


def ground_truth_rows(
    image_name: str,
    transform: dict[str, Any],
    group_index: int,
    group: dict[str, Any],
    applied: dict[str, Any],
    edit_region: np.ndarray,
    mask_ids: np.ndarray,
    names: dict[int, str],
    cad_to_photo: np.ndarray,
    tolerances: dict[str, float],
) -> list[dict[str, Any]]:
    """Ground truth for every assessable part (dominant region >= 400 px) of one generated image."""
    rows = []
    height, width = edit_region.shape
    for value, name in names.items():
        core = largest_component_mask(mask_ids == value)
        if core.sum() < 400:
            continue
        moved = value in group["values"] and transform["label"] != "unchanged"
        moments = cv2.moments(core.astype(np.uint8))
        centroid = cv2.perspectiveTransform(
            np.array([[[moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]]]), cad_to_photo
        )[0, 0]
        row: dict[str, Any] = {"image": image_name, "transform": transform["label"], "group": group_index,
                               "pixel_value": value, "part_name": name}
        if moved and transform.get("remove"):
            row.update(role="removed", expected_outcome="NOT_APPLICABLE")
        elif moved:
            new = applied["matrix"] @ np.array([centroid[0], centroid[1], 1.0])
            dx, dy = float(new[0] - centroid[0]), float(new[1] - centroid[1])
            angle, scale = transform.get("angle", 0.0), transform.get("scale", 1.0)
            expected = {
                "expected_position": _expected(np.hypot(dx, dy), tolerances["position_tolerance_px"]),
                "expected_orientation": _expected(abs(angle), tolerances["orientation_tolerance_deg"]),
                "expected_size": _expected(abs(scale - 1.0), tolerances["size_tolerance_ratio"]),
            }
            row.update(
                role="modified", intended_dx_px=dx, intended_dy_px=dy, intended_offset_px=float(np.hypot(dx, dy)),
                intended_rotation_deg=angle, intended_scale=scale, **expected,
                expected_outcome="OUT_OF_TOLERANCE" if "OUT" in expected.values() else "WITHIN_TOLERANCE",
            )
        else:
            core_photo = cv2.warpPerspective(core.astype(np.uint8), cad_to_photo, (width, height), flags=cv2.INTER_NEAREST) > 0
            row.update(
                role="control_adjacent" if (core_photo & edit_region).any() else "control",
                intended_dx_px=0.0, intended_dy_px=0.0, intended_offset_px=0.0, intended_rotation_deg=0.0,
                intended_scale=1.0, expected_position="WITHIN", expected_orientation="WITHIN",
                expected_size="WITHIN", expected_outcome="WITHIN_TOLERANCE",
            )
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_part(truth: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    """Compare one measured geometry record with its ground-truth row."""
    outcome = measured["status"]
    expected = truth["expected_outcome"]
    row = {**truth, "measured_outcome": outcome, "presence_outcome": measured.get("presence_outcome"),
           "cad_relative_status": measured.get("cad_relative_status"), "diagnostic": measured.get("diagnostic")}
    for key in ("position_offset_dx_px", "position_offset_dy_px", "position_offset_magnitude_px",
                "position_uncertainty_px", "position_result", "orientation_difference_deg",
                "orientation_uncertainty_deg", "orientation_result", "size_ratio", "size_uncertainty",
                "size_result", "cad_relative_dx_px", "cad_relative_dy_px"):
        row[key] = measured.get(key)
    if expected == "NOT_APPLICABLE":
        row["evaluation"] = "CORRECT" if outcome == "NOT_APPLICABLE" else "WRONG"
    elif outcome in {"WITHIN_TOLERANCE", "OUT_OF_TOLERANCE"}:
        if outcome == expected:
            row["evaluation"] = "CORRECT"
        else:
            row["evaluation"] = "FALSE_ALARM" if outcome == "OUT_OF_TOLERANCE" else "MISSED_DEVIATION"
    else:
        row["evaluation"] = "ABSTAINED"
    if truth["role"] == "modified" and measured.get("position_offset_dx_px") is not None:
        row["error_dx_px"] = measured["position_offset_dx_px"] - float(truth["intended_dx_px"])
        row["error_dy_px"] = measured["position_offset_dy_px"] - float(truth["intended_dy_px"])
        if measured.get("orientation_difference_deg") is not None:
            row["error_rotation_deg"] = measured["orientation_difference_deg"] - float(truth["intended_rotation_deg"])
        if measured.get("size_ratio") is not None:
            row["error_scale"] = measured["size_ratio"] - float(truth["intended_scale"])
    for axis in ("position", "orientation", "size"):
        wanted, got = truth.get(f"expected_{axis}"), measured.get(f"{axis}_result")
        if wanted in {"WITHIN", "OUT"} and got in {"WITHIN", "OUT"} and wanted != got:
            row[f"{axis}_axis_error"] = "FALSE_ALARM" if got == "OUT" else "MISSED"
    return row


SCORE_FIELDS = (
    *GROUND_TRUTH_FIELDS, "measured_outcome", "evaluation", "presence_outcome",
    "position_offset_dx_px", "position_offset_dy_px", "position_offset_magnitude_px", "position_uncertainty_px",
    "position_result", "orientation_difference_deg", "orientation_uncertainty_deg", "orientation_result",
    "size_ratio", "size_uncertainty", "size_result", "error_dx_px", "error_dy_px", "error_rotation_deg",
    "error_scale", "position_axis_error", "orientation_axis_error", "size_axis_error",
    "cad_relative_status", "cad_relative_dx_px", "cad_relative_dy_px", "diagnostic",
)


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    array = np.abs(np.asarray(values, float))
    return f"n={len(array)}, mean |err| {array.mean():.2f}, max |err| {array.max():.2f}"


def write_report(path: Path, scored: list[dict[str, Any]], context: dict[str, Any]) -> None:
    """Concise markdown report of measurement errors, false alarms, misses, and abstentions."""
    def subset(**criteria):
        return [row for row in scored if all(row.get(key) in values for key, values in criteria.items())]

    modified = subset(role={"modified"})
    controls = subset(role={"control"})
    adjacent = subset(role={"control_adjacent"})
    removed = subset(role={"removed"})
    lines = [
        "# PartSight Problem 2: semi-synthetic geometry validation",
        "",
        f"Reference: `{context['reference']}` (confirmed known-good). Tolerances: position {context['tolerances']['position_tolerance_px']} px, "
        f"orientation {context['tolerances']['orientation_tolerance_deg']} deg, size {context['tolerances']['size_tolerance_ratio']:.0%}.",
        "Ground truth is the transform applied to a pasted fastener group, relative to the reference photo. "
        "Values are photo pixels, not millimetres. Pasted patches rotate their shading with them and are resampled, "
        "so these results do not establish accuracy on physically displaced parts.",
        "",
        "## Outcomes",
        "",
        "| Role | Parts | Correct | False alarm | Missed deviation | Abstained | Wrong (removed) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, rows in (("modified", modified), ("control", controls), ("control next to edit", adjacent), ("removed", removed)):
        count = lambda name: sum(row["evaluation"] == name for row in rows)  # noqa: E731
        lines.append(f"| {label} | {len(rows)} | {count('CORRECT')} | {count('FALSE_ALARM')} | "
                     f"{count('MISSED_DEVIATION')} | {count('ABSTAINED')} | {count('WRONG')} |")
    lines += ["", "## Measurement error on modified parts (measured - intended)", ""]
    lines.append(f"- Position dx: {_stats([r['error_dx_px'] for r in modified if r.get('error_dx_px') is not None])} px")
    lines.append(f"- Position dy: {_stats([r['error_dy_px'] for r in modified if r.get('error_dy_px') is not None])} px")
    lines.append(f"- Rotation: {_stats([r['error_rotation_deg'] for r in modified if r.get('error_rotation_deg') is not None])} deg")
    lines.append(f"- Scale: {_stats([r['error_scale'] for r in modified if r.get('error_scale') is not None])}")
    lines += ["", "## Per-axis decisions on modified parts", "", "| Axis | Expected OUT: detected / missed / undecided | Expected WITHIN: correct / false alarm / undecided | Not observable |", "|---|---|---|---:|"]
    for axis in ("position", "orientation", "size"):
        out_rows = [r for r in modified if r.get(f"expected_{axis}") == "OUT"]
        in_rows = [r for r in modified if r.get(f"expected_{axis}") == "WITHIN"]
        res = lambda rows, value: sum(r.get(f"{axis}_result") == value for r in rows)  # noqa: E731
        undecided = lambda rows: sum(r.get(f"{axis}_result") not in {"OUT", "WITHIN", "NOT_OBSERVABLE"} for r in rows)  # noqa: E731
        lines.append(
            f"| {axis} | {res(out_rows, 'OUT')} / {res(out_rows, 'WITHIN')} / {undecided(out_rows)} | "
            f"{res(in_rows, 'WITHIN')} / {res(in_rows, 'OUT')} / {undecided(in_rows)} | "
            f"{sum(r.get(f'{axis}_result') == 'NOT_OBSERVABLE' for r in modified)} |"
        )
    measured_controls = [r for r in controls if r.get("position_offset_magnitude_px") is not None]
    lines += [
        "",
        "## Unchanged controls",
        "",
        f"- Reference-relative |offset|: {_stats([r['position_offset_magnitude_px'] for r in measured_controls])} px "
        f"(ideal 0); rotation {_stats([r['orientation_difference_deg'] for r in measured_controls if r.get('orientation_difference_deg') is not None])} deg; "
        f"scale-1 {_stats([r['size_ratio'] - 1 for r in measured_controls if r.get('size_ratio') is not None])}.",
        f"- CAD-relative |offset| of the same controls: "
        f"{_stats([np.hypot(r['cad_relative_dx_px'], r['cad_relative_dy_px']) for r in measured_controls if r.get('cad_relative_dx_px') is not None])} px "
        "(systematic CAD/registration/height offsets that the reference comparison cancels).",
        f"- Controls that a CAD-only comparison would flag OUT_OF_TOLERANCE: "
        f"{sum(r.get('cad_relative_status') == 'OUT_OF_TOLERANCE' for r in controls)} of {len(controls)}.",
        "",
        "## Why parts were not measured",
        "",
    ]
    reasons: dict[str, int] = {}
    for row in scored:
        if row["evaluation"] != "ABSTAINED":
            continue
        text = row.get("diagnostic") or ""
        key = next((label for label, needle in (
            ("presence not established", "contour fit alone"), ("ambiguous fit", "Ambiguous"),
            ("weak edge support", "weak"), ("search boundary", "search boundary"),
            ("reference part not measurable", "Reference part is not measurable"),
            ("uncertainty overlaps tolerance / axis unresolved", "No definitive decision"),
            ("position not resolved", "Position is not resolved"), ("too little mask", "Too few contour"),
        ) if needle in text), "other")
        reasons[key] = reasons.get(key, 0) + 1
    lines += [f"- {key}: {count}" for key, count in sorted(reasons.items(), key=lambda item: -item[1])] or ["- none"]
    lines += ["", "## Incorrect decisions", ""]
    wrong = [r for r in scored if r["evaluation"] in {"FALSE_ALARM", "MISSED_DEVIATION", "WRONG"}]
    lines += [
        f"- {r['image']} part {r['pixel_value']} ({r['role']}, {r['transform']}): expected {r['expected_outcome']}, "
        f"measured {r['measured_outcome']} — {r.get('diagnostic', '')}"
        for r in wrong
    ] or ["- none"]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
