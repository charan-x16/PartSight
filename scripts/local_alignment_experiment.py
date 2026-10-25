"""Experiment: does local rigid-surface alignment around each fastener reduce CAD-relative offsets?

For each part measurable against the CAD outline, CAD edges of the surrounding
rigid surface (large mask regions only, in a ring around the part) are fitted to
photo edges by a local translation. If the systematic part offsets were local
surface misregistration, subtracting that shift would shrink them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import inspect_case, largest_component_mask  # noqa: E402
from src.geometric import edge_distance_map, measure_local_component_geometry  # noqa: E402


def surface_shift(core, ids, cad_gray, cad_to_photo, distance, *, ring=(15, 80), search=15, min_points=80):
    """Translation (photo px) that best aligns nearby rigid-surface CAD edges, or None."""
    large = np.zeros(ids.shape, bool)
    for value in np.unique(ids):
        if value and (ids == value).sum() > 0.02 * ids.size:
            large |= ids == value
    outer = cv2.dilate(core.astype(np.uint8), np.ones((2 * ring[1] + 1,) * 2, np.uint8)) > 0
    inner = cv2.dilate(core.astype(np.uint8), np.ones((2 * ring[0] + 1,) * 2, np.uint8)) > 0
    edges = (cv2.Canny(cad_gray, 50, 140) > 0) & outer & ~inner & large
    ys, xs = np.nonzero(edges)
    if len(xs) < min_points:
        return None, len(xs)
    points = cv2.perspectiveTransform(np.float32(np.column_stack((xs, ys)))[None], cad_to_photo)[0]
    h, w = distance.shape
    best = None
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            px = np.rint(points[:, 0] + dx).astype(int)
            py = np.rint(points[:, 1] + dy).astype(int)
            valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            cost = float(np.minimum(distance[py[valid], px[valid]], 12).mean())
            if best is None or cost < best[0]:
                best = (cost, dx, dy)
    if abs(best[1]) == search or abs(best[2]) == search:
        return None, len(xs)
    return (best[1], best[2]), len(xs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=None, help="case_dir:photo (repeatable)")
    args = parser.parse_args()
    root = PROJECT_ROOT / "data" / "AI_sample_CAD"
    cases = args.case or [f"{root}:BG_00.png", f"{root / 'TestCase2'}:BG_00.png"]
    before, after = [], []
    for spec in cases:
        case_dir, photo = spec.rsplit(":", 1)
        result = inspect_case(Path(case_dir), photo)
        images = result["images"]
        cad_to_photo = np.linalg.inv(images["matrix_photo_to_cad"])
        ids = images["mask_ids"]
        cad_gray = cv2.cvtColor(images["cad"], cv2.COLOR_RGB2GRAY)
        distance = edge_distance_map(images["photo"])
        print(f"{Path(case_dir).name}/{photo}")
        for row in result["parts"]:
            core = largest_component_mask(ids == row["pixel_value"])
            if core.sum() < 2000 or core.sum() > 0.02 * ids.size:
                continue
            m = measure_local_component_geometry(core, images["photo"], cad_to_photo, distance_map=distance)
            if not m.get("measurable"):
                continue
            shift, points = surface_shift(core, ids, cad_gray, cad_to_photo, distance)
            offset = np.array([m["position_offset_dx_px"], m["position_offset_dy_px"]])
            if shift is None:
                print(f"   part {row['pixel_value']:2}: offset {offset} |{np.linalg.norm(offset):.1f}| px; "
                      f"no usable surface structure ({points} edge px)")
                continue
            corrected = offset - np.array(shift)
            before.append(np.linalg.norm(offset))
            after.append(np.linalg.norm(corrected))
            print(f"   part {row['pixel_value']:2}: offset {offset} |{np.linalg.norm(offset):.1f}| px; surface shift {shift} "
                  f"({points} edge px) -> corrected |{np.linalg.norm(corrected):.1f}| px")
    if before:
        print(f"\nParts with a usable surface fit: {len(before)}")
        print(f"median |offset| before {np.median(before):.1f} px, after {np.median(after):.1f} px; "
              f"reduced for {sum(a < b for a, b in zip(after, before))} of {len(before)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
