"""Generate semi-synthetic geometry cases from the known-good photo and score geometry validation.

``--resume`` reuses the saved images, ground truth, and every complete
per-image result (results.json, results.csv, and photo_annotated.png all
present), computes only the missing images, and then scores all of them. It
first regenerates the cases in memory and stops unless the saved ground truth
and images are identical, so completed results are never overwritten or
duplicated.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.detector import inspect_case, result_csv, result_json  # noqa: E402
from src.geometric import inspect_geometry_case  # noqa: E402
from src.geometry_validation import (  # noqa: E402
    GROUND_TRUTH_FIELDS,
    SCORE_FIELDS,
    TRANSFORMS,
    fastener_groups,
    generate_case,
    ground_truth_rows,
    score_part,
    write_report,
    write_rows,
)

RESULT_FILES = ("results.json", "results.csv", "photo_annotated.png")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_save_image(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(array).save(temporary)
    os.replace(temporary, path)


def _csv_text(rows: list[dict], fields) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", type=Path, default=PROJECT_ROOT / "data" / "AI_sample_CAD")
    parser.add_argument("--reference", default="BG_00.png", help="Confirmed known-good photo in the case directory.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "geometry_validation")
    parser.add_argument("--resume", action="store_true", help="Reuse saved cases and complete results; compute the rest.")
    parser.add_argument("--overwrite", action="store_true", help="Allow a fresh run to replace existing results.")
    args = parser.parse_args()
    tolerances = {"position_tolerance_px": 12.0, "orientation_tolerance_deg": 5.0, "size_tolerance_ratio": 0.10}
    image_dir, results_root = args.output / "images", args.output / "results"
    if not args.resume and results_root.exists() and any(results_root.iterdir()) and not args.overwrite:
        parser.error(f"{results_root} already has results; use --resume to continue or --overwrite to replace them.")

    case_dir = args.case_dir.resolve()
    reference = inspect_case(case_dir, args.reference)
    if not reference["alignment"]["success"]:
        parser.error(f"Reference alignment failed: {reference['alignment']['reason']}")
    images = reference["images"]
    cad_to_photo = np.linalg.inv(images["matrix_photo_to_cad"])
    names = {row["pixel_value"]: row["part_name"] for row in reference["parts"]}
    groups = fastener_groups(images["mask_ids"])
    print(f"Found {len(groups)} fastener groups: {[group['values'] for group in groups]}")

    truth: list[dict] = []
    cases: list[tuple[str, np.ndarray]] = []
    for index, transform in enumerate(TRANSFORMS):
        group_index = (index - 1) % len(groups)
        group = groups[group_index]
        edited, edit_region, applied = generate_case(images["photo"], group, transform, cad_to_photo)
        name = f"GV_{index:02d}_{transform['label']}.png"
        truth.extend(ground_truth_rows(
            name, transform, group_index, group, applied, edit_region, images["mask_ids"], names, cad_to_photo, tolerances
        ))
        cases.append((name, edited))
    truth_path = args.output / "geometry_validation.csv"
    if args.resume:
        if not truth_path.is_file():
            parser.error(f"--resume needs the saved ground truth at {truth_path}")
        if truth_path.read_bytes().decode("utf-8") != _csv_text(truth, GROUND_TRUTH_FIELDS):
            parser.error("Saved ground truth differs from the regenerated cases; refusing to resume.")
        for name, edited in cases:
            saved = image_dir / name
            if not saved.is_file() or not np.array_equal(np.asarray(Image.open(saved).convert("RGB")), edited):
                parser.error(f"Saved image {saved} differs from the regenerated case; refusing to resume.")
        print("Resume check passed: saved ground truth and all images match the regenerated cases.")
    else:
        image_dir.mkdir(parents=True, exist_ok=True)
        for name, edited in cases:
            _atomic_save_image(image_dir / name, edited)
        write_rows(truth_path, truth, GROUND_TRUTH_FIELDS)
        print(f"Wrote {len(cases)} images and {len(truth)} ground-truth rows.")

    timings = []
    scored: list[dict] = []
    for name, _ in cases:
        result_dir = results_root / Path(name).stem
        complete = all((result_dir / file).is_file() for file in RESULT_FILES)
        if args.resume and complete:
            status, seconds = "reused", None
        else:
            started = time.perf_counter()
            result = inspect_geometry_case(
                case_dir, str((image_dir / name).resolve()), reference_photo=args.reference,
                reference_confirmed=True, reference_result=reference, **tolerances,
            )
            seconds = time.perf_counter() - started
            result_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(result_dir / "results.json", result_json(result))
            _atomic_write_text(result_dir / "results.csv", result_csv(result))
            _atomic_save_image(result_dir / "photo_annotated.png", result["images"]["photo_annotated"])
            status = "computed"
        parts = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
        by_value = {row["pixel_value"]: row for row in parts["parts"]}
        for row in (row for row in truth if row["image"] == name):
            scored.append(score_part(row, by_value[row["pixel_value"]]))
        timings.append({"image": name, "status": status, "seconds": "" if seconds is None else f"{seconds:.1f}"})
        print(f"{name}: {status}" + ("" if seconds is None else f" in {seconds:.1f} s") + f"; counts={parts['counts']}")
    write_rows(args.output / "geometry_validation_results.csv", scored, SCORE_FIELDS)
    write_rows(args.output / "timings.csv", timings, ("image", "status", "seconds"))
    context = {"reference": args.reference, "tolerances": tolerances}
    write_report(args.output / "geometry_validation_report.md", scored, context)
    (args.output / "groups.json").write_text(json.dumps([group["values"] for group in groups]), encoding="utf-8")
    print(f"Saved results and report under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
