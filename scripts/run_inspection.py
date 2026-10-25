"""Run presence, geometric validation, or both modes on dataset photographs."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_inspection import discover_case_directories, discover_photos  # noqa: E402
from src.detector import inspect_case, result_csv, result_json  # noqa: E402
from src.geometric import inspect_geometry_case  # noqa: E402
from src.reference_comparison import inspect_with_real_reference  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _save_side_by_side(result: dict, destination: Path) -> None:
    """Save a visual comparison; failed alignments are clearly shown unwarped."""

    images = result["images"]
    aligned = images.get("aligned")
    photo = Image.fromarray(aligned if aligned is not None else images["photo"])
    cad = Image.fromarray(images["cad"])
    max_height = 1200
    photo.thumbnail((1600, max_height))
    cad.thumbnail((1600, max_height))
    header = 54
    canvas = Image.new("RGB", (photo.width + cad.width, max(photo.height, cad.height) + header), "#15191f")
    canvas.paste(photo, (0, header))
    canvas.paste(cad, (photo.width, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=28)
    photo_label = "Aligned photo" if aligned is not None else "Photo (unwarped or local geometry view)"
    draw.text((14, 12), photo_label, font=font, fill="white")
    draw.text((photo.width + 14, 12), "CAD reference", font=font, fill="white")
    canvas.save(destination)


def _write_problem1_report(output_dir: Path, source: str) -> Path | None:
    """Problem 1 evaluation of every checked part in every photo.

    Parts listed in ``intended_defects.csv`` take their listed status (a rotated
    part is present); every other checked part is expected to be present.
    Parts not visible in the CAD view are not scored.
    """
    labels_path = output_dir / "intended_defects.csv"
    source_dir = output_dir / source
    if not labels_path.is_file() or not source_dir.is_dir():
        LOGGER.warning("Skipping the Problem 1 report: need %s and %s.", labels_path, source_dir)
        return None
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        labels = {(row["case"], row["photo"], row["pixel_value"]): row for row in csv.DictReader(handle)}
    counts = {"CORRECT": 0, "WRONG": 0, "UNCERTAIN": 0}
    table = []
    for result_path in sorted(source_dir.glob("*/*/results.csv")):
        case, photo = result_path.parent.parent.name, result_path.parent.name
        with result_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["status"] == "NOT_VISIBLE":
                    continue
                label = labels.get((case, photo, row["pixel_value"]))
                expected = "MISSING" if label and label["intended_status"] == "MISSING" else "PRESENT"
                outcome = "UNCERTAIN" if row["status"] == "UNCERTAIN" else ("CORRECT" if row["status"] == expected else "WRONG")
                counts[outcome] += 1
                if label or outcome != "CORRECT":
                    table.append(
                        f"| {case} / {photo} | {row['pixel_value']} {row['part_name']} | "
                        f"{label['defect_type'] if label else 'unchanged'} | {'ABSENT' if expected == 'MISSING' else 'PRESENT'} | "
                        f"{'ABSENT' if row['status'] == 'MISSING' else row['status']} | {outcome} |"
                    )
    method = "CAD comparison (CAD image, mask, and part list only)" if source == "presence" else "known-good photo comparison"
    lines = [
        "# PartSight Problem 1: part presence results",
        "",
        f"Method: {method}.",
        "Expected results: the defects listed in `intended_defects.csv` (deliberate edits to the test images and the "
        "manually reviewed TestCase2 bolts); every other checked part is expected to be present. Parts not visible in "
        "the CAD view are not scored. These are visual-review labels, not official ground truth.",
        "",
        f"Checked parts: {sum(counts.values())}. Correct: {counts['CORRECT']}. Wrong: {counts['WRONG']}. "
        f"Uncertain: {counts['UNCERTAIN']}.",
        "",
        "| Case / photo | Part | Defect | Expected | Result | Outcome |",
        "|---|---|---|---|---|---|",
        *table,
        "",
    ]
    report_path = output_dir / ("problem1_report.md" if source == "presence" else "problem1_report_reference.md")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "AI_sample_CAD"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "inspection")
    parser.add_argument("--mode", choices=("presence", "reference", "geometry", "both", "all"), default="both")
    parser.add_argument(
        "--alignment", choices=("ecc", "sift", "loftr"), default="ecc",
        help="Photo-to-CAD alignment: ECC on edges (default; SIFT fallback), SIFT first, or optional LoFTR.",
    )
    parser.add_argument(
        "--reference-photo", default="BG_00.png",
        help="Case-local photo used as the provisional main-case appearance reference.",
    )
    parser.add_argument(
        "--reference-confirmed", action="store_true",
        help="Mark the selected reference as a confirmed complete assembly.",
    )
    args = parser.parse_args()

    try:
        dataset_root = args.dataset_root.resolve()
        case_dirs = discover_case_directories(dataset_root)
        run_count = 0
        reference_cache: dict[tuple[str, str], dict] = {}
        for case_dir in case_dirs:
            photos = discover_photos(case_dir)
            relative_case = case_dir.relative_to(dataset_root)
            for photo in photos:
                # Geometry reuses the validated alignment and loaded data from
                # the same presence pass rather than repeating SIFT/ECC work.
                presence_result = inspect_case(case_dir, photo.name, alignment_method=args.alignment)
                selected_modes = []
                if args.mode in {"presence", "both", "all"}:
                    selected_modes.append(("presence", presence_result))
                # Never borrow the main-case photograph for another
                # assembly/view. TestCase2 falls back unless it receives an
                # explicitly same-case reference.
                selected_reference = args.reference_photo if relative_case == Path(".") else None
                cached_reference = None
                reference_result = None
                if selected_reference is not None and args.mode in {"reference", "geometry", "both", "all"}:
                    cache_key = (str(case_dir.resolve()), selected_reference)
                    if cache_key not in reference_cache:
                        reference_path = case_dir / selected_reference
                        if reference_path.is_file():
                            reference_cache[cache_key] = inspect_case(
                                case_dir, selected_reference, alignment_method=args.alignment
                            )
                    cached_reference = reference_cache.get(cache_key)
                if args.mode in {"reference", "all"}:
                    reference_result = inspect_with_real_reference(
                        case_dir,
                        photo.name,
                        reference_photo=selected_reference,
                        reference_confirmed=args.reference_confirmed,
                        base_result=presence_result,
                        reference_result=cached_reference,
                        alignment_method=args.alignment,
                    )
                    selected_modes.append(("reference", reference_result))
                if args.mode in {"geometry", "both", "all"}:
                    geometry_result = inspect_geometry_case(
                        case_dir,
                        photo.name,
                        base_result=presence_result,
                        alignment_method=args.alignment,
                        reference_photo=selected_reference,
                        reference_confirmed=args.reference_confirmed,
                        reference_result=cached_reference,
                        presence_result=reference_result,
                    )
                    selected_modes.append(("geometry", geometry_result))
                for mode_name, result in selected_modes:
                    photo_output = args.output / mode_name / (relative_case if str(relative_case) != "." else "main") / photo.stem
                    photo_output.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(result["images"]["cad_annotated"]).save(photo_output / "cad_annotated.png")
                    Image.fromarray(result["images"]["photo_annotated"]).save(photo_output / "photo_annotated.png")
                    if result["images"].get("aligned") is not None:
                        aligned_image = Image.fromarray(result["images"]["aligned"])
                        aligned_image.save(photo_output / "aligned_photo.png")
                        cad_image = Image.fromarray(result["images"]["cad"])
                        Image.blend(aligned_image, cad_image, 0.5).save(photo_output / "alignment_overlay.png")
                    _save_side_by_side(result, photo_output / "side_by_side_alignment.png")
                    for method, visualization in result["images"].get("match_visualizations", {}).items():
                        Image.fromarray(visualization).save(photo_output / f"{method}_matches.png")
                    (photo_output / "results.json").write_text(result_json(result), encoding="utf-8")
                    (photo_output / "results.csv").write_text(result_csv(result), encoding="utf-8")
                    counts = result["counts"]
                    alignment = result["alignment"]
                    print(
                        f"{mode_name}: {case_dir.name}/{photo.name}: selected={alignment['method']} "
                        f"alignment={'ok' if alignment['success'] else 'failed'}; "
                        + ", ".join(f"{name.lower()}={count}" for name, count in counts.items())
                    )
                    for attempt in alignment["attempts"]:
                        print(
                            f"  {attempt['method']}: kp={attempt['keypoints_photo']}/{attempt['keypoints_cad']} "
                            f"desc={attempt['descriptors_photo_available']}/{attempt['descriptors_cad_available']} "
                            f"raw={attempt['raw_matches']} knn-candidates={attempt['knn_candidates']} "
                            f"filtered={attempt['filtered_matches']} "
                            f"correspondences={attempt['correspondences'] or attempt['filtered_matches']} "
                            f"inliers={attempt['ransac_inliers']} ({attempt['inlier_ratio']:.1%}) "
                            f"H-valid={attempt['homography_valid']} "
                            f"reprojection={attempt['median_reprojection_error_px']} "
                            f"spread={attempt['inlier_spread_ratio']} "
                            f"correlation={attempt['correlation_score']} "
                            f"reason={attempt['reason']}"
                        )
                    run_count += 1
        if run_count == 0:
            raise ValueError(f"No BG photographs found under {args.dataset_root}")
        for source, modes in (("presence", {"presence", "both", "all"}), ("reference", {"reference", "all"})):
            if args.mode in modes:
                report_path = _write_problem1_report(args.output, source)
                if report_path:
                    print(f"Saved Problem 1 report to {report_path}")
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved {run_count} inspection result sets under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
