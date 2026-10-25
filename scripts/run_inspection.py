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

from src.data_inspection import discover_case_directories  # noqa: E402
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


def _score_rows(rows: list[dict], prediction_key: str) -> dict[str, float | int]:
    decisive = [row for row in rows if row[prediction_key] != "UNCERTAIN"]
    correct = [row for row in decisive if row[prediction_key] == row["manual_status"]]
    false_positive = [
        row for row in decisive
        if row["manual_status"] == "MISSING" and row[prediction_key] == "PRESENT"
    ]
    false_negative = [
        row for row in decisive
        if row["manual_status"] == "PRESENT" and row[prediction_key] == "MISSING"
    ]
    count = len(rows)
    return {
        "reviewed": count,
        "decisive": len(decisive),
        "correct": len(correct),
        "incorrect": len(decisive) - len(correct),
        "uncertain": count - len(decisive),
        "coverage_percent": 100.0 * len(decisive) / count if count else 0.0,
        "decisive_correct_percent": 100.0 * len(correct) / len(decisive) if decisive else 0.0,
        "false_positives": len(false_positive),
        "false_negatives": len(false_negative),
    }


def _write_reference_comparison_report(output_dir: Path, reference_confirmed: bool) -> Path | None:
    """Compare unchanged human review labels against baseline and candidate rows."""
    labels_path = output_dir / "manual_validation.csv"
    if not labels_path.is_file():
        LOGGER.warning("No manual_validation.csv at %s; skipping baseline comparison report.", labels_path)
        return None
    with labels_path.open(newline="", encoding="utf-8-sig") as source:
        labels = list(csv.DictReader(source))
    joined: list[dict] = []
    for label in labels:
        case_name = label["case"]
        photo = label["photo"]
        case_output = "main" if case_name == "main" else case_name
        result_path = output_dir / "reference" / case_output / photo / "results.csv"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing reference comparison result for reviewed instance: {result_path}")
        with result_path.open(newline="", encoding="utf-8-sig") as source:
            candidates = list(csv.DictReader(source))
        result_row = next((r for r in candidates if r["pixel_value"] == label["pixel_value"]), None)
        if result_row is None:
            raise ValueError(f"No mapped value {label['pixel_value']} in {result_path}")
        joined.append({
            **label,
            "reference_candidate_status": result_row.get("candidate_status", "UNCERTAIN"),
            "reference_final_status": result_row.get("status", "UNCERTAIN"),
            "reference_similarity": result_row.get("reference_similarity", ""),
        })
    baseline = _score_rows(joined, "predicted_status")
    candidate = _score_rows(joined, "reference_candidate_status")

    def metric_cells(metrics: dict) -> str:
        return (
            f"{metrics['decisive']}/{metrics['reviewed']} | {metrics['correct']} | "
            f"{metrics['incorrect']} | {metrics['uncertain']} | "
            f"{metrics['coverage_percent']:.1f}% | {metrics['false_positives']} | {metrics['false_negatives']}"
        )

    lines = [
        "# PartSight real-reference comparison",
        "",
        "This report evaluates the same rows in `manual_validation.csv`. Those labels are small, hand-selected visual-review labels, not official ground truth.",
        "",
        f"Reference completeness confirmed for this run: **{'yes' if reference_confirmed else 'no'}**.",
        "The main case uses its configured reference photo for experimental candidate scores. The default `BG_00.png` completeness is not established. Candidate metrics below assume it is complete; the actual final status remains `UNCERTAIN` when the reference is unconfirmed. TestCase2 uses only its own case-local reference if one is explicitly configured; with none available it falls back to the existing baseline.",
        "",
        "Similarity is the mean of local normalized intensity correlation and gradient-magnitude correlation, after searching small translations in the CAD-aligned image coordinates. Thresholds are fixed configuration values (missing <= 0.20; present >= 0.55); they were not tuned on these labels.",
        "",
        "## Aggregate comparison",
        "",
        "| Method | Decisive / reviewed | Correct | Incorrect | Uncertain | Coverage | False positives | False negatives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Existing baseline | {metric_cells(baseline)} |",
        f"| Reference + baseline candidate (assuming reference complete) | {metric_cells(candidate)} |",
        "",
        "## Per photograph",
        "",
        "| Case / photo | Baseline decisive, correct, uncertain | Candidate decisive, correct, incorrect, uncertain | Candidate false positives / false negatives |",
        "|---|---|---|---:|",
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in joined:
        groups.setdefault((row["case"], row["photo"]), []).append(row)
    for (case_name, photo), rows in groups.items():
        old = _score_rows(rows, "predicted_status")
        new = _score_rows(rows, "reference_candidate_status")
        lines.append(
            f"| {case_name} / {photo} | {old['decisive']}, {old['correct']}, {old['uncertain']} | "
            f"{new['decisive']}, {new['correct']}, {new['incorrect']}, {new['uncertain']} | "
            f"{new['false_positives']} / {new['false_negatives']} |"
        )
    lines.extend([
        "",
        "The candidate is not a reliable accuracy estimate: labels are few, selected by visual review, and the reference has not been confirmed complete. A reference-only missing score is kept `UNCERTAIN` unless the existing baseline also supports absence; a disagreement with baseline presence also becomes `UNCERTAIN`.",
        "Geometry mode is a separate inspection and was not evaluated by these presence labels.",
        "",
    ])
    report_path = output_dir / "reference_comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _defect_outcome(label: dict, row: dict) -> str:
    """DETECTED / MISSED / UNCERTAIN for one intended-defect label."""
    outcome = row.get("inspection_outcome") or row["status"]
    if outcome == "UNCERTAIN":
        return "UNCERTAIN"
    expected = "ROTATED" if label["defect_type"].startswith("rotated") else label["intended_status"]
    return "DETECTED" if outcome == expected else "MISSED"


def _write_intended_defect_report(output_dir: Path, reference_confirmed: bool) -> Path | None:
    """Problem 1 evaluation against the intended test-image defects."""
    labels_path = output_dir / "intended_defects.csv"
    if not labels_path.is_file():
        LOGGER.warning("No intended_defects.csv at %s; skipping Problem 1 defect report.", labels_path)
        return None
    with labels_path.open(newline="", encoding="utf-8-sig") as source:
        labels = list(csv.DictReader(source))
    lines = [
        "# PartSight Problem 1: intended test-image defects",
        "",
        "Reference labels are the defects deliberately introduced into each test image, verified by "
        "pixel-differencing against `BG_00.png`. Every part not listed for a photo is unchanged and must "
        "not be flagged. Geometric validation (Problem 2) is evaluated separately.",
        "",
        f"Reference completeness confirmed for this run: **{'yes' if reference_confirmed else 'no'}**.",
        "",
        "| Photo | Part | Intended | Defect | Reference result | Baseline (CAD) | Outcome |",
        "|---|---|---|---|---|---|---|",
    ]
    outcomes: dict[str, int] = {}
    false_flags: list[str] = []
    for photo in sorted({label["photo"] for label in labels}):
        photo_labels = [label for label in labels if label["photo"] == photo]
        case_output = photo_labels[0]["case"]
        result_path = output_dir / "reference" / case_output / photo / "results.csv"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing reference result for labelled photo: {result_path}")
        with result_path.open(newline="", encoding="utf-8-sig") as source:
            rows = {row["pixel_value"]: row for row in csv.DictReader(source)}
        labelled = {label["pixel_value"] for label in photo_labels}
        for label in photo_labels:
            row = rows[label["pixel_value"]]
            outcome = _defect_outcome(label, row)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            shown = row.get("inspection_outcome") or row["status"]
            lines.append(
                f"| {photo} | {label['pixel_value']} {label['part_name']} | {label['intended_status']} | "
                f"{label['defect_type']} | {shown} | {row['baseline_status']} | {outcome} |"
            )
        for value, row in rows.items():
            flagged = (row.get("inspection_outcome") or row["status"]) in {"MISSING", "ROTATED"}
            if value not in labelled and flagged:
                false_flags.append(f"{photo} part {value} {row['part_name']} ({row['status']})")
    lines += [
        "",
        "## Summary",
        "",
        ", ".join(f"{name.lower()}: {count}" for name, count in sorted(outcomes.items())) or "No labels.",
        "",
        f"Unchanged parts incorrectly flagged: {len(false_flags)}.",
        *[f"- {flag}" for flag in false_flags],
        "",
    ]
    report_path = output_dir / "problem1_defect_report.md"
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
        "--alignment", choices=("sift", "loftr"), default="sift",
        help="Alignment backend to try first. LoFTR is optional and falls back to SIFT/ECC if unavailable or rejected.",
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
            photos = sorted(
                p for p in case_dir.glob("BG_*.*")
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            )
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
        if args.mode in {"reference", "all"}:
            report_path = _write_reference_comparison_report(args.output, args.reference_confirmed)
            if report_path:
                print(f"Saved baseline comparison report to {report_path}")
            defect_report = _write_intended_defect_report(args.output, args.reference_confirmed)
            if defect_report:
                print(f"Saved Problem 1 defect report to {defect_report}")
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved {run_count} inspection result sets under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
