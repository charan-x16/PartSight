"""Run part presence inspection (Problem Statement 1) on every photo of a dataset.

For each real photo, writes the annotated real and CAD images and the per-part
results (CSV and JSON) under ``<output>/presence/<case>/<photo>/``, then a
Problem 1 report scored against ``intended_defects.csv`` when it exists.
``--reference-photo`` with ``--reference-confirmed`` adds the optional
orientation check against a known-good photo of the same assembly.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_inspection import discover_case_directories, discover_photos  # noqa: E402
from src.detector import inspect_case, result_csv, result_json  # noqa: E402
from src.reference_comparison import add_orientation_check, inspect_with_real_reference  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _write_problem1_report(output_dir: Path) -> Path | None:
    """Score every checked part in every photo against the expected presence.

    Parts listed in ``intended_defects.csv`` take their listed status (a rotated
    part is present); every other checked part is expected to be present.
    Parts not visible in the CAD view are not scored.
    """
    labels_path = output_dir / "intended_defects.csv"
    source_dir = output_dir / "presence"
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
    lines = [
        "# PartSight Problem 1: part presence results",
        "",
        "Method: CAD comparison (CAD image, mask, and part list only).",
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
    report_path = output_dir / "problem1_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "AI_sample_CAD")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "inspection")
    parser.add_argument(
        "--reference-photo",
        help="Known-good photo of the same assembly (file name in each case folder) for the orientation check.",
    )
    parser.add_argument(
        "--reference-confirmed", action="store_true",
        help="Confirm the reference photo shows a complete, correctly assembled part set (required to use it).",
    )
    args = parser.parse_args()

    try:
        dataset_root = args.dataset_root.resolve()
        run_count = 0
        for case_dir in discover_case_directories(dataset_root):
            relative_case = case_dir.relative_to(dataset_root)
            case_name = "main" if str(relative_case) == "." else relative_case.as_posix()
            reference_path = case_dir / args.reference_photo if args.reference_photo else None
            reference = None
            for photo in discover_photos(case_dir):
                result = inspect_case(case_dir, photo.name)
                use_reference = (
                    args.reference_confirmed and reference_path is not None
                    and reference_path.is_file() and reference_path.name != photo.name
                )
                if use_reference:
                    if reference is None:
                        reference = inspect_case(case_dir, reference_path.name)
                    try:
                        comparison = inspect_with_real_reference(
                            case_dir, photo.name, reference_photo=reference_path.name, reference_confirmed=True,
                            base_result=result, reference_result=reference,
                        )
                        result = add_orientation_check(result, comparison)
                    except ValueError as exc:  # e.g. the reference shows a different assembly
                        LOGGER.warning("Orientation check skipped for %s: %s", photo.name, exc)
                photo_output = args.output / "presence" / case_name / photo.stem
                photo_output.mkdir(parents=True, exist_ok=True)
                Image.fromarray(result["images"]["photo_annotated"]).save(photo_output / "photo_annotated.png")
                Image.fromarray(result["images"]["cad_annotated"]).save(photo_output / "cad_annotated.png")
                (photo_output / "results.json").write_text(result_json(result), encoding="utf-8")
                (photo_output / "results.csv").write_text(result_csv(result), encoding="utf-8")
                rotated = (result.get("outcome_counts") or {}).get("ROTATED")
                print(
                    f"{case_name}/{photo.name}: alignment {result['alignment']['method']} "
                    f"{'ok' if result['alignment']['success'] else 'failed'}; "
                    + ", ".join(f"{name.lower()}={count}" for name, count in result["counts"].items())
                    + (f", rotated={rotated}" if rotated is not None else "")
                )
                run_count += 1
        if run_count == 0:
            raise ValueError(f"No real photos found under {args.dataset_root}")
        report_path = _write_problem1_report(args.output)
        if report_path:
            print(f"Saved Problem 1 report to {report_path}")
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved {run_count} inspection result sets under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
