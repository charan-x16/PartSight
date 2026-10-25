"""Command-line entry point for PartSight dataset inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_inspection import PartMapping, inspect_dataset  # noqa: E402
from src.visualization import save_mask_visualizations  # noqa: E402


def _format_counts(counts: dict[int, int]) -> str:
    return ", ".join(f"{value}: {count}" for value, count in sorted(counts.items()))


def build_report(results: dict) -> str:
    lines = [
        "# PartSight Dataset Inspection Report",
        "",
        f"Dataset root: `{results['dataset_root']}`",
        f"Cases discovered: **{results['case_count']}**",
        "",
    ]
    for case in results["cases"]:
        lines.extend([f"## Case `{case['name']}`", "", "### Files", ""])
        lines.append("| Role | File | Format | Mode | Dimensions | Channels | dtype |")
        lines.append("|---|---|---|---|---:|---:|---|")
        for image in case["images"]:
            lines.append(
                f"| {image['role']} | `{Path(image['path']).name}` | {image['format']} | "
                f"{image['mode']} | {image['width']}×{image['height']} | "
                f"{image['channels']} | {image['dtype']} |"
            )
        for warning in case["warnings"]:
            lines.extend(["", f"> Warning: {warning}"])

        for json_file in case["json_files"]:
            lines.extend(["", f"### Mapping `{Path(json_file['path']).name}`", ""])
            for mask in json_file["masks"]:
                stats = mask["inspection"]
                comparison = mask["comparison"]
                lines.extend(
                    [
                        f"- Declared mask: `{mask['declared_mask_name']}`",
                        f"- Resolved mask: `{Path(mask['resolved_mask_path']).name}`",
                        f"- Parts mapped: {len(mask['parts'])}",
                        f"- Encoding: Pillow `{stats['selected_channel_name']}` channel "
                        f"(index {stats['selected_channel_index']}); {stats['selection_reason']}",
                        f"- Mask dimensions: {stats['width']}×{stats['height']}; "
                        f"shape `{tuple(stats['shape'])}`; dtype `{stats['dtype']}`",
                    ]
                )
                for channel in stats["channel_stats"]:
                    lines.append(
                        f"- Channel `{channel['name']}` (index {channel['index']}) counts: "
                        f"{_format_counts(channel['value_counts'])}"
                    )
                vector_counts = stats["pixel_vector_counts"]
                if vector_counts is not None:
                    lines.append(
                        "- Full pixel-vector counts (channel order shown above): "
                        + ", ".join(
                            f"({vector}): {count}"
                            for vector, count in vector_counts.items()
                        )
                    )
                absent = comparison["absent_mapped_parts"]
                if absent:
                    absent_text = ", ".join(
                        f"{part['stencil_value']} ({part['name']})" for part in absent
                    )
                    lines.append(f"- Mapped values absent from mask: {absent_text}")
                else:
                    lines.append("- Mapped values absent from mask: none")
                unmapped = comparison["unmapped_mask_values"]
                lines.append(
                    "- Non-background mask values absent from JSON: "
                    + (", ".join(map(str, unmapped)) if unmapped else "none")
                )
        lines.extend(["", "### Dimension compatibility", ""])
        for image in case["images"]:
            masks = image["dimension_compatible_masks"]
            if masks:
                compatibility = "compatible with " + ", ".join(
                    f"`{name}`" for name in masks
                )
            else:
                compatibility = "no compatible mask"
            lines.append(f"- `{Path(image['path']).name}`: {compatibility}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "A JSON-mapped value absent from a CAD mask is a dataset observation requiring "
            "investigation. It is not evidence that the physical part is missing from a photograph.",
            "",
        ]
    )
    return "\n".join(lines)


def _reference_for_case(case: dict, mask_width: int, mask_height: int) -> Path | None:
    for image in case["images"]:
        if (
            image["role"] == "CAD reference image"
            and image["width"] == mask_width
            and image["height"] == mask_height
        ):
            return Path(image["path"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Extracted dataset root")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()

    try:
        results = inspect_dataset(args.dataset_root)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "inspection.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        (args.output / "inspection_report.md").write_text(
            build_report(results), encoding="utf-8"
        )
        for case in results["cases"]:
            case_output = args.output / (
                "root_case" if case["name"] == "." else case["name"]
            )
            for json_file in case["json_files"]:
                for mask in json_file["masks"]:
                    mappings = [PartMapping(**part) for part in mask["parts"]]
                    stats = mask["inspection"]
                    reference = _reference_for_case(case, stats["width"], stats["height"])
                    save_mask_visualizations(
                        mask["resolved_mask_path"], mappings, case_output, reference
                    )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Inspected {results['case_count']} cases. "
        f"Report: {args.output / 'inspection_report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
