"""Streamlit interface for PartSight part presence inspection (Problem Statement 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from src.data_inspection import load_image, load_part_mapping, mask_part_ids, resolve_case_files
from src.detector import discover_inspection_photos, inspect_case, result_csv, result_json
from src.reference_comparison import add_orientation_check, inspect_with_real_reference
from src.visualization import colorize_part_mask, part_mask_legend

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "data" / "AI_sample_CAD"
NO_REFERENCE = "None"
STATUS_LABELS = {
    "PRESENT": "Present", "MISSING": "Absent", "UNCERTAIN": "Uncertain", "NOT_VISIBLE": "Not visible in CAD",
}
STATUS_ORDER = {"MISSING": 0, "UNCERTAIN": 1, "PRESENT": 2, "NOT_VISIBLE": 3}
# Masked-edge ECC is the only alignment that works on the sample photos; SIFT and
# LoFTR remain available from the command line (run_inspection.py --alignment).
ALIGNMENT_METHOD = "ecc"


@st.cache_data(ttl=600, max_entries=16)
def load_case_preview(case_dir_string: str, photo_name: str):
    """Cache source image decoding; analysis only runs after the user submits."""
    files = resolve_case_files(case_dir_string, photo_name)
    mappings = load_part_mapping(files.mapping)[files.mapping_group]
    mask = load_image(files.mask)
    return {
        "photo": load_image(files.photo).array,
        "cad": load_image(files.cad).array,
        "cad_name": files.cad.name,
        "mask_colorized": colorize_part_mask(mask_part_ids(mask, mappings), mappings),
        "legend": part_mask_legend(mappings),
        "mapping_file": files.mapping.name,
    }


def _same_assembly(case_dir: Path, photo_name: str, other_name: str) -> bool:
    """True when two photos use the same CAD image and mask."""
    try:
        first, second = resolve_case_files(case_dir, photo_name), resolve_case_files(case_dir, other_name)
    except (OSError, ValueError):
        return False
    return (first.cad, first.mask) == (second.cad, second.mask)


def _case_options(photos: list[tuple[Path, str]], dataset_root: Path) -> dict[str, Path]:
    options: dict[str, Path] = {}
    for case_dir, _ in photos:
        relative = case_dir.relative_to(dataset_root).as_posix()
        label = "Main case" if relative == "." else relative
        options[label] = case_dir
    return options


def _presence_rows(result: dict) -> tuple[list[dict], list[dict]]:
    """(checked parts, parts not visible in the CAD view), absent parts first."""
    checked, hidden = [], []
    for row in sorted(result["parts"], key=lambda row: (STATUS_ORDER.get(row["status"], 1), row["pixel_value"])):
        entry = {
            "part": row["part_name"],
            "pixel value": row["pixel_value"],
            "result": STATUS_LABELS.get(row["status"], row["status"]),
        }
        if row.get("inspection_outcome") == "ROTATED":
            entry["result"] = "Present (rotated)"
        if row["status"] == "NOT_VISIBLE":
            hidden.append({**entry, "reason": row["diagnostic"]})
            continue
        entry.update({
            "relative evidence": row.get("relative_evidence"),
            "outline support": row.get("shape_support"),
            "reason": row["diagnostic"],
        })
        checked.append(entry)
    return checked, hidden


def _render_results(result: dict) -> None:
    counts = result["counts"]
    outcomes = result.get("outcome_counts")
    metrics = [("Present", counts.get("PRESENT", 0), None)]
    if outcomes is not None:
        # Only shown when an orientation check against a known-good photo ran.
        metrics.append((
            "Rotated", outcomes.get("ROTATED", 0),
            "Present parts that match the known-good photo only after rotation (included in Present).",
        ))
    metrics += [
        ("Absent", counts.get("MISSING", 0), None),
        ("Not visible in CAD", counts.get("NOT_VISIBLE", 0), "Hidden or too small in the CAD view; not checked."),
        ("Uncertain", counts.get("UNCERTAIN", 0), None),
    ]
    for column, (label, value, help_text) in zip(st.columns(len(metrics)), metrics):
        column.metric(label, value, help=help_text)
    st.write(f"Alignment: **{result['alignment']['method']}** — {result['alignment']['reason']}")
    if result.get("reference", {}).get("warning"):
        st.warning(result["reference"]["warning"], title="Orientation check")

    checked, hidden = _presence_rows(result)
    st.dataframe(checked, hide_index=True, alt="Present and absent parts with the reason for each decision")
    if hidden:
        with st.expander(f"Parts not visible in the CAD view ({len(hidden)}): cannot be checked"):
            st.dataframe(hidden, hide_index=True, alt="Parts hidden or too small in the CAD view")

    left, right = st.columns(2)
    with left:
        st.subheader("Real image: present (green) and absent (red) parts")
        st.image(
            result["images"]["photo_annotated"],
            alt="Real photograph with present parts outlined in green and absent parts in red at their expected location",
            width="stretch",
        )
    with right:
        st.subheader("CAD image: present (green) and absent (red) parts")
        st.image(
            result["images"]["cad_annotated"],
            alt="CAD image with present parts outlined in green and absent parts in red",
            width="stretch",
        )

    left, right = st.columns(2)
    left.download_button("Download CSV", result_csv(result), file_name="partsight_results.csv", mime="text/csv")
    right.download_button(
        "Download JSON", result_json(result), file_name="partsight_results.json", mime="application/json"
    )


def main() -> None:
    st.set_page_config(page_title="PartSight", layout="wide")
    st.title("PartSight part presence inspection")
    st.caption(
        "Compares the photo with the CAD image, mask, and part list. After aligning the photo to the CAD, each "
        "part is checked at its expected location: a blank region is absent, and a part much weaker than its "
        "same-type parts (outline, edges, texture, contrast) is absent. Parts hidden in the CAD view are listed "
        "separately."
    )

    dataset_root = Path(st.sidebar.text_input(
        "Dataset folder",
        value=str(DATASET_ROOT),
        help="Folder with the real photos, CAD image, mask, and part list (subfolders are separate cases). "
        "Supports BG_*/FG_* files with one JSON per folder, or <name>_Real/_CAD/_Mask/_PartNamesToPixelMap files.",
    )).expanduser().resolve()
    if not dataset_root.is_dir():
        st.error(f"Dataset folder not found: {dataset_root}")
        st.stop()
    try:
        photos = discover_inspection_photos(dataset_root)
    except (OSError, ValueError) as exc:
        st.error(f"Could not read dataset: {exc}")
        st.stop()
    if not photos:
        st.error("No real photographs were found in the dataset.")
        st.stop()

    st.session_state.setdefault("inspection_result", None)
    st.session_state.setdefault("inspection_key", None)
    case_options = _case_options(photos, dataset_root)
    selector_left, selector_right = st.columns(2)
    with selector_left:
        selected_case = st.selectbox("Dataset case", list(case_options), key="dataset_case")
    case_dir = case_options[selected_case]
    photo_names = sorted({name for directory, name in photos if directory == case_dir})
    with selector_right:
        photo_name = st.selectbox("Real photograph", photo_names, key="dataset_photo")

    reference_photo_name = None
    reference_options = [
        name for name in photo_names if name != photo_name and _same_assembly(case_dir, photo_name, name)
    ]
    if reference_options:
        choices = [NO_REFERENCE, *reference_options]
        default_reference = "BG_00.png" if "BG_00.png" in reference_options else NO_REFERENCE
        chosen = st.selectbox(
            "Known-good photo for orientation check (optional)",
            choices,
            index=choices.index(default_reference),
            key="reference_photo",
            help="A confirmed complete photo of the same assembly from the same viewpoint. Present/absent "
            "decisions always come from the CAD comparison; the known-good photo adds the rotation check.",
        )
        reference_photo_name = None if chosen == NO_REFERENCE else chosen
    else:
        st.caption("No other photo of this assembly is available, so the orientation check is not possible.")

    try:
        preview = load_case_preview(str(case_dir), photo_name)
    except (OSError, ValueError, StopIteration) as exc:
        st.error(f"Could not load case inputs: {exc}")
        st.stop()

    image_columns = st.columns(3)
    image_columns[0].image(
        preview["photo"], caption=photo_name, alt="Selected real assembly photograph", width="stretch"
    )
    image_columns[1].image(
        preview["cad"], caption=preview["cad_name"], alt="CAD reference image", width="stretch"
    )
    image_columns[2].image(
        preview["mask_colorized"],
        caption=f"Colorized part mask (parts from {preview['mapping_file']})",
        alt="Colorized part-ID mask aligned to the CAD reference",
        width="stretch",
    )
    if reference_photo_name is not None:
        reference_preview = load_case_preview(str(case_dir), reference_photo_name)
        with st.expander("Review the selected known-good photo before confirming it"):
            st.image(
                reference_preview["photo"],
                caption=reference_photo_name,
                alt="Selected known-good photograph of the same assembly",
                width="stretch",
            )
    with st.expander("Part mask legend"):
        st.dataframe(preview["legend"], hide_index=True)

    with st.form("inspection_controls", border=False):
        reference_confirmed = False
        if reference_photo_name is not None:
            reference_confirmed = st.checkbox(
                "I confirm the selected known-good photo shows a complete, correctly assembled part set",
                value=False,
                key="reference_confirmed",
            )
            if not reference_confirmed:
                st.caption("Without confirmation the orientation check is skipped.")
        run = st.form_submit_button("Run Inspection", type="primary")

    run_key = (str(case_dir), photo_name, reference_photo_name, reference_confirmed)
    if run:
        with st.spinner("Aligning the photo to the CAD and checking every part…"):
            try:
                result = inspect_case(case_dir, photo_name, alignment_method=ALIGNMENT_METHOD)
                if reference_photo_name is not None and reference_confirmed:
                    comparison = inspect_with_real_reference(
                        case_dir,
                        photo_name,
                        reference_photo=reference_photo_name,
                        reference_confirmed=True,
                        base_result=result,
                        alignment_method=ALIGNMENT_METHOD,
                    )
                    result = add_orientation_check(result, comparison)
                st.session_state["inspection_result"] = result
                st.session_state["inspection_key"] = run_key
            except (OSError, ValueError, RuntimeError, StopIteration, np.linalg.LinAlgError) as exc:
                st.error(f"Inspection failed: {exc}")
                st.stop()

    result = st.session_state.get("inspection_result")
    if result is None or st.session_state.get("inspection_key") != run_key:
        st.info("Select a photograph, then run the inspection.")
        return
    _render_results(result)


if __name__ == "__main__":
    main()
