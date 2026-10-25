"""Streamlit interface for PartSight inspection modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from src.data_inspection import inspect_mask, load_image, load_part_mapping
from src.detector import discover_inspection_photos, inspect_case, result_csv, result_json
from src.geometric import GEOMETRY_OUTCOMES, inspect_geometry_case
from src.reference_comparison import inspect_with_real_reference
from src.visualization import colorize_part_mask, part_mask_legend

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "data" / "AI_sample_CAD"
PRESENCE_MODE = "Part Presence Inspection"
GEOMETRY_MODE = "Geometric Validation"
BASELINE_METHOD = "Existing baseline"
REAL_REFERENCE_METHOD = "Real-photo reference"


@st.cache_data(ttl=600, max_entries=16)
def load_case_preview(case_dir_string: str, photo_name: str):
    """Cache source image decoding; analysis only runs after the user submits."""
    case_dir = Path(case_dir_string)
    mapping_path = next(iter(sorted(case_dir.glob("*.json"))), None)
    cad_path = next(iter(sorted(case_dir.glob("FG_*.*"))), None)
    if mapping_path is None or cad_path is None:
        raise FileNotFoundError(f"Case is missing its JSON mapping or CAD image: {case_dir}")
    groups = load_part_mapping(mapping_path)
    mask_name = next(iter(groups))
    mask_path = case_dir / mask_name
    if not mask_path.is_file():
        candidates = [p for p in case_dir.iterdir() if p.is_file() and "mask" in p.stem.lower()]
        if len(candidates) != 1:
            raise FileNotFoundError(f"Could not uniquely resolve the declared mask {mask_name}")
        mask_path = candidates[0]
    photo = load_image(case_dir / photo_name)
    cad = load_image(cad_path)
    mask = load_image(mask_path)
    stats = inspect_mask(mask)
    channel = stats["selected_channel_index"]
    if channel is None:
        raise ValueError("Mask channels are ambiguous; refusing to guess part IDs.")
    mappings = groups[mask_name]
    colorized = colorize_part_mask(mask.array, mappings, int(channel))
    return {
        "photo": photo.array,
        "cad": cad.array,
        "mask_colorized": colorized,
        "legend": part_mask_legend(mappings),
        "channel": stats["selected_channel_name"],
        "mapping_file": mapping_path.name,
    }


def _case_options(photos: list[tuple[Path, str]]) -> dict[str, Path]:
    options: dict[str, Path] = {}
    for case_dir, _ in photos:
        relative = case_dir.relative_to(DATASET_ROOT).as_posix()
        label = "Main case" if relative == "." else relative
        options[label] = case_dir
    return options


GEOMETRY_TABLE_COLUMNS = [
    "part_name", "pixel_value", "status", "presence_outcome", "geometry_basis",
    "position_offset_magnitude_px", "position_uncertainty_px", "position_result",
    "orientation_difference_deg", "orientation_uncertainty_deg", "orientation_result",
    "size_ratio", "size_uncertainty", "size_result", "diagnostic",
]


def geometry_summary(result: dict) -> dict[str, int]:
    """Metric values for geometry mode, computed from the geometry fields."""
    rows = result["parts"]
    summary = {outcome: sum(row["status"] == outcome for row in rows) for outcome in GEOMETRY_OUTCOMES}
    for axis in ("position", "orientation", "size"):
        summary[f"{axis}_out"] = sum(row.get(f"{axis}_result") == "OUT" for row in rows)
        summary[f"{axis}_inconclusive"] = sum(row.get(f"{axis}_result") == "INCONCLUSIVE" for row in rows)
    return summary


def _render_geometry_metrics(result: dict) -> None:
    summary = geometry_summary(result)
    columns = st.columns(4)
    columns[0].metric("Within tolerance", summary["WITHIN_TOLERANCE"])
    columns[1].metric("Out of tolerance", summary["OUT_OF_TOLERANCE"])
    columns[2].metric("Not measurable", summary["NOT_MEASURABLE"])
    columns[3].metric("Not applicable (not present)", summary["NOT_APPLICABLE"])
    columns = st.columns(3)
    for column, axis in zip(columns, ("position", "orientation", "size")):
        column.metric(
            f"{axis.capitalize()} deviations",
            summary[f"{axis}_out"],
            help=f"{summary[f'{axis}_inconclusive']} part(s) inconclusive: uncertainty overlaps the tolerance.",
        )


def _render_results(result: dict, mode: str) -> None:
    counts = result["counts"]
    outcomes = result.get("outcome_counts")
    deviations = sum(bool(row.get("position_deviation")) for row in result["parts"])
    if mode == GEOMETRY_MODE:
        _render_geometry_metrics(result)
    elif outcomes:
        columns = st.columns(5)
        columns[0].metric("Present", outcomes["PRESENT"])
        columns[1].metric("Present, rotated", outcomes["ROTATED"])
        columns[2].metric("Missing", outcomes["MISSING"])
        columns[3].metric("Uncertain", outcomes["UNCERTAIN"])
        columns[4].metric("Position deviations", deviations)
    else:
        columns = st.columns(4)
        columns[0].metric("Present", counts["PRESENT"])
        columns[1].metric("Missing", counts["MISSING"])
        columns[2].metric("Uncertain", counts["UNCERTAIN"])
        columns[3].metric("Position deviations", deviations)
    st.write(f"Alignment: **{result['alignment']['method']}** — {result['alignment']['reason']}")
    loftr_attempt = next(
        (attempt for attempt in result["alignment"].get("attempts", []) if attempt.get("method") == "LoFTR"),
        None,
    )
    if loftr_attempt and not loftr_attempt.get("success"):
        st.info(f"LoFTR was not used: {loftr_attempt['reason']} The pipeline continued with its fallback aligner.")
    if result.get("reference", {}).get("warning"):
        st.warning(result["reference"]["warning"], title="Reference limitation")
    if mode == GEOMETRY_MODE:
        registration = result.get("registration", {})
        st.write(
            f"Geometry basis: **{result.get('geometry_basis')}**; presence from **{result.get('presence_source')}**. "
            f"CAD registration uncertainty ≈ {registration.get('cad_registration_uncertainty_px', 0):.1f} px"
            + (
                f"; photo-to-photo residual ≈ {registration['photo_registration_residual_px']:.1f} px."
                if registration.get("photo_registration_residual_px") is not None
                else "."
            )
        )
        st.caption(
            "Position, orientation, and size are image-space estimates in photo pixels, not millimetres. "
            "Orientation is positive counter-clockwise. NOT_MEASURABLE: weak, ambiguous, or boundary fits, or "
            "uncertainty overlapping a tolerance. NOT_APPLICABLE: the part is reported missing. "
            "NOT_OBSERVABLE orientation: the outline is rotationally symmetric (e.g. a round washer)."
        )
        rows = [{field: row.get(field) for field in GEOMETRY_TABLE_COLUMNS} for row in result["parts"]]
        st.dataframe(rows, hide_index=True, alt="Per-part geometric validation results")
    else:
        visible_columns = [
            "part_name", "pixel_value", "inspection_outcome", "status", "candidate_status", "baseline_status",
            "reference_similarity", "reference_rotated_similarity", "reference_rotation_deg",
            "position_deviation", "position_offset_magnitude_px", "localization_edge_support",
            "evidence_score", "diagnostic",
        ]
        rows = result["parts"]
        if rows:
            fields = [field for field in visible_columns if field in rows[0]]
            rows = [{field: row.get(field) for field in fields} for row in rows]
        st.dataframe(rows, hide_index=True, alt="Part presence, local position, and evidence results")

    left, right = st.columns(2)
    with left:
        st.subheader("Annotated real photograph")
        st.image(
            result["images"]["photo_annotated"],
            alt="Real photograph annotated with inspection statuses and expected locations",
            width="stretch",
        )
    with right:
        st.subheader("Annotated CAD reference")
        st.image(
            result["images"]["cad_annotated"],
            alt="CAD reference annotated with mapped component regions and statuses",
            width="stretch",
        )

    csv_data, json_data = result_csv(result), result_json(result)
    left, right = st.columns(2)
    left.download_button(
        "Download CSV", csv_data, file_name="partsight_results.csv", mime="text/csv"
    )
    right.download_button(
        "Download JSON", json_data, file_name="partsight_results.json", mime="application/json"
    )


def main() -> None:
    st.set_page_config(page_title="PartSight", layout="wide")
    st.title("PartSight inspection")
    st.caption("Use image-space results as screening evidence. Review uncertain cases manually.")

    if not DATASET_ROOT.is_dir():
        st.error(f"Dataset folder not found: {DATASET_ROOT}")
        st.stop()
    try:
        photos = discover_inspection_photos(DATASET_ROOT)
    except (OSError, ValueError) as exc:
        st.error(f"Could not read dataset: {exc}")
        st.stop()
    if not photos:
        st.error("No real photographs were found in the dataset.")
        st.stop()

    st.session_state.setdefault("inspection_result", None)
    st.session_state.setdefault("inspection_key", None)
    mode = st.segmented_control(
        "Inspection mode",
        [PRESENCE_MODE, GEOMETRY_MODE],
        default=PRESENCE_MODE,
        key="inspection_mode",
        width="stretch",
    )
    presence_method = BASELINE_METHOD
    if mode == PRESENCE_MODE:
        presence_method = st.segmented_control(
            "Presence evidence method",
            [BASELINE_METHOD, REAL_REFERENCE_METHOD],
            default=BASELINE_METHOD,
            key="presence_evidence_method",
            width="stretch",
        )
        if presence_method == BASELINE_METHOD:
            st.caption(
                "The baseline compares the photo with CAD edges only. It cannot tell a covered or "
                "painted-over part from an intact one, or report rotation; use a confirmed same-case "
                "reference photograph for that."
            )
    alignment_method_label = st.segmented_control(
        "Alignment method",
        ["SIFT then ECC", "LoFTR then SIFT/ECC"],
        default="SIFT then ECC",
        key="alignment_method",
        width="stretch",
    ) or "SIFT then ECC"
    alignment_method = "loftr" if alignment_method_label.startswith("LoFTR") else "sift"
    case_options = _case_options(photos)
    selector_left, selector_right = st.columns(2)
    with selector_left:
        selected_case = st.selectbox("Dataset case", list(case_options), key="dataset_case")
    case_dir = case_options[selected_case]
    photo_names = sorted({name for directory, name in photos if directory == case_dir})
    with selector_right:
        photo_name = st.selectbox("Real photograph", photo_names, key="dataset_photo")

    reference_photo_name = None
    uses_reference = (mode == PRESENCE_MODE and presence_method == REAL_REFERENCE_METHOD) or mode == GEOMETRY_MODE
    if uses_reference:
        reference_options = [name for name in photo_names if name != photo_name]
        if reference_options:
            default_reference = "BG_00.png" if "BG_00.png" in reference_options else reference_options[0]
            reference_photo_name = st.selectbox(
                "Same-case reference photograph",
                reference_options,
                index=reference_options.index(default_reference),
                key="reference_photo",
            )
        else:
            st.info("No separate same-case photograph is available; the baseline will be used.")

    try:
        preview = load_case_preview(str(case_dir), photo_name)
    except (OSError, ValueError, StopIteration) as exc:
        st.error(f"Could not load case inputs: {exc}")
        st.stop()

    image_columns = st.columns(3)
    image_columns[0].image(
        preview["photo"], caption=photo_name, alt="Selected real assembly photograph", width="stretch"
    )
    cad_path = next(case_dir.glob("FG_*.*"))
    image_columns[1].image(
        preview["cad"], caption=cad_path.name, alt="CAD reference image", width="stretch"
    )
    image_columns[2].image(
        preview["mask_colorized"],
        caption=f"Colorized part mask ({preview['channel']} channel IDs)",
        alt="Colorized part-ID mask aligned to the CAD reference",
        width="stretch",
    )
    if uses_reference and reference_photo_name is not None:
        reference_preview = load_case_preview(str(case_dir), reference_photo_name)
        with st.expander("Review the selected reference photograph before confirming it"):
            st.image(
                reference_preview["photo"],
                caption=reference_photo_name,
                alt="Selected same-case real photograph used as the appearance reference",
                width="stretch",
            )
    with st.expander("Part mask legend"):
        st.dataframe(preview["legend"], hide_index=True)

    with st.form("inspection_controls", border=False):
        if mode == GEOMETRY_MODE:
            a, b, c = st.columns(3)
            position_tolerance = a.number_input(
                "Position tolerance (pixels)", min_value=1.0, max_value=100.0, value=12.0, step=1.0
            )
            size_tolerance_percent = b.number_input(
                "Image-size tolerance (%)", min_value=1.0, max_value=50.0, value=10.0, step=1.0
            )
            orientation_tolerance = c.number_input(
                "Orientation tolerance (degrees)", min_value=1.0, max_value=30.0, value=5.0, step=1.0
            )
        else:
            position_tolerance = 12.0
            size_tolerance_percent = 10.0
            orientation_tolerance = 5.0
        reference_confirmed = False
        if uses_reference and reference_photo_name is not None:
            reference_confirmed = st.checkbox(
                "I confirm the selected reference shows a complete, correctly assembled part set",
                value=False,
                key="reference_confirmed",
            )
            if not reference_confirmed:
                st.caption(
                    "Without confirmation, reference-based presence stays UNCERTAIN and geometry is "
                    "measured relative to the CAD outline only."
                )
        run = st.form_submit_button("Run Inspection", type="primary")

    run_key = (
        mode,
        str(case_dir),
        photo_name,
        float(position_tolerance),
        float(size_tolerance_percent),
        float(orientation_tolerance),
        presence_method,
        reference_photo_name,
        reference_confirmed,
        alignment_method,
    )
    if run:
        with st.spinner("Aligning images and evaluating component evidence…"):
            try:
                if mode == PRESENCE_MODE:
                    if presence_method == REAL_REFERENCE_METHOD:
                        result = inspect_with_real_reference(
                            case_dir,
                            photo_name,
                            reference_photo=reference_photo_name,
                            reference_confirmed=reference_confirmed,
                            alignment_method=alignment_method,
                        )
                    else:
                        result = inspect_case(
                            case_dir, photo_name, alignment_method=alignment_method
                        )
                else:
                    result = inspect_geometry_case(
                        case_dir,
                        photo_name,
                        position_tolerance_px=float(position_tolerance),
                        size_tolerance_ratio=float(size_tolerance_percent) / 100.0,
                        orientation_tolerance_deg=float(orientation_tolerance),
                        alignment_method=alignment_method,
                        reference_photo=reference_photo_name,
                        reference_confirmed=reference_confirmed,
                    )
                st.session_state["inspection_result"] = result
                st.session_state["inspection_key"] = run_key
            except (OSError, ValueError, RuntimeError, StopIteration, np.linalg.LinAlgError) as exc:
                st.error(f"Inspection failed: {exc}")
                st.stop()

    result = st.session_state.get("inspection_result")
    if result is None or st.session_state.get("inspection_key") != run_key:
        st.info("Select a mode and photograph, then run the inspection.")
        return
    _render_results(result, mode)


if __name__ == "__main__":
    main()
