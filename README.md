# PartSight

PartSight compares real assembly photographs with CAD references and part-ID masks. The app provides **Part Presence Inspection** and **Geometric Validation** modes. The default alignment remains SIFT/RANSAC followed by masked-edge ECC; an optional LoFTR/RANSAC experiment is available from the CLI.

## Setup and run

`pyproject.toml` is the authoritative dependency definition. uv uses the ignored project-local `.uv-cache` directory. Install with `uv sync`, then run all photographs in the sample dataset with:

```powershell
uv run python scripts/run_inspection.py
```

Launch the Streamlit interface with:

```powershell
uv run streamlit run app.py
```

The command-line inspection runs the existing presence and geometry modes by default. Use `--mode reference` to run the experimental real-photo comparison, or `--mode all` for all three. Reference results are written under `outputs/inspection/reference/`; baseline and geometry outputs retain their existing paths. The reference experiment compares local normalized intensity and gradient structure after each real photo is aligned into CAD coordinates, allowing a small translation search.

Try LoFTR alignment for the full dataset with:

```powershell
uv pip install torch kornia
uv run python scripts/run_inspection.py --mode presence --alignment loftr --output outputs/inspection/loftr
```

LoFTR is optional because PyTorch adds a large platform-specific dependency and the pretrained outdoor weights may need to download on first use. If dependencies, weights, or alignment checks are unavailable, the command logs the reason and falls back to SIFT and ECC. LoFTR inliers that are nearly collinear are rejected because the homography is unconstrained away from them. An accepted LoFTR homography is also compared with masked-edge ECC on a label-free registration score (median distance from CAD foreground edges to aligned-photo edges, `foreground_edge_distance_px`), and the better registration is used. On the sample data ECC scores better for every photograph. This is a matcher comparison, not evidence that presence detection is accurate.

Presence results no longer accept global RANSAC inliers alone as proof that a component is present. For positive local edge evidence, the pipeline attempts a bounded local silhouette fit and reports image-space `position_offset_*_px` and `position_deviation` fields. A part can be present but outside the positional tolerance; this is a geometric deviation, not a missing-part label. Weak or ambiguous localization remains `UNCERTAIN`.

For the main case, `--mode reference` uses `BG_00.png` as a **provisional** reference. Its completeness is not confirmed by the supplied data. Therefore the default run leaves final reference-mode statuses `UNCERTAIN`, but records proposed candidate statuses for evaluation. Only pass `--reference-confirmed` after an operator verifies the chosen same-case reference is complete. Override the reference filename with `--reference-photo <filename>`. TestCase2 never borrows the main-case reference and falls back to the existing baseline unless a same-case reference is explicitly added to the code/UI selection.

Reference comparison uses each part's largest connected mask component and skips parts under 400 px, as the CAD baseline does. A part that differs from the reference is re-compared with the reference rotated about the part's axis, taken as the centre of the minimum enclosing circle of its outline (5 degree steps, within a +/-24 px window). The centroid is not used because a partly hidden part, such as a washer under a nut, has its centroid off the axis. If a rotation matches, the part keeps `status` `PRESENT` and gets `inspection_outcome` `ROTATED`. `inspection_outcome` keeps present, rotated, missing and uncertain distinct in the CSV, the Streamlit table and the annotations. A confirmed reference is only used when the rest of the photo also matches it (median part similarity at least the present limit); otherwise the photo is treated as not correctly aligned and reference statuses stay `UNCERTAIN`. With a confirmed reference, a part may be reported `MISSING` without baseline agreement only when no tested shift or rotation brings it above the missing similarity limit and the rest of the photo still matches the reference. CAD-edge evidence alone cannot reject a part whose region is covered by unrelated texture.

For Problem 1, `outputs/inspection/intended_defects.csv` lists the defects deliberately introduced into the three test images, verified by pixel-differencing against `BG_00.png`. Running `--mode reference --reference-confirmed` writes `problem1_defect_report.md`, scoring each defect as detected, missed or uncertain and listing any unchanged part that was flagged. `tests/test_problem1_defects.py` checks the same cases and is skipped without the sample data.

The command writes `outputs/inspection/reference_comparison_report.md` when `manual_validation.csv` is available. This compares the baseline with proposed reference candidates on the unchanged visual-review rows; it is not an accuracy estimate. The Streamlit app keeps its two inspection modes and adds a presence evidence selector for the baseline or real-photo reference method.

## Geometric validation (Problem 2)

Geometry outcomes are separate from presence: `WITHIN_TOLERANCE`, `OUT_OF_TOLERANCE`, `NOT_MEASURABLE` and `NOT_APPLICABLE`. `NOT_APPLICABLE` means the part is reported missing, so its geometry is never called correct. Position, orientation and size each get a value, an uncertainty and a result (`WITHIN`, `OUT`, `INCONCLUSIVE`, `NOT_OBSERVABLE` or `NOT_MEASURABLE`), and any `OUT` axis, including orientation, makes the part `OUT_OF_TOLERANCE`. A tolerance decision is made only when value ± uncertainty is entirely on one side of the tolerance; otherwise the part is `NOT_MEASURABLE` with the reason.

- **Conventions:** position is the displacement of the part centroid in photo pixels (+x right, +y down). Orientation is in degrees, positive counter-clockwise as displayed (same sign as `cv2.getRotationMatrix2D`). Size is a uniform scale ratio. None of these are millimetres.
- **Search:** ranges and resolution are fixed (±28 px, ±15°, 0.80–1.25) and independent of the tolerances. A coarse grid is refined to 1 px, 0.5° and 0.01. No pose prior is added; equal costs are tie-broken toward the smaller deviation. Uncertainty comes from each axis's cost profile and is never below the resolution. A fit at a search limit, or a distinctly different pose that fits almost as well, makes that axis (or, for position, the part) `NOT_MEASURABLE`. Rotation of a rotationally symmetric outline is `NOT_OBSERVABLE`.
- **CAD-relative** (no confirmed reference): the CAD outline is fitted to photo edges after global alignment, and the global registration error is added to the position uncertainty. On the sample photos this is mostly `NOT_MEASURABLE` for rotation and scale, because outline-to-edge fits are ambiguous there.
- **Reference-relative** (confirmed same-viewpoint known-good photo, e.g. `--reference-confirmed` with `BG_00.png`): presence comes from the confirmed-reference comparison, the photos are registered directly (SIFT/RANSAC), and each part's appearance is matched against the reference part. This cancels CAD-placement and height offsets shared with the reference, but viewpoint changes between the photos are modelled only by the photo-to-photo registration. A part that presence found rotated beyond the local search range gets its orientation from that rotation search (±2.5°).

Geometry mode in the app uses the selected alignment method and an optional confirmed reference photo, and shows outcome counts and per-axis deviation counts.

`uv run python scripts/run_geometry_validation.py` builds semi-synthetic cases from the confirmed `BG_00.png`. Fastener groups are found from the mask and pasted back with known shifts, rotations and scale changes, or removed, with all other parts as unchanged controls. The ground truth goes to `outputs/geometry_validation/geometry_validation.csv`, scored results to `geometry_validation_results.csv`, and a summary to `geometry_validation_report.md`. Pasted patches rotate their shading with them and are resampled, so these results do not establish accuracy on physically displaced parts. `scripts/local_alignment_experiment.py` tests local rigid-surface re-alignment around each fastener; on the sample photos it did not reduce offsets and is not used.

Run unit tests with:

```powershell
uv run pytest -q
```

If a large wheel download is interrupted, retry in PowerShell with a longer timeout and more retries:

```powershell
$env:UV_HTTP_TIMEOUT = "120"
$env:UV_HTTP_RETRIES = "8"
uv sync
```

## Baseline limits

The supplied masks decode as RGB `uint8` with part IDs in the red channel. The detector reads `StencilValue` and `Name` from each case's JSON mapping. Presence decisions use local edge correspondence and contrast evidence; a successful global alignment alone is never treated as proof of presence. Geometric validation is described above; its values are image-space estimates, not millimetres, and it has only semi-synthetic validation. Neither mode is validated against real ground-truth measurements. Reflections, occlusion, perspective, different assembly depths, and fragmented mask IDs remain limitations; review results manually.

`scripts/inspect_dataset.py` remains available for dataset and mask diagnostics. Source utilities are in `src/data_inspection.py`, `src/visualization.py`, `src/detector.py`, and `src/geometric.py`. Local data and generated outputs are ignored by Git.
