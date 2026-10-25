# PartSight

PartSight checks a real photograph of an assembly against its CAD image, part-ID mask, and part list. **Part Presence Inspection** (Problem Statement 1) decides for every part in the JSON whether it is present or absent in the photo and highlights the result on both images. Problem Statement 2 (position, orientation, and size) is answered by the architecture in [`docs/problem2_architecture.md`](docs/problem2_architecture.md), informed by a command-line geometry prototype.

## Setup and run

`pyproject.toml` is the authoritative dependency definition. uv uses the ignored project-local `.uv-cache` directory. Install with `uv sync`, then run all photographs in the sample dataset with:

```powershell
uv run python scripts/run_inspection.py
```

Launch the Streamlit interface with:

```powershell
uv run streamlit run app.py
```

The command line writes per-photo results (`results.csv`, `results.json`, annotated real and CAD images) under `outputs/inspection/<mode>/<case>/<photo>/`. `--mode presence` runs Problem 1 only; the default also runs geometry, and `--mode all` adds the optional known-good-photo comparison.

## Part presence (Problem 1)

Inputs per case: the real photo, the CAD image, the part-ID mask, and the JSON part list. No other photo is needed. Two file layouts are recognised:

- **Sample data layout:** one folder per case with `BG_*` real photos, an `FG_*` CAD image, one JSON (`{"<mask file>": [{"Name": ..., "StencilValue": ...}]}`), and the mask it names (or the folder's only `*mask*` file).
- **Assignment naming:** `<name>_Real.<ext>`, `<name>_CAD.<ext>`, `<name>_Mask.<ext>`, `<name>_PartNamesToPixelMap.json`; several `<name>` sets may share one folder.

The JSON may also be a plain `{"<part name>": <pixel value>}` map or a list of parts. A pixel value is a scalar ID (read from the single mask channel that carries IDs) or an RGB colour `[r, g, b]` matched exactly in a colour mask. Point the app's sidebar "Dataset folder", or `run_inspection.py --dataset-root`, at another folder to inspect other data.

1. **Align** the photo to the CAD with masked-edge ECC, which registers the edges of the photo and the CAD inside the assembly area. Keypoint matching (SIFT) between a real photo and a rendered CAD image fails on the sample data, so it is only a fallback (`--alignment sift` tries it first). Optional LoFTR alignment (`--alignment loftr`, needs `uv pip install torch kornia`) is kept only when it registers the CAD edges better than ECC.
2. **Locate** each part from its main region in the mask; stray boundary pixels are ignored. Parts with no pixels, or fewer than 400 px, in the CAD view are reported `NOT_VISIBLE` (hidden in the CAD view) and are not guessed.
3. **Measure four local cues** at the part's expected location in the aligned photo: outline support (CAD edges matched by photo edges within 3 px, allowing a ±16 px local shift), edge density, texture, and contrast with the surrounding ring.
4. **Decide.** A blank region is absent. Otherwise the part is compared with its same-type parts in the same photo (names that differ only in a trailing number, e.g. `large_washer_1`..`_5`): if its cues relative to theirs combine to below 0.6, it is absent, whatever removed or covered it. A part with no same-type part is present when at least 50% of its CAD outline is found, otherwise uncertain. A rotated part is present.
5. **Display.** Present parts are outlined in green and absent parts in red, at their expected location, on both the real and the CAD image. A table gives each part's result and reason.

The decision limits were chosen by inspecting the sample images and have no independent validation. Known limits: a part type whose instances are all missing cannot be judged against siblings; parts hidden in the CAD view cannot be checked; results depend on the photo being taken from roughly the CAD viewpoint.

`outputs/inspection/intended_defects.csv` lists the expected absent parts (deliberate edits to the main test images and the manually reviewed TestCase2 bolts). `run_inspection.py` writes `problem1_report.md`, which scores every checked part against these expectations; every unlisted part is expected to be present.

**Optional orientation check.** When a confirmed complete photo of the same assembly is available (the app's "Known-good photo for orientation check", or `--mode all --reference-confirmed --reference-photo <file>`), each present part is also compared with it, including a rotation search; parts that match only after rotating are shown as present but rotated (blue), with the angle. Present/absent decisions still come from the CAD comparison, and the photo is used only when the rest of it matches the reference.

## Geometric validation (Problem 2)

The proposed architecture for checking location and dimensions against the CAD is in [`docs/problem2_architecture.md`](docs/problem2_architecture.md). The rest of this section describes the 2D prototype in `src/geometric.py` whose results informed that design.

Geometry outcomes are separate from presence: `WITHIN_TOLERANCE`, `OUT_OF_TOLERANCE`, `NOT_MEASURABLE` and `NOT_APPLICABLE`. `NOT_APPLICABLE` means the part is reported missing, so its geometry is never called correct. Position, orientation and size each get a value, an uncertainty and a result (`WITHIN`, `OUT`, `INCONCLUSIVE`, `NOT_OBSERVABLE` or `NOT_MEASURABLE`), and any `OUT` axis, including orientation, makes the part `OUT_OF_TOLERANCE`. A tolerance decision is made only when value ± uncertainty is entirely on one side of the tolerance; otherwise the part is `NOT_MEASURABLE` with the reason.

- **Conventions:** position is the displacement of the part centroid in photo pixels (+x right, +y down). Orientation is in degrees, positive counter-clockwise as displayed (same sign as `cv2.getRotationMatrix2D`). Size is a uniform scale ratio. None of these are millimetres.
- **Search:** ranges and resolution are fixed (±28 px, ±15°, 0.80–1.25) and independent of the tolerances. A coarse grid is refined to 1 px, 0.5° and 0.01. No pose prior is added; equal costs are tie-broken toward the smaller deviation. Uncertainty comes from each axis's cost profile and is never below the resolution. A fit at a search limit, or a distinctly different pose that fits almost as well, makes that axis (or, for position, the part) `NOT_MEASURABLE`. Rotation of a rotationally symmetric outline is `NOT_OBSERVABLE`.
- **CAD-relative** (no confirmed reference): the CAD outline is fitted to photo edges after global alignment, and the global registration error is added to the position uncertainty. On the sample photos this is mostly `NOT_MEASURABLE` for rotation and scale, because outline-to-edge fits are ambiguous there.
- **Reference-relative** (confirmed same-viewpoint known-good photo, e.g. `--reference-confirmed` with `BG_00.png`): presence comes from the confirmed-reference comparison, the photos are registered directly (SIFT/RANSAC), and each part's appearance is matched against the reference part. This cancels CAD-placement and height offsets shared with the reference, but viewpoint changes between the photos are modelled only by the photo-to-photo registration. A part that presence found rotated beyond the local search range gets its orientation from that rotation search (±2.5°).

The prototype runs from the command line (`run_inspection.py --mode geometry`); the app covers Problem 1 only.

Local rigid-surface re-alignment around each fastener was tested and did not reduce offsets on the sample photos, so it is not used.

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

## Notes

The supplied masks decode as RGB `uint8` with part IDs in the red channel; part names and IDs come from each case's JSON (`Name`, `StencilValue`). Geometry prototype values are image-space estimates, not millimetres. Local data and generated outputs are ignored by Git.
