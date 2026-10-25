# PartSight

PartSight checks a real photograph of an assembly against its CAD image, part-ID mask, and part list. **Part Presence Inspection** (Problem Statement 1) decides for every part in the JSON whether it is present or absent in the photo and highlights the result on both images. Problem Statement 2 (position, orientation, and size) is answered by the architecture in [`docs/problem2_architecture.md`](docs/problem2_architecture.md).

## Setup and run

`pyproject.toml` is the authoritative dependency definition. uv uses the ignored project-local `.uv-cache` directory. Install with `uv sync`, then launch the Streamlit interface with:

```powershell
uv run streamlit run app.py
```

Run all photographs in the sample dataset from the command line with:

```powershell
uv run python scripts/run_inspection.py
```

Options: `--dataset-root <folder>` inspects other data, `--output <folder>` changes the output folder (default `outputs/inspection`), and `--reference-photo BG_00.png --reference-confirmed` adds the optional orientation check. For each photo the command writes `photo_annotated.png`, `cad_annotated.png`, `results.json` and `results.csv` under `<output>/presence/<case>/<photo>/`, then `problem1_report.md`.

## Part presence (Problem 1)

Inputs per case: the real photo, the CAD image, the part-ID mask, and the JSON part list. No other photo is needed. Two file layouts are recognised:

- **Sample data layout:** one folder per case with `BG_*` real photos, an `FG_*` CAD image, one JSON (`{"<mask file>": [{"Name": ..., "StencilValue": ...}]}`), and the mask it names (or the folder's only `*mask*` file).
- **Assignment naming:** `<name>_Real.<ext>`, `<name>_CAD.<ext>`, `<name>_Mask.<ext>`, `<name>_PartNamesToPixelMap.json`; several `<name>` sets may share one folder.

The JSON may also be a plain `{"<part name>": <pixel value>}` map or a list of parts. A pixel value is a scalar ID (read from the single mask channel that carries IDs) or an RGB colour `[r, g, b]` matched exactly in a colour mask. Point the app's sidebar "Dataset folder", or `run_inspection.py --dataset-root`, at another folder to inspect other data.

1. **Align** the photo to the CAD with masked-edge ECC, which registers the edges of the photo and the CAD inside the assembly area. Keypoint matching (SIFT) between a real photo and a rendered CAD image fails on the sample data, so it is only tried when ECC fails.
2. **Locate** each part from its main region in the mask; stray boundary pixels are ignored. Parts with no pixels, or fewer than 400 px, in the CAD view are reported `NOT_VISIBLE` (hidden in the CAD view) and are not guessed.
3. **Measure four local cues** at the part's expected location in the aligned photo: outline support (CAD edges matched by photo edges within 3 px, allowing a ±16 px local shift), edge density, texture, and contrast with the surrounding ring.
4. **Decide.** A blank region is absent. Otherwise the part is compared with its same-type parts in the same photo (names that differ only in a trailing number, e.g. `large_washer_1`..`_5`): if its cues relative to theirs combine to below 0.6, it is absent, whatever removed or covered it. A part with no same-type part is present when at least 50% of its CAD outline is found, otherwise uncertain. A rotated part is present.
5. **Display.** Present parts are outlined in green and absent parts in red, at their expected location, on both the real and the CAD image. A table gives each part's result and reason.

The decision limits were chosen by inspecting the sample images and have no independent validation. Known limits: a part type whose instances are all missing cannot be judged against siblings; parts hidden in the CAD view cannot be checked; results depend on the photo being taken from roughly the CAD viewpoint.

`outputs/inspection/intended_defects.csv` lists the expected absent parts (deliberate edits to the main test images and the manually reviewed TestCase2 bolts). `run_inspection.py` writes `problem1_report.md`, which scores every checked part against these expectations; every unlisted part is expected to be present.

**Optional orientation check.** When a confirmed complete photo of the same assembly is available (the app's "Known-good photo for orientation check", or `--reference-photo <file> --reference-confirmed`), each present part is also compared with it, including a rotation search; parts that match only after rotating are shown as present but rotated (blue), with the angle. Present/absent decisions still come from the CAD comparison, and the photo is used only when the rest of it matches the reference.

## Tests

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

The supplied masks decode as RGB `uint8` with part IDs in the red channel; part names and IDs come from each case's JSON (`Name`, `StencilValue`). Local data and generated outputs are ignored by Git.
