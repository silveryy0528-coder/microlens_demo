# Microlens characterization: methodology and data contracts

## Goal and scope

Analyze a 2D height map of spherical microlenses, then export a table of per-lens parameters and diagnostic image artifacts. The first version uses **synthetic data only** and reports two physical measurements: **radius of curvature** and **lens height**.

Initial representative dataset:
- Field of view: approximately **6 µm × 6 µm** (6000 nm × 6000 nm).
- Image: approximately **1024 × 1024** samples.
- Array: approximately **3 × 3** spherical microlenses.
- Lens height: approximately **500 nm**.
- Physical lateral and height units: **nm**. The image metadata, not hardcoded constants, determines coordinate scaling.

These are starting conditions, not fixed requirements for all images.

## Data model: `Image`

Use a lightweight `Image` class (suggested location: `src/microlens/image.py`) to carry both the measured values and their physical sampling metadata:

```python
image = Image(
    values=height_map,       # 2D NumPy array: [row_y, column_x]
    pixel_size_x=5.86,       # lateral spacing between columns
    pixel_size_y=5.86,       # lateral spacing between rows
    xy_unit="nm",           # units of pixel_size_x and pixel_size_y
    data_unit="nm",         # units of height values
)
```

The numerical pixel sizes above are **illustrative**, approximating a 6000 nm field of view sampled at 1024 pixels. The synthetic generator must supply its actual values consistently.

Coordinate convention for the initial implementation:
- `values.shape == (n_y, n_x)`; columns represent `x`, rows represent `y`.
- Pixel-center coordinates are `x = column * pixel_size_x` and `y = row * pixel_size_y`.
- `xy_unit` applies to both lateral pixel sizes; `data_unit` applies to `values`.
- Preserve these fields through processing. Do not assume that lateral and height units are interchangeable or that pixel spacing is isotropic.
- Initially, **record units without converting them**. Unit-conversion methods are explicitly out of scope for the first version.

## Processing methodology

### 1. Background correction

1. Identify **low-height substrate pixels** and exclude microlens regions from the fit.
2. Fit a plane to the selected substrate pixels in physical `x, y, z` coordinates.
3. Subtract that fitted plane to flatten the substrate.

The precise substrate-pixel selection/masking rule remains to be specified. Do not fit a plane to all pixels indiscriminately.

### 2. Lens detection

1. Apply **percentile-based thresholding** to the background-corrected height map.
2. Identify individual microlens regions to analyze.

The threshold percentile and any component-filtering rules remain to be specified.

### 3. Apex localization and curvature

1. Fit a **sphere in physical 3D coordinates** to each detected lens region.
2. Use the fitted sphere to localize the apex at subpixel precision.
3. Report the **radius of curvature `R` (nm)** from the fitted sphere, **not** the reciprocal curvature `1/R`.

Fit-region selection, sign conventions, and optimization details will be specified when implementing this stage.

### 4. Apex-centered profile and lens height

1. Extract an **interpolated 2D height profile through the fitted apex**; the apex need not coincide with a pixel center.
2. Include the intended lens and surrounding baseline in the sampled profile.
3. Define lens height as **`P99(profile) - P5(profile)`**, using all valid sampled profile values.

Profile orientation, total length, sample spacing, and out-of-bounds handling remain to be specified. Do not substitute a fitted sphere's cap height for this measured profile statistic.

### 5. Results and artifacts

- Report one row per detected lens, including lens ID, apex coordinates, radius of curvature (nm), and lens height (nm).
- Write a machine-readable results table (CSV initially) and diagnostic visualizations (for example, corrected height map, detection overlay, and extracted profiles).
- Preserve relevant pixel metadata and configuration for interpretability and reproducibility.

## Intended software organization

- `Image` stores the array and sampling metadata; it should not contain the entire analysis pipeline.
- `UseCaseMicrolens` orchestrates `preprocess()` and `process()`. `process()` should perform the complete analysis, calling preprocessing as needed.
- Keep plane fitting, detection, sphere fitting, and profile extraction in small testable numerical functions; a single `algorithms.py` is sufficient initially.
- Expose `scripts/process_single_image.py` and `scripts/batch_process_images.py` as entry points, without a notebook or GUI.
- Keep file I/O, artifact export, and plotting separate from the numerical implementation.

The precise Python method signatures and result containers should be agreed on before implementation; no large interface hierarchy is required.

## Verification strategy

**Software tests:** Check small contracts independently (for example, `Image` array/metadata handling, coordinate mapping, and bilinear interpolation on a known grid).

**Scientific validation:** Generate spherical-cap height maps with known radius, apex, and substrate tilt. Run the pipeline and compare recovered values with analytical ground truth. Include subpixel apex locations and later vary noise, sampling, and lens geometry.

For the **height** metric, compute the expected `P99 - P5` from an independently specified synthetic reference profile with the same sampling convention; do not automatically equate it to the ideal spherical-cap apex-to-rim height.

## Decisions to make during implementation

- Rule for choosing substrate pixels for plane fitting (including lens exclusion).
- Detection percentile and rules for separating/filtering lenses.
- Sphere-fitting mask, numerical method, and apex convention.
- Profile direction, extent, interpolation method, and sampling density.
- Handling failed fits, edge lenses, NaNs, and missing background pixels.
