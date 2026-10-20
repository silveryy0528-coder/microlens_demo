# Microlens characterization: methodology

## Goal and scope

Characterize spherical microlenses in 2D height maps and export per-lens **radius of curvature** and **profile-based height**, together with diagnostic figures. Initially, use synthetic images with analytical ground truth.

A representative dataset covers approximately **6000 × 6000 nm** at **1024 × 1024** pixels, with a periodic array of lenses about **500 nm** high. These are example conditions, not fixed constraints. Physical scaling comes from image metadata.

## Image data model

The `Image` class stores:

- `values`: 2D NumPy array indexed `[row_y, column_x]`.
- `pixel_size_x`, `pixel_size_y`: physical spacing between sample centers.
- `xy_unit`: lateral distance unit; `data_unit`: height unit.

Pixel-center coordinates are `x = column * pixel_size_x` and `y = row * pixel_size_y`. Preserve metadata during processing; do not assume isotropic spacing or identical lateral and height units. Initially, record units without conversion.

## Synthetic reference geometry

Generate a rectangular periodic lattice of identical spherical-cap lenses on a flat substrate, viewed through a fixed field of view. Lens centers are

\[
x_i = x_{\mathrm{offset}} + i p_x, \qquad y_j = y_{\mathrm{offset}} + j p_y,
\]

where `i` and `j` may be negative. Shifting the offsets changes which lenses are visible, including those cut off by image boundaries.

For footprint radius `a`, geometric height `h`, and radial distance `r` from a lens center:

\[
R = \frac{a^2 + h^2}{2h}, \qquad
z(r) =
\begin{cases}
\sqrt{R^2-r^2}-(R-h), & r<a,\\
0, & r\ge a.
\end{cases}
\]

Require `0 < h <= a` and `pitch_x, pitch_y >= 2a` to model single-valued, nonoverlapping caps.

Render complete and partial lenses when at least one sampled pixel center lies strictly inside their footprint (`r < a`). **Ground truth includes only complete lenses**, with their lattice indices, physical centers, radius `R`, footprint radius, and geometric height. Under the current sampled-image convention, a lens is considered partial if its footprint contains a sampled pixel center in the first or last image row or column. Partial lenses remain in the image but are outside the intended characterization and validation scope.

### Substrate tilt

An optional planar tilt `z = a x + b y` is added to the ideal image, using the pixel-center coordinates above, so it is zero at pixel `(0, 0)`. The slope `(a, b)` is in `data_unit` per `xy_unit`, without unit conversion. Ground truth is unchanged: lens heights are relative to the local substrate, not absolute apex `z`.
## Processing methodology

### 1. Background correction

Select low-height **substrate pixels**, excluding microlenses. Fit a plane in physical `(x, y, z)` coordinates and subtract it. The substrate-selection rule remains to be defined.

### 2. Lens detection

Apply percentile-based thresholding to the corrected height map and identify individual lens regions. Threshold and region-filtering criteria remain to be defined. Exclude partial lenses from characterization.

### 3. Apex localization and radius

Fit a sphere in physical 3D coordinates to each selected lens and use the fit to locate its apex with subpixel precision. Report **sphere radius `R` in nm**, not reciprocal curvature. Fit-region and numerical-method choices remain to be defined.

### 4. Profile-based height

Interpolate a 1D profile through the fitted apex, including the lens and surrounding substrate. Compute **lens height = P99(profile) − P5(profile)** over valid samples. This is *not* necessarily the analytical spherical-cap height. Profile orientation, extent, and sampling remain to be defined.

### 5. Outputs

Export one result row per characterized lens (ID, apex position, radius, and height), plus diagnostic figures. Preserve relevant image metadata and processing configuration.

## Verification strategy

**Software tests** check isolated contracts, such as image metadata, coordinate mapping, and interpolation on known grids.

**Scientific validation** compares the analysis outputs with independent expectations from synthetic spherical caps, including subpixel centers and substrate tilt, noise, and varied sampling. Radius is compared with analytical `R`; expected profile-based height must be calculated using the **same profile sampling convention**, not assumed equal to geometric cap height. Partial lenses are excluded from ground-truth comparisons.
