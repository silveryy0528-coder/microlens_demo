"""Generate and save synthetic microlens datasets under ``data/synthetic``.

All images share the same shape, FoV, pitch, and lens geometry. Image ``i`` of
``N`` shifts the lattice offset by ``i * pitch / N`` (modulo the pitch) along
each axis, starting from the base offset, to produce different views.

Per image ``NN`` the script writes:

- ``data_file_synthetic_NN.csv``: height values only, no headers or indices.
- ``metadata_synthetic_NN.json``: image metadata, generation parameters, and
  the analytical ground truth of the complete (non-border-touching) lenses.
- ``2d_image_plot_synthetic_NN.png`` and ``1d_cross_section_synthetic_NN.png``.
"""

import argparse
import inspect
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from microlens.synthetic import generate_synthetic_image

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"
DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(generate_synthetic_image).parameters.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--image-shape", type=int, nargs=2, metavar=("N_Y", "N_X"),
                        default=DEFAULTS["image_shape"])
    parser.add_argument("--fov-nm", type=float, nargs=2, metavar=("X", "Y"),
                        default=DEFAULTS["fov_nm"])
    parser.add_argument("--lens-pitch-nm", type=float, nargs=2, metavar=("X", "Y"),
                        default=DEFAULTS["lens_pitch_nm"])
    parser.add_argument("--base-offset-nm", type=float, nargs=2, metavar=("X", "Y"),
                        default=DEFAULTS["offset_nm"])
    parser.add_argument("--footprint-radius-nm", type=float,
                        default=DEFAULTS["footprint_radius_nm"])
    parser.add_argument("--lens-height-nm", type=float,
                        default=DEFAULTS["lens_height_nm"])
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing output files instead of failing")
    args = parser.parse_args()
    if args.num_images < 1:
        parser.error("--num-images must be at least 1")
    return args


def phase_shifted_offset(base_offset_nm, pitch_nm, index, num_images):
    """Return the lattice offset for image `index`, in (x, y) order."""
    return tuple(
        (base + index * pitch / num_images) % pitch
        for base, pitch in zip(base_offset_nm, pitch_nm)
    )


def horizontal_profile(image, y_nm):
    """Linearly interpolate the height row at physical `y_nm` (pixel-center convention)."""
    row = y_nm / image.pixel_size_y
    row_below = math.floor(row)
    row_above = min(row_below + 1, image.values.shape[0] - 1)
    weight = row - row_below
    return (1 - weight) * image.values[row_below] + weight * image.values[row_above]


def save_height_map_plot(image, path):
    n_y, n_x = image.values.shape
    half_x, half_y = image.pixel_size_x / 2, image.pixel_size_y / 2
    # Pixel centers lie at column * pixel_size_x and row * pixel_size_y.
    extent = (-half_x, (n_x - 1) * image.pixel_size_x + half_x,
              (n_y - 1) * image.pixel_size_y + half_y, -half_y)
    fig, ax = plt.subplots(figsize=(6, 5))
    mappable = ax.imshow(image.values, extent=extent, origin="upper", aspect="equal")
    ax.set_xlabel(f"X ({image.xy_unit})")
    ax.set_ylabel(f"Y ({image.xy_unit})")
    fig.colorbar(mappable, ax=ax, label=f"Height ({image.data_unit})")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_cross_section_plot(image, ground_truth, path):
    """Plot the horizontal profile through the complete-lens center row nearest mid-FoV."""
    fov_middle_y_nm = (image.values.shape[0] - 1) * image.pixel_size_y / 2
    profile_y_nm = min(
        (lens["center_y_nm"] for lens in ground_truth),
        key=lambda y: abs(y - fov_middle_y_nm),
    )
    x_nm = np.arange(image.values.shape[1]) * image.pixel_size_x
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_nm, horizontal_profile(image, profile_y_nm))
    ax.set_xlabel(f"X ({image.xy_unit})")
    ax.set_ylabel(f"Height ({image.data_unit})")
    ax.set_title(f"Cross section at Y = {profile_y_nm:.1f} {image.xy_unit}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return profile_y_nm


def main() -> None:
    args = parse_args()
    width = max(2, len(str(args.num_images)))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prefixes = [
        "data_file_synthetic", "metadata_synthetic",
        "2d_image_plot_synthetic", "1d_cross_section_synthetic",
    ]
    suffixes = [".csv", ".json", ".png", ".png"]
    paths = [
        [OUTPUT_DIR / f"{prefix}_{number:0{width}d}{suffix}"
         for prefix, suffix in zip(prefixes, suffixes)]
        for number in range(1, args.num_images + 1)
    ]
    existing = [p for group in paths for p in group if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"{len(existing)} output file(s) already exist, e.g. {existing[0]}. "
            "Use --overwrite to replace them."
        )

    for index, (csv_path, json_path, map_path, profile_path) in enumerate(paths):
        offset_nm = phase_shifted_offset(
            args.base_offset_nm, args.lens_pitch_nm, index, args.num_images
        )
        image, ground_truth = generate_synthetic_image(
            image_shape=tuple(args.image_shape),
            fov_nm=tuple(args.fov_nm),
            lens_pitch_nm=tuple(args.lens_pitch_nm),
            offset_nm=offset_nm,
            footprint_radius_nm=args.footprint_radius_nm,
            lens_height_nm=args.lens_height_nm,
        )
        if not ground_truth:
            raise SystemExit(
                f"Image {index + 1} has no complete lens; cannot choose a cross section."
            )

        np.savetxt(csv_path, image.values, delimiter=",")
        profile_y_nm = save_cross_section_plot(image, ground_truth, profile_path)
        save_height_map_plot(image, map_path)
        metadata = {
            "image_index": index + 1,
            "image": {
                "shape_n_y_n_x": list(image.values.shape),
                "pixel_size_x": image.pixel_size_x,
                "pixel_size_y": image.pixel_size_y,
                "xy_unit": image.xy_unit,
                "data_unit": image.data_unit,
            },
            "generation_parameters": {
                "image_shape_n_y_n_x": list(args.image_shape),
                "fov_nm_x_y": list(args.fov_nm),
                "lens_pitch_nm_x_y": list(args.lens_pitch_nm),
                "base_offset_nm_x_y": list(args.base_offset_nm),
                "offset_nm_x_y": list(offset_nm),
                "footprint_radius_nm": args.footprint_radius_nm,
                "lens_height_nm": args.lens_height_nm,
                "num_images": args.num_images,
            },
            "cross_section_y_nm": profile_y_nm,
            "ground_truth_note": "Complete lenses only; lenses touching the image border are omitted.",
            "ground_truth": ground_truth,
        }
        json_path.write_text(json.dumps(metadata, indent=2))
        print(f"Image {index + 1}/{args.num_images}: offset_nm={offset_nm}, "
              f"{len(ground_truth)} complete lenses -> {csv_path.parent}")


if __name__ == "__main__":
    main()
