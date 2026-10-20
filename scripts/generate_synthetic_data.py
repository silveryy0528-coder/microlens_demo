"""Generate and save synthetic microlens datasets under ``data/synthetic``.

All images share the same shape, FoV, pitch, and lens geometry.
Per image ``NN`` the script writes:

- ``data_file_synthetic_NN.csv``: height values only, no headers or indices.
- ``metadata_synthetic_NN.json``: image metadata, generation parameters, and
  the analytical ground truth of the complete (non-border-touching) lenses.
- ``2d_image_plot_synthetic_NN.png`` and ``1d_cross_section_synthetic_NN.png``.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from microlens.settings import read_yaml
from microlens.synthetic import generate_synthetic_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "synthetic_generation.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"
CONFIG_KEYS = {
    "num_images", "image_shape", "fov_nm", "lens_pitch_nm", "base_offset_nm",
    "footprint_radius_nm", "lens_height_nm", "overwrite",
}
PAIR_KEYS = ("image_shape", "fov_nm", "lens_pitch_nm", "base_offset_nm")


def load_config(path: Path) -> dict:
    """Read the YAML config, require all keys, and convert pairs to tuples."""
    config = read_yaml(path)
    missing = CONFIG_KEYS - config.keys()
    unknown = config.keys() - CONFIG_KEYS
    if missing or unknown:
        raise ValueError(
            f"Invalid {path.name}: missing keys {sorted(missing)}, "
            f"unknown keys {sorted(unknown)}."
        )
    num_images = config["num_images"]
    if isinstance(num_images, bool) or not isinstance(num_images, int) or num_images < 1:
        raise ValueError("num_images must be an integer >= 1.")
    if not isinstance(config["overwrite"], bool):
        raise ValueError("overwrite must be true or false.")
    for key in PAIR_KEYS:
        value = config[key]
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{key} must be a list of two values.")
        config[key] = tuple(value)
    return config


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
    config = load_config(CONFIG_PATH)
    num_images = config["num_images"]
    width = max(2, len(str(num_images)))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prefixes = [
        "data_file_synthetic", "metadata_synthetic",
        "2d_image_plot_synthetic", "1d_cross_section_synthetic",
    ]
    suffixes = [".csv", ".json", ".png", ".png"]
    paths = [
        [OUTPUT_DIR / f"{prefix}_{number:0{width}d}{suffix}"
         for prefix, suffix in zip(prefixes, suffixes)]
        for number in range(1, num_images + 1)
    ]
    existing = [p for group in paths for p in group if p.exists()]
    if existing and not config["overwrite"]:
        raise SystemExit(
            f"{len(existing)} output file(s) already exist, e.g. {existing[0]}. "
            "Set overwrite: true in the config to replace them."
        )

    for index, (csv_path, json_path, map_path, profile_path) in enumerate(paths):
        offset_nm = phase_shifted_offset(
            config["base_offset_nm"], config["lens_pitch_nm"], index, num_images
        )
        image, ground_truth = generate_synthetic_image(
            image_shape=config["image_shape"],
            fov_nm=config["fov_nm"],
            lens_pitch_nm=config["lens_pitch_nm"],
            offset_nm=offset_nm,
            footprint_radius_nm=config["footprint_radius_nm"],
            lens_height_nm=config["lens_height_nm"],
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
                "image_shape_n_y_n_x": list(config["image_shape"]),
                "fov_nm_x_y": list(config["fov_nm"]),
                "lens_pitch_nm_x_y": list(config["lens_pitch_nm"]),
                "base_offset_nm_x_y": list(config["base_offset_nm"]),
                "offset_nm_x_y": list(offset_nm),
                "footprint_radius_nm": config["footprint_radius_nm"],
                "lens_height_nm": config["lens_height_nm"],
                "num_images": num_images,
            },
            "cross_section_y_nm": profile_y_nm,
            "ground_truth_note": "Complete lenses only; lenses touching the image border are omitted.",
            "ground_truth": ground_truth,
        }
        json_path.write_text(json.dumps(metadata, indent=2))
        print(f"Image {index + 1}/{num_images}: offset_nm={offset_nm}, "
              f"{len(ground_truth)} complete lenses -> {csv_path.parent}")


if __name__ == "__main__":
    main()
