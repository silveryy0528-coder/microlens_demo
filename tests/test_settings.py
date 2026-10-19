from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from microlens.settings import (
    PreprocessingSettings,
    SubstrateSettings,
    read_yaml,
    write_yaml,
)
from microlens.usecasemicrolens import MicrolensSettings


def test_preprocessing_settings_WHEN_method_is_supported_THEN_constructs_valid_settings():
    substrate = PreprocessingSettings(
        method="substrate",
        substrate=SubstrateSettings(percentile=5.0, distance=5.0),
    )
    all_pixels = PreprocessingSettings(method="all_pixels")
    substrate_settings = MicrolensSettings(preprocessing=substrate)
    all_pixels_settings = MicrolensSettings(preprocessing=all_pixels)

    assert substrate_settings.preprocessing.method == "substrate"
    assert substrate_settings.preprocessing.substrate.percentile == 5.0
    assert substrate_settings.preprocessing.substrate.distance == 5.0
    assert all_pixels_settings.preprocessing.method == "all_pixels"


@pytest.mark.parametrize(
    ("percentile", "distance"),
    [
        (True, 5.0),
        (False, 5.0),
        (float("nan"), 5.0),
        (float("inf"), 5.0),
        (float("-inf"), 5.0),
        (-0.1, 5.0),
        (100.1, 5.0),
        (5.0, True),
        (5.0, False),
        (5.0, float("nan")),
        (5.0, float("inf")),
        (5.0, float("-inf")),
        (5.0, -0.1),
    ],
)
def test_substrate_settings_WHEN_parameters_are_invalid_THEN_raises(
    percentile, distance
):
    with pytest.raises((TypeError, ValueError)):
        SubstrateSettings(percentile=percentile, distance=distance)


@pytest.mark.parametrize(
    ("method", "substrate"),
    [
        ("unknown", None),
        ("substrate", None),
        ("all_pixels", SubstrateSettings(percentile=5.0, distance=5.0)),
        ("substrate", {"percentile": 5.0, "distance": 5.0}),
        (1, None),
    ],
)
def test_preprocessing_settings_WHEN_method_or_configuration_is_invalid_THEN_raises(
    method, substrate
):
    with pytest.raises((TypeError, ValueError)):
        PreprocessingSettings(method=method, substrate=substrate)


def test_settings_WHEN_frozen_instance_is_reassigned_THEN_raises_frozen_instance_error():
    settings = SubstrateSettings(percentile=5.0, distance=5.0)

    with pytest.raises(FrozenInstanceError):
        settings.percentile = 10.0


def test_yaml_io_WHEN_dictionary_is_written_and_read_THEN_round_trips(tmp_path):
    data = {"preprocessing": {"method": "all_pixels"}, "enabled": True}
    path = tmp_path / "settings.yaml"

    write_yaml(path, data)

    assert read_yaml(str(path)) == data


def test_microlens_settings_from_dict_WHEN_method_is_supported_THEN_constructs_settings():
    substrate = MicrolensSettings.from_dict(
        {
            "preprocessing": {
                "method": "substrate",
                "substrate": {"percentile": 5.0, "distance": 5.0},
            }
        }
    )
    all_pixels = MicrolensSettings.from_dict(
        {"preprocessing": {"method": "all_pixels"}}
    )

    assert substrate.preprocessing == PreprocessingSettings(
        method="substrate",
        substrate=SubstrateSettings(percentile=5.0, distance=5.0),
    )
    assert all_pixels.preprocessing == PreprocessingSettings(method="all_pixels")


def test_yaml_and_microlens_settings_WHEN_input_is_malformed_THEN_raises_clear_errors(
    tmp_path,
):
    malformed_yaml = tmp_path / "malformed.yaml"
    malformed_yaml.write_text("preprocessing: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        read_yaml(malformed_yaml)

    non_dictionary_yaml = tmp_path / "sequence.yaml"
    non_dictionary_yaml.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root.*dictionary"):
        read_yaml(non_dictionary_yaml)

    with pytest.raises(FileNotFoundError):
        read_yaml(tmp_path / "missing.yaml")
    with pytest.raises(TypeError, match="dictionary"):
        write_yaml(tmp_path / "not-a-dictionary.yaml", [])
    with pytest.raises(TypeError, match="cannot be serialized"):
        write_yaml(tmp_path / "unsupported.yaml", {"value": object()})

    invalid_settings = [
        [],
        {},
        {"preprocessing": {"method": "substrate"}},
        {
            "preprocessing": {
                "method": "substrate",
                "substrate": {"percentile": 5.0},
            }
        },
        {
            "preprocessing": {
                "method": "all_pixels",
                "substrate": {"percentile": 5.0, "distance": 5.0},
            }
        },
        {"preprocessing": {"method": "all_pixels", "extra": 1}},
    ]
    for data in invalid_settings:
        with pytest.raises((TypeError, ValueError)):
            MicrolensSettings.from_dict(data)


def test_microlens_settings_WHEN_serialized_and_reconstructed_THEN_preserves_values(
    tmp_path,
):
    config_path = Path(__file__).parents[1] / "configs" / "settings.yaml"
    settings = MicrolensSettings.from_dict(read_yaml(config_path))
    saved_path = tmp_path / "saved.yaml"

    write_yaml(saved_path, asdict(settings))

    restored = MicrolensSettings.from_dict(read_yaml(saved_path))
    assert restored == settings
