from dataclasses import FrozenInstanceError

import pytest

from microlens.settings import (
    PreprocessingSettings,
    SubstrateSettings,
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
