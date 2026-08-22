"""Intentional audio perturbations stay deterministic and format-preserving."""

from patientqa.audiofx import make_road_noise


def test_road_noise_is_deterministic_non_identity_and_length_preserving() -> None:
    audio = bytes(range(256)) * 4
    first = make_road_noise(42)(audio)
    second = make_road_noise(42)(audio)

    assert first == second
    assert first != audio
    assert len(first) == len(audio)
