"""Deterministic outbound-audio perturbations for intentional call probes."""

from __future__ import annotations

import random
from collections.abc import Callable

from patientqa.calllog.ulaw import pcm16_to_ulaw_bytes, ulaw_bytes_to_pcm


def make_road_noise(seed: int, *, amplitude: int = 650) -> Callable[[bytes], bytes]:
    """Return a stateful, deterministic low road-noise transform for μ-law audio.

    The signal stays intelligible: a slowly varying rumble is mixed below the
    speech instead of clipping or dropping frames. Stateful RNG/filter values
    keep chunk boundaries from resetting the sound.
    """

    rng = random.Random(seed)
    rumble = 0.0

    def transform(ulaw: bytes) -> bytes:
        nonlocal rumble
        noisy: list[int] = []
        for sample in ulaw_bytes_to_pcm(ulaw):
            rumble = 0.985 * rumble + rng.uniform(-amplitude, amplitude) * 0.06
            white = rng.uniform(-amplitude, amplitude) * 0.22
            mixed = int(sample + rumble + white)
            noisy.append(max(-32768, min(32767, mixed)))
        return pcm16_to_ulaw_bytes(noisy)

    return transform
