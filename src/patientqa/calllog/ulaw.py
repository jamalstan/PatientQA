"""G.711 μ-law ⇄ 16-bit PCM translation (DESIGN.md §3.1).

Session audio is stored byte-exact in Twilio's wire format (8 kHz μ-law);
this module is the single translation point used when finalizing a playable
recording, because browsers and editors want PCM. The algorithms are the
canonical ITU G.711 reference (Sun Microsystems' g711.c), so frames round-trip
identically with Twilio's own codec stack. Pure Python on purpose: zero new
dependencies, and finalization runs once per call — never in the hot path.
"""

from array import array
from collections.abc import Iterable

_BIAS = 0x84
_CLIP = 8159
_QUANT_MASK = 0x0F
_SEG_MASK = 0x70
_SEG_SHIFT = 4
_SIGN_BIT = 0x80
_SEG_END = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)


def _decode_byte(u_val: int) -> int:
    """One μ-law byte → 16-bit PCM sample (reference decoder)."""
    u_val = ~u_val & 0xFF
    t = ((u_val & _QUANT_MASK) << 3) + _BIAS
    t <<= (u_val & _SEG_MASK) >> _SEG_SHIFT
    return (_BIAS - t) if (u_val & _SIGN_BIT) else (t - _BIAS)


#: ``DECODE_TABLE[byte]`` → PCM sample; precomputed so buffers decode in C-speed map steps.
DECODE_TABLE: tuple[int, ...] = tuple(_decode_byte(u) for u in range(256))


def ulaw_byte_to_pcm16(u_val: int) -> int:
    """One μ-law byte → 16-bit PCM sample."""
    return DECODE_TABLE[u_val & 0xFF]


def pcm16_to_ulaw_byte(sample: int) -> int:
    """One 16-bit PCM sample → μ-law byte (reference encoder)."""
    sample >>= 2
    if sample < 0:
        sample = -sample
        mask = 0x7F
    else:
        mask = 0xFF
    if sample > _CLIP:
        sample = _CLIP
    sample += _BIAS >> 2
    for seg, end in enumerate(_SEG_END):
        if sample <= end:
            return ((seg << 4) | ((sample >> (seg + 1)) & _QUANT_MASK)) ^ mask
    return 0x7F ^ mask


def ulaw_bytes_to_pcm(data: bytes) -> array:
    """A μ-law buffer → ``array('h')`` of 16-bit PCM samples, one per byte."""
    return array("h", map(DECODE_TABLE.__getitem__, data))


def pcm16_to_ulaw_bytes(samples: Iterable[int]) -> bytes:
    """16-bit PCM samples → μ-law bytes, one per sample."""
    return bytes(pcm16_to_ulaw_byte(s) for s in samples)
