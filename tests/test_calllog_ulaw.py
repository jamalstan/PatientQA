from patientqa.calllog.ulaw import (
    DECODE_TABLE,
    pcm16_to_ulaw_byte,
    pcm16_to_ulaw_bytes,
    ulaw_byte_to_pcm16,
    ulaw_bytes_to_pcm,
)


def test_decode_known_anchors() -> None:
    assert ulaw_byte_to_pcm16(0xFF) == 0  # positive silence
    assert ulaw_byte_to_pcm16(0x7F) == 0  # negative silence
    assert ulaw_byte_to_pcm16(0x80) == 32124  # +max of the G.711 range
    assert ulaw_byte_to_pcm16(0x00) == -32124  # -max


def test_encode_known_anchors() -> None:
    assert pcm16_to_ulaw_byte(0) == 0xFF
    assert pcm16_to_ulaw_byte(8) == 0xFE  # smallest non-silent step
    assert pcm16_to_ulaw_byte(32767) == 0x80
    assert pcm16_to_ulaw_byte(-32768) == 0x00


def test_round_trip_within_one_quantization_step() -> None:
    # max segment step is 256 in the 14-bit domain -> 1024 in 16-bit samples
    for sample in range(-32768, 32768, 997):
        decoded = ulaw_byte_to_pcm16(pcm16_to_ulaw_byte(sample))
        assert abs(sample - decoded) <= 1030, (sample, decoded)


def test_table_matches_function() -> None:
    assert DECODE_TABLE[0x80] == ulaw_byte_to_pcm16(0x80)
    assert len(DECODE_TABLE) == 256


def test_buffer_round_trip() -> None:
    samples = [0, 8, 16128, 32000, 32767, -8, -16128, -32000, -32768]
    ulaw = pcm16_to_ulaw_bytes(samples)
    assert len(ulaw) == len(samples)
    assert list(ulaw_bytes_to_pcm(ulaw)) == [
        ulaw_byte_to_pcm16(u) for u in ulaw
    ]


def test_silence_buffer_decodes_to_zeros() -> None:
    assert list(ulaw_bytes_to_pcm(b"\xff" * 100)) == [0] * 100
