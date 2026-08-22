"""Cloudflared discovery and output parsing stay offline and deterministic."""

from patientqa.tunnel import find_cloudflared, parse_tunnel_url


def test_parse_tunnel_url_and_ignore_unrelated_output() -> None:
    assert parse_tunnel_url("still starting") is None
    assert (
        parse_tunnel_url("INF https://quiet-tree-123.trycloudflare.com is ready")
        == "https://quiet-tree-123.trycloudflare.com"
    )


def test_find_cloudflared_prefers_repo_local_binary(tmp_path) -> None:
    tools = tmp_path / ".tools"
    tools.mkdir()
    binary = tools / "cloudflared.exe"
    binary.write_bytes(b"binary")
    assert find_cloudflared(tmp_path) == str(binary)
