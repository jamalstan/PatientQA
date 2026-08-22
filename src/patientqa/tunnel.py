"""The public-wss bridge Twilio dials back on (a cloudflared quick tunnel).

Twilio Media Streams must reach us over a public ``wss://`` URL. A cloudflared
quick tunnel provides one for free with no account (DESIGN.md §3.1's
"predictability over cleverness"): run ``cloudflared tunnel --url
http://127.0.0.1:PORT``, scrape the printed ``*.trycloudflare.com`` URL, and
swap ``https`` for ``wss``. The binary is looked up in ``.tools/`` (its
committed-by-.gitignore home on this machine), ``.tmp/``, then PATH.

The URL parser is pure so tests can pin it; :func:`start_tunnel` is the
asyncio glue that owns the subprocess.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_STARTUP_TIMEOUT_S = 60.0

_BINARY_CANDIDATES = (".tools/cloudflared.exe", ".tmp/cloudflared.exe")


@dataclass
class Tunnel:
    """A running quick tunnel; ``stop()`` is safe to call twice."""

    process: asyncio.subprocess.Process
    https_url: str

    @property
    def wss_url(self) -> str:
        return self.https_url.replace("https://", "wss://", 1)

    def stop(self) -> None:
        if self.process.returncode is None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass


def parse_tunnel_url(text: str) -> str | None:
    """First ``https://…trycloudflare.com`` URL in cloudflared's output, if any."""
    match = _URL_RE.search(text)
    return match.group(0) if match else None


def find_cloudflared(root: Path | None = None) -> str | None:
    """Locate the binary: repo-local ``.tools``/``.tmp`` first, then PATH."""
    root = root or Path.cwd()
    for candidate in _BINARY_CANDIDATES:
        path = root / candidate
        if path.is_file():
            return str(path)
    return shutil.which("cloudflared")


async def start_tunnel(
    port: int, *, timeout_s: float = _STARTUP_TIMEOUT_S, binary: str | None = None
) -> Tunnel:
    """Run a quick tunnel to ``port`` and wait for its public URL."""
    binary = binary or find_cloudflared()
    if binary is None:
        raise RuntimeError(
            "cloudflared not found — put the binary at .tools/cloudflared(.exe) "
            "or on PATH (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)"
        )
    process = await asyncio.create_subprocess_exec(
        binary,
        "tunnel",
        "--url",
        f"http://127.0.0.1:{port}",
        "--no-autoupdate",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        https_url = await asyncio.wait_for(_await_url(process), timeout_s)
    except BaseException:
        if process.returncode is None:
            process.kill()
        raise
    return Tunnel(process=process, https_url=https_url)


async def _await_url(process: asyncio.subprocess.Process) -> str:
    assert process.stderr is not None
    while True:
        line = await process.stderr.readline()
        if not line:  # EOF: the tunnel died before printing a URL
            raise RuntimeError(
                f"cloudflared exited early with code {process.returncode}"
            )
        url = parse_tunnel_url(line.decode("utf-8", errors="replace"))
        if url:
            return url
