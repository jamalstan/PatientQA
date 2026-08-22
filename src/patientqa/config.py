"""Credential loading.

Secrets live in ``secrets.toml`` at the project root (git-ignored). Contributors
copy ``secrets.example.toml`` to ``secrets.toml`` and fill in their own keys.
Environment variables override file values (``ELEVENLABS_API_KEY`` etc.) so CI
and deployments can inject credentials without a file.
"""

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

SECRETS_FILENAME = "secrets.toml"


def find_secrets_file(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) until ``secrets.toml`` is found."""
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        path = candidate / SECRETS_FILENAME
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{SECRETS_FILENAME} not found in {directory} or any parent directory. "
        "Copy secrets.example.toml to secrets.toml and fill in your keys."
    )


def load_secrets(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Parse secrets.toml into ``{provider: {key: value}}``."""
    secrets_path = path if path is not None else find_secrets_file()
    with secrets_path.open("rb") as fh:
        return tomllib.load(fh)


def get_secret(section: str, key: str, *, path: Path | None = None) -> str:
    """Return ``secrets.toml[section][key]``, with an environment-variable override."""
    env_name = f"{section}_{key}".upper()
    if env_name in os.environ:
        return os.environ[env_name]
    try:
        return load_secrets(path)[section][key]
    except KeyError as exc:
        raise KeyError(
            f"secrets.toml is missing [{section}] {key}. "
            "See secrets.example.toml for the expected shape."
        ) from exc
