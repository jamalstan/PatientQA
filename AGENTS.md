# AGENTS.md

Synthetic-patient voice bot ("Patient Simulator") that calls Pretty Good AI's
medical-scheduling test line and stress-tests their voice agent. Requirements
live in `Pretty Good AI — AI Engineering Challenge.md`; architecture and
decision log in [DESIGN.md](DESIGN.md) — read the relevant DESIGN.md section
before changing architecture, config, or the data pipeline (§6–§7 for datagen).

## Layout

- `src/patientqa/` — the package (uv src layout; dist and console script are
  also `patientqa`).
  - `config.py` — `secrets.toml` loader (`load_secrets`, `get_secret`); env vars override the file.
  - `orchestrator.py` — wires the providers (`build_patient_simulator`); owns the
    gated/streaming engagement (`respond`/`respond_streaming`, DESIGN.md §5.3–§5.4).
  - `turns.py` — turn-taking core (state machine, epoch guard, energy gate,
    speakable chunker); transport-free by design (§5.4).
  - `stt.py` + `deepgram/`, `elevenlabs/`, `twilio/`, `cerebras/` — thin provider
    clients (settings object in, typed client out). STT events are provider-agnostic
    (`Partial`/`Final`); language/keywords pin to the manifest persona (§3.4).
  - `datagen/` — offline manifest pipeline; has its own README — read it before touching datagen.
- `tests/` — mirrors the package; `tests/datagen/` shares `factories.py` and a
  `secrets_file` fixture. Provider tests use fakes (FakeBrain/FakeVoice); no network.
- `manifests/` — generated JSONL manifests, starter sets, and reports
  (deterministic per `--seed`).
- `secrets.example.toml` — committed template. Real keys go only in git-ignored
  `secrets.toml`; never hardcode or commit credentials.

## Commands

```bash
uv sync                        # setup (Python 3.10 managed by uv)
uv run pytest                  # tests (addopts -q)
uv run ruff check .            # lint
uv run python -m patientqa     # smoke-check credentials (prints provider names only)
uv run python -m patientqa.datagen generate --count 60 --out manifest.jsonl
uv run python -m patientqa.datagen starters manifest.jsonl --limit 8
uv run python -m patientqa.datagen validate manifest.jsonl
```

## Conventions & gotchas

- Python 3.10 minimum: no 3.11+-only stdlib (`tomllib` is backported via `tomli`).
- Ruff: line-length 100, rules `E,F,I,UP,B` (imports are sorted; absolute
  imports `from patientqa.…`). `tests/datagen/test_schemas.py` alone ignores
  E501 — it embeds the DESIGN.md §6.3 manifest example verbatim.
- Imports/commands use `patientqa`, but the domain vocabulary keeps "simulator":
  the `PatientSimulator` class and `build_patient_simulator()` are intentional
  and were not part of the package rename.
- Windows + Git Bash: `uv` is at `~/.local/bin/uv.exe` and may not be on PATH
  — prefix commands with `export PATH="$USERPROFILE/.local/bin:$PATH"`.
- Git auth is a fine-grained PAT stored as a local `extraheader` in
  `.git/config` (see commit a8e59f9) — don't overwrite or strip it.
