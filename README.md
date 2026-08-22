# Patient Simulator

A synthetic-patient voice bot that calls Pretty Good AI's test line
(+1-805-439-8008), plays realistic patients, and stress-tests their medical
scheduling agent. Architecture and decision log: [DESIGN.md](DESIGN.md).

## Submission artifacts

- [Call index](deliverables/calls/INDEX.md) — 11 full, two-sided conversations
  with MP3 recordings and timestamped transcripts
- [Bug report](BUGS.md) — human-reviewed findings linked to exact transcript
  and audio moments
- The two-paragraph system overview is in [Architecture](#architecture); the
  longer tradeoff and decision record is in [DESIGN.md](DESIGN.md)

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.10 is downloaded automatically):

```bash
uv sync
```

Create your local secrets file and fill in your own credentials
(`secrets.toml` is git-ignored — never commit real keys):

```bash
cp secrets.example.toml secrets.toml
```

Verify everything works — prints which providers are configured:

```bash
uv run python -m patientqa
```

The same settings may be supplied as environment variables; see
[`.env.example`](.env.example). Deepgram is the recommended STT provider for
the campaign. Setting `STT_PROVIDER=scribe` uses ElevenLabs Scribe instead and
consumes roughly 330 ElevenLabs credits per call-minute.

## Run the campaign

The committed 12-call campaign spans all ten stress classes. Every scenario is
a bounded adversarial experiment with a named hypothesis, registered red-team
techniques, a multi-turn escalation ladder, and machine-checkable success
criteria. It covers instruction-hierarchy attacks, verification bypass and PHI
extraction attempts, false premises, context poisoning, destructive-action
confirmation, state/temporal confusion, ASR and multilingual perturbation,
barge-in races, and emotional pressure. Stable verification facts, four staged
follow-ups, and a final read-back produce an 8–12-turn conversation that
naturally targets three minutes without repetition or filler.

Preview every persona and opener without dialing:

```bash
uv run python -m patientqa --campaign manifests/campaign.jsonl --dry-run
```

Then run the full campaign with one command:

```bash
uv run python -m patientqa --campaign manifests/campaign.jsonl
```

The runner starts a local Media Streams server and a cloudflared quick tunnel,
casts a persona-appropriate shared-library voice, and sequentially calls the
`live_test_number` from `secrets.toml`. The Twilio client refuses destinations
outside `allowed_numbers`; for this assessment, set both values only to
`+18054398008`. Completed `call-NNN_<timestamp>` folders are detected on rerun,
but only when the recorded manifest seed matches the current scenario; regenerating
the campaign cannot silently reuse stale evidence. An interrupted unchanged campaign
therefore resumes rather than redialing it. Use `--limit 1`
for a single live canary and `--stream-url wss://...` to reuse an existing
tunnel. A call only satisfies resume when it lasts at least 60 seconds, has at
least four turns on each side, and avoids infrastructure end states; shorter
attempts are retried once by default (`--attempts`).

Voice casting is gender-checked. If the detailed library query has no match,
the runner broadens the search without dropping the gender constraint; its
last-resort voices are Will (male), Sarah (female), and River
(neutral/nonbinary). `meta.json`, `call.json`, and the post-call report record
the persona gender, resolved voice gender, voice ID/name, source, and match
verdict. A declared mismatch is rejected before dialing.

Calls target 180 seconds by default and can never be configured above five
minutes. `--max-seconds` is rejected above 300, while Twilio receives the same
server-side `time_limit`; the local watchdog and final hang-up are additional
backstops.

To replace an earlier campaign after behavior changes, bypass resume explicitly:

```bash
uv run python -m patientqa --campaign manifests/campaign.jsonl --no-resume
```

Use `--no-resume --limit 1 --attempts 1` for one fresh canary. Export always
selects the latest quality-passing session for each `call-NNN`, so the newer
recordings replace earlier runs without deleting raw evidence.

## Architecture

The bot is a thin live transport around a provider-independent turn-taking
core. Twilio Media Streams sends byte-exact 8 kHz μ-law audio to the local call
loop; Deepgram (or Scribe) commits remote-agent turns, Cerebras plays a pinned
synthetic patient, and ElevenLabs streams that patient's speech back to Twilio.
An energy gate, semantic end-of-turn classifier, generation epoch, Twilio mark
echoes, and a lost-mark watchdog jointly handle silence, barge-in, cancellation,
and floor release without coupling those decisions to any provider SDK.

Offline, a deterministic seeded pipeline combines diverse demographics,
realistic medical clusters, a bug-bait taxonomy, identity facts, and four
scenario-specific follow-ups into a validated JSONL manifest. The campaign
runner converts each entry into a persona prompt, opener, voice, and call; the
session logger records both audio legs, aligned turns, and latency events.
Export and analysis then produce reviewer-ready compressed audio, transcripts,
an index, and an evidence-linked bug report. The longer rationale and decision
log are in [DESIGN.md](DESIGN.md).

## Synthetic data generation

Generate the campaign manifest (personas + stress-test objectives,
[DESIGN.md §6–§7](DESIGN.md)):

```bash
uv run python -m patientqa.datagen generate --count 12 --seed 20260820 --out manifest.jsonl
uv run python -m patientqa.datagen starters manifest.jsonl --count 4
uv run python -m patientqa.datagen validate manifest.jsonl
```

Deterministic per `--seed`; uses LLM elaboration (Cerebras) when a key is
configured and falls back to offline template prose otherwise. Details:
[`src/patientqa/datagen/README.md`](src/patientqa/datagen/README.md).

## Call recordings & the viewer

Every call writes a self-contained session folder under `calls/`
([DESIGN.md §12](DESIGN.md)):

```
calls/call-014_20260817T153001Z/
├── meta.json           persona + objective snapshot, written at call start
├── session.jsonl       append-only event log — turns, latencies, lifecycle
├── audio/inbound.ulaw  agent leg, byte-exact 8 kHz μ-law as received
├── audio/outbound.ulaw patient leg, byte-exact as sent
├── recording.wav       mono mixdown on close, aligned to the event timebase
├── transcript.json     the dialogue, extracted on close
├── call.json           outcome + latency stats
├── analysis.json       evidence-anchored findings + detector status
└── analysis.md         readable report with seekable audio timestamps
```

Event lines are `{"seq", "t_ms", "wall", "type", "data"}`; `t_ms` maps 1:1 onto
position in `recording.wav`, so the viewer can keep transcript, timeline and
audio in sync. The log is append-only — a crash mid-call still leaves a
playable, inspectable session (the viewer rebuilds audio from the raw μ-law
legs when `recording.wav` is missing).

Every finalized `run_call()` is post-processed automatically. Deterministic
checks and the evidence-guarded Cerebras judge write `analysis.json` and
`analysis.md`; `calls/ISSUES.md` is then refreshed from those per-call files.
The judge's proposed time is never trusted: its verbatim quote must resolve to
one transcript turn, whose logged `t_ms` becomes the issue's audio position.
Intentional injected behavior is copied into the analysis metadata and listed
separately from agent issues. A judge failure is explicit in detector status
and never prevents the heuristic report from being written.

Browse sessions in the dark-mode static viewer (all local, nothing uploaded):

```bash
uv run python -m patientqa.calllog demo --out calls   # optional: synthetic demo call
uv run python -m patientqa.calllog viewer calls       # writes + opens viewer.html
```

Drop call folders (or the whole `calls/` folder) onto the page. Click any turn
or event to jump the audio there; the waveform shows who spoke when (blue =
agent, violet = patient); the event timeline carries every logged moment with
per-stage latencies.

Bake a single-file report — events + audio embedded, shareable as a submission
artifact:

```bash
uv run python -m patientqa.calllog report calls/call-014_*
```

After the campaign, build the GitHub deliverables. Re-running the analyzer is
also the backfill command for older finalized sessions:

```bash
uv run python -m patientqa.calllog export calls --campaign-only --out deliverables/calls
uv run python -m patientqa.analyze calls --campaign-only --out BUGS.md
```

Each exported folder contains `recording.ogg` when ffmpeg is installed and
`recording.mp3` otherwise, plus a timestamped two-sided `transcript.txt`.
`deliverables/calls/INDEX.md` summarizes duration, turns, outcome, persona, and
objective. The analyzer runs deterministic tripwires first, then an
evidence-guarded Cerebras review; every reported issue must quote the transcript
and points to the matching audio timestamp. Listen to each flagged moment before
submitting `BUGS.md`.

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv add <package>         # add a runtime dependency
```

## Secrets convention

- Real credentials live only in `secrets.toml` (git-ignored).
- `secrets.example.toml` is the committed template for contributors.
- Environment variables override the file (`ELEVENLABS_API_KEY`, `TWILIO_API_KEY_SID`, ...).
