# whisper-subtitle-studio

A desktop application for producing word-accurate subtitles, built on top of
[`whisper-timestamped`](https://github.com/linto-ai/whisper-timestamped).

---

## Project Overview

### Upstream: `whisper-timestamped`

OpenAI's Whisper models are trained to emit approximate timestamps at the level
of speech *segments*, typically accurate to about one second. They cannot
natively predict where an individual word begins and ends.

`whisper-timestamped` closes that gap. It applies Dynamic Time Warping (DTW) to
the cross-attention weights of the decoder to align each decoded token against
the audio, recovering per-word start and end times along with a confidence
score for every word and segment. Where possible the alignment runs on the fly
after each segment is decoded, so no additional inference pass is required, and
memory use stays close to that of plain Whisper even on long files.

The upstream project is a Python library and a pair of command-line tools
(`whisper_timestamped`, `whisper_timestamped_make_subtitles`). It produces JSON,
SRT, VTT and other formats from a terminal invocation. It has no interactive
component: there is no way to hear the audio while reading the transcript, to
see which word the model believes is being spoken at a given instant, or to
correct a mistake without editing an output file by hand.

### This fork: `whisper-subtitle-studio`

This fork adds `studio.py`, a standalone PyQt6 desktop application that wraps
the upstream transcription core in an interactive workspace. The transcript
becomes a synchronized, editable document played against the media rather than
a file emitted at the end of a batch run.

The upstream library, CLI and API are unmodified and remain fully usable; see
the [upstream README](https://github.com/linto-ai/whisper-timestamped#readme)
for their documentation.

---

## Architectural and Functional Enhancements

### GUI and concurrency

A single-window PyQt6 application (`studio.py`). Transcription runs in a
`TranscribeWorker` moved onto its own `QThread`, so the event loop is never
blocked and the window stays responsive throughout a multi-minute decode.
Whisper and torch are imported inside the worker's `run` method, which moves
roughly three seconds of import cost off application startup.

Worker lifecycle is guarded explicitly. A cancelled worker's signals may already
be queued in the event loop when it is retired, and Qt does not retract posted
events on disconnect, so stale signals are filtered by sender identity rather
than by managing connections. Retired threads are held in a set until they
unwind, which prevents Qt from collecting a `QThread` that is still running.
`closeEvent` cancels every in-flight worker before joining it, so shutdown waits
on a single forward pass instead of a full transcription.

### Studio workspace

The content area is a horizontal `QSplitter` between a video pane and the
lyrics column. With video loaded the split opens to 70/30; the divider is
draggable, and a ratio the user chooses is preserved across window resizes.

The video pane holds a `QVideoWidget` set to `KeepAspectRatio`, so the picture
is letterboxed and centred at its natural proportions whatever width the split
gives it. Media type is resolved in two stages: the file extension decides
immediately, which avoids the pane flashing open on an audio file, and
`QMediaPlayer.hasVideo()` corrects that guess once the real track list has
loaded. An `.mp4` carrying no video stream, or a container the platform cannot
decode, correctly stays collapsed.

For audio-only media the left pane collapses and is hidden outright, so the
splitter drops its handle and the lyrics occupy the full width. Transitions in
both directions are animated through a single `videoShare` property, since a
splitter's sizes are a list rather than an animatable Qt property.

The top bar (file, model, language, transcribe/cancel) and the bottom transport
and export bar span the full window width above and below the split.

### Synchronized lyrics view

The transcript is rendered as one text block per Whisper segment in a style
modelled on a music player's lyrics display: inactive lines dimmed, the active
line bright, and the word currently being spoken highlighted within it.

Every line and word records its character range once at build time, so the
per-frame cost during playback is two binary searches and a selection swap,
never a re-render of the document. The active line is glided to the vertical
centre of the viewport by a re-targeted `QPropertyAnimation` on the scrollbar.
Clicking any word seeks the player to that word's start time. Type size is
adjustable from `A-` and `A+` in the lyrics pane header, between 11 and 34 px.

### Hallucination and drift mitigation

Whisper tends to latch onto its own output over music, noise and silence,
emitting repeated phrases that are absent from the audio. Several independent
guards address this:

| Mechanism | Purpose |
| --- | --- |
| Voice activity detection (`auditok`) | Strips non-speech regions before decoding. A pure-Python energy gate, so the application stays self-contained with no model download or trust prompt. |
| `condition_on_previous_text=False` | Stops previous output being fed back as context, which is what sustains a repetition loop. |
| Temperature fallback `(0.0, 0.2, 0.4)` | Retries a degenerate decode at a higher temperature to break out of a loop. |
| `is_degenerate` | Per-line gzip compression-ratio test. Real sentences measure around 0.8-0.95; repetition loops exceed 2.4. Applied per line because an energy gate cannot distinguish music from speech. |
| `collapse_repeats` | Folds runs of identical consecutive lines into one cue spanning the whole run, catching loops whose individual lines are too short for the entropy test. |
| `SILENCE_GAP` (1.5 s) | Past this much silence after a line ends, nothing is highlighted. Holding the last line lit through an instrumental break is what reads as tracking drift. |
| `remove_empty_words=True` | Drops the zero-length words Whisper appends to segment ends, the main source of word-highlight drift. |

### Editing and export

Edit mode turns clicks from seeks into caret placement and makes the transcript
directly editable. Because one text block corresponds to one cue, keystrokes
that would split or merge a block are blocked, and pastes are flattened to a
single line; a stray newline would otherwise silently desynchronize every
following line from its timestamps.

Corrections that preserve word count keep their original timings exactly. When
words are added or removed the line's own span is redistributed across the new
words by length, which is approximate but keeps click-to-seek and highlighting
working.

Export produces SRT and WebVTT from the edited document. Cues are sorted, cues
shorter than 0.2 s are stretched to that readability floor, and any cue whose
end runs past its successor's start is trimmed, since overlapping cues render as
two stacked subtitles in most players. Files are written with explicit `\n`
line endings on every host.

### Deterministic interruption

Whisper exposes no cancellation flag, and a decode can run for minutes. Checking
a flag only at stage boundaries would leave the user waiting long after they
clicked Cancel.

Instead, the cancellation checkpoint is registered as a PyTorch forward hook on
`model.encoder.conv1` and `model.decoder.token_embedding`. The encoder hook runs
once per 30-second window and the decoder hook once per decoded token, so the
worst case between the click and the unwind is a single token. Raising from the
hook unwinds inference exactly as any other exception would, and
`whisper-timestamped` removes its own hooks in a `finally` block, so nothing is
left attached to the model. Measured latency from click to thread exit is
0.01-0.08 s.

The UI returns to its ready state in the same event-loop turn as the click
rather than freezing while the worker unwinds. A cancel left pending at window
close is joined during `closeEvent`, so no thread is orphaned.

### Localization

UI text lives in a per-locale dictionary keyed by string name, covering English,
Turkish, Spanish and German. Keys missing from a locale fall back to English, so
a partial translation degrades to English rather than displaying a raw key.

Status messages are stored as a key plus its parameters and re-rendered on every
locale change, so a message already on screen switches language with the rest of
the window. Quantities are worded through `<name>_one` / `<name>_other` key
pairs resolved at render time rather than by appending a suffix, because the
rule belongs to the locale: Turkish takes no plural after a numeral at all
("1 kelime", "32 kelime"), and the German plural of "Untertitel" is
"Untertitel".

---

## Installation and Usage

### System dependencies

`ffmpeg` is required for audio and video decoding.

```bash
# macOS
brew install ffmpeg

# Debian / Ubuntu
sudo apt update && sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

### Python environment

Python 3.9 or newer.

```bash
git clone https://github.com/<your-account>/whisper-subtitle-studio.git
cd whisper-subtitle-studio

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### Dependencies

```bash
pip install -r requirements.txt          # transcription core
pip install -r requirements-studio.txt   # desktop application
pip install -e .                         # whisper_timestamped package
```

`requirements-studio.txt` holds the two dependencies the desktop application
adds on top of the core: `PyQt6` and `auditok`.

### Running

```bash
python studio.py
```

Open a media file, choose a model, and start the transcription. The first run
with a given model downloads its weights. Models are listed smallest first;
`tiny` is the default so that iteration stays fast, since transcription
currently runs on CPU.

| Step | Action |
| --- | --- |
| Load | `Open Media` — audio or video |
| Transcribe | Select a model, then `Transcribe`; `Cancel` stops a run in progress |
| Review | Play back and follow the highlighted word; click any word to seek |
| Correct | `Edit` to fix errors in place; timings are preserved |
| Export | `Export SRT` or `Export VTT` |

---

## Roadmap and Performance Optimizations

Transcription currently runs on CPU through the reference `openai-whisper`
implementation, which is the dominant cost in the application. The items below
address throughput, memory footprint and distribution.

### `faster-whisper` (CTranslate2) backend

Reimplementing the inference path on
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) would reduce
transcription time substantially — its maintainers report roughly a fourfold
speedup over `openai-whisper` at equal accuracy, with a lower memory footprint.
The main design question is word-level alignment: CTranslate2 does not expose
cross-attention weights in the form the current DTW alignment consumes, so
either its own word-timestamp support is adopted or the alignment step is
reworked against its API.

### Quantization (INT8 / FP16)

Selectable INT8 and FP16 compute types would cut memory bandwidth, which is the
binding constraint for CPU inference and for entry-level GPUs with limited VRAM.
INT8 in particular makes the larger models practical on machines that cannot
hold them in float32. This follows naturally from a CTranslate2 backend, which
supports both directly.

### Hardware acceleration auto-detection

Device selection is currently fixed to CPU. Detecting and defaulting to the best
available backend — `mps` on Apple Silicon, `cuda` on NVIDIA hardware, CPU
otherwise — with a manual override in the UI, would remove the largest
performance gap on consumer machines. Upstream has no MPS path today, so Apple
Silicon support requires validating the alignment code on that backend.

### Audio pre-processing pipeline

Word recognition degrades sharply on music-heavy material. An optional
pre-processing stage that isolates vocals before transcription would improve
results directly. This is worth implementing in two tiers: lightweight
band-pass and dynamics filtering, which is cheap enough to enable by default,
and full stem separation via [Demucs](https://github.com/adefossez/demucs) as
an opt-in for difficult sources, given its significant runtime cost.

### Standalone binary distribution

Packaging with PyInstaller or Nuitka would produce a self-contained executable
per platform, removing Python installation, virtual environment setup and
dependency resolution from the end user's path. The notable packaging problems
are the bundled `ffmpeg` binary and model weight handling: weights should be
fetched on first run rather than embedded, to keep the download to a reasonable
size.

---

## License and Attribution

This project is a fork of
[`whisper-timestamped`](https://github.com/linto-ai/whisper-timestamped) by
LINAGORA, distributed under the GNU General Public License v3. The upstream work
is by Jérôme Louradour; the DTW-on-cross-attention alignment approach derives
from a notebook by Jong Wook Kim. Whisper itself is by OpenAI.

See [LICENSE](LICENSE) for full terms. If you use this work in research, cite
the upstream project:

```bibtex
@misc{lintoai2023whispertimestamped,
  title  = {whisper-timestamped},
  author = {Louradour, J{\'e}r{\^o}me},
  journal = {GitHub repository},
  year   = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/linto-ai/whisper-timestamped}}
}
```
