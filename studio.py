#!/usr/bin/env python3
"""Whisper Subtitle Studio -- a desktop GUI for word-level transcription.

Built on whisper-timestamped: transcribes media, then shows the result as a
Spotify-style lyrics view -- dimmed lines, a bright active line, the spoken word
highlighted, and click-anywhere-to-seek.
"""

import os
import re
import sys
import zlib
from dataclasses import dataclass, field

from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    Qt,
    QThread,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QTextBlockFormat, QTextCharFormat, QTextCursor
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Whisper model sizes, smallest first. 'tiny' is the default so iteration stays
# fast -- everything runs on CPU on Apple Silicon (no MPS path upstream).
MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]

# UI display languages. Names are endonyms, so they read the same whichever
# locale is active and are never themselves translated.
LOCALES = [("English", "en"), ("Türkçe", "tr"), ("Español", "es"), ("Deutsch", "de")]

DEFAULT_LOCALE = "en"

# Lyrics type scale, in px. Users nudge within these bounds via A- / A+.
FONT_MIN, FONT_MAX, FONT_STEP, FONT_DEFAULT = 11, 34, 2, 16

AUDIO_EXT = ["mp3", "wav", "m4a", "flac", "ogg", "opus", "aac", "wma", "aiff"]
VIDEO_EXT = ["mp4", "mov", "mkv", "webm", "avi", "m4v", "mpg", "mpeg", "wmv"]

def _glob(exts):
    return " ".join(f"*.{e}" for e in exts)

MEDIA_FILTER = ";;".join([
    f"Media ({_glob(AUDIO_EXT + VIDEO_EXT)})",
    f"Audio ({_glob(AUDIO_EXT)})",
    f"Video ({_glob(VIDEO_EXT)})",
    "All files (*)",
])

# Voice activity detection. auditok is a pure-Python energy gate: no model, no
# download, so the app stays self-contained. It strips quiet, not music -- an
# energy gate cannot tell the two apart -- so the repetition guards below carry
# the rest. Upstream's other option, "silero", is a real speech classifier but
# fetches itself through torch.hub behind an interactive trust prompt, which
# would block this worker thread on a question nobody can answer.
VAD_METHOD = "auditok"

# Temperature fallback: if a decode looks degenerate (bad compression ratio or
# low logprob) Whisper retries hotter, which breaks it out of repetition loops.
# More than one value forces upstream's naive (two-pass) path -- see README note.
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4)

# Past this much silence after a line ends, nothing is highlighted -- otherwise
# the last line stays lit through an instrumental break and reads as drift.
SILENCE_GAP = 1.5   # seconds

# Subtitle cues shorter than this are stretched; it is the usual readability floor.
MIN_CUE_DURATION = 0.2   # seconds

# A line that compresses better than this is a repetition loop, not speech.
# Same heuristic (and value) Whisper applies per decode, applied again per line
# because VAD cannot tell music from speech -- it only strips quiet.
REPETITION_RATIO = 2.4
REPETITION_MIN_CHARS = 32   # below this, gzip overhead makes the ratio meaningless

# UI strings per locale. Keys missing from a locale fall back to English, so a
# partial translation degrades to English rather than showing a raw key.
STRINGS = {
    "en": {
        "open_media": "Open Media",
        "no_file": "No file loaded",
        "model": "Model",
        "language": "Language",
        "transcribe": "Transcribe",
        "transcribing_btn": "Transcribing…",
        "volume": "Vol",
        "export_srt": "Export SRT",
        "export_vtt": "Export VTT",
        "model_tip": "Larger models are more accurate but much slower on CPU",
        "language_tip": "Display language of this window",
        "font_smaller_tip": "Decrease transcript text size",
        "font_larger_tip": "Increase transcript text size",
        "placeholder": "Open a media file, pick a model, and hit Transcribe — the transcript appears here as lyrics.",
        "ready": "Ready",
        "loaded": "Loaded {name}",
        "loading_whisper": "Loading Whisper…",
        "loading_model": "Loading model '{model}' (first run downloads it)…",
        "transcribing": "Transcribing with '{model}'…",
        "transcribed": "Transcribed {count} words in {lines} lines with '{model}'",
        "no_speech": "No speech found in this file",
        "failed": "Transcription failed — {error}",
        "jumped": "Jumped to {secs:.2f}s — “{word}”",
        "playback_error": "Playback error: {error}",
        "edit": "Edit",
        "edit_tip": "Correct the transcript — exports use your edits",
        "edit_on": "Edit mode — click places the cursor; line breaks are locked",
        "edit_off": "Edit mode off — click seeks again",
        "save_srt": "Save SRT",
        "save_vtt": "Save VTT",
        "exported": "Exported {count} cues to {name}",
        "export_failed": "Export failed — {error}",
        "vad_unavailable": "Voice detection unavailable, transcribing without it…",
    },
    "tr": {
        "open_media": "Medya Aç",
        "no_file": "Dosya yüklenmedi",
        "model": "Model",
        "language": "Dil",
        "transcribe": "Deşifre Et",
        "transcribing_btn": "Deşifre ediliyor…",
        "volume": "Ses",
        "export_srt": "SRT Dışa Aktar",
        "export_vtt": "VTT Dışa Aktar",
        "model_tip": "Büyük modeller daha doğrudur ama CPU'da çok daha yavaştır",
        "language_tip": "Bu pencerenin görüntüleme dili",
        "font_smaller_tip": "Metin boyutunu küçült",
        "font_larger_tip": "Metin boyutunu büyüt",
        "placeholder": "Bir medya dosyası açın, model seçin ve Deşifre Et'e basın — metin burada şarkı sözü gibi görünür.",
        "ready": "Hazır",
        "loaded": "{name} yüklendi",
        "loading_whisper": "Whisper yükleniyor…",
        "loading_model": "'{model}' modeli yükleniyor (ilk çalıştırmada indirilir)…",
        "transcribing": "'{model}' ile deşifre ediliyor…",
        "transcribed": "{count} kelime, {lines} satır '{model}' ile deşifre edildi",
        "no_speech": "Bu dosyada konuşma bulunamadı",
        "failed": "Deşifre başarısız — {error}",
        "jumped": "{secs:.2f}sn konumuna atlandı — “{word}”",
        "playback_error": "Oynatma hatası: {error}",
        "edit": "Düzenle",
        "edit_tip": "Metni düzeltin — dışa aktarım düzenlemelerinizi kullanır",
        "edit_on": "Düzenleme modu — tıklama imleci taşır; satır sonları kilitli",
        "edit_off": "Düzenleme modu kapalı — tıklama yine atlar",
        "save_srt": "SRT Kaydet",
        "save_vtt": "VTT Kaydet",
        "exported": "{count} altyazı {name} dosyasına aktarıldı",
        "export_failed": "Dışa aktarma başarısız — {error}",
        "vad_unavailable": "Ses algılama kullanılamıyor, onsuz deşifre ediliyor…",
    },
    "es": {
        "open_media": "Abrir medio",
        "no_file": "Ningún archivo cargado",
        "model": "Modelo",
        "language": "Idioma",
        "transcribe": "Transcribir",
        "transcribing_btn": "Transcribiendo…",
        "volume": "Vol",
        "export_srt": "Exportar SRT",
        "export_vtt": "Exportar VTT",
        "model_tip": "Los modelos grandes son más precisos pero mucho más lentos en CPU",
        "language_tip": "Idioma de visualización de esta ventana",
        "font_smaller_tip": "Reducir el tamaño del texto",
        "font_larger_tip": "Aumentar el tamaño del texto",
        "placeholder": "Abre un archivo, elige un modelo y pulsa Transcribir — la transcripción aparece aquí como letra.",
        "ready": "Listo",
        "loaded": "{name} cargado",
        "loading_whisper": "Cargando Whisper…",
        "loading_model": "Cargando el modelo '{model}' (la primera vez se descarga)…",
        "transcribing": "Transcribiendo con '{model}'…",
        "transcribed": "{count} palabras en {lines} líneas transcritas con '{model}'",
        "no_speech": "No se encontró voz en este archivo",
        "failed": "Error en la transcripción — {error}",
        "jumped": "Saltado a {secs:.2f}s — “{word}”",
        "playback_error": "Error de reproducción: {error}",
        "edit": "Editar",
        "edit_tip": "Corrige la transcripción — la exportación usa tus cambios",
        "edit_on": "Modo edición — el clic coloca el cursor; los saltos están bloqueados",
        "edit_off": "Modo edición desactivado — el clic vuelve a saltar",
        "save_srt": "Guardar SRT",
        "save_vtt": "Guardar VTT",
        "exported": "{count} subtítulos exportados a {name}",
        "export_failed": "Error al exportar — {error}",
        "vad_unavailable": "Detección de voz no disponible, transcribiendo sin ella…",
    },
    "de": {
        "open_media": "Medien öffnen",
        "no_file": "Keine Datei geladen",
        "model": "Modell",
        "language": "Sprache",
        "transcribe": "Transkribieren",
        "transcribing_btn": "Transkribiere…",
        "volume": "Lautst.",
        "export_srt": "SRT exportieren",
        "export_vtt": "VTT exportieren",
        "model_tip": "Größere Modelle sind genauer, auf der CPU aber deutlich langsamer",
        "language_tip": "Anzeigesprache dieses Fensters",
        "font_smaller_tip": "Schriftgröße verkleinern",
        "font_larger_tip": "Schriftgröße vergrößern",
        "placeholder": "Datei öffnen, Modell wählen und auf Transkribieren klicken — das Transkript erscheint hier als Liedtext.",
        "ready": "Bereit",
        "loaded": "{name} geladen",
        "loading_whisper": "Whisper wird geladen…",
        "loading_model": "Modell '{model}' wird geladen (beim ersten Mal heruntergeladen)…",
        "transcribing": "Transkribiere mit '{model}'…",
        "transcribed": "{count} Wörter in {lines} Zeilen mit '{model}' transkribiert",
        "no_speech": "In dieser Datei wurde keine Sprache gefunden",
        "failed": "Transkription fehlgeschlagen — {error}",
        "jumped": "Zu {secs:.2f}s gesprungen — „{word}“",
        "playback_error": "Wiedergabefehler: {error}",
        "edit": "Bearbeiten",
        "edit_tip": "Transkript korrigieren — der Export übernimmt Ihre Änderungen",
        "edit_on": "Bearbeitungsmodus — Klick setzt den Cursor; Zeilenumbrüche gesperrt",
        "edit_off": "Bearbeitungsmodus aus — Klick springt wieder",
        "save_srt": "SRT speichern",
        "save_vtt": "VTT speichern",
        "exported": "{count} Untertitel nach {name} exportiert",
        "export_failed": "Export fehlgeschlagen — {error}",
        "vad_unavailable": "Spracherkennung nicht verfügbar, Transkription ohne sie…",
    },
}


@dataclass
class Word:
    """One transcribed word, tied to both its audio time and its place in the document."""
    text: str
    start: float          # seconds
    end: float            # seconds
    doc_start: int = 0    # character offset in the transcript document
    doc_end: int = 0


@dataclass
class Line:
    """One Whisper segment, rendered as a single lyrics line (one text block)."""
    words: list = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    block: int = 0        # text-block index, used to locate the line for scrolling
    doc_start: int = 0
    doc_end: int = 0

    @property
    def text(self):
        return " ".join(w.text for w in self.words)


def format_time(ms):
    """Milliseconds -> m:ss (or h:mm:ss past an hour)."""
    if ms < 0:
        ms = 0
    total = int(ms // 1000)
    h, m, s = total // 3600, (total // 60) % 60, total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def condense_error(message):
    """Squeeze a multi-line tool error down to one status-bar line.

    Whisper surfaces ffmpeg failures by embedding ffmpeg's whole stderr -- banner,
    build flags and all -- so the real cause is the last line, not the first.
    """
    lines = [l.strip() for l in str(message).splitlines() if l.strip()]
    if not lines:
        return str(message)

    detail = [
        l for l in lines[1:]
        if re.search(r"(?i)\berror\b|no such file|not found|permission denied", l)
    ]
    if detail:
        return f"{lines[0].split(':')[0]}: {detail[-1]}"
    return lines[0]


def is_degenerate(text):
    """True when a line is a repetition loop ("la la la la…") rather than speech.

    Whisper latches onto its own output over music and beats. gzip separates the
    two cleanly: real sentences measured here land at 0.8-0.95, loops at 2.9+.
    """
    body = text.strip()
    if len(body) < REPETITION_MIN_CHARS:
        return False
    raw = body.encode("utf-8")
    return len(raw) / max(1, len(zlib.compress(raw))) > REPETITION_RATIO


def collapse_repeats(lines):
    """Fold runs of identical consecutive lines into one spanning cue.

    A loop over music often comes out as short lines ("Okay." "Okay." "Okay.")
    that individually are too short for the gzip test to judge. The run itself
    is the signal. The surviving line keeps the whole run's time span, so no
    audio stops being covered -- it is just no longer repeated on screen.
    """
    kept = []
    for line in lines:
        key = " ".join(line.text.lower().split())
        if kept and key == kept[-1][1]:
            kept[-1][0].end = max(kept[-1][0].end, line.end)
            continue
        kept.append([line, key])
    return [line for line, _ in kept]


def format_timestamp(seconds, millis_sep=","):
    """Seconds -> HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (WebVTT)."""
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{millis:03d}"


def build_cues(entries):
    """Turn (start, end, text) triples into ordered, non-overlapping cues.

    Whisper segment ends can run past the next segment's start, which players
    render as two subtitles stacked on screen -- so each cue is trimmed to its
    successor. Empty text is dropped rather than exported as a blank cue.
    """
    cues = []
    for start, end, text in entries:
        text = " ".join(str(text).split())   # cues must be single-line
        if text:
            cues.append([float(start), float(end), text])

    cues.sort(key=lambda cue: cue[0])
    for i, cue in enumerate(cues):
        if cue[1] - cue[0] < MIN_CUE_DURATION:
            cue[1] = cue[0] + MIN_CUE_DURATION
        if i + 1 < len(cues) and cue[1] > cues[i + 1][0]:
            cue[1] = max(cue[0] + 0.001, cues[i + 1][0])
    return cues


def build_srt(cues):
    """SubRip: 1-based index, comma decimal separator, blank line between cues."""
    blocks = [
        f"{i}\n"
        f"{format_timestamp(start, ',')} --> {format_timestamp(end, ',')}\n"
        f"{text}\n"
        for i, (start, end, text) in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


def build_vtt(cues):
    """WebVTT: required header, dot decimal separator, no cue numbering."""
    blocks = ["WEBVTT\n"] + [
        f"{format_timestamp(start, '.')} --> {format_timestamp(end, '.')}\n"
        f"{text}\n"
        for start, end, text in cues
    ]
    return "\n".join(blocks)


class LyricsView(QTextEdit):
    """Lyrics-style transcript: dim lines, one bright active line, one hot word.

    Every line and word records its character range once at build time, so the
    per-frame work while playing is two range lookups plus a selection swap --
    never a re-render of the document.
    """

    DIM = QColor(255, 255, 255, 89)          # rgba(255,255,255,0.35)
    ACTIVE_LINE = QColor("#ffffff")
    ACTIVE_WORD_BG = QColor("#1ed760")
    ACTIVE_WORD_FG = QColor("#08130c")

    def __init__(self, on_seek):
        super().__init__()
        self._on_seek = on_seek
        self._lines: list[Line] = []
        self._active_line = -1
        self._active_word = -1
        self._editing = False
        self._building = False   # suppresses resync while set_lines builds
        self.font_px = FONT_DEFAULT

        self.setReadOnly(True)
        self.setMouseTracking(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.apply_font_size(self.font_px)

        self._line_fmt = QTextCharFormat()
        self._line_fmt.setForeground(self.ACTIVE_LINE)

        self._word_fmt = QTextCharFormat()
        self._word_fmt.setBackground(self.ACTIVE_WORD_BG)
        self._word_fmt.setForeground(self.ACTIVE_WORD_FG)

        # Smooth auto-scroll. Re-targeted rather than re-created on each line.
        self._scroll = QPropertyAnimation(self.verticalScrollBar(), b"value", self)
        self._scroll.setDuration(340)
        self._scroll.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self.document().contentsChanged.connect(self._resync)

    # -- content ----------------------------------------------------------

    @property
    def lines(self):
        return self._lines

    @property
    def words(self):
        return [w for line in self._lines for w in line.words]

    def apply_font_size(self, px):
        """Set the lyrics type size. Line spacing is proportional, so it follows."""
        self.font_px = max(FONT_MIN, min(FONT_MAX, px))
        self.setStyleSheet(f"""
            QTextEdit {{
                background: #121212;
                border: none;
                padding: 28px 34px;
                font-size: {self.font_px}px;
                font-weight: 600;
                color: rgba(255, 255, 255, 0.35);
                selection-background-color: rgba(30, 215, 96, 0.30);
            }}
        """)
        if self._active_line >= 0:
            self._scroll_to_line(self._active_line, animate=False)

    def set_lines(self, lines):
        """Render `lines` as one text block each, recording all character ranges."""
        self._building = True
        self.clear()
        self._lines = lines
        self._active_line = -1
        self._active_word = -1
        self.setExtraSelections([])
        if not lines:
            self._building = False
            return

        block_fmt = QTextBlockFormat()
        block_fmt.setLineHeight(
            140, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
        )
        block_fmt.setBottomMargin(12)

        cursor = self.textCursor()
        for i, line in enumerate(lines):
            if i:
                cursor.insertBlock()
            cursor.setBlockFormat(block_fmt)
            line.block = cursor.blockNumber()
            line.doc_start = cursor.position()
            for j, word in enumerate(line.words):
                if j:
                    cursor.insertText(" ")
                word.doc_start = cursor.position()
                cursor.insertText(word.text)
                word.doc_end = cursor.position()
            line.doc_end = cursor.position()

        self.moveCursor(QTextCursor.MoveOperation.Start)
        self.verticalScrollBar().setValue(0)
        self._building = False

    # -- editing ----------------------------------------------------------

    def set_editing(self, on):
        """Toggle correction mode: clicks place a caret instead of seeking."""
        self._editing = bool(on) and bool(self._lines)
        self.setReadOnly(not self._editing)
        if self._editing:
            self._scroll.stop()   # auto-scroll would yank the caret away

    @property
    def editing(self):
        return self._editing

    def _resync(self):
        """Re-derive character ranges after an edit, keeping timings attached.

        Lines map one-to-one onto text blocks, so a block's current text is the
        source of truth for its cue; only the offsets and word texts move.
        """
        if self._building or not self._lines:
            return
        document = self.document()
        for line in self._lines:
            block = document.findBlockByNumber(line.block)
            if block.isValid():
                self._retokenize(line, block)

    def _retokenize(self, line, block):
        base, text = block.position(), block.text()
        line.doc_start = base
        line.doc_end = base + len(text)

        tokens = [
            (m.group(), base + m.start(), base + m.end())
            for m in re.finditer(r"\S+", text)
        ]
        if not tokens:
            line.words = []
            return

        if len(tokens) == len(line.words):
            # Same word count: a typo fix. Timings stay exactly as transcribed.
            for word, (word_text, start, end) in zip(line.words, tokens):
                word.text, word.doc_start, word.doc_end = word_text, start, end
            return

        # Words were added or removed, so the original one-to-one mapping is
        # gone. Spread the line's own span across the new words by length --
        # approximate, but it keeps click-to-seek and highlighting alive.
        span = max(line.end - line.start, MIN_CUE_DURATION)
        total = sum(len(word_text) for word_text, _, _ in tokens)
        words, clock = [], line.start
        for word_text, start, end in tokens:
            width = span * len(word_text) / total
            words.append(Word(word_text, clock, clock + width, start, end))
            clock += width
        line.words = words

    def cues(self):
        """Current transcript as export-ready cues, including any edits."""
        document = self.document()
        entries = []
        for line in self._lines:
            block = document.findBlockByNumber(line.block)
            entries.append(
                (line.start, line.end, block.text() if block.isValid() else line.text)
            )
        return build_cues(entries)

    def keyPressEvent(self, event):
        """Block anything that would split or merge a block.

        One block == one cue, so a stray Enter or a Backspace across a boundary
        would silently desync every line from its timestamps.
        """
        if not self._editing:
            super().keyPressEvent(event)
            return

        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return
        cursor = self.textCursor()
        if cursor.hasSelection():
            if "\u2029" in cursor.selectedText():   # spans a block boundary
                return
        elif key == Qt.Key.Key_Backspace and cursor.atBlockStart():
            return
        elif key == Qt.Key.Key_Delete and cursor.atBlockEnd():
            return
        super().keyPressEvent(event)

    def insertFromMimeData(self, source):
        """Paste as a single line -- pasted newlines would create new cues."""
        self.insertPlainText(" ".join(source.text().split()))

    # -- lookups ----------------------------------------------------------

    def line_at_time(self, seconds):
        """Index of the line playing at `seconds`, or None before the first one.

        Binary search for the last line starting at or before `seconds`. A short
        gap keeps that line active so the view does not flicker between lines,
        but past SILENCE_GAP nothing is highlighted -- holding the last line lit
        through an instrumental break is what reads as tracking drift.
        """
        lo, hi, found = 0, len(self._lines) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._lines[mid].start <= seconds:
                found = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if found is not None and seconds - self._lines[found].end > SILENCE_GAP:
            return None
        return found

    def word_in_line_at_time(self, line_index, seconds):
        """Index (within the line) of the word at `seconds`, or None."""
        words = self._lines[line_index].words
        lo, hi, found = 0, len(words) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if words[mid].start <= seconds:
                found = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if found is not None and seconds - words[found].end > SILENCE_GAP:
            return None
        return found

    def locate(self, pos):
        """Map a document offset to (line_index, word_index|None), or None."""
        for i, line in enumerate(self._lines):
            if line.doc_start <= pos <= line.doc_end:
                for j, word in enumerate(line.words):
                    if word.doc_start <= pos <= word.doc_end:
                        return i, j
                return i, None
        return None

    # -- highlighting -----------------------------------------------------

    def highlight_at(self, seconds):
        """Light up the line and word playing at `seconds`, scrolling if needed."""
        line_index = self.line_at_time(seconds) if self._lines else None
        if line_index is None:
            self._set_active(-1, -1)
            return
        word_index = self.word_in_line_at_time(line_index, seconds)
        self._set_active(line_index, -1 if word_index is None else word_index)

    def _set_active(self, line_index, word_index):
        if (line_index, word_index) == (self._active_line, self._active_word):
            return
        line_changed = line_index != self._active_line
        self._active_line, self._active_word = line_index, word_index

        if line_index < 0:
            self.setExtraSelections([])
            return

        line = self._lines[line_index]
        selections = [self._selection(line.doc_start, line.doc_end, self._line_fmt)]
        if word_index >= 0:
            word = line.words[word_index]
            # Appended last so the word paints over the line highlight.
            selections.append(
                self._selection(word.doc_start, word.doc_end, self._word_fmt)
            )
        self.setExtraSelections(selections)

        if line_changed:
            self._scroll_to_line(line_index)

    def _selection(self, start, end, fmt):
        cursor = self.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        selection = QTextEdit.ExtraSelection()
        selection.cursor = cursor
        selection.format = fmt
        return selection

    def _scroll_to_line(self, line_index, animate=True):
        """Glide the active line to the vertical centre of the viewport."""
        if self._editing or not (0 <= line_index < len(self._lines)):
            return
        block = self.document().findBlockByNumber(self._lines[line_index].block)
        if not block.isValid():
            return

        cursor = QTextCursor(block)
        rect = self.cursorRect(cursor)          # viewport coordinates
        bar = self.verticalScrollBar()
        target = int(
            bar.value() + rect.top() + rect.height() / 2 - self.viewport().height() / 2
        )
        target = max(bar.minimum(), min(bar.maximum(), target))

        self._scroll.stop()
        if not animate or abs(target - bar.value()) < 2:
            bar.setValue(target)
            return
        self._scroll.setStartValue(bar.value())
        self._scroll.setEndValue(target)
        self._scroll.start()

    # -- interaction ------------------------------------------------------

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if (event.button() != Qt.MouseButton.LeftButton
                or self._editing or not self._lines):
            return
        hit = self.locate(self.cursorForPosition(event.pos()).position())
        if hit is None:
            return
        line_index, word_index = hit
        line = self._lines[line_index]
        # Clicking the gap between words still seeks -- to the line's own start.
        target = line.words[word_index] if word_index is not None else line
        self._on_seek(target.start, getattr(target, "text", ""))

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._editing or not self._lines:
            return
        over = self.locate(self.cursorForPosition(event.pos()).position())
        self.viewport().setCursor(
            Qt.CursorShape.PointingHandCursor if over
            else Qt.CursorShape.IBeamCursor
        )


class TranscribeWorker(QObject):
    """Runs whisper-timestamped off the GUI thread.

    Lives in its own QThread; whisper/torch are imported inside `run` so the
    ~3s import cost lands here instead of delaying app startup.
    """

    # progress carries a STRINGS key plus its format params, so the status bar
    # renders in whatever locale is active when the signal arrives.
    progress = pyqtSignal(str, dict)
    finished = pyqtSignal(list)   # list[Line]
    failed = pyqtSignal(str)

    def __init__(self, path, model_name):
        super().__init__()
        self.path = path
        self.model_name = model_name

    def run(self):
        try:
            self.progress.emit("loading_whisper", {})
            import whisper_timestamped as whisper

            self.progress.emit("loading_model", {"model": self.model_name})
            model = whisper.load_model(self.model_name)

            # VAD needs its backend installed; without it we still transcribe,
            # just without the anti-hallucination gate.
            vad = VAD_METHOD
            try:
                from whisper_timestamped.transcribe import check_vad_method
                check_vad_method(vad)
            except Exception:
                self.progress.emit("vad_unavailable", {})
                vad = False

            self.progress.emit("transcribing", {"model": self.model_name})
            result = whisper.transcribe(
                model,
                self.path,
                language=None,   # Whisper always detects the spoken language itself
                vad=vad,
                temperature=TEMPERATURE_FALLBACK,
                # Music and noise make Whisper latch onto its own last output and
                # loop it; not feeding the previous text back breaks the cycle.
                condition_on_previous_text=False,
                # Drops zero-length words Whisper tacks onto segment ends -- the
                # main source of word-highlight drift.
                remove_empty_words=True,
                verbose=None,
            )

            lines = []
            for segment in result.get("segments", []):
                words = [
                    Word(w["text"], float(w["start"]), float(w["end"]))
                    for w in segment.get("words", [])
                ]
                if not words:       # segments can be empty on silence
                    continue
                line = Line(words=words, start=words[0].start, end=words[-1].end)
                if is_degenerate(line.text):
                    continue        # a loop Whisper spun over music or noise
                lines.append(line)
            self.finished.emit(collapse_repeats(lines))
        except Exception as exc:
            # Anything from a bad codec to an OOM lands here; the UI stays alive.
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class Studio(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Subtitle Studio")
        self.resize(1020, 720)

        self.media_path = None
        self._seeking = False   # suppress player updates while dragging the slider
        self._thread = None
        self._worker = None
        self._busy = False
        self.locale = DEFAULT_LOCALE
        self._status = ("ready", {})   # (key, params), re-rendered on locale change

        self._build_player()
        self._build_ui()
        self.retranslate()
        self._update_actions()

    # -- setup ------------------------------------------------------------

    def _build_player(self):
        self.audio = QAudioOutput()
        self.audio.setVolume(0.8)
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio)

        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_playback_state)
        self.player.errorOccurred.connect(self._on_player_error)

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_topbar())
        self.transcript = LyricsView(on_seek=self.seek_to)
        layout.addWidget(self.transcript, stretch=1)
        layout.addWidget(self._build_bottombar())

        self.setCentralWidget(root)
        self.setStyleSheet(STYLESHEET)

    def _build_topbar(self):
        bar = QFrame()
        bar.setObjectName("topBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(20, 14, 20, 14)
        row.setSpacing(10)

        self.open_btn = QPushButton()
        self.open_btn.clicked.connect(self.open_media)

        self.file_label = QLabel()
        self.file_label.setObjectName("fileLabel")

        self.font_down_btn = QPushButton("A−")
        self.font_down_btn.setObjectName("chip")
        self.font_down_btn.setFixedWidth(38)
        self.font_down_btn.clicked.connect(lambda: self.scale_font(-FONT_STEP))

        self.font_up_btn = QPushButton("A+")
        self.font_up_btn.setObjectName("chip")
        self.font_up_btn.setFixedWidth(38)
        self.font_up_btn.clicked.connect(lambda: self.scale_font(+FONT_STEP))

        self.edit_btn = QPushButton()
        self.edit_btn.setCheckable(True)
        self.edit_btn.clicked.connect(self.toggle_edit)

        self.model_label = QLabel()
        self.model_box = QComboBox()
        self.model_box.addItems(MODELS)
        self.model_box.setCurrentText("tiny")

        self.language_label = QLabel()
        self.language_box = QComboBox()
        for label, code in LOCALES:
            self.language_box.addItem(label, code)
        self.language_box.setCurrentIndex([c for _, c in LOCALES].index(DEFAULT_LOCALE))
        self.language_box.currentIndexChanged.connect(self._on_locale_changed)

        self.transcribe_btn = QPushButton()
        self.transcribe_btn.setObjectName("primary")
        self.transcribe_btn.clicked.connect(self.transcribe)

        row.addWidget(self.open_btn)
        row.addWidget(self.file_label, stretch=1)
        row.addWidget(self.font_down_btn)
        row.addWidget(self.font_up_btn)
        row.addSpacing(6)
        row.addWidget(self.edit_btn)
        row.addSpacing(6)
        row.addWidget(self.model_label)
        row.addWidget(self.model_box)
        row.addSpacing(6)
        row.addWidget(self.language_label)
        row.addWidget(self.language_box)
        row.addSpacing(6)
        row.addWidget(self.transcribe_btn)
        return bar

    def _build_bottombar(self):
        bar = QFrame()
        bar.setObjectName("bottomBar")
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(20, 14, 20, 14)
        outer.setSpacing(12)

        # transport
        transport = QHBoxLayout()
        transport.setSpacing(14)

        self.play_btn = QPushButton("▶")
        self.play_btn.setObjectName("play")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.clicked.connect(self.toggle_play)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderPressed.connect(self._on_seek_start)
        self.position_slider.sliderReleased.connect(self._on_seek_end)
        self.position_slider.sliderMoved.connect(self._on_seek_move)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("timeLabel")

        self.volume_label = QLabel()
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(96)
        self.volume_slider.valueChanged.connect(lambda v: self.audio.setVolume(v / 100))

        transport.addWidget(self.play_btn)
        transport.addWidget(self.position_slider, stretch=1)
        transport.addWidget(self.time_label)
        transport.addSpacing(8)
        transport.addWidget(self.volume_label)
        transport.addWidget(self.volume_slider)

        # status + exports
        footer = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setObjectName("status")
        self.srt_btn = QPushButton()
        self.srt_btn.clicked.connect(lambda: self.export("srt"))
        self.vtt_btn = QPushButton()
        self.vtt_btn.clicked.connect(lambda: self.export("vtt"))
        footer.addWidget(self.status_label, stretch=1)
        footer.addWidget(self.srt_btn)
        footer.addWidget(self.vtt_btn)

        outer.addLayout(transport)
        outer.addLayout(footer)
        return bar

    # -- i18n -------------------------------------------------------------

    def tr(self, key, **params):
        """Render a UI string in the active locale, falling back to English."""
        template = STRINGS[self.locale].get(key) or STRINGS[DEFAULT_LOCALE][key]
        return template.format(**params)

    def _on_locale_changed(self):
        self.locale = self.language_box.currentData()
        self.retranslate()

    def retranslate(self):
        """Re-render every user-visible string in the active locale.

        Called once at startup and again on each locale change, so there is a
        single place where UI text is set -- widgets are built without text.
        """
        self.open_btn.setText(self.tr("open_media"))
        self.model_label.setText(self.tr("model"))
        self.language_label.setText(self.tr("language"))
        self.volume_label.setText(self.tr("volume"))
        self.srt_btn.setText(self.tr("export_srt"))
        self.vtt_btn.setText(self.tr("export_vtt"))
        self.edit_btn.setText(self.tr("edit"))

        self.model_box.setToolTip(self.tr("model_tip"))
        self.language_box.setToolTip(self.tr("language_tip"))
        self.font_down_btn.setToolTip(self.tr("font_smaller_tip"))
        self.font_up_btn.setToolTip(self.tr("font_larger_tip"))
        self.edit_btn.setToolTip(self.tr("edit_tip"))
        self.transcript.setPlaceholderText(self.tr("placeholder"))

        # A loaded filename is data, not UI text -- leave it as-is.
        if self.media_path is None:
            self.file_label.setText(self.tr("no_file"))

        key, params = self._status
        self.status_label.setText(self.tr(key, **params))

        # Owns the busy/idle wording of the Transcribe button.
        self._update_actions()

    def set_status(self, key, **params):
        """Record the status as (key, params) so it survives a locale switch."""
        self._status = (key, params)
        self.status_label.setText(self.tr(key, **params))
        if key != "failed":
            self.status_label.setToolTip("")

    # -- actions ----------------------------------------------------------

    def open_media(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("open_media"), "", MEDIA_FILTER)
        if not path:
            return

        self.media_path = path
        self.player.setSource(QUrl.fromLocalFile(path))
        self._leave_edit_mode()
        self.transcript.set_lines([])
        self.file_label.setText(path.rsplit("/", 1)[-1])
        self.set_status("loaded", name=path.rsplit("/", 1)[-1])
        self._update_actions()

    def toggle_edit(self):
        """Correction mode. Off, clicks seek; on, they place a caret."""
        on = self.edit_btn.isChecked()
        self.transcript.set_editing(on)
        self.set_status("edit_on" if on else "edit_off")
        if not on:
            # Re-sync the highlight to wherever playback actually is.
            self.transcript.highlight_at(self.player.position() / 1000)

    def export(self, fmt):
        """Write the transcript -- edits included -- as SRT or WebVTT."""
        cues = self.transcript.cues()
        if not cues:
            return

        source = self.media_path or "transcript"
        stem = os.path.splitext(os.path.basename(source))[0]
        suggested = os.path.join(os.path.dirname(source), f"{stem}.{fmt}")
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("save_srt" if fmt == "srt" else "save_vtt"),
            suggested,
            f"{fmt.upper()} (*.{fmt});;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(f".{fmt}"):
            path += f".{fmt}"

        body = build_srt(cues) if fmt == "srt" else build_vtt(cues)
        try:
            # newline="" keeps the "\n" endings both formats expect on any host.
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(body)
        except OSError as exc:
            self.set_status("export_failed", error=str(exc))
            return
        self.set_status("exported", count=len(cues), name=os.path.basename(path))

    def scale_font(self, delta):
        self.transcript.apply_font_size(self.transcript.font_px + delta)
        self._update_actions()

    def transcribe(self):
        if self._busy or not self.media_path:
            return

        self._busy = True
        self._leave_edit_mode()
        self.transcript.set_lines([])
        self._update_actions()

        self._thread = QThread(self)
        self._worker = TranscribeWorker(self.media_path, self.model_box.currentText())
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda key, params: self.set_status(key, **params))
        self._worker.finished.connect(self._on_transcribed)
        self._worker.failed.connect(self._on_transcribe_failed)

        # Both outcomes stop the thread; deleteLater keeps the objects alive
        # until the event loop has finished dispatching their signals.
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_done)

        self._thread.start()

    def _leave_edit_mode(self):
        self.edit_btn.setChecked(False)
        self.transcript.set_editing(False)

    def _on_transcribed(self, lines):
        self.transcript.set_lines(lines)
        if lines:
            self.set_status(
                "transcribed",
                count=sum(len(l.words) for l in lines),
                lines=len(lines),
                model=self.model_box.currentText(),
            )
        else:
            self.set_status("no_speech")

    def _on_transcribe_failed(self, message):
        self.set_status("failed", error=condense_error(message))
        self.status_label.setToolTip(message)

    def _on_thread_done(self):
        self._busy = False
        self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._update_actions()

    def seek_to(self, seconds, label=""):
        self.player.setPosition(int(seconds * 1000))
        self.set_status("jumped", secs=seconds, word=label)

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _update_actions(self):
        has_media = self.media_path is not None
        has_lines = bool(self.transcript.lines)
        self.open_btn.setEnabled(not self._busy)
        self.model_box.setEnabled(not self._busy)
        self.transcribe_btn.setEnabled(has_media and not self._busy)
        self.transcribe_btn.setText(
            self.tr("transcribing_btn" if self._busy else "transcribe")
        )
        self.play_btn.setEnabled(has_media)
        self.position_slider.setEnabled(has_media)
        self.srt_btn.setEnabled(has_lines)
        self.vtt_btn.setEnabled(has_lines)
        self.edit_btn.setEnabled(has_lines and not self._busy)
        self.font_down_btn.setEnabled(self.transcript.font_px > FONT_MIN)
        self.font_up_btn.setEnabled(self.transcript.font_px < FONT_MAX)

    # -- player signals ---------------------------------------------------

    def _on_position(self, ms):
        if not self._seeking:
            self.position_slider.setValue(ms)
        self._refresh_time(ms)
        if self.transcript.lines:
            self.transcript.highlight_at(ms / 1000)

    def _on_duration(self, ms):
        self.position_slider.setRange(0, ms)
        self._refresh_time(self.player.position())

    def _on_playback_state(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_btn.setText("⏸" if playing else "▶")

    def _on_player_error(self, error, message):
        if error != QMediaPlayer.Error.NoError:
            self.set_status("playback_error", error=message or error.name)

    def _refresh_time(self, ms):
        self.time_label.setText(
            f"{format_time(ms)} / {format_time(self.player.duration())}"
        )

    def _on_seek_start(self):
        self._seeking = True

    def _on_seek_move(self, ms):
        self._refresh_time(ms)

    def _on_seek_end(self):
        self._seeking = False
        self.player.setPosition(self.position_slider.value())

    # -- shutdown ---------------------------------------------------------

    def closeEvent(self, event):
        """Let a running transcription finish unwinding before we tear down.

        Destroying a QThread that is still running aborts the process, and a
        Whisper pass cannot be interrupted mid-inference.
        """
        self.player.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(15000)
        event.accept()


STYLESHEET = """
QWidget { background: #121212; color: #ffffff; font-size: 13px; }

QFrame#topBar    { background: #181818; border-bottom: 1px solid #282828; }
QFrame#bottomBar { background: #181818; border-top: 1px solid #282828; }

QLabel { color: #b3b3b3; background: transparent; }
QLabel#fileLabel { color: #ffffff; font-weight: 600; background: transparent; }
QLabel#status    { color: #7a7a7a; background: transparent; }
QLabel#timeLabel {
    color: #b3b3b3; background: transparent;
    font-family: Menlo, monospace;   /* Qt has no 'ui-monospace' generic */
}

QPushButton {
    background: #1e1e1e; border: 1px solid #333333; border-radius: 9px;
    padding: 8px 16px; color: #ffffff; font-weight: 500;
}
QPushButton:hover:enabled  { background: #2a2a2a; border-color: #454545; }
QPushButton:pressed:enabled { background: #171717; }
QPushButton:disabled { color: #5a5a5a; background: #1a1a1a; border-color: #262626; }

QPushButton#primary {
    background: #1ed760; border-color: #1ed760; color: #08130c; font-weight: 700;
}
QPushButton#primary:hover:enabled  { background: #24e96b; border-color: #24e96b; }
QPushButton#primary:pressed:enabled { background: #17b850; }
QPushButton#primary:disabled { background: #1a1a1a; border-color: #262626; color: #5a5a5a; }

QPushButton#chip { padding: 8px 0; font-weight: 700; border-radius: 8px; }

QPushButton:checked:enabled {
    background: #1ed760; border-color: #1ed760; color: #08130c; font-weight: 700;
}
QPushButton:checked:hover:enabled { background: #24e96b; border-color: #24e96b; }

QPushButton#play {
    background: #ffffff; border: none; border-radius: 20px;
    color: #000000; font-size: 15px; padding: 0;
}
QPushButton#play:hover:enabled { background: #f0f0f0; }
QPushButton#play:disabled { background: #2a2a2a; color: #5a5a5a; }

QComboBox {
    background: #1e1e1e; border: 1px solid #333333; border-radius: 9px;
    padding: 7px 10px; min-width: 104px; color: #ffffff;
}
QComboBox:hover:enabled { border-color: #454545; }
QComboBox:disabled { color: #5a5a5a; background: #1a1a1a; }
QComboBox QAbstractItemView {
    background: #1e1e1e; border: 1px solid #333333; border-radius: 8px;
    padding: 4px; outline: none;
    selection-background-color: #1ed760; selection-color: #08130c;
}

QSlider::groove:horizontal { height: 4px; background: #404040; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #1ed760; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #ffffff; width: 12px; height: 12px;
    margin: -4px 0; border-radius: 6px;
}
QSlider::groove:horizontal:disabled { background: #262626; }
QSlider::sub-page:horizontal:disabled { background: #3a3a3a; }
QSlider::handle:horizontal:disabled { background: #4a4a4a; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical {
    background: #3a3a3a; border-radius: 5px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #4d4d4d; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

QToolTip {
    background: #282828; color: #ffffff;
    border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px 8px;
}
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Whisper Subtitle Studio")
    window = Studio()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
