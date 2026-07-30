"""Regenerate the demo page's Baselines-vs-PP-DG transcripts with WhisperX large-v3.

The comparison clips (static/demo/<CODE>/<model>.mp3) are dual-channel:

    left  channel = user simulator (the shared interlocutor)
    right channel = AI model under test (moshi / pp / ppft)

so speaker identity is known exactly and must never be guessed by diarization.
Each channel is split out, transcribed independently with WhisperX large-v3, and
force-aligned (wav2vec2) for word-level timestamps. Words are then regrouped into
utterances on pauses, the two channels are merged on the shared clip timeline, and
the result is written as window.CMP_TRANSCRIPTS.

Output: static/js/comparison_transcripts.js  (+ .json sidecar for inspection)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SCENARIOS = ["TEA", "PLN", "INT", "NEG", "PER", "SOC"]
MODELS = ["moshi", "pp", "ppft"]

# channel index -> speaker label used by index.html's renderTranscript()
CHANNEL_SPEAKER = {0: "Human", 1: "AI Assistant"}

SR = 16000
# Split an utterance when the speaker pauses longer than this (seconds).
PAUSE_GAP = 0.6
# ...and hard-split anything longer than this so bubbles stay readable.
MAX_UTT_SEC = 14.0
# Hallucination guard: WhisperX can emit text over near-silence. Drop a segment
# whose own channel is quieter than this RMS (roughly -46 dBFS) across its span.
MIN_RMS = 0.005
# Coverage audit: a channel is "speaking" in a 0.1 s frame above this RMS. Any
# speaking stretch longer than MIN_GAP_REPORT with no transcribed words over it is
# reported as a miss, so silent ASR drop-outs cannot slip into the page unnoticed.
FRAME = 0.1
ACTIVE_RMS = 0.01
MIN_GAP_REPORT = 0.8


def split_channels(mp3: Path, outdir: Path) -> list[Path]:
    """Decode mp3 to one 16 kHz mono wav per channel (left first)."""
    left, right = outdir / "ch0.wav", outdir / "ch1.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
            "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[l][r]",
            "-map", "[l]", "-ar", str(SR), "-ac", "1", str(left),
            "-map", "[r]", "-ar", str(SR), "-ac", "1", str(right),
        ],
        check=True,
    )
    return [left, right]


def words_from(aligned: dict) -> list[dict]:
    """Flatten aligned segments to words, filling in any missing timestamps."""
    words: list[dict] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            if not text:
                continue
            start, end = w.get("start"), w.get("end")
            if start is None or end is None:
                # Unaligned word (usually pure punctuation or a number token):
                # glue it onto the previous word so no text is lost.
                if words:
                    words[-1]["text"] += ("" if text[0] in ",.?!'" else " ") + text
                continue
            words.append({"text": text, "start": float(start), "end": float(end)})
    return words


def group_utterances(words: list[dict]) -> list[dict]:
    """Regroup words into utterances on pauses / length, keeping true timestamps."""
    utts: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if not cur:
            return
        utts.append({
            "text": " ".join(w["text"] for w in cur),
            "start": cur[0]["start"],
            "end": cur[-1]["end"],
        })
        cur.clear()

    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            too_long = w["end"] - cur[0]["start"] > MAX_UTT_SEC
            ends_sentence = cur[-1]["text"][-1:] in ".?!"
            if gap > PAUSE_GAP or too_long or (ends_sentence and gap > 0.25):
                flush()
        cur.append(w)
    flush()

    # Tidy the spacing that word-level joins introduce around punctuation.
    for u in utts:
        for p in (",", ".", "?", "!", "'", "%"):
            u["text"] = u["text"].replace(" " + p, p)
        u["text"] = u["text"].replace("$ ", "$").strip()
    return utts


def rms(audio: np.ndarray, start: float, end: float) -> float:
    a, b = int(max(0.0, start) * SR), int(max(0.0, end) * SR)
    chunk = audio[a:b]
    return float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0


def active_spans(audio: np.ndarray) -> list[tuple[float, float]]:
    """Energy-based speech spans, used only to audit ASR coverage."""
    n = int(FRAME * SR)
    frames = [audio[i:i + n] for i in range(0, len(audio) - n + 1, n)]
    hot = [float(np.sqrt(np.mean(f ** 2))) > ACTIVE_RMS for f in frames]
    spans: list[list[float]] = []
    for i, is_hot in enumerate(hot):
        if not is_hot:
            continue
        t0, t1 = i * FRAME, (i + 1) * FRAME
        if spans and t0 - spans[-1][1] <= 0.3:
            spans[-1][1] = t1
        else:
            spans.append([t0, t1])
    return [(a, b) for a, b in spans]


def uncovered(audio: np.ndarray, utts: list[dict]) -> list[tuple[float, float]]:
    """Speaking stretches with no transcribed words over them."""
    misses = []
    for a, b in active_spans(audio):
        cur = a
        for u in sorted(utts, key=lambda u: u["start"]):
            if u["end"] <= cur or u["start"] >= b:
                continue
            if u["start"] > cur and u["start"] - cur >= MIN_GAP_REPORT:
                misses.append((cur, u["start"]))
            cur = max(cur, u["end"])
        if b - cur >= MIN_GAP_REPORT:
            misses.append((cur, b))
    return misses


def transcribe_channel(wav: Path, model, align_model, align_meta, device: str):
    audio, sr = sf.read(str(wav), dtype="float32")
    assert sr == SR, sr
    if float(np.sqrt(np.mean(audio ** 2))) < MIN_RMS:
        return [], []  # silent channel: nothing to transcribe

    import whisperx

    result = model.transcribe(audio, batch_size=8, language="en")
    if not result.get("segments"):
        return [], uncovered(audio, [])
    aligned = whisperx.align(
        result["segments"], align_model, align_meta, audio, device,
        return_char_alignments=False,
    )
    utts = group_utterances(words_from(aligned))
    dur = len(audio) / SR
    kept = []
    for u in utts:
        if u["start"] >= dur:
            continue
        u["end"] = min(u["end"], dur)
        if rms(audio, u["start"], u["end"]) < MIN_RMS:
            continue  # hallucinated over silence
        kept.append(u)
    return kept, uncovered(audio, kept)


def parse_only(spec: str | None) -> set[tuple[str, str]] | None:
    """"TEA:ppft,PLN/moshi" -> {("TEA","ppft"), ("PLN","moshi")}."""
    if not spec:
        return None
    out = set()
    for item in spec.split(","):
        item = item.strip().replace("/", ":")
        if not item:
            continue
        code, _, mkey = item.partition(":")
        code, mkey = code.upper(), mkey.lower()
        if code not in SCENARIOS or mkey not in MODELS:
            raise SystemExit(f"--only: bad cell {item!r} (expect <{'|'.join(SCENARIOS)}>:<{'|'.join(MODELS)}>)")
        out.add((code, mkey))
    return out or None


def load_existing(site: Path) -> dict[str, dict[str, list[dict]]]:
    """Previously generated transcripts, so --only can leave other cells untouched."""
    sidecar = Path(__file__).resolve().parent / "comparison_transcripts.review.json"
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    js = site / "static" / "js" / "comparison_transcripts.js"
    if js.is_file():
        text = js.read_text(encoding="utf-8")
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    # the repo root, i.e. the parent of tools/; override to run against another checkout
    ap.add_argument("--site", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vad", default="pyannote", choices=["pyannote", "silero"])
    ap.add_argument(
        "--only",
        help="Re-transcribe just these cells, e.g. 'PER:moshi,PLN:ppft'. Other cells keep "
             "their previously generated transcripts, so swapping one clip does not churn "
             "the other 17 through a nondeterministic re-run.",
    )
    args = ap.parse_args()

    import whisperx

    site = Path(args.site)
    only = parse_only(args.only)
    if only:
        print(f"[asr] only re-transcribing: {', '.join(f'{c}/{m}' for c, m in sorted(only))}", flush=True)
    compute_type = "float16" if args.device == "cuda" else "int8"
    print(f"[asr] loading whisper {args.model} ({compute_type})", flush=True)
    asr = whisperx.load_model(
        args.model, args.device, compute_type=compute_type, language="en",
        vad_method=args.vad,
        asr_options={
            # Temperature fallback matters here: a single greedy pass silently
            # returns nothing on the hardest overlapped chunks.
            "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "condition_on_previous_text": False,
            # Style hint so casing/punctuation stay consistent across chunks.
            "initial_prompt": "Hello, how can I help you today? Sure, that works for me.",
        },
    )
    print("[asr] loading wav2vec2 alignment model", flush=True)
    align_model, align_meta = whisperx.load_align_model(language_code="en", device=args.device)

    out: dict[str, dict[str, list[dict]]] = load_existing(site) if only else {}
    for code in SCENARIOS:
        out.setdefault(code, {})
        for mkey in MODELS:
            if only and (code, mkey) not in only:
                out[code].setdefault(mkey, [])
                continue
            mp3 = site / "static" / "demo" / code / f"{mkey}.mp3"
            if not mp3.exists():
                print(f"[warn] missing {mp3}", file=sys.stderr)
                out[code][mkey] = []
                continue
            segs: list[dict] = []
            gaps: list[str] = []
            with tempfile.TemporaryDirectory() as td:
                for ch, wav in enumerate(split_channels(mp3, Path(td))):
                    utts, misses = transcribe_channel(wav, asr, align_model, align_meta, args.device)
                    for u in utts:
                        text = u["text"]
                        if text[:1].islower():
                            text = text[0].upper() + text[1:]
                        segs.append({
                            "speaker": CHANNEL_SPEAKER[ch],
                            "text": text,
                            "start": round(u["start"], 1),
                            "end": round(u["end"], 1),
                        })
                    gaps += [f"{CHANNEL_SPEAKER[ch]} {a:.1f}-{b:.1f}" for a, b in misses]
            segs.sort(key=lambda s: (s["start"], s["end"]))
            out[code][mkey] = segs
            note = f"  MISSED: {', '.join(gaps)}" if gaps else ""
            print(f"[asr] {code}/{mkey}: {len(segs)} utterances{note}", flush=True)

    js = site / "static" / "js" / "comparison_transcripts.js"
    payload = json.dumps(out, ensure_ascii=False)
    js.write_text(
        "// Generated by tools/build_comparison_transcripts.py: WhisperX large-v3,\n"
        "// transcribed per channel (left = user simulator, right = AI model) so that\n"
        "// speaker labels come from the channel, never from diarization guesswork.\n"
        "window.CMP_TRANSCRIPTS = " + payload + ";\n",
        encoding="utf-8",
    )
    # Readable sidecar for review; kept out of static/ so the page ships only the .js.
    sidecar = Path(__file__).resolve().parent / "comparison_transcripts.review.json"
    sidecar.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[asr] wrote {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
