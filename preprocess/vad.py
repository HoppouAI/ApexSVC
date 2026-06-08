"""Silero-VAD based silence trimming for reference audio.

The reference clip is often live recording / streaming capture with long
silences, breaths, or non-vocal lead-ins. Including those frames in the kNN
matching pool dilutes the timbre estimate. We use silero-vad to keep only
speech/sing-active segments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from svc.utils.audio import load_wav

_VAD_SR = 16000


def _load_vad():
    # lazy import / cache; silero-vad ships a tiny torch model
    from silero_vad import load_silero_vad
    return load_silero_vad()


_VAD_MODEL = None


def _model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        _VAD_MODEL = _load_vad()
    return _VAD_MODEL


def trim_silence(wav: np.ndarray, sr: int, *, min_speech_ms: int = 500,
                 min_silence_ms: int = 200, speech_pad_ms: int = 100) -> np.ndarray:
    """Drop silent / non-vocal regions from `wav`. Returns a new concatenated
    waveform at the same sample rate. Operates internally at 16 kHz which is
    what silero-vad expects.

    Falls through unchanged (no trim) if the model returns no speech regions,
    e.g. if the audio is mostly hummed pitch that the VAD does not flag.
    """
    from silero_vad import get_speech_timestamps

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != _VAD_SR:
        import resampy
        vad_wav = resampy.resample(wav, sr, _VAD_SR, axis=0)
    else:
        vad_wav = wav

    vad_tensor = torch.from_numpy(vad_wav.astype(np.float32))
    timestamps = get_speech_timestamps(
        vad_tensor, _model(), sampling_rate=_VAD_SR,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    if not timestamps:
        return wav

    # convert 16k indices back to source-sr indices and concatenate
    pieces: list[np.ndarray] = []
    for seg in timestamps:
        start = int(seg["start"] / _VAD_SR * sr)
        end = int(seg["end"] / _VAD_SR * sr)
        pieces.append(wav[start:end])
    return np.concatenate(pieces, axis=0) if pieces else wav


def trim_silence_file(in_path: str | Path, sr: int = 16000) -> np.ndarray:
    """Convenience: load + trim + return waveform at `sr`."""
    wav, fs = load_wav(in_path, sr=sr)
    return trim_silence(wav, sr=fs)
