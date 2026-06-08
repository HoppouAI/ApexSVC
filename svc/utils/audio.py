"""Audio I/O and voiced-area detection. Trimmed from upstream
utils/spectrogram.py. Only the helpers actually needed for inference."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import resampy
import soundfile as sf


def load_wav(wav_path: str | Path, sr: int = 24000) -> tuple[np.ndarray, int]:
    """Read a wav, mix to mono if needed, resample to `sr`, and peak-normalize
    to keep |x| <= 1 (only when the file actually clipped that bound).

    Returns (waveform float32 in [-1, 1], sr).
    """
    wav, fs = sf.read(str(wav_path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return wav.astype(np.float32), fs


def stft_mag(x: np.ndarray, n_fft: int = 2048, hop: int = 480,
             win: int = 2048, window: str = "hann") -> np.ndarray:
    """Magnitude STFT, shape (T, F)."""
    spec = librosa.stft(x.astype(np.float32), n_fft=n_fft, hop_length=hop,
                        win_length=win, window=window, center=True, pad_mode="reflect")
    return np.abs(spec).T


def extract_voiced_area(wav_path: str | Path, n_fft: int = 2048, hop_size: int = 480,
                        hi_freq: int = 1000, energy_thres: float = 0.5) -> np.ndarray:
    """Per-frame boolean mask of "voiced" regions using sub-1kHz band energy.
    Frame rate = sr / hop_size. With sr=24kHz, hop=480 -> 50 Hz (matches WavLM).
    Lifted from upstream `utils/spectrogram.py::extract_voiced_area`.
    """
    x, sr = load_wav(wav_path, sr=24000)
    mag = stft_mag(x, n_fft=n_fft, hop=hop_size)
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    filtered = mag[:, freq_axis <= hi_freq]
    loudness = np.log10(np.mean(np.power(10.0, filtered / 10.0), axis=-1) + 1e-5)
    return loudness >= energy_thres


def a_weighted_loudness(x: np.ndarray, sr: int, n_fft: int = 1024,
                        hop: int = 240) -> np.ndarray:
    """A-weighted log loudness per frame. Used by training-time conditioning;
    kept available for future use even though inference does not need it."""
    mag = stft_mag(x, n_fft=n_fft, hop=hop)
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    perceptual = librosa.perceptual_weighting(mag ** 2, freq_axis, ref=1.0)
    return np.log10(np.mean(np.power(10.0, perceptual / 10.0), axis=-1) + 1e-5)
