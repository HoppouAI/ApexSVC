"""Per-frame F0 extraction.

The upstream NeuCoSVC2 model was trained with the median of PYIN + ParselMouth
+ REAPER. REAPER is painful on Windows so we drop it. We default to a modern
singing-grade neural F0 tracker (FCPE, RMVPE family) instead of classical
autocorrelation: vibrato + breaths + soft consonants make PYIN/ParselMouth
produce octave jumps and false-voiced frames, which show up as the RVC-style
buzzy pitch artifacts in the converted audio.

Available methods:
  "fcpe"   -- TorchFCPE neural pitch tracker (default, GPU, singing-grade)
  "praat"  -- ParselMouth (fast classical, fine for clean speech)
  "pyin"   -- librosa PYIN (slow CPU viterbi)
  "median" -- median of PYIN + ParselMouth (closest to upstream NeuCoSVC2)

All frame rates are at 100 Hz (frame_period = 10 ms) on 24 kHz audio, matching
what the NSF vocoder was trained on (hop_size = 240 samples = 10 ms).
"""

from __future__ import annotations

import librosa
import numpy as np
import parselmouth
import torch

from ..utils.audio import load_wav

DEFAULT_SR = 24000
DEFAULT_FRAME_PERIOD = 0.01  # seconds
FMIN_HZ = librosa.note_to_hz("C2")
FMAX_HZ = librosa.note_to_hz("C7")

# Lazy singleton: FCPE bundled model. ~30 MB, downloads on first use.
_fcpe_model = None
_fcpe_device = None


def _get_fcpe(device: str | torch.device | None = None):
    global _fcpe_model, _fcpe_device
    from torchfcpe import spawn_bundled_infer_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if _fcpe_model is None or _fcpe_device != device:
        _fcpe_model = spawn_bundled_infer_model(device=str(device))
        _fcpe_device = device
    return _fcpe_model


def _fcpe_f0(wav: np.ndarray, sr: int, target_len: int,
             device: str | torch.device | None = None) -> np.ndarray:
    """FCPE pitch inference. Returns Hz, 0 for unvoiced, length = target_len."""
    model = _get_fcpe(device)
    dev = _fcpe_device
    # FCPE expects 16k mono shaped (1, T, 1). Resample to 16k if needed (cheap).
    if sr != 16000:
        wav16 = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    else:
        wav16 = wav.astype(np.float32)
    x = torch.from_numpy(wav16).float().unsqueeze(0).unsqueeze(-1).to(dev)
    with torch.inference_mode():
        f0 = model.infer(
            x, sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
            f0_min=float(FMIN_HZ),
            f0_max=float(FMAX_HZ),
            interp_uv=False,
            output_interp_target_length=int(target_len),
        )
    f0 = f0.squeeze().detach().cpu().numpy().astype(np.float64)
    # FCPE returns 0 for unvoiced already (interp_uv=False).
    return f0


def _pyin_f0(wav: np.ndarray, sr: int, frame_period: float) -> np.ndarray:
    frame_length = int(sr * frame_period * 4)
    f0, *_ = librosa.pyin(wav, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sr,
                          frame_length=frame_length)
    return np.where(np.isnan(f0), 0.0, f0).astype(np.float64)


def _parselmouth_f0(wav: np.ndarray, sr: int, frame_period: float) -> np.ndarray:
    sound = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=frame_period, pitch_floor=65.0, pitch_ceiling=1000.0)
    return pitch.selected_array["frequency"].astype(np.float64)


def _pad_to(arrays: list[np.ndarray], target_len: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for arr in arrays:
        if len(arr) >= target_len:
            out.append(arr[:target_len])
        else:
            pad = target_len - len(arr)
            left = pad // 2
            right = pad - left
            out.append(np.pad(arr, (left, right), mode="edge"))
    return out


def compute_f0(wav_path: str, sr: int = DEFAULT_SR,
               frame_period: float = DEFAULT_FRAME_PERIOD,
               method: str = "fcpe",
               device: str | torch.device | None = None) -> np.ndarray:
    """F0 contour from a wav file. Returns (T,) in Hz, 0 for unvoiced frames.

    method:
      "fcpe"   -- TorchFCPE neural tracker (default, singing-grade)
      "praat"  -- ParselMouth only (fast classical)
      "pyin"   -- librosa PYIN only (CPU viterbi, slow but robust)
      "median" -- median of PYIN + ParselMouth (closest to upstream NeuCoSVC2)
    """
    wav, fs = load_wav(wav_path, sr=sr)
    target_len = wav.shape[0] // int(frame_period * fs) + 1

    if method == "fcpe":
        return _fcpe_f0(wav, sr=fs, target_len=target_len, device=device)
    if method == "praat":
        out = _parselmouth_f0(wav, sr=fs, frame_period=frame_period)
        return _pad_to([out], target_len)[0]
    if method == "pyin":
        out = _pyin_f0(wav, sr=fs, frame_period=frame_period)
        return _pad_to([out], target_len)[0]
    if method == "median":
        candidates = [
            _pyin_f0(wav, sr=fs, frame_period=frame_period),
            _parselmouth_f0(wav, sr=fs, frame_period=frame_period),
        ]
        candidates = _pad_to(candidates, target_len)
        return np.median(np.stack(candidates, axis=0), axis=0)
    raise ValueError(f"unknown F0 method: {method!r}")


def coarse_f0(f0: np.ndarray, f0_bins: int = 256, f0_min: float = 65.0,
              f0_max: float = 1000.0) -> np.ndarray:
    """Bucket continuous F0 (Hz) into integer bin indices on a mel scale, to
    feed the `pitch_emb` embedding in the vocoder. Reproduces upstream
    `coarse_f0`. Returns int64 array shape (T,) with values in [1, f0_bins-1].
    Unvoiced (F0==0) frames map to bin 1."""
    f0_mel_min = 1127.0 * np.log(1.0 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1.0 + f0_max / 700.0)
    f0_mel = 1127.0 * np.log(1.0 + np.asarray(f0, dtype=np.float64) / 700.0)
    voiced = f0_mel > 0
    f0_mel[voiced] = (f0_mel[voiced] - f0_mel_min) * (f0_bins - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bins - 1] = f0_bins - 1
    return np.rint(f0_mel).astype(np.int64)


def compute_pitch_shift_factor(source_f0: np.ndarray, ref_wav_path: str,
                               speech_enroll: bool = False) -> float:
    """Compute a multiplicative F0 shift factor from source -> reference so the
    converted singing sits in the reference singer's typical pitch range.

    Uses median voiced F0 (way more robust than mean against long sustained
    notes, breaths, octave errors), then snaps the resulting shift to whole
    semitones and folds it into the closest octave so a source/ref in the same
    register doesn't get unnecessarily transposed. With `speech_enroll=True`
    the snapped shift gets +3 semitones to bridge speaking -> singing range.
    """
    nonzero = source_f0[np.nonzero(source_f0)]
    if nonzero.size == 0:
        return 1.0
    source_med = float(np.median(nonzero))

    ref_wav, ref_fs = load_wav(ref_wav_path, sr=DEFAULT_SR)
    ref_f0 = _parselmouth_f0(ref_wav, sr=ref_fs, frame_period=DEFAULT_FRAME_PERIOD)
    ref_nonzero = ref_f0[np.nonzero(ref_f0)]
    if ref_nonzero.size == 0:
        return 1.0
    ref_med = float(np.median(ref_nonzero))

    # raw shift in semitones (continuous)
    raw_st = 12.0 * np.log2(ref_med / source_med)
    # fold into [-6, +6) so cross-octave refs don't pull the song an octave
    # away from the original key. wraps via modulo on the semitone axis.
    folded_st = ((raw_st + 6.0) % 12.0) - 6.0
    # snap to nearest whole semitone so output lands in-key instead of between
    snapped_st = float(np.round(folded_st))
    if speech_enroll:
        snapped_st += 3.0  # speaking voices sit ~3-4 st below singing
    return semitone_factor(snapped_st)


def semitone_factor(semitones: float) -> float:
    """Convert a semitone offset to a multiplicative F0 factor."""
    return float(2.0 ** (semitones / 12.0))
